"""Fine-tune a CRNN on word crops from several sources.

Sources mix freely:

* full-image datasets with word polygons (``--ic15``, ``--totaltext``):
  every legible care word is rectified from its full image with
  :func:`torchocr.ops.crop_quads` -- the path ``OCRPipeline`` uses at
  inference -- after jittering the quad's corners so the model tolerates
  imprecise detector boxes;
* pre-cropped words in the deep-text-recognition-benchmark LMDB layout
  (``--lmdb``, repeatable), resized by the recognizer's preset.

Words with characters outside the model's charset are skipped. Example::

    python references/recognition/train.py --ic15 ~/data/icdar2015 \
        --totaltext ~/data/totaltext --lmdb ~/data/mlt2019/Latin \
        --init CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2 --output runs/crnn_mix

Writes ``last.pth`` and a resumable ``checkpoint.pth`` (``--resume``) every epoch, and logs test word accuracy on ground-truth
quads every ``--eval-every`` epochs (monitoring only: no checkpoint is
selected on the test set).
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image

from torchocr import CRNNLoss, CTCGreedyDecoder, load_charset
from torchocr.datasets import ICDAR2015, LMDBWords, TotalText
from torchocr.metrics import RecognitionAccuracy
from torchocr.models import CRNN, get_weight
from torchocr.models.hub import zero_subnormals_
from torchocr.ops import crop_quads, order_quad_vertices
from torchocr.transforms import RecognitionPreset


def as_quad(polygon: torch.Tensor) -> torch.Tensor:
    """Reading-order quad for a (possibly padded) polygon.

    Four distinct vertices are kept as annotated (ICDAR and Total-Text list them
    in reading order); curved polygons are replaced by their minimum-area
    rectangle.
    """
    distinct = torch.unique_consecutive(polygon, dim=0)
    if len(distinct) > 1 and torch.equal(distinct[0], distinct[-1]):
        distinct = distinct[:-1]
    if len(distinct) == 4:
        return distinct
    rect = cv2.boxPoints(cv2.minAreaRect(polygon.numpy().astype(np.float32)))
    return order_quad_vertices(torch.from_numpy(rect))


class WordCrops(Dataset):
    """``(crop, label indices, text)`` for every legible care word of a full-image dataset.

    ``dataset`` must expose ``image_paths`` and ``load_target(index)``
    (:class:`ICDAR2015`, :class:`TotalText`).
    """

    def __init__(self, dataset, preset: RecognitionPreset, charset: list[str], jitter: float) -> None:
        self.dataset, self.preset, self.jitter = dataset, preset, jitter
        index = {c: i for i, c in enumerate(charset) if i > 0}  # 0 is the CTC blank
        self.words: list[tuple[int, torch.Tensor, list[int], str]] = []
        for image_idx in range(len(dataset)):
            target = dataset.load_target(image_idx)  # no image decoding
            for polygon, text, ignored in zip(target.polygons, target.texts, target.ignore.tolist()):
                if ignored or not any(c.isalnum() for c in text) or any(c not in index for c in text):
                    continue
                self.words.append((image_idx, as_quad(polygon), [index[c] for c in text], text))

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, list[int], str]:
        image_idx, quad, label, text = self.words[i]
        page = self.preset.normalize(read_image(str(self.dataset.image_paths[image_idx]), mode=ImageReadMode.RGB))
        if self.jitter > 0:
            height = torch.linalg.vector_norm(quad[0] - quad[3])
            quad = quad + (torch.rand(4, 2) * 2 - 1) * self.jitter * height
        crop, _ = crop_quads(page[None], quad[None], torch.zeros(1, dtype=torch.long),
                             height=self.preset.height, max_width=self.preset.max_width)
        return crop[0], label, text


class CroppedWords(Dataset):
    """``(crop, label indices, text)`` from an :class:`LMDBWords` corpus of pre-cropped words."""

    def __init__(self, root: Path, preset: RecognitionPreset, charset: list[str], max_length: int) -> None:
        index = {c: i for i, c in enumerate(charset) if i > 0}
        self.index, self.preset = index, preset
        self.words = LMDBWords(
            root, label_filter=lambda t: 0 < len(t) <= max_length and all(c in index for c in t)
        )

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, list[int], str]:
        image, text = self.words[i]
        return self.preset(image), [self.index[c] for c in text], text


def parse_lmdb_spec(spec: str) -> tuple[Path, int | None]:
    """``"PATH"`` or ``"PATH:N"`` (draw N random samples of the corpus per epoch)."""
    path, _, count = spec.rpartition(":")
    if path and count.isdigit():
        return Path(path).expanduser(), int(count)
    return Path(spec).expanduser(), None


def epoch_sampler(sizes: list[int], caps: list[int | None]) -> torch.utils.data.Sampler:
    """Sample a concatenation of sources: uncapped ones whole, capped ones ``cap`` times per epoch.

    Each epoch draws, with replacement, as many samples as the uncapped sizes plus the caps, so a
    9M-word synthetic corpus can sit next to a few thousand real words without drowning them.
    """
    if all(cap is None for cap in caps):
        return torch.utils.data.RandomSampler(range(sum(sizes)))
    weights = torch.cat([torch.full((size,), 1.0 if cap is None else cap / size) for size, cap in zip(sizes, caps)])
    total = sum(size if cap is None else cap for size, cap in zip(sizes, caps))
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=total, replacement=True)


def collate(batch):
    crops, labels, texts = zip(*batch)
    return (
        torch.stack(crops),
        torch.tensor([c for label in labels for c in label], dtype=torch.long),
        torch.tensor([len(label) for label in labels], dtype=torch.long),
        list(texts),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ic15", "--root", dest="ic15", required=True, help="ICDAR-2015 root (train + test).")
    parser.add_argument("--totaltext", help="Total-Text root: adds its training words, and its test words to monitoring.")
    parser.add_argument("--lmdb", action="append", default=[],
                        help="Pre-cropped LMDB word corpus, PATH or PATH:N to draw N samples per epoch (repeatable).")
    parser.add_argument("--ic15-repeat", type=int, default=2, help="Times the ICDAR-2015 words appear per epoch.")
    parser.add_argument("--init", default="CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=0.1, help="Corner jitter, as a fraction of word height.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast for the forward pass (~1.8x faster).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Continue from <output>/checkpoint.pth if present.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: CRNN, words: WordCrops, decoder: CTCGreedyDecoder, device: torch.device) -> float:
    model.eval()
    metric = RecognitionAccuracy(case_sensitive=False, alphanumeric_only=True)
    loader = DataLoader(words, batch_size=128, num_workers=4, collate_fn=collate)
    for crops, _, _, texts in loader:
        metric.update(decoder(model(crops.to(device))), texts)
    model.train()
    return metric.compute().word_accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    weights = get_weight(args.init)
    model = CRNN(weights=weights).to(device)
    preset = weights.transforms()
    charset = load_charset(weights.meta["charset"], weights.meta["num_classes"])
    decoder = CTCGreedyDecoder(charset)

    # CTC needs at least one frame per label (more for repeats): keep labels under T / 2.
    max_length = preset.max_width // 4 // 2
    sources = {"icdar2015": WordCrops(ICDAR2015(args.ic15, split="train"), preset, charset, jitter=args.jitter)}
    test_sets = {"icdar2015": WordCrops(ICDAR2015(args.ic15, split="test"), preset, charset, jitter=0.0)}
    if args.totaltext:
        sources["totaltext"] = WordCrops(TotalText(args.totaltext, split="train"), preset, charset, jitter=args.jitter)
        test_sets["totaltext"] = WordCrops(TotalText(args.totaltext, split="test"), preset, charset, jitter=0.0)
    caps = {}
    for spec in args.lmdb:
        path, cap = parse_lmdb_spec(spec)
        sources[path.name], caps[path.name] = CroppedWords(path, preset, charset, max_length), cap
    parts = [("icdar2015", sources["icdar2015"])] * args.ic15_repeat + [(k, v) for k, v in sources.items() if k != "icdar2015"]
    train_words = ConcatDataset([dataset for _, dataset in parts])
    sampler = epoch_sampler([len(dataset) for _, dataset in parts], [caps.get(name) for name, _ in parts])
    loader = DataLoader(train_words, batch_size=args.batch_size, sampler=sampler, drop_last=True,
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    print(json.dumps({"train_words": {k: len(v) for k, v in sources.items()}, "ic15_repeat": args.ic15_repeat,
                      "per_epoch_caps": {k: c for k, c in caps.items() if c}, "samples_per_epoch": len(sampler),
                      "test_words": {k: len(v) for k, v in test_sets.items()}}), flush=True)

    criterion = CRNNLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, int(args.warmup_epochs * len(loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup_steps
        if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))),
    )
    log = (args.output / "log.jsonl").open("a")

    def report(epoch: int) -> None:
        record = {"epoch": epoch, **{f"{name}_test_word_accuracy": round(evaluate(model, words, decoder, device), 4)
                                     for name, words in test_sets.items()}}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    first_epoch, checkpoint_path = 1, args.output / "checkpoint.pth"
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
        first_epoch = state["epoch"] + 1
        print(json.dumps({"resumed_from_epoch": state["epoch"]}), flush=True)
    else:
        report(0)
    model.train()
    for epoch in range(first_epoch, args.epochs + 1):
        start, total = time.perf_counter(), 0.0
        for crops, targets, target_lengths, _ in loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                logits = model(crops.to(device, non_blocking=True))
            # CTC loss in fp32: the cast below happens outside autocast.
            input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long)
            loss = criterion(logits.float(), targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        print(json.dumps({"epoch": epoch, "loss": round(total / len(loader), 4), "lr": round(scheduler.get_last_lr()[0], 7),
                          "epoch_seconds": round(time.perf_counter() - start, 1)}), flush=True)
        zero_subnormals_(model)
        torch.save(model.state_dict(), args.output / "last.pth")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": torch.get_rng_state(), "epoch": epoch},
                   checkpoint_path.with_suffix(".tmp"))
        checkpoint_path.with_suffix(".tmp").replace(checkpoint_path)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            report(epoch)
    log.close()


if __name__ == "__main__":
    main()
