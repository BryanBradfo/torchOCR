"""Fine-tune a CRNN on ICDAR-2015 word crops.

Every care word of the training split is rectified from its full image
with :func:`torchocr.ops.crop_quads` -- the same path ``OCRPipeline``
uses at inference -- after jittering the quad's corners so the model
tolerates imprecise detector boxes. Example::

    python references/recognition/train.py --root ~/data/icdar2015 \\
        --init CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2 --output runs/crnn_ic15

Writes ``last.pth`` every epoch and logs test word accuracy on
ground-truth quads every ``--eval-every`` epochs (monitoring only: no
checkpoint is selected on the test set).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image

from torchocr import CRNNLoss, CTCGreedyDecoder, load_charset
from torchocr.datasets import ICDAR2015
from torchocr.metrics import RecognitionAccuracy
from torchocr.models import CRNN, get_weight
from torchocr.models.hub import zero_subnormals_
from torchocr.ops import crop_quads
from torchocr.transforms import RecognitionPreset


class WordCrops(Dataset):
    """``(crop, label indices, text)`` for every legible care word of an ICDAR-2015 split."""

    def __init__(self, dataset: ICDAR2015, preset: RecognitionPreset, charset: list[str], jitter: float) -> None:
        self.dataset, self.preset, self.jitter = dataset, preset, jitter
        index = {c: i for i, c in enumerate(charset) if i > 0}  # 0 is the CTC blank
        self.words: list[tuple[int, torch.Tensor, list[int], str]] = []
        for image_idx in range(len(dataset)):
            # Ground truth only: avoid decoding the image here.
            target = _ground_truth(dataset, image_idx)
            for quad, text, ignored in zip(target.polygons, target.texts, target.ignore.tolist()):
                if ignored or not any(c.isalnum() for c in text) or any(c not in index for c in text):
                    continue
                self.words.append((image_idx, quad, [index[c] for c in text], text))

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


def _ground_truth(dataset: ICDAR2015, image_idx: int):
    from torchocr.datasets.icdar import _parse_ground_truth

    stem = dataset.image_paths[image_idx].stem
    return _parse_ground_truth(dataset.gt_dir / f"gt_{stem}.txt")


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
    parser.add_argument("--root", required=True)
    parser.add_argument("--init", default="CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=0.1, help="Corner jitter, as a fraction of word height.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
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

    train_words = WordCrops(ICDAR2015(args.root, split="train"), preset, charset, jitter=args.jitter)
    test_words = WordCrops(ICDAR2015(args.root, split="test"), preset, charset, jitter=0.0)
    loader = DataLoader(train_words, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    print(json.dumps({"train_words": len(train_words), "test_words": len(test_words)}), flush=True)

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
        record = {"epoch": epoch, "test_word_accuracy": round(evaluate(model, test_words, decoder, device), 4)}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()

    report(0)
    model.train()
    for epoch in range(1, args.epochs + 1):
        start, total = time.perf_counter(), 0.0
        for crops, targets, target_lengths, _ in loader:
            logits = model(crops.to(device, non_blocking=True))
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
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            report(epoch)
    log.close()


if __name__ == "__main__":
    main()
