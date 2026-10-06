"""Fine-tune (or train) DBNet on ICDAR-2015.

Example -- fine-tune the English PP-OCRv3 detector::

    python references/detection/train.py --root ~/data/icdar2015 \\
        --init DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN --output runs/db_mv3_ic15

Writes ``last.pth`` (state_dict) after every epoch and one JSON line per
evaluation to ``log.jsonl``. The test split is only *monitored*: no
checkpoint is selected on it, so the final number is not optimistically
biased (ICDAR-2015 has no validation split).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from presets import DBTrainPreset
from torch.utils.data import DataLoader

from evaluate import evaluate_detector
from torchocr import DBLoss, DBPostProcessor
from torchocr.datasets import ICDAR2015
from torchocr.models import DBNet, get_weight
from torchocr.models.detection import DBNetOutput
from torchocr.transforms import DBTargets, DetectionPreset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory with the extracted ICDAR-2015 archives.")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--init", help="Weights enum to fine-tune, e.g. DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN.")
    start.add_argument("--backbone", help="Train this DBNet backbone from random init instead.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--size", type=int, default=640, help="Training crop side.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10, help="Epochs between test-set evaluations.")
    parser.add_argument("--eval-max-side", type=int, default=1280, help="Longer side at evaluation (PaddleOCR: 736x1280).")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast for the forward pass.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate(batch: list[tuple[torch.Tensor, DBTargets]]) -> tuple[torch.Tensor, DBTargets]:
    images, targets = zip(*batch)
    return torch.stack(images), DBTargets(
        probability=torch.stack([t.probability for t in targets]),
        threshold=torch.stack([t.threshold for t in targets]),
        mask=torch.stack([t.mask for t in targets]),
    )


def build_model(args: argparse.Namespace) -> tuple[DBNet, DetectionPreset]:
    if args.init:
        weights = get_weight(args.init)
        return DBNet(weights=weights), weights.transforms()
    return DBNet(backbone=args.backbone), DetectionPreset(channel_order="bgr")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    model, preset = build_model(args)
    model.to(device)
    train_set = ICDAR2015(
        args.root,
        split="train",
        transforms=DBTrainPreset(size=args.size, mean=preset.mean, std=preset.std, channel_order=preset.channel_order),
    )
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0,
    )
    test_set = ICDAR2015(args.root, split="test")
    eval_preset = DetectionPreset(max_side=args.eval_max_side, mean=preset.mean, std=preset.std,
                                  channel_order=preset.channel_order)
    postprocess = DBPostProcessor(threshold=0.3, box_thresh=0.6, unclip_ratio=1.5)

    # PaddleOCR weighs the terms 5 (BCE) : 10 (threshold L1) : 1 (dice); same ratios here.
    criterion = DBLoss(threshold_weight=2.0, binary_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, int(args.warmup_epochs * len(loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup_steps
        if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))),
    )

    log = (args.output / "log.jsonl").open("a")

    def evaluate(epoch: int) -> None:
        model.eval()
        start = time.perf_counter()
        result = evaluate_detector(model, test_set, eval_preset, postprocess, device)
        record = {"epoch": epoch, "precision": round(result.precision, 4), "recall": round(result.recall, 4),
                  "hmean": round(result.hmean, 4), "eval_seconds": round(time.perf_counter() - start, 1)}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + "\n")
        log.flush()
        model.train()

    evaluate(0)
    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        start, running = time.perf_counter(), {}
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                output = model(images)
            # BCE is not autocast-safe: compute the loss in fp32.
            output = DBNetOutput(output.probability.float(), output.threshold.float())
            losses = criterion(
                output,
                targets.probability.to(device, non_blocking=True),
                targets.threshold.to(device, non_blocking=True),
                targets.mask.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            scheduler.step()
            step += 1
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + value.item()

        summary = {name: round(total / len(loader), 4) for name, total in running.items()}
        print(json.dumps({"epoch": epoch, "step": step, "lr": round(scheduler.get_last_lr()[0], 6),
                          "epoch_seconds": round(time.perf_counter() - start, 1), **summary}), flush=True)
        torch.save(model.state_dict(), args.output / "last.pth")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(epoch)
    log.close()


if __name__ == "__main__":
    main()
