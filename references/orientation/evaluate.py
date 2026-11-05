"""Evaluate page-orientation correction on DocLayNet (CDLA-Permissive-1.0).

Every page is fed upright and turned 90, 180 and 270 degrees; the engine must say how far it is
turned. Two views are reported:

- ``accuracy``: the engine's answer, whatever its confidence, per document category;
- ``decision``: what happens when a page is only turned at ``--min-confidence`` or above:
  upright pages wrongly turned, and turned pages set upright.

To compare engines fairly, choose each one's threshold on the *validation* split with
``--max-upright-turned`` (the lowest threshold that turns at most that share of upright pages),
then score the test split once with ``--min-confidence``. ``--save`` / ``--predictions`` keep
the raw answers so thresholds can be studied without running the engine again.

Engines (see ``baselines.py``): ``torchocr`` (default), ``doctr`` and ``tesseract``. Shards are
the parquet files of https://huggingface.co/datasets/docling-project/DocLayNet-v1.2. Example::

    python references/orientation/evaluate.py --data "~/datasets/doclaynet/test-*.parquet"
"""

import argparse
import collections
import glob
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
from baselines import NO_ANSWER, DocTR, Tesseract, TorchOCR
from torchocr.io.image import decode_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="Glob of DocLayNet parquet shards.")
    parser.add_argument(
        "--predictions", help="Answers saved by --save, instead of running an engine."
    )
    parser.add_argument("--save", help="Write the engine's answers to this .pt file.")
    parser.add_argument(
        "--engine", choices=["torchocr", "doctr", "tesseract"], default="torchocr"
    )
    parser.add_argument(
        "--weights", default="PPLCNet_x1_0_Weights.DEFAULT", help="torchocr weights."
    )
    parser.add_argument(
        "--checkpoint", help="torchocr: local state_dict instead of the published file."
    )
    parser.add_argument(
        "--scale", type=float, default=2.0, help="tesseract: upscale pages first."
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="tesseract: parallel processes."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Default: the weights' own (torchocr), else 0.",
    )
    parser.add_argument(
        "--max-upright-turned",
        type=float,
        help="Also choose the threshold for this rate.",
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N pages (smoke test)."
    )
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=32,
        help="Pages per engine call (x4 orientations).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (args.data is None) == (args.predictions is None):
        parser.error("give exactly one of --data and --predictions")
    return args


def pages(pattern: str, limit: int | None):
    """``(uint8 RGB page, document category)`` for every page of the shards, in file order."""
    count = 0
    for shard in sorted(glob.glob(os.path.expanduser(pattern))):
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=32, columns=["image", "metadata"]
        ):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    return
                yield decode_image(row["image"]["bytes"]), row["metadata"]["doc_category"]
                count += 1


def build_engine(args: argparse.Namespace):
    if args.engine == "doctr":
        return DocTR(args.device)
    if args.engine == "tesseract":
        return Tesseract(args.scale, args.workers)
    return TorchOCR(args.weights, args.checkpoint, args.device)


def run(engine, args: argparse.Namespace) -> dict:
    """The engine's answers for every page in all four orientations."""
    classes, confidences, targets, categories = [], [], [], []
    batch, start = [], time.perf_counter()

    def flush() -> None:
        c, p = engine(batch)
        classes.append(c), confidences.append(p)
        batch.clear()
        print(
            f"{len(targets) // 4} pages, {time.perf_counter() - start:.0f} s",
            file=sys.stderr,
            flush=True,
        )

    for page, category in pages(args.data, args.limit):
        # Turning the upright page k quarter turns counterclockwise = (4 - k) quarter turns clockwise.
        batch += [torch.rot90(page, k, dims=(-2, -1)) for k in range(4)]
        targets += [(4 - k) % 4 for k in range(4)]
        categories += [category] * 4
        if len(batch) >= 4 * args.batch_pages:
            flush()
    if batch:
        flush()
    return {
        "engine": engine.name,
        "classes": torch.cat(classes),
        "confidences": torch.cat(confidences),
        "targets": torch.tensor(targets),
        "categories": categories,
        "seconds_per_image": (time.perf_counter() - start) / len(targets),
    }


def decide(
    classes: torch.Tensor, confidences: torch.Tensor, min_confidence: float
) -> torch.Tensor:
    """Turn a page only when the engine is confident enough; otherwise keep it (class 0)."""
    return torch.where(
        confidences >= min_confidence, classes, torch.zeros_like(classes)
    )


def decision_metrics(
    decisions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    upright, turned = targets == 0, targets != 0
    return {
        "upright_wrongly_turned": round(
            (decisions[upright] != 0).float().mean().item(), 4
        ),
        "turned_set_upright": round(
            (decisions[turned] == targets[turned]).float().mean().item(), 4
        ),
        "turned_to_a_wrong_angle": round(
            ((decisions[turned] != 0) & (decisions[turned] != targets[turned]))
            .float()
            .mean()
            .item(),
            4,
        ),
    }


def threshold_for(classes, confidences, targets, max_upright_turned: float) -> float:
    """Lowest threshold at which at most ``max_upright_turned`` of upright pages are turned."""
    upright = targets == 0
    turned_upright = confidences[upright & (classes != 0)].sort(descending=True).values
    allowed = int(
        max_upright_turned * upright.sum().item()
    )  # wrongly turned upright pages we accept
    if allowed >= len(turned_upright):
        return float(confidences.min().clamp(min=0))
    # Just above the confidence of the first upright page we are not allowed to turn.
    return torch.nextafter(turned_upright[allowed], torch.tensor(float("inf"))).item()


def main() -> None:
    args = parse_args()
    if args.predictions:
        answers = torch.load(args.predictions, weights_only=True)
    else:
        with torch.inference_mode():
            answers = run(build_engine(args), args)
        if args.save:
            torch.save(answers, args.save)
    classes, confidences, targets = (
        answers["classes"],
        answers["confidences"],
        answers["targets"],
    )

    if args.min_confidence is not None:
        min_confidence = args.min_confidence
    elif answers["engine"].startswith("torchocr"):
        min_confidence = TorchOCR.default_min_confidence(args.weights)
    else:
        min_confidence = 0.0

    correct = classes == targets
    by_category = collections.defaultdict(list)
    for category, ok in zip(answers["categories"], correct.tolist()):
        by_category[category].append(ok)
    report = {
        "engine": answers["engine"],
        "pages": len(targets) // 4,
        "accuracy": round(correct.float().mean().item(), 4),
        "accuracy_by_category": {
            c: round(sum(v) / len(v), 4) for c, v in sorted(by_category.items())
        },
        "no_answer": round((confidences == NO_ANSWER).float().mean().item(), 4),
        "min_confidence": min_confidence,
        "decision": decision_metrics(
            decide(classes, confidences, min_confidence), targets
        ),
        "seconds_per_image": round(answers["seconds_per_image"], 4),
    }
    if args.max_upright_turned is not None:
        chosen = threshold_for(classes, confidences, targets, args.max_upright_turned)
        report["chosen"] = {
            "max_upright_turned": args.max_upright_turned,
            "min_confidence": chosen,
            "decision": decision_metrics(decide(classes, confidences, chosen), targets),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
