"""Evaluate page-orientation correction on DocLayNet (CDLA-Permissive-1.0).

Every page is fed upright and turned 90, 180 and 270 degrees; the classifier
must say how far it is turned. Two views are reported:

- ``accuracy``: argmax of the classifier, per document category;
- ``decision``: what ``PageOrientation`` does at its ``min_confidence``:
  upright pages wrongly turned, and turned pages set upright.

``--sweep`` adds the decision metrics for a range of thresholds; run it on
the *validation* split to choose ``min_confidence``, then score the test
split once. Shards are the parquet files of
https://huggingface.co/datasets/docling-project/DocLayNet-v1.2. Example::

    python references/orientation/evaluate.py --data "~/datasets/doclaynet/test-*.parquet"
"""

import argparse
import collections
import glob
import json
import os
import time

import pyarrow.parquet as pq
import torch
from torchvision.io import ImageReadMode, decode_image

from torchocr import PageOrientation
from torchocr.orientation import _decide
from torchocr.models import PPLCNet, PPLCNet_x1_0_Weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Glob of DocLayNet parquet shards.")
    parser.add_argument("--weights", default="PPLCNet_x1_0_Weights.DEFAULT")
    parser.add_argument("--checkpoint", help="Local state_dict to evaluate instead of the published file.")
    parser.add_argument("--min-confidence", type=float, help="Default: the weights' own threshold.")
    parser.add_argument("--sweep", action="store_true", help="Also report a range of thresholds.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N pages (smoke test).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pages(pattern: str, limit: int | None):
    """``(uint8 RGB page, document category)`` for every page of the shards, in file order."""
    count = 0
    for shard in sorted(glob.glob(os.path.expanduser(pattern))):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["image", "metadata"]):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    return
                data = torch.frombuffer(bytearray(row["image"]["bytes"]), dtype=torch.uint8)
                yield decode_image(data, ImageReadMode.RGB), row["metadata"]["doc_category"]
                count += 1


def decision_metrics(decisions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    upright, turned = targets == 0, targets != 0
    return {
        "upright_wrongly_turned": round((decisions[upright] != 0).float().mean().item(), 4),
        "turned_set_upright": round((decisions[turned] == targets[turned]).float().mean().item(), 4),
        "turned_to_a_wrong_angle": round(((decisions[turned] != 0) & (decisions[turned] != targets[turned])).float().mean().item(), 4),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    weights = PPLCNet_x1_0_Weights.verify(args.weights)
    if args.checkpoint:
        classifier = PPLCNet(num_classes=len(weights.meta["categories"]))
        classifier.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        classifier = PPLCNet(weights=weights)
    min_confidence = weights.meta["min_confidence"] if args.min_confidence is None else args.min_confidence
    orientation = PageOrientation(classifier.eval().to(args.device), weights.transforms(), min_confidence)

    probabilities, targets, categories = [], [], []
    start = time.perf_counter()
    for page, category in pages(args.data, args.limit):
        page = page.to(args.device)
        # Turning the upright page k quarter turns counterclockwise = (4 - k) quarter turns clockwise.
        probabilities.append(orientation.probabilities([torch.rot90(page, k, dims=(-2, -1)) for k in range(4)]).cpu())
        targets += [(4 - k) % 4 for k in range(4)]
        categories += [category] * 4
    elapsed = time.perf_counter() - start
    probabilities, targets = torch.cat(probabilities), torch.tensor(targets)

    correct = probabilities.argmax(-1) == targets
    by_category = collections.defaultdict(list)
    for category, ok in zip(categories, correct.tolist()):
        by_category[category].append(ok)
    decisions = _decide(probabilities, min_confidence)
    report = {
        "weights": repr(weights),
        "checkpoint": args.checkpoint,
        "pages": len(targets) // 4,
        "accuracy": round(correct.float().mean().item(), 4),
        "accuracy_by_category": {c: round(sum(v) / len(v), 4) for c, v in sorted(by_category.items())},
        "min_confidence": min_confidence,
        "decision": decision_metrics(decisions, targets),
        "seconds_per_page": round(elapsed / max(len(targets) // 4, 1) / 4, 4),
    }
    if args.sweep:
        report["sweep"] = {
            f"{t:.2f}": decision_metrics(_decide(probabilities, t), targets)
            for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
