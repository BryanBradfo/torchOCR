"""Pick DBPostProcessor's ``box_thresh`` on the training split, then score the test split once.

The threshold that maximizes train-split hmean is applied unchanged to
the test split, so the reported test number involves no test-set tuning.
Example::

    python references/detection/calibrate.py --root ~/data/icdar2015 \\
        --backbone mobilenet_v3_large_05 --checkpoint runs/db_mv3_en_ic15/last.pth
"""

import argparse
import json

import torch

from evaluate import evaluate_detector
from torchocr import DBPostProcessor
from torchocr.datasets import ICDAR2015
from torchocr.metrics import DetectionHmean
from torchocr.models import DBNet
from torchocr.transforms import DetectionPreset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="mobilenet_v3_large_05")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--channel-order", default="bgr", choices=["rgb", "bgr"])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def sweep(model, dataset, preset, thresholds, device) -> dict[float, DetectionHmean]:
    """One forward pass per image; every threshold post-processes the same output."""
    processors = {t: DBPostProcessor(threshold=0.3, box_thresh=t, unclip_ratio=1.5) for t in thresholds}
    metrics = {t: DetectionHmean() for t in thresholds}
    for index in range(len(dataset)):
        image, target = dataset[index]
        batch = preset(image.to(device)).unsqueeze(0)
        output = model(batch)
        scale = torch.tensor([image.shape[-1] / batch.shape[-1], image.shape[-2] / batch.shape[-2]], device=device)
        for t, processor in processors.items():
            metrics[t].update((processor.quadrilaterals(output)[0] * scale).cpu(), target.polygons, target.ignore)
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = DBNet(backbone=args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)
    preset = DetectionPreset(max_side=args.max_side, channel_order=args.channel_order)

    train_scores = {t: m.compute() for t, m in sweep(model, ICDAR2015(args.root, split="train"), preset,
                                                    args.thresholds, device).items()}
    for t, r in train_scores.items():
        print(json.dumps({"split": "train", "box_thresh": t, "precision": round(r.precision, 4),
                          "recall": round(r.recall, 4), "hmean": round(r.hmean, 4)}))
    best = max(train_scores, key=lambda t: train_scores[t].hmean)

    test = evaluate_detector(model, ICDAR2015(args.root, split="test"), preset,
                             DBPostProcessor(threshold=0.3, box_thresh=best, unclip_ratio=1.5), device)
    print(json.dumps({"split": "test", "box_thresh": best, "selected_on": "train", "precision": round(test.precision, 4),
                      "recall": round(test.recall, 4), "hmean": round(test.hmean, 4)}))


if __name__ == "__main__":
    main()
