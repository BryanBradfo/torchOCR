"""Evaluate a DBNet checkpoint on ICDAR-2015 with the official IoU protocol.

This is the script behind every ``meta["_metrics"]`` entry of torchocr's
detection weights. Example::

    python references/detection/evaluate.py --root ~/data/icdar2015 \\
        --weights DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2

    # Unpublished / locally converted checkpoint:
    python references/detection/evaluate.py --root ~/data/icdar2015 \\
        --backbone resnet18_vd --checkpoint dbnet_resnet18_vd.pth \\
        --max-side 1280 --channel-order bgr

Flags left unset fall back to the preset bound to ``--weights``.
"""

import argparse
import json
import time

import torch

from torchocr import DBPostProcessor
from torchocr.datasets import ICDAR2015
from torchocr.metrics import DetectionHmean, HmeanResult
from torchocr.models import DBNet, get_weight
from torchocr.transforms import DetectionPreset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory with the extracted ICDAR-2015 archives.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", help="Weights enum, e.g. DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2.")
    source.add_argument("--checkpoint", help="Path to a local torchocr .pth state_dict.")
    parser.add_argument("--backbone", default="resnet18_vd", help="DBNet backbone (with --checkpoint).")
    parser.add_argument("--max-side", type=int, help="Longer-side resize target (0 = native size).")
    parser.add_argument("--channel-order", choices=["rgb", "bgr"])
    # Post-processing defaults come from weights.meta["postprocess"] (PaddleOCR's otherwise).
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--box-thresh", type=float)
    parser.add_argument("--unclip-ratio", type=float)
    parser.add_argument("--axis-aligned", action="store_true", help="Score xyxy boxes instead of rotated quads.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images (smoke test).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build(args: argparse.Namespace) -> tuple[DBNet, DetectionPreset]:
    if args.weights:
        weights = get_weight(args.weights)
        model = DBNet(backbone=weights.meta["backbone"], weights=weights)
        preset = weights.transforms()
    else:
        model = DBNet(backbone=args.backbone)
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        preset = DetectionPreset()
    if args.max_side is not None:
        preset.max_side = args.max_side or None
    if args.channel_order is not None:
        preset.channel_order = args.channel_order
    return model, preset


@torch.inference_mode()
def evaluate_detector(
    model: DBNet,
    dataset: ICDAR2015,
    preset: DetectionPreset,
    postprocess: DBPostProcessor,
    device: torch.device,
    limit: int | None = None,
    axis_aligned: bool = False,
) -> HmeanResult:
    """Score ``model`` (already in eval mode, on ``device``) on ``dataset``."""
    metric = DetectionHmean()
    for index in range(min(len(dataset), limit or len(dataset))):
        image, target = dataset[index]
        batch = preset(image.to(device)).unsqueeze(0)
        output = model(batch)
        scale = torch.tensor(
            [image.shape[-1] / batch.shape[-1], image.shape[-2] / batch.shape[-2]], device=device
        )
        if axis_aligned:
            preds = postprocess(output)[:, 1:] * scale.repeat(2)
        else:
            preds = postprocess.quadrilaterals(output)[0] * scale
        metric.update(preds.cpu(), target.polygons, target.ignore)
    return metric.compute()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, preset = build(args)
    model.eval().to(device)
    settings = {"threshold": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5}
    if args.weights:
        settings.update(get_weight(args.weights).meta.get("postprocess", {}))
    for name in settings:
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    postprocess = DBPostProcessor(**settings)

    dataset = ICDAR2015(args.root, split=args.split)
    num_images = min(len(dataset), args.limit or len(dataset))
    start = time.perf_counter()
    result = evaluate_detector(model, dataset, preset, postprocess, device, args.limit, args.axis_aligned)
    elapsed = time.perf_counter() - start

    report = {
        "source": args.weights or args.checkpoint,
        "split": args.split,
        "images": num_images,
        "preset": preset.extra_repr(),
        "postprocess": {
            **settings,
            "axis_aligned": args.axis_aligned,
        },
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "hmean": round(result.hmean, 4),
        "matched": result.num_matched,
        "num_gt": result.num_gt,
        "num_pred": result.num_pred,
        "seconds_per_image": round(elapsed / num_images, 4),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
