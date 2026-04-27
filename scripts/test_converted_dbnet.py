"""End-to-end smoke test for a converted PaddleOCR DBNet checkpoint.

Loads converted weights into ``DBNet(backbone="resnet18_vd")``, runs the
contour-based post-processor, and writes an annotated copy of the input
image with bounding boxes drawn over detected text regions. Useful for
eyeballing whether a freshly-converted ``.pth`` actually works before
publishing it to the model hub.

Usage:
    python scripts/test_converted_dbnet.py \\
        --weights /tmp/dbnet_resnet18_vd.pth \\
        --image examples/chinese_receipt.jpg \\
        --output /tmp/chinese_receipt_boxes.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from torchocr import DBPostProcessor
from torchocr.models import DBNet


# PaddleOCR detector-side preprocessing constants.
# (BGR image -> resize so longest side ~= max_side and divisible by 32 ->
#  BGR->RGB -> /255 -> ImageNet normalize.)
_MAX_SIDE = 960
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr: np.ndarray) -> tuple[torch.Tensor, float, float]:
    """Resize, normalize, and convert ``bgr`` to a (1, 3, H, W) tensor.

    Returns the tensor plus the (scale_x, scale_y) ratios used to map
    network-space boxes back to original-image pixel coordinates.
    """
    h, w = bgr.shape[:2]
    scale = _MAX_SIDE / max(h, w)
    new_h = max(int(round(h * scale / 32) * 32), 32)
    new_w = max(int(round(w * scale / 32) * 32), 32)
    resized = cv2.resize(bgr, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, w / new_w, h / new_h


def draw_boxes(bgr: np.ndarray, boxes: torch.Tensor, scale_x: float, scale_y: float) -> np.ndarray:
    """Overlay axis-aligned boxes on a copy of ``bgr``."""
    canvas = bgr.copy()
    for row in boxes.tolist():
        _, x1, y1, x2, y2 = row
        x1 = int(round(x1 * scale_x))
        x2 = int(round(x2 * scale_x))
        y1 = int(round(y1 * scale_y))
        y2 = int(round(y2 * scale_y))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help="Converted .pth file.")
    parser.add_argument("--image", type=Path, required=True, help="Input document image.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the annotated image.")
    parser.add_argument("--threshold", type=float, default=0.3, help="DB binarization threshold (default 0.3).")
    parser.add_argument("--box-thresh", type=float, default=0.7, help="Box confidence threshold (default 0.7).")
    parser.add_argument("--unclip-ratio", type=float, default=1.5, help="Polygon expansion ratio (default 1.5).")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        sys.exit(f"ERROR: could not read image at {args.image}")

    print(f"Image: {args.image}  shape={bgr.shape}")

    tensor, scale_x, scale_y = preprocess(bgr)
    print(f"Preprocessed tensor: shape={tuple(tensor.shape)}  scale=({scale_x:.3f}, {scale_y:.3f})")

    model = DBNet(backbone="resnet18_vd")
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.train(False)
    print(f"Loaded {len(state)} tensors from {args.weights}")

    with torch.no_grad():
        out = model(tensor)

    prob = out.probability[0, 0]
    print(
        f"\nProbability map: shape={tuple(prob.shape)}  "
        f"min={prob.min():.4f}  mean={prob.mean():.4f}  max={prob.max():.4f}  "
        f"std={prob.std():.4f}"
    )
    print(f"  pixels above threshold={args.threshold}: {(prob > args.threshold).sum().item()} / {prob.numel()}")

    post = DBPostProcessor(
        threshold=args.threshold,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )
    boxes = post(out)
    print(f"\nDetected boxes: {boxes.shape[0]}")

    annotated = draw_boxes(bgr, boxes, scale_x, scale_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
