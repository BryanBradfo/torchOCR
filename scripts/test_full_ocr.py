"""End-to-end OCR smoke test using converted PaddleOCR weights.

Loads the converted DBNet detector + CRNN recognizer, runs them on a
document image with the *correct* per-stage preprocessing (ImageNet
normalization for the detector, ``(x-127.5)/127.5`` for the recognizer),
decodes with PaddleOCR's Chinese charset, and writes an annotated
visualization with the recognized text overlaid.

This bypasses :class:`torchocr.OCRPipeline` because the pipeline
currently feeds one normalized tensor to both stages -- a refactor to
handle stage-specific preprocessing is Phase C work.

Usage:
    python scripts/test_full_ocr.py \\
        --det-weights /tmp/dbnet_resnet18_vd.pth \\
        --rec-weights /tmp/crnn_resnet34_vd.pth \\
        --image examples/chinese_receipt.jpg \\
        --output /tmp/chinese_receipt_full.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from torchocr import CTCGreedyDecoder, DBPostProcessor, load_ppocr_keys_v1
from torchocr.models import CRNN, DBNet


# Detector preprocessing (ImageNet, BGR -> RGB).
_DET_MAX_SIDE = 960
_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Recognizer preprocessing (PaddleOCR's [-1, 1] normalization).
_REC_HEIGHT = 32
_REC_MAX_WIDTH = 320


def preprocess_detector(bgr: np.ndarray) -> tuple[Tensor, float, float]:
    """PaddleOCR-style detector preprocessing.

    Note: PaddleOCR trains on cv2-loaded *BGR* images and applies the
    ImageNet mean/std directly to BGR pixels (no channel swap). We
    replicate that to match the trained weights.
    """
    h, w = bgr.shape[:2]
    scale = _DET_MAX_SIDE / max(h, w)
    nh = max(int(round(h * scale / 32) * 32), 32)
    nw = max(int(round(w * scale / 32) * 32), 32)
    resized = cv2.resize(bgr, (nw, nh)).astype(np.float32) / 255.0
    resized = (resized - _DET_MEAN) / _DET_STD
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, w / nw, h / nh


def preprocess_recognizer(bgr_crop: np.ndarray) -> Tensor:
    """PaddleOCR-style recognizer preprocessing: BGR -> [-1, 1], resize to 32xW."""
    h, w = bgr_crop.shape[:2]
    if h == 0 or w == 0:
        return torch.zeros(3, _REC_HEIGHT, _REC_MAX_WIDTH)
    target_w = max(min(int(round(_REC_HEIGHT * w / max(h, 1))), _REC_MAX_WIDTH), 8)
    resized = cv2.resize(bgr_crop, (target_w, _REC_HEIGHT)).astype(np.float32)
    resized = (resized - 127.5) / 127.5
    tensor = torch.from_numpy(resized.transpose(2, 0, 1))
    # Right-pad with zeros so all crops in a batch have width=_REC_MAX_WIDTH.
    if target_w < _REC_MAX_WIDTH:
        padded = torch.zeros(3, _REC_HEIGHT, _REC_MAX_WIDTH)
        padded[:, :, :target_w] = tensor
        tensor = padded
    return tensor


def crop_aabb(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    xa = int(max(0, np.floor(x1)))
    ya = int(max(0, np.floor(y1)))
    xb = int(min(w, np.ceil(x2) + 1))
    yb = int(min(h, np.ceil(y2) + 1))
    return bgr[ya:yb, xa:xb]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--det-weights", type=Path, required=True)
    parser.add_argument("--rec-weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-thresh", type=float, default=0.5)
    parser.add_argument("--unclip-ratio", type=float, default=1.6)
    parser.add_argument("--num-classes", type=int, default=6625)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    print(f"loaded {args.image}: shape={bgr.shape}")

    # Detector
    detector = DBNet(backbone="resnet18_vd")
    detector.load_state_dict(torch.load(args.det_weights, weights_only=True), strict=True)
    detector.train(False)

    det_tensor, sx, sy = preprocess_detector(bgr)
    with torch.no_grad():
        det_output = detector(det_tensor)
    boxes = DBPostProcessor(box_thresh=args.box_thresh, unclip_ratio=args.unclip_ratio)(det_output)
    print(f"detected {boxes.shape[0]} boxes")
    if boxes.shape[0] == 0:
        print("no detections; nothing to recognize")
        cv2.imwrite(str(args.output), bgr)
        return

    # Recognizer
    recognizer = CRNN(num_classes=args.num_classes, backbone="resnet34_vd")
    recognizer.load_state_dict(torch.load(args.rec_weights, weights_only=True), strict=True)
    recognizer.train(False)
    decoder = CTCGreedyDecoder(charset=load_ppocr_keys_v1(args.num_classes))

    crops = []
    box_pixels = []
    for row in boxes.tolist():
        _, x1, y1, x2, y2 = row
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        bgr_crop = crop_aabb(bgr, x1, y1, x2, y2)
        crops.append(preprocess_recognizer(bgr_crop))
        box_pixels.append((int(x1), int(y1), int(x2), int(y2)))
    rec_input = torch.stack(crops, dim=0)

    with torch.no_grad():
        logits = recognizer(rec_input)
    texts = decoder(logits)

    # Visualization
    canvas = bgr.copy()
    for (x1, y1, x2, y2), text in zip(box_pixels, texts):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2 default fonts cannot render Chinese; print to stdout instead.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"\nwrote annotated boxes -> {args.output}")
    print("\nrecognized text per box:")
    for i, ((x1, y1, x2, y2), text) in enumerate(zip(box_pixels, texts)):
        print(f"  [{i:3}] ({x1:4},{y1:4})-({x2:4},{y2:4})  {text!r}")


if __name__ == "__main__":
    main()
