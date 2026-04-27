"""End-to-end visual demo for torchocr.

Loads a real document image from ``examples/`` (Chinese by default,
since the converted PP-OCR weights ship with strong Chinese coverage),
runs the OCR pipeline, and writes an annotated visualization next to
the input.

Without ``--weights`` the detector and recognizer use random
initialization; the pipeline composes end-to-end but boxes will be
noise. With a converted DBNet checkpoint (see
``scripts/convert_paddle_dbnet.py``) the boxes track real text.

Run from any working directory:
    # pipeline shape check (random init, demonstrates the API):
    python examples/demo_inference.py

    # with real weights converted from PaddleOCR:
    python examples/demo_inference.py --weights /tmp/dbnet_resnet18_vd.pth

    # try a different image:
    python examples/demo_inference.py --image examples/english_doc.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes

from torchocr import CTCGreedyDecoder, DBPostProcessor, OCRPipeline
from torchocr.models import CRNN, DBNet


SCRIPT_DIR = Path(__file__).resolve().parent

# PaddleOCR detection-side preprocessing.
_MAX_SIDE = 960
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_detector(bgr: np.ndarray) -> torch.Tensor:
    h, w = bgr.shape[:2]
    scale = _MAX_SIDE / max(h, w)
    nh = max(int(round(h * scale / 32) * 32), 32)
    nw = max(int(round(w * scale / 32) * 32), 32)
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--image",
        type=Path,
        default=SCRIPT_DIR / "chinese_receipt.jpg",
        help="Input document image (default: examples/chinese_receipt.jpg).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional converted DBNet .pth file (run scripts/convert_paddle_dbnet.py to make one).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the annotated image (default: <image>_annotated.jpg next to the input).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    output_path = args.output or args.image.with_name(f"{args.image.stem}_annotated.jpg")

    bgr = cv2.imread(str(args.image))
    print(f"loaded {args.image.name}: shape={bgr.shape}")

    # Build detector. Use the ResNet-VD path when real weights are provided,
    # since that's the only path with PaddleOCR-compatible parameter shapes.
    if args.weights is not None:
        if not args.weights.exists():
            raise FileNotFoundError(f"Weights file not found: {args.weights}")
        detector = DBNet(backbone="resnet18_vd")
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        detector.load_state_dict(state, strict=True)
        print(f"loaded converted detector weights from {args.weights} ({len(state)} tensors)")
    else:
        detector = DBNet()  # random init, torchvision ResNet-18
        print("using random-init DBNet (boxes will be noise; pass --weights to use real weights)")
    detector.train(False)

    # Recognizer is always random-init at v0.2.0 -- decoded text will be noise.
    charset = ["-"] + [chr(c) for c in range(32, 127)]
    recognizer = CRNN(num_classes=len(charset)).train(False)

    pipeline = OCRPipeline(
        detector=detector,
        recognizer=recognizer,
        post_processor=DBPostProcessor(),
        decoder=CTCGreedyDecoder(charset),
    )

    image_tensor = preprocess_for_detector(bgr)
    document = pipeline(image_tensor)
    box_count = 0 if document.bounding_boxes is None else document.bounding_boxes.shape[0]
    print(f"detected {box_count} box(es); decoded text (random recognizer): {document.text}")

    # Map detection-space boxes back to original image pixels.
    detector_h, detector_w = image_tensor.shape[-2:]
    src_h, src_w = bgr.shape[:2]
    sx, sy = src_w / detector_w, src_h / detector_h
    rgb_uint8 = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()
    if box_count > 0:
        boxes = document.bounding_boxes.clone()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        labels = [t if t else "<noise>" for t in document.text]
        annotated = draw_bounding_boxes(rgb_uint8, boxes, labels=labels, colors="red", width=3)
    else:
        print("no detections; saving the input image unannotated")
        annotated = rgb_uint8

    write_jpeg(annotated, str(output_path), quality=92)
    print(f"wrote {output_path}")

    if args.weights is None:
        print()
        print("NOTE: Without --weights the detector and recognizer are random-init.")
        print("Pass a converted .pth from scripts/convert_paddle_dbnet.py to see real boxes.")


if __name__ == "__main__":
    main()
