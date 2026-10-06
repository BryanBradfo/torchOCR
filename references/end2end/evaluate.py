"""Evaluate a full OCRPipeline (image -> text) on ICDAR-2015.

Each test image goes through ``OCRPipeline.from_pretrained`` exactly as a
user would call it. A word counts when its region matches a ground-truth
word (ICDAR IoU protocol) *and* is read correctly (case-insensitive,
letters and digits only, generic vocabulary). Detection-only hmean of the
same run is reported alongside. Example::

    python references/end2end/evaluate.py --root ~/data/icdar2015 \\
        --detector DBNet_MobileNetV3_Large_05_Weights.ICDAR2015 \\
        --recognizer CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2
"""

import argparse
import json
import time

import torch

from torchocr import OCRPipeline
from torchocr.datasets import ICDAR2015
from torchocr.metrics import DetectionHmean, EndToEndHmean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory with the extracted ICDAR-2015 archives.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--detector", default="DBNet_MobileNetV3_Large_05_Weights.ICDAR2015")
    parser.add_argument("--recognizer", default="CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images (smoke test).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = OCRPipeline.from_pretrained(args.detector, args.recognizer, device=args.device)
    dataset = ICDAR2015(args.root, split=args.split)
    num_images = min(len(dataset), args.limit or len(dataset))

    end2end, detection = EndToEndHmean(), DetectionHmean()
    start = time.perf_counter()
    for index in range(num_images):
        image, target = dataset[index]
        doc = pipeline(image)
        polygons = doc.polygons.cpu() if doc.polygons is not None else torch.zeros(0, 4, 2)
        end2end.update(polygons, target.polygons, target.ignore, doc.text, target.texts)
        detection.update(polygons, target.polygons, target.ignore)
    elapsed = time.perf_counter() - start

    e2e, det = end2end.compute(), detection.compute()
    print(json.dumps({
        "detector": args.detector,
        "recognizer": args.recognizer,
        "split": args.split,
        "images": num_images,
        "end2end": {"precision": round(e2e.precision, 4), "recall": round(e2e.recall, 4),
                    "hmean": round(e2e.hmean, 4), "correct": e2e.num_matched},
        "detection_hmean": round(det.hmean, 4),
        "seconds_per_image": round(elapsed / num_images, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
