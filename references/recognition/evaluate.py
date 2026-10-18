"""Evaluate a CRNN checkpoint on ICDAR-2015 ground-truth word quads.

Crops every care word of the test split straight from the full image
using its ground-truth quadrilateral, recognizes it, and reports
case-insensitive alphanumeric word accuracy (the usual scene-text
protocol) plus character error rate. Words with no letter or digit are
skipped. This is *not* the official Task 4.3 benchmark, which ships its
own pre-cropped word images.

``--crop quad`` rectifies each quad with :func:`torchocr.ops.crop_quads`;
``--crop box`` crops the quad's axis-aligned hull instead (what an
``roi_align``-based pipeline sees). Everything else is identical, so the
difference isolates the cost of not rectifying rotated text. Example::

    python references/recognition/evaluate.py --root ~/data/icdar2015 \\
        --weights CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2 --crop quad
"""

import argparse
import json
import time

import torch
from torchvision.transforms.v2 import functional as F

from torchocr import CTCGreedyDecoder, load_charset
from torchocr.datasets import ICDAR2015
from torchocr.metrics import RecognitionAccuracy
from torchocr.models import CRNN, get_weight
from torchocr.ops import crop_quads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory with the extracted ICDAR-2015 archives.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--weights", default="CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2")
    parser.add_argument("--crop", default="quad", choices=["quad", "box"])
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images (smoke test).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    weights = get_weight(args.weights)
    model = CRNN(weights=weights).eval().to(device)
    preset = weights.transforms()
    decoder = CTCGreedyDecoder(load_charset(weights.meta["charset"], weights.meta["num_classes"]))

    dataset = ICDAR2015(args.root, split=args.split)
    num_images = min(len(dataset), args.limit or len(dataset))
    metric = RecognitionAccuracy(case_sensitive=False, alphanumeric_only=True)
    start = time.perf_counter()
    for index in range(num_images):
        image, target = dataset[index]
        keep = ~target.ignore & torch.tensor([any(c.isalnum() for c in t) for t in target.texts], dtype=torch.bool)
        if not keep.any():
            continue
        quads = target.polygons[keep]
        if args.crop == "box":
            lo, hi = quads.amin(1), quads.amax(1)
            quads = torch.stack([lo, torch.stack([hi[:, 0], lo[:, 1]], -1), hi, torch.stack([lo[:, 0], hi[:, 1]], -1)], 1)

        # The preset normalizes per pixel, which commutes with bilinear
        # cropping: normalize the page once, then crop every word.
        page = image.to(device)
        if preset.channel_order == "bgr":
            page = page.flip(0)
        page = F.normalize(F.to_dtype(page, torch.float32, scale=True), preset.mean, preset.std)
        crops, _ = crop_quads(
            page[None], quads.to(device), torch.zeros(len(quads), dtype=torch.long, device=device),
            height=preset.height, max_width=preset.max_width,
        )
        texts = [t for t, k in zip(target.texts, keep.tolist()) if k]
        metric.update(decoder(model(crops)), texts)
    elapsed = time.perf_counter() - start

    result = metric.compute()
    print(json.dumps({
        "weights": args.weights,
        "split": args.split,
        "crop": args.crop,
        "images": num_images,
        "words": result.num_words,
        "word_accuracy": round(result.word_accuracy, 4),
        "char_error_rate": round(result.char_error_rate, 4),
        "seconds_per_image": round(elapsed / num_images, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
