"""Inference CLI for torchocr.

Process a PDF or image end-to-end with the OCRPipeline:
  1. Load: ``load_pdf`` for ``.pdf``, ``load_image`` for image suffixes.
  2. Per-page top-left crop to a multiple of 32 (DBNet input contract).
  3. Run the pipeline on the requested device.
  4. Overlay boxes + decoded text via ``draw_bounding_boxes``.
  5. Save annotated JPGs to ``--output-dir``.

Models load with ``weights="DEFAULT"`` -- when the published checkpoint
is not yet available, the hub prints a warning and falls back to random
init. The CLI keeps running so users can validate the pipeline path.

Examples:
    python scripts/infer.py --input docs/contract.pdf
    python scripts/infer.py --input page.jpg --output-dir runs/ --device cuda
    python scripts/infer.py --input scan.pdf --dpi 300 --device cuda:0
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes

from torchocr import (
    CTCGreedyDecoder,
    DBPostProcessor,
    OCRPipeline,
    load_image,
    load_pdf,
)
from torchocr.models import CRNN, DBNet


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="torchocr inference CLI: PDF/image -> annotated JPGs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a PDF or image file (.pdf, .jpg, .png, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Directory for annotated images. Default ./outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render resolution for PDFs. Default 200.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (e.g. cuda, cuda:0, cpu, mps). "
        "Default: cuda if available, else cpu.",
    )
    return parser.parse_args()


def resolve_device(arg: str | None) -> torch.device:
    if arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested device '{arg}' but CUDA is unavailable.")
    return device


def load_document(path: Path, dpi: int) -> list[Tensor]:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, dpi=dpi)
    if suffix in _IMAGE_SUFFIXES:
        return [load_image(path)]
    raise SystemExit(
        f"Unsupported file extension '{suffix}'. "
        f"Expected .pdf or one of {sorted(_IMAGE_SUFFIXES)}."
    )


def crop_to_multiple_of_32(image: Tensor) -> Tensor:
    h_crop = (image.shape[1] // 32) * 32
    w_crop = (image.shape[2] // 32) * 32
    if h_crop == 0 or w_crop == 0:
        raise SystemExit(
            f"Image too small after 32-multiple crop: {tuple(image.shape)}."
        )
    return image[:, :h_crop, :w_crop]


def build_pipeline(device: torch.device) -> OCRPipeline:
    """Construct OCRPipeline with both models on ``device``.

    OCRPipeline is not an nn.Module, so we move detector and recognizer
    explicitly *before* assembling the pipeline.
    """
    charset = ["-"] + [chr(c) for c in range(32, 127)]   # blank + 95 ASCII printables
    detector = DBNet(weights="DEFAULT").train(False).to(device)
    recognizer = CRNN(num_classes=len(charset), weights="DEFAULT").train(False).to(device)
    return OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(threshold=0.3),
        CTCGreedyDecoder(charset),
    )


def annotate(canvas: Tensor, boxes: Tensor | None, texts: list[str]) -> Tensor:
    """Draw boxes/labels on ``canvas`` (uint8). Empty strings render as ``<noise>``."""
    canvas = canvas.cpu().contiguous()
    if boxes is None or boxes.shape[0] == 0:
        return canvas
    labels = [text if text else "<noise>" for text in texts]
    return draw_bounding_boxes(
        canvas,
        boxes.detach().cpu(),
        labels=labels,
        colors="red",
        width=3,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device: {device}")

    pages = load_document(args.input, args.dpi)
    print(f"loaded {len(pages)} page(s) from {args.input.name}")
    if not pages:
        raise SystemExit("No pages to process.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(device)

    saved_paths: list[Path] = []
    for i, page in enumerate(pages):
        cropped = crop_to_multiple_of_32(page)
        image_float = cropped.float().div_(255.0).to(device)
        document = pipeline(image_float)
        n_boxes = 0 if document.bounding_boxes is None else document.bounding_boxes.shape[0]
        print(f"  page {i+1}/{len(pages)}: {tuple(cropped.shape)} -> {n_boxes} box(es)")

        annotated = annotate(page, document.bounding_boxes, document.text)
        out_path = args.output_dir / f"page_{i+1}_annotated.jpg"
        write_jpeg(annotated, str(out_path), quality=92)
        saved_paths.append(out_path)

    print()
    print(
        f"Processed {len(pages)} page(s). "
        f"Annotated images saved to {args.output_dir.resolve()}:"
    )
    for path in saved_paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
