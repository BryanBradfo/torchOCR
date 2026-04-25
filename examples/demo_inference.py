"""End-to-end visual demo for torchocr.

The script generates a synthetic document image, saves it as
``sample_doc.jpg``, runs it through :class:`OCRPipeline`, and writes
an annotated visualization to ``demo_output.jpg`` next to this script.

NOTE: Models are currently randomly initialized. The detected boxes
and text are noise, but this proves the end-to-end tensor pipeline
and shape contracts function perfectly.

Run from any working directory:
    python examples/demo_inference.py
"""

from pathlib import Path

import torch
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes

from torchocr import CTCGreedyDecoder, DBPostProcessor, OCRPipeline
from torchocr.io import load_image
from torchocr.models import CRNN, DBNet


SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = SCRIPT_DIR / "sample_doc.jpg"
OUTPUT_PATH = SCRIPT_DIR / "demo_output.jpg"


def make_synthetic_document(height: int = 800, width: int = 608) -> torch.Tensor:
    """Build a (3, H, W) uint8 white image with dark text-line bars.

    H and W must be multiples of 32 to satisfy DBNet's input contract.
    """
    if height % 32 or width % 32:
        raise ValueError(
            f"Synthetic document dims must be divisible by 32; got {height}x{width}."
        )

    image = torch.full((3, height, width), 255, dtype=torch.uint8)
    rng = torch.Generator().manual_seed(0)

    # Title bar across the top.
    image[:, 64:96, 80 : width - 80] = 30

    # 10 mock paragraph lines with varied left/right margins.
    line_height = 22
    margin_top = 144
    margin_bottom = 80
    available = height - margin_top - margin_bottom
    line_count = 10
    spacing = available // line_count
    for i in range(line_count):
        y = margin_top + i * spacing
        line_width = int(
            torch.randint(width // 2, width - 100, (1,), generator=rng).item()
        )
        x = int(
            torch.randint(60, max(61, width - line_width - 60), (1,), generator=rng).item()
        )
        image[:, y : y + line_height, x : x + line_width] = 50
    return image


def main() -> None:
    torch.manual_seed(0)

    # 1. Synthesize a document image and persist it as JPEG.
    sample = make_synthetic_document()
    write_jpeg(sample, str(SAMPLE_PATH), quality=92)
    print(f"wrote synthetic document -> {SAMPLE_PATH.name} {tuple(sample.shape)}")

    # 2. Load the image back through torchocr's I/O utility.
    image = load_image(SAMPLE_PATH)
    print(f"loaded {SAMPLE_PATH.name}: shape={tuple(image.shape)} dtype={image.dtype}")

    # 3. uint8 -> float [0, 1] for the pipeline's conv layers.
    image_float = image.float().div_(255.0)

    # 4. Build the pipeline. blank + 95 ASCII printables = 96 classes.
    charset = ["-"] + [chr(c) for c in range(32, 127)]
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(charset)).train(False)
    pipeline = OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(threshold=0.3),
        CTCGreedyDecoder(charset),
    )

    # 5. Run the full pipeline.
    document = pipeline(image_float)
    box_count = 0 if document.bounding_boxes is None else document.bounding_boxes.shape[0]
    print(f"detected {box_count} box(es); decoded text: {document.text}")

    # 6. Visualize on the uint8 sample (draw_bounding_boxes wants uint8).
    if box_count > 0:
        labels = [text if text else "<noise>" for text in document.text]
        annotated = draw_bounding_boxes(
            sample,
            document.bounding_boxes,
            labels=labels,
            colors="red",
            width=3,
        )
    else:
        print("no detections — saving the input image unannotated")
        annotated = sample

    write_jpeg(annotated, str(OUTPUT_PATH), quality=92)
    print(f"wrote annotated visualization -> {OUTPUT_PATH.name}")

    # 7. Disclaimer (also at the top of this file).
    print()
    print("NOTE: Models are currently randomly initialized. The detected boxes")
    print("and text are noise, but this proves the end-to-end tensor pipeline")
    print("and shape contracts function perfectly.")


if __name__ == "__main__":
    main()
