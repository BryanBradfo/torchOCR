"""End-to-end demo: synthesize a 2-page PDF, OCR the first page.

NOTE: Models are randomly initialized; the decoded text is noise. The
demo proves that the PDF -> tensor -> pipeline path works end-to-end.

Run from any working directory:
    python examples/demo_pdf.py
"""

from pathlib import Path

import fitz

from torchocr import (
    CTCGreedyDecoder,
    DBPostProcessor,
    OCRPipeline,
    load_pdf,
)
from torchocr.models import CRNN, DBNet


SCRIPT_DIR = Path(__file__).resolve().parent
PDF_PATH = SCRIPT_DIR / "sample_doc.pdf"


def write_synthetic_pdf(path: Path) -> None:
    """Create a 2-page A4 PDF with placeholder text via PyMuPDF."""
    document = fitz.open()
    try:
        for i in range(2):
            page = document.new_page(width=595, height=842)
            page.insert_text(
                (72, 110),
                f"Page {i + 1} - torchocr PDF demo",
                fontsize=20,
            )
            for line in range(8):
                page.insert_text(
                    (72, 160 + line * 28),
                    f"Lorem ipsum dolor sit amet line {line + 1}.",
                    fontsize=14,
                )
        document.save(str(path))
    finally:
        document.close()


def main() -> None:
    write_synthetic_pdf(PDF_PATH)
    print(f"wrote synthetic PDF -> {PDF_PATH.name}")

    # 100 dpi keeps the demo fast on CPU; production OCR can use the loader's
    # default 200 dpi. At 100 dpi an A4 page is roughly 826 x 1169 pixels.
    pages = load_pdf(PDF_PATH, dpi=100)
    print(f"loaded {len(pages)} page(s) via load_pdf")
    for i, page_tensor in enumerate(pages):
        print(f"  page {i+1}: shape={tuple(page_tensor.shape)} dtype={page_tensor.dtype}")

    # DBNet requires H, W divisible by 32; crop the top-left to satisfy that.
    page = pages[0]
    h_crop = (page.shape[1] // 32) * 32
    w_crop = (page.shape[2] // 32) * 32
    page_cropped = page[:, :h_crop, :w_crop]
    print(f"cropped page 1 -> {tuple(page_cropped.shape)}")

    # uint8 -> float [0, 1] for the pipeline's conv layers.
    image_float = page_cropped.float().div_(255.0)

    charset = ["-"] + [chr(c) for c in range(32, 127)]   # blank + 95 ASCII printables
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(charset)).train(False)
    pipeline = OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(threshold=0.3),
        CTCGreedyDecoder(charset),
    )

    document = pipeline(image_float)
    n_boxes = 0 if document.bounding_boxes is None else document.bounding_boxes.shape[0]
    print(f"pipeline succeeded: detected {n_boxes} box(es); decoded text: {document.text}")
    print()
    print("OK -- PDF -> tensor -> OCR pipeline path works end-to-end.")
    print("(Models are random-init; recognized text is noise.)")


if __name__ == "__main__":
    main()
