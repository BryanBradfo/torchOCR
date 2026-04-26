"""load_pdf round-trip: PyMuPDF synthesizes -> torchocr loads."""

import fitz
import pytest
import torch

from torchocr import load_pdf


def _write_synthetic_pdf(path, n_pages: int = 2) -> None:
    document = fitz.open()
    try:
        for i in range(n_pages):
            page = document.new_page(width=400, height=400)
            page.insert_text((50, 50), f"page {i + 1}", fontsize=20)
        document.save(str(path))
    finally:
        document.close()


def test_load_pdf_returns_uint8_chw_tensors(tmp_path):
    pdf_path = tmp_path / "synth.pdf"
    _write_synthetic_pdf(pdf_path, n_pages=2)

    pages = load_pdf(pdf_path, dpi=72)
    assert len(pages) == 2
    for page in pages:
        assert page.dtype == torch.uint8
        assert page.ndim == 3
        assert page.shape[0] == 3   # RGB channel-first


def test_load_pdf_dpi_scales_resolution(tmp_path):
    pdf_path = tmp_path / "synth.pdf"
    _write_synthetic_pdf(pdf_path, n_pages=1)

    low = load_pdf(pdf_path, dpi=72)[0]
    high = load_pdf(pdf_path, dpi=144)[0]
    # Doubling DPI doubles each spatial dim (with possible 1px rounding).
    assert abs(high.shape[1] - 2 * low.shape[1]) <= 2
    assert abs(high.shape[2] - 2 * low.shape[2]) <= 2


def test_load_pdf_rejects_non_positive_dpi(tmp_path):
    with pytest.raises(ValueError):
        load_pdf(tmp_path / "anything.pdf", dpi=0)
    with pytest.raises(ValueError):
        load_pdf(tmp_path / "anything.pdf", dpi=-1)
