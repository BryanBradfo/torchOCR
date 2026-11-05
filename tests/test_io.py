"""load_pdf round-trip: PyMuPDF synthesizes -> torchocr loads."""

import subprocess
import sys

import pymupdf
import pytest
import torch

from torchocr import load_pdf


def _write_synthetic_pdf(path, n_pages: int = 2) -> None:
    document = pymupdf.open()
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


def test_load_image_converts_grayscale_and_rgba_to_rgb(tmp_path):
    from PIL import Image
    from torchvision.io import ImageReadMode

    from torchocr.io import load_image

    Image.new("L", (20, 10), 128).save(tmp_path / "gray.png")
    Image.new("RGBA", (20, 10), (10, 20, 30, 0)).save(tmp_path / "rgba.png")
    assert load_image(tmp_path / "gray.png").shape == (3, 10, 20)
    assert load_image(tmp_path / "rgba.png").shape == (3, 10, 20)
    assert load_image(tmp_path / "gray.png", mode="UNCHANGED").shape == (1, 10, 20)
    assert load_image(tmp_path / "rgba.png", mode="UNCHANGED").shape == (4, 10, 20)
    # torchvision's enum is still accepted.
    assert load_image(tmp_path / "gray.png", mode=ImageReadMode.UNCHANGED).shape == (1, 10, 20)


def test_load_image_reads_bytes_tiff_and_16_bit_scans(tmp_path):
    import numpy as np
    from PIL import Image

    from torchocr.io import load_image

    rgb = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    Image.fromarray(rgb).save(tmp_path / "scan.tif")
    expected = torch.from_numpy(rgb).permute(2, 0, 1)
    assert torch.equal(load_image(tmp_path / "scan.tif"), expected)
    data = (tmp_path / "scan.tif").read_bytes()
    assert torch.equal(load_image(data), expected)
    assert torch.equal(load_image(torch.frombuffer(bytearray(data), dtype=torch.uint8)), expected)

    deep = np.full((10, 20), 0xABCD, dtype=np.uint16)
    Image.fromarray(deep).save(tmp_path / "deep.png")
    assert load_image(tmp_path / "deep.png", mode="UNCHANGED").unique().tolist() == [0xAB]


def test_load_image_applies_exif_orientation_unless_asked_not_to(tmp_path):
    from PIL import Image

    from torchocr.io import load_image

    image = Image.new("RGB", (20, 10))
    exif = image.getexif()
    exif[0x0112] = 6  # stored sideways: viewers turn it 90 degrees clockwise
    image.save(tmp_path / "phone.jpg", exif=exif)
    assert load_image(tmp_path / "phone.jpg").shape == (3, 20, 10)
    assert load_image(tmp_path / "phone.jpg", apply_exif_orientation=False).shape == (3, 10, 20)


def test_load_image_rejects_unknown_modes(tmp_path):
    from torchocr.io import load_image

    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "missing.png", mode="CMYK")


def test_image_decoding_does_not_use_torchvision_io(tmp_path):
    """torchvision deprecated its image decoders in 0.29; torchocr must not depend on them.

    The decoders are removed before torchocr is imported, in a fresh interpreter, so that a
    ``from torchvision.io import decode_image`` anywhere in torchocr would pick up the stub.
    """
    script = f"""
import torchvision.io
def removed(*args, **kwargs):
    raise AssertionError("torchvision.io image decoding was called")
for name in ("decode_image", "read_image", "decode_jpeg", "decode_png", "read_file"):
    setattr(torchvision.io, name, removed)
from PIL import Image
import torchocr.datasets, torchocr.io
Image.new("RGB", (4, 4)).save({str(tmp_path / "a.png")!r})
assert torchocr.io.load_image({str(tmp_path / "a.png")!r}).shape == (3, 4, 4)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_emits_no_pymupdf_deprecation():
    # PyMuPDF warns once per process when imported as `fitz`: check in a fresh interpreter.
    subprocess.run(
        [sys.executable, "-W", "error::FutureWarning", "-c", "import torchocr"], check=True
    )
