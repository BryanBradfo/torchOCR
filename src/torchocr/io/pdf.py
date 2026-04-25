"""PDF loading via PyMuPDF (``import fitz``)."""

from pathlib import Path

import fitz
import numpy as np
import torch
from torch import Tensor


def load_pdf(path: str | Path, dpi: int = 200) -> list[Tensor]:
    """Load a PDF and return one ``(3, H, W) uint8`` RGB tensor per page.

    Each page is rendered to a PyMuPDF ``Pixmap`` at the requested DPI
    and converted into a fresh PyTorch tensor that does not alias the
    pixmap's underlying memory. The pixmap and its sample buffer are
    released between pages, keeping peak memory bounded by the largest
    single page rather than the whole document.

    Args:
        path: Filesystem path to the PDF.
        dpi: Render resolution in dots-per-inch. Default 200 -- a sane
            OCR baseline. PyMuPDF's native unit is the PostScript point
            (1/72 inch); the conversion is ``Matrix(dpi/72, dpi/72)``.

    Returns:
        A list of ``(3, H, W) uint8`` tensors, one per page in document
        order. Channel order is RGB.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pages: list[Tensor] = []

    with fitz.open(str(path)) as document:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            arr_hwc = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3
            )
            arr_chw = np.transpose(arr_hwc, (2, 0, 1)).copy()
            pages.append(torch.from_numpy(arr_chw))

    return pages
