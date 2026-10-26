"""I/O helpers for reading OCR inputs."""

from pathlib import Path

from torch import Tensor
from torchvision.io import ImageReadMode, decode_image

from .pdf import load_pdf


def load_image(path: str | Path, mode: ImageReadMode = ImageReadMode.RGB) -> Tensor:
    """Load an image as a ``(C, H, W)`` ``uint8`` tensor, RGB by default.

    Grayscale scans and RGBA screenshots are converted to the 3-channel RGB
    that pretrained models and :class:`~torchocr.OCRPipeline` expect; pass
    ``mode=ImageReadMode.UNCHANGED`` to keep the file's own channels.
    """
    return decode_image(str(path), mode=mode)


__all__ = ["load_image", "load_pdf"]
