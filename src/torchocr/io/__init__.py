"""I/O helpers for reading OCR inputs."""

from pathlib import Path

from torch import Tensor
from torchvision.io import read_image

from .pdf import load_pdf


def load_image(path: str | Path) -> Tensor:
    """Load an image as a ``(C, H, W)`` tensor (typically ``uint8``)."""
    return read_image(str(path))


__all__ = ["load_image", "load_pdf"]
