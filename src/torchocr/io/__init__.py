"""I/O helpers for reading OCR inputs."""

from pathlib import Path

from torch import Tensor

from .image import decode_image
from .pdf import load_pdf


def load_image(
    source: str | Path | bytes | Tensor,
    mode: str = "RGB",
    *,
    apply_exif_orientation: bool = True,
) -> Tensor:
    """Load an image as a ``(C, H, W)`` ``uint8`` tensor, RGB by default.

    Grayscale scans and RGBA screenshots are converted to the 3-channel RGB
    that pretrained models and :class:`~torchocr.OCRPipeline` expect; pass
    ``mode="UNCHANGED"`` to keep the file's own channels. Phone photos are
    turned the way their EXIF tag says, as image viewers show them.

    Args:
        source: Path, encoded bytes, or a 1-D ``uint8`` tensor of encoded bytes.
            JPEG, PNG, TIFF, WebP, BMP, GIF (first frame) and the other formats
            Pillow reads.
        mode: ``"RGB"``, ``"GRAY"``, ``"RGB_ALPHA"``, ``"GRAY_ALPHA"`` or
            ``"UNCHANGED"``. A ``torchvision.io.ImageReadMode`` member also works.
        apply_exif_orientation: Apply the EXIF orientation tag. Turn it off
            when annotations refer to the stored pixels, as in datasets.
    """
    return decode_image(source, mode, apply_exif_orientation=apply_exif_orientation)


__all__ = ["load_image", "load_pdf"]
