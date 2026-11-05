"""Image decoding with Pillow.

torchvision deprecated its image decoders in 0.29. Pillow is already installed
with torchvision, decodes JPEG exactly as ``torchvision.io.decode_image`` did
(same libjpeg-turbo output), and also reads TIFF, the usual format of scans.
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

# torchocr mode -> Pillow mode; same names as torchvision.io.ImageReadMode.
_MODES = {"RGB": "RGB", "GRAY": "L", "RGB_ALPHA": "RGBA", "GRAY_ALPHA": "LA", "UNCHANGED": None}


def _unchanged(image: Image.Image) -> Image.Image:
    """The file's own channels as 8-bit L, LA, RGB or RGBA."""
    if image.mode in ("L", "LA", "RGB", "RGBA"):
        return image
    if image.mode == "P":  # palette: expand, keeping transparency if there is any
        return image.convert("RGBA" if "transparency" in image.info else "RGB")
    if image.mode.startswith("I;16") or image.mode == "I":  # 16/32-bit gray: keep the 8 high bits
        bits = 16 if image.mode.startswith("I;16") else 32
        return Image.fromarray((np.asarray(image).astype(np.uint32) >> (bits - 8)).astype(np.uint8), "L")
    if image.mode == "PA":
        return image.convert("RGBA")
    return image.convert("L" if image.mode == "1" else "RGB")  # 1-bit, CMYK, YCbCr, LAB, HSV, F


def decode_image(
    source: str | Path | bytes | Tensor, mode: str = "RGB", *, apply_exif_orientation: bool = False
) -> Tensor:
    """Decode an image file, or its encoded bytes, to a ``(C, H, W)`` ``uint8`` tensor.

    Args:
        source: Path, encoded bytes, or a 1-D ``uint8`` tensor of encoded bytes.
        mode: ``"RGB"``, ``"GRAY"``, ``"RGB_ALPHA"``, ``"GRAY_ALPHA"`` or
            ``"UNCHANGED"`` (a ``torchvision.io.ImageReadMode`` member also works).
        apply_exif_orientation: Turn the image the way its EXIF tag says, as
            image viewers do.
    """
    name = getattr(mode, "name", mode)
    if name not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}; got {mode!r}.")
    if isinstance(source, Tensor):
        source = source.numpy().tobytes()
    with Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as image:
        if apply_exif_orientation:
            image = ImageOps.exif_transpose(image)
        image = _unchanged(image)
        if _MODES[name] is not None:
            image = image.convert(_MODES[name])
        pixels = np.array(image)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
