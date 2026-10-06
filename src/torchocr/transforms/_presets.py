"""Inference presets: the preprocessing a set of pretrained weights expects."""

from typing import Literal

import torch
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DetectionPreset(nn.Module):
    """Resize + normalize ``uint8`` RGB images for a text detector.

    The image is scaled (aspect ratio kept) so its longer side is about
    ``max_side``, then each side is rounded to the nearest multiple of
    32 -- the stride DBNet requires. Map predictions back to the input
    with ``scale = input_size / output_size`` per axis.

    Args:
        max_side: Target length of the longer side. ``None`` keeps the
            input scale and only rounds to multiples of 32.
        mean: Per-channel mean applied after scaling to ``[0, 1]``.
        std: Per-channel standard deviation.
        channel_order: ``"rgb"`` or ``"bgr"``. PaddleOCR-trained weights
            saw OpenCV-decoded BGR pixels normalized with RGB-ordered
            ImageNet statistics; ``"bgr"`` reproduces that by flipping
            channels before normalizing.

    Forward:
        ``(..., 3, H, W)`` ``uint8`` RGB -> ``(..., 3, H', W')`` ``float32``.
    """

    def __init__(
        self,
        max_side: int | None = 960,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        channel_order: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        super().__init__()
        if channel_order not in ("rgb", "bgr"):
            raise ValueError(f"channel_order must be 'rgb' or 'bgr'; got '{channel_order}'.")
        self.max_side = max_side
        self.mean = list(mean)
        self.std = list(std)
        self.channel_order = channel_order

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim < 3 or image.shape[-3] != 3:
            raise ValueError(f"Expected (..., 3, H, W) images; got {tuple(image.shape)}.")
        if image.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 images; got {image.dtype}.")

        height, width = image.shape[-2:]
        scale = 1.0 if self.max_side is None else self.max_side / max(height, width)
        size = [max(32, round(side * scale / 32) * 32) for side in (height, width)]
        image = F.resize(image, size, antialias=True)
        if self.channel_order == "bgr":
            image = image.flip(-3)
        image = F.to_dtype(image, torch.float32, scale=True)
        return F.normalize(image, self.mean, self.std)

    def extra_repr(self) -> str:
        return (
            f"max_side={self.max_side}, mean={self.mean}, std={self.std}, "
            f"channel_order='{self.channel_order}'"
        )


class RecognitionPreset(nn.Module):
    """Resize + normalize one ``uint8`` RGB text crop for a CTC recognizer.

    The crop is scaled to ``height`` with its aspect ratio kept (width
    clamped to ``[16, max_width]``), normalized to ``[-1, 1]`` by
    default, then right-padded with zeros to ``max_width`` so crops of
    different lengths stack into one batch.

    Args:
        height: Output height. Default 32 (CRNN's contract).
        max_width: Output width after padding. Default 320.
        mean: Per-channel mean applied after scaling to ``[0, 1]``.
        std: Per-channel standard deviation.
        channel_order: ``"rgb"`` or ``"bgr"``; see :class:`DetectionPreset`.

    Forward:
        ``(3, H, W)`` ``uint8`` RGB -> ``(3, height, max_width)`` ``float32``.
    """

    def __init__(
        self,
        height: int = 32,
        max_width: int = 320,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        channel_order: Literal["rgb", "bgr"] = "rgb",
    ) -> None:
        super().__init__()
        if channel_order not in ("rgb", "bgr"):
            raise ValueError(f"channel_order must be 'rgb' or 'bgr'; got '{channel_order}'.")
        self.height = height
        self.max_width = max_width
        self.mean = list(mean)
        self.std = list(std)
        self.channel_order = channel_order

    def forward(self, crop: Tensor) -> Tensor:
        if crop.ndim != 3 or crop.shape[0] != 3:
            raise ValueError(f"Expected a (3, H, W) crop; got {tuple(crop.shape)}.")
        if crop.dtype != torch.uint8:
            raise ValueError(f"Expected a uint8 crop; got {crop.dtype}.")

        h, w = crop.shape[-2:]
        width = min(self.max_width, max(16, round(self.height * w / max(h, 1))))
        crop = F.resize(crop, [self.height, width], antialias=True)
        if self.channel_order == "bgr":
            crop = crop.flip(0)
        crop = F.normalize(F.to_dtype(crop, torch.float32, scale=True), self.mean, self.std)
        return torch.nn.functional.pad(crop, (0, self.max_width - width))

    def extra_repr(self) -> str:
        return (
            f"height={self.height}, max_width={self.max_width}, mean={self.mean}, "
            f"std={self.std}, channel_order='{self.channel_order}'"
        )
