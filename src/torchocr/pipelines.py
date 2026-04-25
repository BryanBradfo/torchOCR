"""High-level orchestration of detection, recognition, and decoding."""

from typing import Callable

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align

from .core.structures import DocumentTensor
from .models.detection import DBNetOutput


_PostProcessor = Callable[[DBNetOutput], Tensor]
_Decoder = Callable[[Tensor], list[str]]


class OCRPipeline:
    """End-to-end OCR pipeline: image -> populated :class:`DocumentTensor`.

    Composes a detector, post-processor, recognizer, and decoder. The
    post-processor is expected to return ``(K, 5)`` boxes formatted as
    ``[batch_idx, x1, y1, x2, y2]``; these are passed straight to
    :func:`torchvision.ops.roi_align`, which crops and resizes every
    box to ``crop_size`` in one batched, on-device call. The resulting
    crops form a single batch fed to the recognizer.

    Args:
        detector: Module returning a :class:`DBNetOutput`.
        recognizer: Module mapping ``(K, C, H, W)`` crops to
            ``(T, K, num_classes)`` logits.
        post_processor: Callable mapping a :class:`DBNetOutput` to a
            ``(K, 5) [batch_idx, x1, y1, x2, y2]`` box tensor.
        decoder: Callable mapping ``(T, K, num_classes)`` logits to a
            list of K strings.
        crop_size: Target ``(H, W)`` of recognizer crops.
            Default ``(32, 128)`` — height matches the CRNN contract,
            width yields ``T = 32`` sequence steps.

    Note:
        Crops are stretched to ``crop_size`` regardless of their
        original aspect ratio. Pad-and-bucket strategies that preserve
        aspect ratio belong in a future ``AspectPreservingOCRPipeline``.
    """

    def __init__(
        self,
        detector: nn.Module,
        recognizer: nn.Module,
        post_processor: _PostProcessor,
        decoder: _Decoder,
        *,
        crop_size: tuple[int, int] = (32, 128),
    ) -> None:
        if crop_size[0] != 32:
            raise ValueError(
                f"crop_size height must equal 32 to match the CRNN contract; got {crop_size[0]}."
            )
        if crop_size[1] <= 0 or crop_size[1] % 4:
            raise ValueError(
                f"crop_size width must be a positive multiple of 4; got {crop_size[1]}."
            )
        self.detector = detector
        self.recognizer = recognizer
        self.post_processor = post_processor
        self.decoder = decoder
        self.crop_size = crop_size

    def __call__(self, image: Tensor) -> DocumentTensor:
        """Run the full pipeline on a single ``(3, H, W)`` image."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected (3, H, W) image; got {tuple(image.shape)}.")

        batched = image.unsqueeze(0)
        with torch.no_grad():
            detection = self.detector(batched)
            boxes = self.post_processor(detection)

            if boxes.shape[0] == 0:
                return DocumentTensor(pixels=image)

            crops = roi_align(
                batched,
                boxes,
                output_size=self.crop_size,
                spatial_scale=1.0,
                aligned=True,
            )
            logits = self.recognizer(crops)
            texts = self.decoder(logits)

        return DocumentTensor(
            pixels=image,
            text=texts,
            bounding_boxes=boxes[:, 1:].clone(),
        )
