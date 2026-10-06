"""High-level orchestration of detection, recognition, and decoding."""

from typing import Callable

import torch
from torch import Tensor, nn

from .charsets import load_charset
from .core.structures import DocumentTensor
from .decoders import CTCGreedyDecoder
from .models import CRNN, CRNN_ResNet34_VD_Weights, DBNet, DBNet_MobileNetV3_Large_05_Weights, WeightsEnum, get_weight
from .models.detection import DBNetOutput
from .ops import crop_quads, quad_to_box
from .postprocess import DBPostProcessor
from .transforms import RecognitionPreset


_PostProcessor = Callable[[DBNetOutput], Tensor]
_Decoder = Callable[[Tensor], list[str]]


class OCRPipeline:
    """End-to-end OCR pipeline: image -> populated :class:`DocumentTensor`.

    Composes a detector, post-processor, recognizer, and decoder:

    1. ``detector_transforms`` (if given) resizes/normalizes the image for
       the detector, e.g. ``detector_weights.transforms()``.
    2. The post-processor yields text regions. Rotated quads are used when it
       offers ``quadrilaterals()`` (as :class:`DBPostProcessor` does);
       otherwise its ``(K, 5) [batch_idx, x1, y1, x2, y2]`` boxes are used.
       Regions are mapped back to input-image coordinates.
    3. ``recognizer_transforms.normalize`` (if given) normalizes the
       *original-resolution* image, and every region is rectified out of it
       with :func:`torchocr.ops.crop_quads` -- aspect-preserving, padded,
       one batched call on the model's device.
    4. The recognizer and decoder turn crops into strings.

    Use :meth:`from_pretrained` to assemble all four from published weights.

    Args:
        detector: Module returning a :class:`DBNetOutput`.
        recognizer: Module mapping ``(K, C, H, W)`` crops to
            ``(T, K, num_classes)`` logits.
        post_processor: Callable mapping a :class:`DBNetOutput` to boxes
            (see above).
        decoder: Callable mapping ``(T, K, num_classes)`` logits to a
            list of K strings.
        crop_size: ``(height, max_width)`` of recognizer crops. Default:
            from ``recognizer_transforms`` when given, else ``(32, 128)``.
        detector_transforms: Optional preset applied to the input image
            before the detector (e.g. :class:`~torchocr.transforms.DetectionPreset`).
        recognizer_transforms: Optional :class:`~torchocr.transforms.RecognitionPreset`
            whose ``normalize`` is applied to the page before cropping.

    Without presets the input is fed to both models as-is, so it must
    already be normalized and sized for the detector (the v0.1 behavior).
    """

    def __init__(
        self,
        detector: nn.Module,
        recognizer: nn.Module,
        post_processor: _PostProcessor,
        decoder: _Decoder,
        *,
        crop_size: tuple[int, int] | None = None,
        detector_transforms: Callable[[Tensor], Tensor] | None = None,
        recognizer_transforms: RecognitionPreset | None = None,
    ) -> None:
        if crop_size is None:
            crop_size = (
                (recognizer_transforms.height, recognizer_transforms.max_width)
                if recognizer_transforms is not None
                else (32, 128)
            )
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
        self.detector_transforms = detector_transforms
        self.recognizer_transforms = recognizer_transforms

    @classmethod
    def from_pretrained(
        cls,
        detector_weights: WeightsEnum | str | None = None,
        recognizer_weights: WeightsEnum | str | None = None,
        device: torch.device | str | None = None,
    ) -> "OCRPipeline":
        """Assemble a pipeline from published weights, in eval mode.

        Each stage gets its own preprocessing (``weights.transforms()``); the
        post-processor uses ``detector_weights.meta["postprocess"]`` and the
        decoder the charset named in ``recognizer_weights.meta``.

        Args:
            detector_weights: DBNet weights enum member or ``"Cls.NAME"``.
                Default ``DBNet_MobileNetV3_Large_05_Weights.DEFAULT``.
            recognizer_weights: CRNN weights enum member or ``"Cls.NAME"``.
                Default ``CRNN_ResNet34_VD_Weights.DEFAULT``.
            device: Where to place both models. Default: CPU.
        """
        det_weights = _resolve(detector_weights, DBNet_MobileNetV3_Large_05_Weights)
        rec_weights = _resolve(recognizer_weights, CRNN_ResNet34_VD_Weights)
        detector = DBNet(weights=det_weights).eval().to(device)
        recognizer = CRNN(weights=rec_weights).eval().to(device)
        charset = load_charset(rec_weights.meta["charset"], rec_weights.meta["num_classes"])
        return cls(
            detector,
            recognizer,
            DBPostProcessor(**det_weights.meta.get("postprocess", {})),
            CTCGreedyDecoder(charset),
            detector_transforms=det_weights.transforms(),
            recognizer_transforms=rec_weights.transforms(),
        )

    def __call__(self, image: Tensor) -> DocumentTensor:
        """Run the full pipeline on a single ``(3, H, W)`` image.

        With presets, ``image`` is a ``uint8`` RGB tensor (as returned by
        :func:`torchocr.load_image`). It is returned unmodified in
        ``DocumentTensor.pixels``; regions are in its pixel coordinates.
        """
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected (3, H, W) image; got {tuple(image.shape)}.")
        device = next(self.detector.parameters(), image).device

        with torch.inference_mode():
            pixels = image.to(device)
            detector_input = (self.detector_transforms(pixels) if self.detector_transforms else pixels)[None]
            detection = self.detector(detector_input)
            quads = self._regions(detection).to(device=device, dtype=torch.float32)
            if quads.shape[0] == 0:
                return DocumentTensor(pixels=image)
            scale = torch.tensor(
                [pixels.shape[-1] / detector_input.shape[-1], pixels.shape[-2] / detector_input.shape[-2]],
                device=device,
            )
            quads = quads * scale

            page = self.recognizer_transforms.normalize(pixels) if self.recognizer_transforms else pixels
            crops, _ = crop_quads(
                page[None].float(),
                quads,
                torch.zeros(quads.shape[0], dtype=torch.long, device=device),
                height=self.crop_size[0],
                max_width=self.crop_size[1],
            )
            texts = self.decoder(self.recognizer(crops))

        return DocumentTensor(pixels=image, text=texts, bounding_boxes=quad_to_box(quads), polygons=quads)

    def _regions(self, detection: DBNetOutput) -> Tensor:
        """``(K, 4, 2)`` quads for the single image of ``detection``."""
        if hasattr(self.post_processor, "quadrilaterals"):
            return self.post_processor.quadrilaterals(detection)[0]
        boxes = self.post_processor(detection)[:, 1:]
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([torch.stack(p, -1) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))], dim=1)


def _resolve(weights: WeightsEnum | str | None, default: type[WeightsEnum]) -> WeightsEnum:
    if weights is None:
        return default.DEFAULT
    if isinstance(weights, str) and "." in weights:
        return get_weight(weights)
    if isinstance(weights, WeightsEnum):
        return weights
    return default.verify(weights)
