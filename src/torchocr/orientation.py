"""Page orientation: detect pages scanned sideways or upside down, and set them upright."""

import torch
from torch import Tensor, nn

from .models import PPLCNet, PPLCNet_x1_0_Weights, WeightsEnum


class PageOrientation(nn.Module):
    """Predict how far each page is turned, and turn it back.

    Wraps a 4-way classifier (0, 90, 180, 270 degrees) and its preprocessing.
    A page is only turned when the classifier is confident enough
    (``min_confidence``): most real pages arrive upright, and turning an
    upright page costs more than leaving a turned one as it is.

    Args:
        classifier: Module mapping ``(B, 3, h, w)`` preprocessed pages to
            ``(B, 4)`` logits, class ``c`` meaning "turned ``c * 90`` degrees
            clockwise".
        transforms: Preprocessing for one ``(3, H, W)`` ``uint8`` RGB page,
            e.g. ``weights.transforms()``.
        min_confidence: Smallest softmax probability at which a page is
            turned; below it the page is kept as it is (angle 0).

    Forward:
        ``(3, H, W)`` page, ``(B, 3, H, W)`` batch or list of pages (sizes
        may differ), ``uint8`` RGB -> ``(B,)`` ``int64`` clockwise angles in
        ``{0, 90, 180, 270}``.

    Example::

        orientation = PageOrientation.from_pretrained()
        page, angle = orientation.upright(load_image("scan.jpg"))
    """

    def __init__(self, classifier: nn.Module, transforms: nn.Module, min_confidence: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1]; got {min_confidence}.")
        self.classifier = classifier
        self.transforms = transforms
        self.min_confidence = min_confidence

    @classmethod
    def from_pretrained(
        cls,
        weights: WeightsEnum | str | None = None,
        device: torch.device | str | None = None,
        min_confidence: float | None = None,
    ) -> "PageOrientation":
        """Classifier, preset and threshold from published weights, in eval mode.

        Args:
            weights: :class:`~torchocr.models.PPLCNet_x1_0_Weights` member or
                name. Default ``PPLCNet_x1_0_Weights.DEFAULT``.
            device: Where to place the classifier. Default: CPU.
            min_confidence: Default ``weights.meta["min_confidence"]``.
        """
        weights = PPLCNet_x1_0_Weights.verify(weights) or PPLCNet_x1_0_Weights.DEFAULT
        if min_confidence is None:
            min_confidence = weights.meta["min_confidence"]
        classifier = PPLCNet(weights=weights).eval().to(device)
        return cls(classifier, weights.transforms(), min_confidence=min_confidence)

    @torch.inference_mode()
    def probabilities(self, pages: Tensor | list[Tensor]) -> Tensor:
        """``(B, 4)`` softmax probabilities of 0, 90, 180 and 270 degrees clockwise."""
        if isinstance(pages, Tensor):
            pages = [pages] if pages.ndim == 3 else list(pages)
        device = next(self.classifier.parameters()).device
        batch = torch.stack([self.transforms(page.to(device)) for page in pages])
        return self.classifier(batch).softmax(-1)

    def forward(self, pages: Tensor | list[Tensor]) -> Tensor:
        return _decide(self.probabilities(pages), self.min_confidence) * 90

    @torch.inference_mode()
    def upright(self, page: Tensor) -> tuple[Tensor, int]:
        """``(page set upright, clockwise angle it was turned)`` for one ``(3, H, W)`` page."""
        if page.ndim != 3 or page.shape[0] != 3:
            raise ValueError(f"Expected a (3, H, W) page; got {tuple(page.shape)}.")
        angle = int(self(page)[0])
        return rotate_page(page, angle), angle

    def extra_repr(self) -> str:
        return f"min_confidence={self.min_confidence}"


def rotate_page(page: Tensor, angle: int) -> Tensor:
    """Undo a clockwise turn of ``angle`` degrees (a multiple of 90): rotate ``(..., H, W)`` back."""
    if angle % 90:
        raise ValueError(f"angle must be a multiple of 90; got {angle}.")
    return torch.rot90(page, (angle // 90) % 4, dims=(-2, -1))


def _decide(probabilities: Tensor, min_confidence: float) -> Tensor:
    """``(B,)`` class to apply for ``(B, 4)`` probabilities (0 keeps the page as it is).

    The most likely class is applied only when its probability reaches
    ``min_confidence``. On DocLayNet validation a single threshold did as well
    as a stricter one for 180 degrees (the most frequent costly error), so
    there is one knob.
    """
    confidence, classes = probabilities.max(-1)
    return torch.where(confidence >= min_confidence, classes, torch.zeros_like(classes))
