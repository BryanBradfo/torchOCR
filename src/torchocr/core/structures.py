"""Core tensor structures for OCR pipelines."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class DocumentTensor:
    """Container for OCR-ready document data."""

    pixels: Tensor
    text: list[str] = field(default_factory=list)
    bounding_boxes: Tensor | None = None
