"""Core tensor structures for OCR pipelines."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class DocumentTensor:
    """Container for OCR-ready document data."""

    pixels: Tensor
    text: list[str] = field(default_factory=list)
    bounding_boxes: Tensor | None = None

    def to(self, *args: object, **kwargs: object) -> "DocumentTensor":
        """Return a copy moved to the requested device/dtype."""
        return DocumentTensor(
            pixels=self.pixels.to(*args, **kwargs),
            text=list(self.text),
            bounding_boxes=self.bounding_boxes.to(*args, **kwargs) if self.bounding_boxes is not None else None,
        )
