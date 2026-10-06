"""Core tensor structures for OCR pipelines."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class TextDetectionTarget:
    """Ground-truth text regions for one image.

    Attributes:
        polygons: ``(N, P, 2)`` float vertices in pixel ``(x, y)`` order.
            ICDAR-2015 regions are quadrilaterals (``P = 4``).
        texts: ``N`` transcriptions, aligned with ``polygons``.
        ignore: ``(N,)`` bool mask of don't-care regions (``###`` in
            ICDAR files). Evaluators exclude them from both recall and
            precision.
    """

    polygons: Tensor
    texts: list[str]
    ignore: Tensor

    def __post_init__(self) -> None:
        if self.polygons.ndim != 3 or self.polygons.shape[2] != 2:
            raise ValueError(f"Expected (N, P, 2) polygons; got {tuple(self.polygons.shape)}.")
        n = self.polygons.shape[0]
        if len(self.texts) != n or tuple(self.ignore.shape) != (n,):
            raise ValueError(
                f"polygons ({n}), texts ({len(self.texts)}) and ignore "
                f"{tuple(self.ignore.shape)} must describe the same N regions."
            )


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
