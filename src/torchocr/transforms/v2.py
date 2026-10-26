"""Polygon-aware augmentation on top of :mod:`torchvision.transforms.v2`.

torchvision's v2 transforms move :class:`~torchvision.tv_tensors.KeyPoints`
of any leading shape, so ``(N, P, 2)`` text polygons follow flips,
rotations, affines, resizes and crops once wrapped. :class:`TextDetectionTarget`
is a pytree node, so its ``texts`` and ``ignore`` ride along untouched.
This module adds what is specific to text:

- :func:`wrap_sample` wraps a dataset sample as tv_tensors;
- :class:`RandomTextCrop` crops without cutting through any text region
  (PaddleOCR's ``EastRandomCropData``);
- :class:`SanitizeTextPolygons` drops regions moved off the canvas and
  marks partly visible ones don't-care, keeping ``texts`` and ``ignore``
  aligned.

Example::

    from torchvision.transforms import v2
    from torchocr.transforms.v2 import RandomTextCrop, SanitizeTextPolygons, wrap_sample

    augment = v2.Compose([
        v2.RandomRotation(10), v2.RandomAffine(0, scale=(0.5, 3.0)),
        RandomTextCrop(), v2.Resize(size=None, max_size=640), SanitizeTextPolygons(),
    ])
    image, target = augment(*wrap_sample(*dataset[0]))
    db_targets = MakeDBTargets()(target.polygons, target.ignore, image.shape[-2:])

Pass images as ``tv_tensors.Image`` (what :func:`wrap_sample` does): a
plain tensor would make torchvision treat the first plain tensor it meets
as the image, which could be the ``ignore`` mask.
"""

from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2._utils import query_size

from ..core.structures import TextDetectionTarget

__all__ = ["RandomTextCrop", "SanitizeTextPolygons", "wrap_sample"]


def wrap_sample(image: Tensor, target: TextDetectionTarget) -> tuple[tv_tensors.Image, TextDetectionTarget]:
    """Wrap a ``(C, H, W)`` image and its target as tv_tensors for :mod:`torchvision.transforms.v2`."""
    if image.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) image; got {tuple(image.shape)}.")
    polygons = tv_tensors.KeyPoints(target.polygons.float(), canvas_size=tuple(image.shape[-2:]))
    return tv_tensors.Image(image), TextDetectionTarget(polygons, list(target.texts), target.ignore)


class RandomTextCrop(v2.Transform):
    """Random crop whose borders never cut through a text polygon.

    Crop edges are drawn from rows and columns that no polygon's
    axis-aligned extent covers, so every region ends up fully inside or
    fully outside the crop on each axis, and the crop holds at least one
    region. After ``max_tries`` misses (e.g. text everywhere) the image is
    returned whole. Every polygon in the sample counts, don't-care ones
    included.

    Args:
        min_crop_side_ratio: Minimum crop side, as a fraction of the image side.
        max_tries: Attempts before giving up and keeping the whole image.
    """

    def __init__(self, min_crop_side_ratio: float = 0.1, max_tries: int = 50) -> None:
        super().__init__()
        if not 0 < min_crop_side_ratio <= 1:
            raise ValueError(f"min_crop_side_ratio must be in (0, 1]; got {min_crop_side_ratio}.")
        self.min_crop_side_ratio = min_crop_side_ratio
        self.max_tries = max_tries

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = query_size(flat_inputs)
        whole = {"top": 0, "left": 0, "height": height, "width": width}
        polys = [_as_polygons(i) for i in flat_inputs if isinstance(i, tv_tensors.KeyPoints)]
        polys = torch.cat(polys) if polys else torch.zeros(0, 1, 2)
        rows = torch.zeros(height, dtype=torch.bool)
        cols = torch.zeros(width, dtype=torch.bool)
        for poly in polys.round().long():
            x0, x1 = poly[:, 0].min().clamp(0, width), poly[:, 0].max().clamp(0, width)
            y0, y1 = poly[:, 1].min().clamp(0, height), poly[:, 1].max().clamp(0, height)
            cols[x0:x1] = True
            rows[y0:y1] = True
        free_rows, free_cols = (~rows).nonzero().flatten(), (~cols).nonzero().flatten()
        if len(polys) == 0 or len(free_rows) == 0 or len(free_cols) == 0:
            return whole
        for _ in range(self.max_tries):
            x0, x1 = _pick_span(free_cols)
            y0, y1 = _pick_span(free_rows)
            if x1 - x0 < self.min_crop_side_ratio * width or y1 - y0 < self.min_crop_side_ratio * height:
                continue
            if _overlaps(polys, x0, y0, x1, y1).any():
                return {"top": y0, "left": x0, "height": y1 - y0, "width": x1 - x0}
        return whole

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(F.crop, inpt, **params)


class SanitizeTextPolygons(v2.Transform):
    """Drop text regions outside the canvas; mark partly visible ones don't-care.

    Run it last, after the geometric transforms. A region is dropped when
    its axis-aligned extent misses the canvas (or when it has fewer than
    ``min_size`` pixels of extent along either axis); a region that
    sticks out of the canvas has a clipped word whose transcription no
    longer matches the pixels, so its ``ignore`` flag is set (with
    ``partial="ignore"``) or it is dropped (``partial="drop"``).
    Vertices are not clamped: :class:`~torchocr.transforms.MakeDBTargets`
    rasterizes polygons that overhang the canvas correctly.

    Args:
        partial: ``"ignore"`` (default), ``"drop"`` or ``"keep"`` for
            regions partly outside the canvas.
        min_size: Minimum width and height, in pixels, of a kept region.
    """

    def __init__(self, partial: str = "ignore", min_size: float = 1.0) -> None:
        super().__init__()
        if partial not in ("ignore", "drop", "keep"):
            raise ValueError(f'partial must be "ignore", "drop" or "keep"; got {partial!r}.')
        self.partial = partial
        self.min_size = min_size

    def forward(self, *inputs: Any) -> Any:
        # Like v2.SanitizeBoundingBoxes: filtering changes N, so it works on whole
        # targets rather than leaf by leaf.
        flat, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0],
                                  is_leaf=lambda x: isinstance(x, TextDetectionTarget))
        return tree_unflatten([self._sanitize(x) if isinstance(x, TextDetectionTarget) else x for x in flat], spec)

    def _sanitize(self, target: TextDetectionTarget) -> TextDetectionTarget:
        polygons = target.polygons
        if not isinstance(polygons, tv_tensors.KeyPoints):
            raise TypeError("SanitizeTextPolygons needs tv_tensors.KeyPoints polygons; wrap samples with wrap_sample().")
        if len(polygons) == 0:
            return target
        height, width = polygons.canvas_size
        x, y = polygons[..., 0].as_subclass(Tensor), polygons[..., 1].as_subclass(Tensor)
        x0, x1, y0, y1 = x.min(1).values, x.max(1).values, y.min(1).values, y.max(1).values
        visible = (x1 > 0) & (x0 < width) & (y1 > 0) & (y0 < height)
        big_enough = (x1.clamp(0, width) - x0.clamp(0, width) >= self.min_size) & (
            y1.clamp(0, height) - y0.clamp(0, height) >= self.min_size
        )
        inside = (x0 >= 0) & (x1 <= width) & (y0 >= 0) & (y1 <= height)
        keep = visible & big_enough
        ignore = target.ignore.clone()
        if self.partial == "drop":
            keep &= inside
        elif self.partial == "ignore":
            ignore |= ~inside
        texts = [text for text, k in zip(target.texts, keep.tolist()) if k]
        kept = tv_tensors.wrap(polygons.as_subclass(Tensor)[keep], like=polygons)  # indexing unwraps tv_tensors
        return TextDetectionTarget(kept, texts, ignore[keep])


def _as_polygons(points: tv_tensors.KeyPoints) -> Tensor:
    """``(N, P, 2)`` view of keypoints of any leading shape (a lone point is a 1-vertex polygon)."""
    points = points.as_subclass(Tensor)
    return points.reshape(-1, 1, 2) if points.ndim == 2 else points.reshape(-1, points.shape[-2], 2)


def _pick_span(free: Tensor) -> tuple[int, int]:
    """Two positions among sorted free indices, from two (possibly equal) runs so a crop can span text."""
    runs = torch.tensor_split(free, ((free.diff() != 1).nonzero().flatten() + 1).tolist())
    if len(runs) > 1:
        picks = [run[torch.randint(len(run), ())] for run in (runs[i] for i in torch.randint(len(runs), (2,)))]
    else:
        picks = list(free[torch.randint(len(free), (2,))])
    return int(min(picks)), int(max(picks))


def _overlaps(polys: Tensor, x0: int, y0: int, x1: int, y1: int) -> Tensor:
    """``(N,)`` mask of polygons whose extent overlaps the inside of the ``[x0, x1] x [y0, y1]`` rectangle."""
    x, y = polys[..., 0], polys[..., 1]
    return (x.max(1).values > x0) & (x.min(1).values < x1) & (y.max(1).values > y0) & (y.min(1).values < y1)
