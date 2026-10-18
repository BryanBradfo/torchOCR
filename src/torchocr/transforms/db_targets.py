"""Training targets for Differentiable Binarization (Liao et al., AAAI 2020)."""

from dataclasses import dataclass

import cv2
import numpy as np
import pyclipper
import torch
from torch import Tensor


@dataclass
class DBTargets:
    """Per-image supervision for :class:`torchocr.DBLoss`.

    Attributes:
        probability: ``(1, H, W)`` 0/1 map of the *shrunk* text polygons.
        threshold: ``(1, H, W)`` border map, in ``[thresh_min, thresh_max]``
            inside the dilated band around each polygon and 0 elsewhere
            (``DBLoss`` only supervises its non-zero pixels).
        mask: ``(1, H, W)`` valid-pixel mask; ``False`` over ignored
            (``###``) and too-small text so neither class is learned there.
    """

    probability: Tensor
    threshold: Tensor
    mask: Tensor


class MakeDBTargets:
    """Rasterize text polygons into DB probability / threshold / mask maps.

    Follows PaddleOCR's ``MakeShrinkMap`` + ``MakeBorderMap``:

    * Each polygon is shrunk by ``D = A (1 - r^2) / L`` (area ``A``,
      perimeter ``L``, ``r = shrink_ratio``) with :mod:`pyclipper`; if that
      splits it, ``r = 2 * shrink_ratio`` is tried before giving up and
      ignoring it. The shrunk region is the probability target.
    * The threshold target lives in the polygon dilated by ``D``: one minus
      the distance to the nearest edge divided by ``D``, clipped to
      ``[0, 1]``, then rescaled to ``[thresh_min, thresh_max]``. The exact
      point-to-segment distance is used.
    * Ignored polygons, polygons with a side under ``min_text_size`` and
      polygons whose shrink fails are zeroed in ``mask``.

    Polygons are clipped to the canvas first. Coordinates follow the
    pixel-index convention of :func:`cv2.fillPoly`.

    Args:
        shrink_ratio: ``r`` above. Default 0.4.
        min_text_size: Minimum polygon height/width in pixels. Default 8.
        thresh_min: Threshold target far from the border. Default 0.3.
        thresh_max: Threshold target on the border. Default 0.7.
    """

    def __init__(
        self,
        shrink_ratio: float = 0.4,
        min_text_size: int = 8,
        thresh_min: float = 0.3,
        thresh_max: float = 0.7,
    ) -> None:
        if not 0.0 < shrink_ratio < 1.0:
            raise ValueError(f"shrink_ratio must lie in (0, 1); got {shrink_ratio}.")
        if not 0.0 <= thresh_min < thresh_max <= 1.0:
            raise ValueError(
                f"Need 0 <= thresh_min < thresh_max <= 1; got thresh_min={thresh_min}, thresh_max={thresh_max}."
            )
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, polygons: Tensor | np.ndarray, ignore: Tensor | np.ndarray, size: tuple[int, int]) -> DBTargets:
        """Build targets for one ``size = (H, W)`` canvas.

        Args:
            polygons: ``(N, P, 2)`` vertices.
            ignore: ``(N,)`` bool don't-care flags.
            size: Output ``(H, W)``.
        """
        polys = np.asarray(polygons, dtype=np.float64)
        ignored = np.asarray(ignore, dtype=bool).reshape(-1).copy()
        if polys.ndim != 3 or polys.shape[-1] != 2:
            raise ValueError(f"Expected polygons of shape (N, P, 2); got {tuple(polys.shape)}.")
        if ignored.shape[0] != polys.shape[0]:
            raise ValueError(f"ignore has {ignored.shape[0]} entries but there are {polys.shape[0]} polygons.")

        height, width = size
        probability = np.zeros((height, width), dtype=np.float32)
        mask = np.ones((height, width), dtype=np.float32)
        border = np.zeros((height, width), dtype=np.float32)
        border_mask = np.zeros((height, width), dtype=np.uint8)

        for poly, is_ignored in zip(polys, ignored):
            poly = poly.copy()
            poly[:, 0] = poly[:, 0].clip(0, width - 1)
            poly[:, 1] = poly[:, 1].clip(0, height - 1)
            area, perimeter = _area(poly), _perimeter(poly)
            extent = poly.max(0) - poly.min(0)
            if is_ignored or area < 1 or extent.min() < self.min_text_size:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 0.0)
                continue

            shrunk = self._shrink(poly, area, perimeter)
            if shrunk is None:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 0.0)
                continue
            cv2.fillPoly(probability, [shrunk.astype(np.int32)], 1.0)
            self._draw_border(poly, area * (1 - self.shrink_ratio**2) / perimeter, border, border_mask)

        threshold = np.where(border_mask > 0, border * (self.thresh_max - self.thresh_min) + self.thresh_min, 0.0)
        return DBTargets(
            probability=torch.from_numpy(probability)[None],
            threshold=torch.from_numpy(threshold.astype(np.float32))[None],
            mask=torch.from_numpy(mask > 0)[None],
        )

    def _shrink(self, poly: np.ndarray, area: float, perimeter: float) -> np.ndarray | None:
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        for ratio in (self.shrink_ratio, 2 * self.shrink_ratio):
            if ratio >= 1:
                break
            shrunk = offset.Execute(-area * (1 - ratio**2) / perimeter)
            if len(shrunk) == 1:
                return np.asarray(shrunk[0], dtype=np.float64)
        return None

    @staticmethod
    def _draw_border(poly: np.ndarray, distance: float, canvas: np.ndarray, canvas_mask: np.ndarray) -> None:
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated = offset.Execute(distance)
        if not dilated:
            return
        dilated = np.asarray(dilated[0], dtype=np.int32)
        cv2.fillPoly(canvas_mask, [dilated], 1)

        height, width = canvas.shape
        x0, y0 = np.clip(dilated.min(0), 0, [width - 1, height - 1])
        x1, y1 = np.clip(dilated.max(0), 0, [width - 1, height - 1])
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1].astype(np.float64)
        nearest = np.full(xs.shape, np.inf)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            nearest = np.minimum(nearest, _segment_distance(xs, ys, a, b))
        closeness = 1 - np.clip(nearest / distance, 0, 1)
        region = canvas[y0 : y1 + 1, x0 : x1 + 1]
        np.maximum(region, closeness.astype(np.float32), out=region)


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _perimeter(poly: np.ndarray) -> float:
    return float(np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1).sum())


def _segment_distance(xs: np.ndarray, ys: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance from every ``(xs, ys)`` point to segment ``ab``."""
    ab = b - a
    length2 = max(float(ab @ ab), 1e-12)
    t = np.clip(((xs - a[0]) * ab[0] + (ys - a[1]) * ab[1]) / length2, 0, 1)
    return np.hypot(xs - a[0] - t * ab[0], ys - a[1] - t * ab[1])
