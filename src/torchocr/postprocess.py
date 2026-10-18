"""Post-processing utilities for detector outputs."""

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from torch import Tensor

from .models.detection import DBNetOutput
from .ops import order_quad_vertices


class DBPostProcessor:
    """Extract per-region bounding boxes from DBNet probability maps.

    Implements PaddleOCR's contour-based DB post-processing flow:
    threshold the probability map, run :func:`cv2.findContours`, fit
    a rotated minimum-area rectangle to each contour, score it by
    averaging the probability map under the rectangle mask, expand
    the polygon outward by ``unclip_ratio`` via :mod:`pyclipper`, and
    refit the rectangle. Boxes whose score is below ``box_thresh`` or
    whose minor side is below ``min_size`` are discarded.

    Calling the processor projects the rotated rectangles to
    axis-aligned ``(x1, y1, x2, y2)`` boxes (the v0.1 contract);
    :meth:`quadrilaterals` returns the rotated quads themselves.

    The output is shaped ``(K, 5)`` with columns
    ``[batch_idx, x1, y1, x2, y2]`` so it composes directly with
    :func:`torchvision.ops.roi_align`. ``K`` is the *total* number of
    accepted boxes across the batch (not capped at one per image).
    Coordinates are floats in input-pixel units, with ``x2``/``y2``
    being inclusive (``x2 = x1`` is a one-pixel-wide column) to match
    the convention of :func:`torchvision.ops.masks_to_boxes`.

    Note:
        ``cv2`` and ``pyclipper`` run on CPU. GPU probability maps are
        moved to CPU for processing, then the resulting box tensor is
        placed back on the original device.

    Args:
        threshold: Probability threshold for binarization. Default 0.3,
            matching PaddleOCR.
        box_thresh: Minimum mean probability under a candidate box for
            it to be kept. Default 0.7.
        max_candidates: Maximum number of contours examined per image.
            Default 1000.
        unclip_ratio: Polygon expansion ratio used by
            :func:`pyclipper.PyclipperOffset.Execute`. Default 1.5
            (matches PaddleOCR v2 weights). Larger values produce
            looser boxes.
        min_size: Reject candidates whose minor rotated-rectangle side
            is below this many pixels. Default 3.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1]; got {threshold}.")
        if not 0.0 <= box_thresh <= 1.0:
            raise ValueError(f"box_thresh must lie in [0, 1]; got {box_thresh}.")
        if max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive; got {max_candidates}.")
        if unclip_ratio <= 0:
            raise ValueError(f"unclip_ratio must be positive; got {unclip_ratio}.")
        if min_size < 1:
            raise ValueError(f"min_size must be >= 1; got {min_size}.")
        self.threshold = threshold
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def __call__(self, output: DBNetOutput) -> Tensor:
        """Return ``(K, 5)`` ``[batch_idx, x1, y1, x2, y2]`` axis-aligned boxes."""
        device = output.probability.device
        rows = [
            torch.cat([torch.full((quads.shape[0], 1), float(batch_idx)), quads.amin(1), quads.amax(1)], dim=1)
            for batch_idx, quads in enumerate(self._quadrilaterals_cpu(output))
        ]
        return torch.cat(rows).to(device) if rows else torch.zeros((0, 5), device=device)

    def quadrilaterals(self, output: DBNetOutput) -> list[Tensor]:
        """Return one ``(N_b, 4, 2)`` tensor of rotated text quads per image.

        Vertices are ``(x, y)`` input-pixel coordinates clipped to the
        page, in reading order (top-left, top-right, bottom-right,
        bottom-left) so they feed :func:`torchocr.ops.crop_quads`. Unlike
        :meth:`__call__`, the rotation of each region is preserved, which
        is what polygon-based benchmarks such as ICDAR-2015 score.
        """
        device = output.probability.device
        return [order_quad_vertices(quads).to(device) for quads in self._quadrilaterals_cpu(output)]

    def _quadrilaterals_cpu(self, output: DBNetOutput) -> list[Tensor]:
        probability = output.probability
        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError(
                f"Expected probability of shape (B, 1, H, W); got {tuple(probability.shape)}."
            )

        prob_np = probability[:, 0].detach().cpu().numpy()
        per_image: list[Tensor] = []
        for page in prob_np:
            h, w = page.shape
            mask = (page > self.threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            quads: list[np.ndarray] = []
            for contour in contours[: self.max_candidates]:
                box = self._rotated_box(contour)
                if box is None:
                    continue
                score = self._score(page, box)
                if score < self.box_thresh:
                    continue
                expanded = self._unclip(box)
                if expanded is None:
                    continue
                refit = self._rotated_box(expanded)
                if refit is None:
                    continue
                refit[:, 0] = np.clip(refit[:, 0], 0, w - 1)
                refit[:, 1] = np.clip(refit[:, 1], 0, h - 1)
                quads.append(refit)
            per_image.append(
                torch.from_numpy(np.stack(quads)) if quads else torch.zeros((0, 4, 2))
            )
        return per_image

    def _rotated_box(self, contour: np.ndarray) -> np.ndarray | None:
        """Fit a 4-point rotated rectangle to ``contour`` if it is large enough."""
        if contour.shape[0] < 4:
            return None
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        if min(w, h) < self.min_size:
            return None
        return cv2.boxPoints(rect).astype(np.float32)

    def _score(self, prob_map: np.ndarray, box: np.ndarray) -> float:
        """Average probability under the polygon defined by ``box`` (4x2)."""
        h, w = prob_map.shape
        xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        shifted = box.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
        return float(cv2.mean(prob_map[ymin : ymax + 1, xmin : xmax + 1], mask=mask)[0])

    def _unclip(self, box: np.ndarray) -> np.ndarray | None:
        """Expand ``box`` by ``unclip_ratio`` via :mod:`pyclipper` offsetting."""
        polygon = Polygon(box)
        if polygon.length < 1e-6:
            return None
        distance = polygon.area * self.unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0], dtype=np.float32).reshape(-1, 1, 2)
