"""Text-detection metrics."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import shapely
from torch import Tensor

from .recognition import normalize_text


@dataclass(frozen=True)
class HmeanResult:
    """Dataset-level detection scores under the ICDAR IoU protocol.

    ``num_gt`` and ``num_pred`` exclude don't-care regions and invalid
    polygons; they are the denominators of ``recall`` and ``precision``.
    """

    precision: float
    recall: float
    hmean: float
    num_matched: int
    num_gt: int
    num_pred: int


class DetectionHmean:
    """ICDAR-2015 text-detection precision / recall / hmean.

    Reproduces the official RRC "IoU" evaluation (the protocol behind
    every published ICDAR-2015 detection number, and PaddleOCR's
    ``DetectionIoUEvaluator``):

    1. Polygons that are not valid (e.g. self-intersecting) are dropped.
    2. A prediction is *don't care* if more than ``ignore_threshold`` of
       its own area lies inside a single ignored ground-truth region
       (``###`` transcriptions in ICDAR files).
    3. Remaining pairs are matched greedily and one-to-one in
       (ground truth, prediction) order whenever IoU > ``iou_threshold``.
    4. Matches and care counts are summed over all images before the
       ratios are taken, so large images are not averaged away.

    Usage follows the stateful-metric idiom::

        metric = DetectionHmean()
        for preds, target in loader:
            metric.update(preds, target.polygons, target.ignore)
        result = metric.compute()  # HmeanResult

    Args:
        iou_threshold: Minimum IoU (exclusive) for a match. Default 0.5.
        ignore_threshold: Fraction (exclusive) of a prediction's area that
            must overlap an ignored region to discard it. Default 0.5.
    """

    def __init__(self, iou_threshold: float = 0.5, ignore_threshold: float = 0.5) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must lie in [0, 1]; got {iou_threshold}.")
        if not 0.0 <= ignore_threshold <= 1.0:
            raise ValueError(f"ignore_threshold must lie in [0, 1]; got {ignore_threshold}.")
        self.iou_threshold = iou_threshold
        self.ignore_threshold = ignore_threshold
        self.reset()

    def reset(self) -> None:
        self._matched = 0
        self._gt_care = 0
        self._pred_care = 0

    def update(self, preds: Tensor, targets: Tensor, ignore: Tensor | None = None) -> None:
        """Accumulate one image.

        Args:
            preds: Predicted regions, either ``(N, P, 2)`` polygons with
                ``P >= 3`` vertices or ``(N, 4)`` ``xyxy`` boxes.
            targets: Ground-truth regions in the same formats.
            ignore: Optional ``(M,)`` bool mask over ``targets`` marking
                don't-care regions. Default: nothing ignored.
        """
        pairs, gt_care, pred_care = self._match(preds, targets, ignore)
        self._matched += len(pairs)
        self._gt_care += gt_care
        self._pred_care += pred_care

    def _match(
        self, preds: Tensor, targets: Tensor, ignore: Tensor | None
    ) -> tuple[list[tuple[int, int]], int, int]:
        """Return matched ``(target_index, pred_index)`` pairs and the care counts."""
        pred_polys = _to_polygons(preds, "preds")
        gt_polys = _to_polygons(targets, "targets")
        if ignore is None:
            gt_ignore = np.zeros(len(gt_polys), dtype=bool)
        else:
            gt_ignore = ignore.detach().cpu().numpy().astype(bool).reshape(-1)
            if gt_ignore.shape[0] != len(gt_polys):
                raise ValueError(
                    f"ignore has {gt_ignore.shape[0]} entries but targets has {len(gt_polys)} regions."
                )

        gt_index = np.flatnonzero(shapely.is_valid(gt_polys))
        pred_index = np.flatnonzero(shapely.is_valid(pred_polys))
        gt_polys, gt_ignore = gt_polys[gt_index], gt_ignore[gt_index]
        pred_polys = pred_polys[pred_index]

        pred_ignore = np.zeros(len(pred_polys), dtype=bool)
        if gt_ignore.any() and len(pred_polys):
            dont_care = gt_polys[gt_ignore]
            overlap = shapely.area(shapely.intersection(pred_polys[:, None], dont_care[None, :]))
            pred_area = shapely.area(pred_polys)[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction = np.where(pred_area > 0, overlap / pred_area, 0.0)
            pred_ignore = (fraction > self.ignore_threshold).any(axis=1)

        pairs: list[tuple[int, int]] = []
        if len(gt_polys) and len(pred_polys):
            inter = shapely.area(shapely.intersection(gt_polys[:, None], pred_polys[None, :]))
            union = shapely.area(shapely.union(gt_polys[:, None], pred_polys[None, :]))
            with np.errstate(divide="ignore", invalid="ignore"):
                iou = np.where(union > 0, inter / union, 0.0)
            # Greedy in index order -- intentionally not Hungarian, to stay
            # bit-compatible with the official script.
            pred_taken = pred_ignore.copy()
            for g in np.flatnonzero(~gt_ignore):
                for p in np.flatnonzero(~pred_taken & (iou[g] > self.iou_threshold)):
                    pred_taken[p] = True
                    pairs.append((int(gt_index[g]), int(pred_index[p])))
                    break
        return pairs, int((~gt_ignore).sum()), int((~pred_ignore).sum())

    def compute(self) -> HmeanResult:
        recall = self._matched / self._gt_care if self._gt_care else 0.0
        precision = self._matched / self._pred_care if self._pred_care else 0.0
        hmean = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return HmeanResult(
            precision=precision,
            recall=recall,
            hmean=hmean,
            num_matched=self._matched,
            num_gt=self._gt_care,
            num_pred=self._pred_care,
        )


def _to_polygons(regions: Tensor, name: str) -> np.ndarray:
    """Convert ``(N, P, 2)`` polygons or ``(N, 4)`` xyxy boxes to shapely polygons."""
    coords = regions.detach().cpu().double().numpy()
    if coords.ndim == 2 and coords.shape[1] == 4:
        x1, y1, x2, y2 = coords.T
        coords = np.stack([np.stack([x1, y1], -1), np.stack([x2, y1], -1),
                           np.stack([x2, y2], -1), np.stack([x1, y2], -1)], axis=1)
    if coords.ndim != 3 or coords.shape[2] != 2 or coords.shape[1] < 3:
        raise ValueError(
            f"{name} must be (N, P, 2) polygons with P >= 3 or (N, 4) xyxy boxes; "
            f"got {tuple(regions.shape)}."
        )
    if coords.shape[0] == 0:
        return np.empty(0, dtype=object)
    return shapely.polygons(coords)


class EndToEndHmean(DetectionHmean):
    """End-to-end text spotting hmean: a detection counts only if it is also read correctly.

    Regions are matched exactly as in :class:`DetectionHmean`; a matched pair
    is a true positive only when the normalized transcriptions are equal.
    This follows the ICDAR-2015 end-to-end task with a generic vocabulary
    (no lexicon), simplified: text equality uses
    :class:`~torchocr.metrics.RecognitionAccuracy`'s normalization instead of
    the official script's special-character rules.

    Args:
        iou_threshold, ignore_threshold: As in :class:`DetectionHmean`.
        case_sensitive, alphanumeric_only: Text normalization before
            comparison. Defaults follow the scene-text convention
            (case-insensitive, letters and digits only).
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        ignore_threshold: float = 0.5,
        case_sensitive: bool = False,
        alphanumeric_only: bool = True,
    ) -> None:
        super().__init__(iou_threshold, ignore_threshold)
        self.case_sensitive = case_sensitive
        self.alphanumeric_only = alphanumeric_only

    def update(  # type: ignore[override]
        self,
        preds: Tensor,
        targets: Tensor,
        ignore: Tensor | None,
        pred_texts: Sequence[str],
        target_texts: Sequence[str],
    ) -> None:
        """Accumulate one image; ``pred_texts`` / ``target_texts`` align with the regions."""
        if len(pred_texts) != preds.shape[0]:
            raise ValueError(f"pred_texts has {len(pred_texts)} entries for {preds.shape[0]} predictions.")
        if len(target_texts) != targets.shape[0]:
            raise ValueError(f"target_texts has {len(target_texts)} entries for {targets.shape[0]} targets.")
        pairs, gt_care, pred_care = self._match(preds, targets, ignore)
        norm = lambda t: normalize_text(t, self.case_sensitive, self.alphanumeric_only)  # noqa: E731
        self._matched += sum(norm(pred_texts[p]) == norm(target_texts[g]) for g, p in pairs)
        self._gt_care += gt_care
        self._pred_care += pred_care
