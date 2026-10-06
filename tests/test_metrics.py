"""DetectionHmean: ICDAR-2015 IoU protocol (greedy one-to-one matching at
IoU > 0.5, don't-care regions excluded, counts aggregated over the dataset)."""

import pytest
import torch

from torchocr.metrics import DetectionHmean, HmeanResult


def _quad(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    """Clockwise axis-aligned quadrilateral, ICDAR point order."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _quads(*boxes: tuple[float, float, float, float]) -> torch.Tensor:
    if not boxes:
        return torch.zeros(0, 4, 2)
    return torch.tensor([_quad(*b) for b in boxes], dtype=torch.float32)


def test_perfect_detection_scores_one():
    gt = _quads((0, 0, 10, 10), (20, 20, 40, 30))
    metric = DetectionHmean()
    metric.update(gt.clone(), gt)
    result = metric.compute()

    assert isinstance(result, HmeanResult)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.hmean == pytest.approx(1.0)
    assert (result.num_matched, result.num_gt, result.num_pred) == (2, 2, 2)


@pytest.mark.parametrize(("pred_height", "matched"), [(4.9, 0), (5.1, 1)])
def test_iou_threshold_is_strictly_greater_than_half(pred_height, matched):
    """IoU of a 10x10 GT with a 10xh prediction sharing a corner is h / 10."""
    metric = DetectionHmean()
    metric.update(_quads((0, 0, 10, pred_height)), _quads((0, 0, 10, 10)))
    assert metric.compute().num_matched == matched


def test_each_ground_truth_matches_at_most_one_prediction():
    """Two duplicates of the same GT: one TP, one FP."""
    gt = _quads((0, 0, 10, 10))
    metric = DetectionHmean()
    metric.update(_quads((0, 0, 10, 10), (0, 0, 10, 9)), gt)
    result = metric.compute()

    assert result.num_matched == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.hmean == pytest.approx(2 / 3)


def test_prediction_inside_dont_care_region_is_excluded():
    """A prediction covering a ``###`` region counts neither as TP nor FP."""
    gt = _quads((0, 0, 10, 10), (50, 50, 90, 70))
    ignore = torch.tensor([False, True])
    preds = _quads((0, 0, 10, 10), (52, 52, 88, 68))

    metric = DetectionHmean()
    metric.update(preds, gt, ignore)
    result = metric.compute()

    assert (result.num_matched, result.num_gt, result.num_pred) == (1, 1, 1)
    assert result.hmean == pytest.approx(1.0)


def test_prediction_mostly_outside_dont_care_is_a_false_positive():
    """Overlap with ``###`` must exceed half the *prediction's* area to be ignored."""
    gt = _quads((0, 0, 10, 10))
    ignore = torch.tensor([True])
    # 10x10 prediction; 4x10 of it (40%) overlaps the don't-care region.
    preds = _quads((6, 0, 16, 10))

    metric = DetectionHmean()
    metric.update(preds, gt, ignore)
    result = metric.compute()

    assert (result.num_matched, result.num_gt, result.num_pred) == (0, 0, 1)
    assert result.precision == 0.0


def test_counts_are_aggregated_across_images_not_averaged():
    metric = DetectionHmean()
    # Image 1: 2/2 found. Image 2: 0/2 found, no predictions.
    metric.update(_quads((0, 0, 10, 10), (20, 0, 30, 10)), _quads((0, 0, 10, 10), (20, 0, 30, 10)))
    metric.update(_quads(), _quads((0, 0, 10, 10), (20, 0, 30, 10)))
    result = metric.compute()

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.5)
    assert result.hmean == pytest.approx(2 / 3)


def test_axis_aligned_xyxy_boxes_are_accepted():
    gt = _quads((0, 0, 10, 10))
    metric = DetectionHmean()
    metric.update(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), gt)
    assert metric.compute().num_matched == 1


def test_self_intersecting_polygons_are_skipped_like_the_official_script():
    """Bow-tie quads are invalid; ICDAR's evaluator drops them silently."""
    bowtie = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]])
    metric = DetectionHmean()
    metric.update(bowtie, _quads((0, 0, 10, 10)))
    result = metric.compute()
    assert (result.num_gt, result.num_pred) == (1, 0)


def test_empty_dataset_returns_zeros():
    result = DetectionHmean().compute()
    assert (result.precision, result.recall, result.hmean) == (0.0, 0.0, 0.0)


def test_reset_clears_state():
    gt = _quads((0, 0, 10, 10))
    metric = DetectionHmean()
    metric.update(gt, gt)
    metric.reset()
    assert metric.compute().num_gt == 0


def test_rotated_quads_match():
    diamond = torch.tensor([[[5.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]]])
    metric = DetectionHmean()
    metric.update(diamond + 0.2, diamond)
    assert metric.compute().num_matched == 1


@pytest.mark.parametrize(
    "bad",
    [torch.zeros(3, 4, 3), torch.zeros(3, 2, 2), torch.zeros(3, 5), torch.zeros(4, 2)],
)
def test_bad_polygon_shape_raises(bad):
    with pytest.raises(ValueError, match=r"\(N, P, 2\)"):
        DetectionHmean().update(bad, _quads((0, 0, 10, 10)))


def test_ignore_mask_length_must_match_ground_truth():
    with pytest.raises(ValueError, match="ignore"):
        DetectionHmean().update(_quads(), _quads((0, 0, 1, 1)), torch.tensor([True, False]))


@pytest.mark.parametrize("kwarg", ["iou_threshold", "ignore_threshold"])
def test_thresholds_are_validated(kwarg):
    with pytest.raises(ValueError, match=kwarg):
        DetectionHmean(**{kwarg: 1.5})
