"""DBPostProcessor: probability map -> (K, 5) [batch_idx, x1, y1, x2, y2] boxes
via contour detection + polygon offset + min-area-rectangle scoring."""

import pytest
import torch

from torchocr import DBPostProcessor
from torchocr.models.detection import DBNetOutput


def _strong(prob: torch.Tensor) -> DBNetOutput:
    """Wrap a synthetic probability map for the post-processor."""
    return DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))


def test_contiguous_region_yields_single_box():
    """A solid foreground rectangle in each image becomes one box per image."""
    prob = torch.zeros(3, 1, 96, 96)
    prob[0, 0, 20:60, 30:80] = 0.95  # text in image 0
    # image 1 has no foreground, gets dropped
    prob[2, 0, 10:50, 20:60] = 0.95  # text in image 2

    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (2, 5)
    # Image 1 has no detections, so the boxes are from images {0, 2}.
    assert sorted(int(b.item()) for b in boxes[:, 0]) == [0, 2]


def test_multiple_disconnected_regions_in_one_image():
    """Two well-separated blobs in one image yield two distinct boxes."""
    prob = torch.zeros(1, 1, 96, 192)
    prob[0, 0, 20:60, 20:60] = 0.95  # left blob
    prob[0, 0, 20:60, 130:170] = 0.95  # right blob (gap of 70 px)

    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (2, 5)
    assert (boxes[:, 0] == 0).all()

    # Sort by x1; loose containment of original blob with expansion tolerance.
    sorted_boxes = boxes[boxes[:, 1].argsort()]
    left, right = sorted_boxes[0], sorted_boxes[1]
    # Left blob center is (40, 40) - box should contain it.
    assert left[1] <= 40 <= left[3]
    assert left[2] <= 40 <= left[4]
    # Right blob center is (40, 150) - second box should contain it.
    assert right[1] <= 150 <= right[3]
    assert right[2] <= 40 <= right[4]
    # Boxes should not overlap on the x axis.
    assert left[3] < right[1]


def test_components_distributed_across_batch_elements():
    """Each batch image's components contribute to the global K count."""
    prob = torch.zeros(2, 1, 96, 96)
    prob[0, 0, 10:40, 10:40] = 0.95  # blob A in image 0
    prob[0, 0, 55:85, 55:85] = 0.95  # blob B in image 0
    prob[1, 0, 30:60, 30:60] = 0.95  # blob in image 1

    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (3, 5)
    assert (boxes[:, 0] == 0).sum().item() == 2
    assert (boxes[:, 0] == 1).sum().item() == 1


def test_empty_batch_returns_empty_tensor():
    prob = torch.zeros(2, 1, 64, 64)
    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (0, 5)


def test_box_xyxy_columns_ordered():
    prob = torch.zeros(1, 1, 96, 96)
    prob[0, 0, 20:50, 30:70] = 0.95
    boxes = DBPostProcessor()(_strong(prob))
    _, x1, y1, x2, y2 = boxes[0]
    assert x1 < x2
    assert y1 < y2


def test_low_confidence_regions_filtered_by_box_thresh():
    """Regions whose mean probability is below box_thresh are dropped."""
    prob = torch.zeros(1, 1, 96, 96)
    # Above threshold for binarization (0.3) but below box_thresh (0.7).
    prob[0, 0, 20:50, 20:50] = 0.4
    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (0, 5)


def test_box_contains_original_region():
    """Unclip expands the box; it must still cover the original blob."""
    prob = torch.zeros(1, 1, 128, 128)
    prob[0, 0, 30:60, 40:80] = 0.95  # blob spans rows 30..59, cols 40..79
    boxes = DBPostProcessor()(_strong(prob))
    assert boxes.shape == (1, 5)
    _, x1, y1, x2, y2 = boxes[0]
    assert x1 <= 40 and x2 >= 79
    assert y1 <= 30 and y2 >= 59


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_rejects_threshold_outside_unit_range(threshold):
    with pytest.raises(ValueError):
        DBPostProcessor(threshold=threshold)


@pytest.mark.parametrize("box_thresh", [-0.1, 1.5])
def test_rejects_box_thresh_outside_unit_range(box_thresh):
    with pytest.raises(ValueError):
        DBPostProcessor(box_thresh=box_thresh)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_candidates": 0},
        {"unclip_ratio": 0.0},
        {"min_size": 0},
    ],
)
def test_rejects_non_positive_params(kwargs):
    with pytest.raises(ValueError):
        DBPostProcessor(**kwargs)


def test_rejects_bad_input_shape():
    bad = DBNetOutput(
        probability=torch.zeros(2, 64, 64),
        threshold=torch.zeros(2, 64, 64),
    )
    with pytest.raises(ValueError):
        DBPostProcessor()(bad)
