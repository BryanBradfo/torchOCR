"""DBPostProcessor: probability map -> (K, 5) [batch_idx, x1, y1, x2, y2] boxes via connected components."""

import pytest
import torch

from torchocr import DBPostProcessor
from torchocr.models.detection import DBNetOutput


def test_contiguous_region_yields_single_box():
    """A solid foreground rectangle in each image becomes one box per image."""
    prob = torch.zeros(3, 1, 64, 64)
    prob[0, 0, 10:30, 20:50] = 0.9   # text in image 0
    # image 1 has no foreground, gets dropped
    prob[2, 0, 5:15, 5:25] = 0.9     # text in image 2
    out = DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))

    boxes = DBPostProcessor(threshold=0.3)(out)
    assert boxes.shape == (2, 5)
    assert int(boxes[0, 0].item()) == 0   # image 1 dropped, indices skip
    assert int(boxes[1, 0].item()) == 2


def test_multiple_disconnected_regions_in_one_image():
    """Two well-separated blobs in one image yield two distinct boxes."""
    prob = torch.zeros(1, 1, 64, 128)
    prob[0, 0, 10:30, 10:30] = 0.9   # left blob
    prob[0, 0, 10:30, 80:100] = 0.9  # right blob (gap of 50 px between them)
    out = DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))

    boxes = DBPostProcessor(threshold=0.3)(out)
    assert boxes.shape == (2, 5)
    assert (boxes[:, 0] == 0).all()   # both from batch 0

    # Sort by x1 to make the assertion order-independent.
    sorted_boxes = boxes[boxes[:, 1].argsort()]
    # Left blob: x in [10, 29], y in [10, 29].
    assert sorted_boxes[0].tolist() == [0.0, 10.0, 10.0, 29.0, 29.0]
    # Right blob: x in [80, 99], y in [10, 29].
    assert sorted_boxes[1].tolist() == [0.0, 80.0, 10.0, 99.0, 29.0]


def test_components_distributed_across_batch_elements():
    """Each batch image's components contribute to the global K count."""
    prob = torch.zeros(2, 1, 32, 32)
    # Image 0 has two non-touching blobs.
    prob[0, 0, 5:10, 5:10] = 0.9
    prob[0, 0, 20:25, 20:25] = 0.9
    # Image 1 has a single blob.
    prob[1, 0, 10:20, 10:20] = 0.9
    out = DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))

    boxes = DBPostProcessor(threshold=0.3)(out)
    assert boxes.shape == (3, 5)
    assert (boxes[:, 0] == 0).sum().item() == 2
    assert (boxes[:, 0] == 1).sum().item() == 1


def test_empty_batch_returns_empty_tensor():
    prob = torch.zeros(2, 1, 32, 32)
    out = DBNetOutput(probability=prob, threshold=prob)
    boxes = DBPostProcessor(threshold=0.3)(out)
    assert boxes.shape == (0, 5)


def test_box_xyxy_columns_ordered():
    prob = torch.zeros(1, 1, 64, 64)
    prob[0, 0, 10:30, 20:50] = 0.9
    out = DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))
    boxes = DBPostProcessor(threshold=0.3)(out)
    _, x1, y1, x2, y2 = boxes[0]
    assert x1 < x2
    assert y1 < y2


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_rejects_threshold_outside_unit_range(threshold):
    with pytest.raises(ValueError):
        DBPostProcessor(threshold=threshold)


def test_rejects_bad_input_shape():
    bad = DBNetOutput(
        probability=torch.zeros(2, 64, 64),
        threshold=torch.zeros(2, 64, 64),
    )
    with pytest.raises(ValueError):
        DBPostProcessor()(bad)
