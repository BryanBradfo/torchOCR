"""DBPostProcessor: probability map -> (K, 5) [batch_idx, x1, y1, x2, y2] boxes."""

import pytest
import torch

from torchocr import DBPostProcessor
from torchocr.models.detection import DBNetOutput


def test_extracts_one_aabb_per_image_with_text():
    prob = torch.zeros(3, 1, 64, 64)
    prob[0, 0, 10:30, 20:50] = 0.9   # text in image 0
    # image 1: empty, gets dropped
    prob[2, 0, 5:15, 5:25] = 0.9     # text in image 2
    out = DBNetOutput(probability=prob, threshold=torch.zeros_like(prob))

    boxes = DBPostProcessor(threshold=0.3)(out)
    assert boxes.shape == (2, 5)
    # batch indices are the original 0 and 2 (image 1 dropped)
    assert int(boxes[0, 0].item()) == 0
    assert int(boxes[1, 0].item()) == 2


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
    # Columns: batch_idx, x1, y1, x2, y2
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
