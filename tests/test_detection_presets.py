"""Training augmentation for DB (references/detection/presets.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references" / "detection"))

from presets import DBTrainPreset, random_text_crop  # noqa: E402

from torchocr.datasets import TextDetectionTarget  # noqa: E402
from torchocr.transforms import DBTargets  # noqa: E402


def _sample(height: int = 180, width: int = 320) -> tuple[torch.Tensor, TextDetectionTarget]:
    image = torch.zeros(3, height, width, dtype=torch.uint8)
    image[:, 40:70, 30:150] = 255
    image[:, 100:130, 180:300] = 255
    polygons = torch.tensor(
        [[[30, 40], [150, 40], [150, 70], [30, 70]], [[180, 100], [300, 100], [300, 130], [180, 130]]],
        dtype=torch.float32,
    )
    return image, TextDetectionTarget(polygons, ["one", "two"], torch.tensor([False, False]))


def test_output_contract():
    torch.manual_seed(0)
    image, target = _sample()
    out, targets = DBTrainPreset(size=128)(image, target)

    assert out.shape == (3, 128, 128) and out.dtype == torch.float32
    assert isinstance(targets, DBTargets)
    assert targets.probability.shape == (1, 128, 128)


def test_text_survives_augmentation():
    """Over many random draws, the targets still land on bright (text) pixels."""
    image, target = _sample()
    preset = DBTrainPreset(size=128, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), channel_order="rgb")
    torch.manual_seed(1)
    hits = 0
    for _ in range(20):
        out, targets = preset(image, target)
        text = targets.probability[0] > 0
        if text.any():
            hits += 1
            # Probability targets sit on the white bars, never on black background.
            assert out[0][text].mean() > 0.9
    assert hits >= 15


def test_random_text_crop_never_cuts_through_text():
    rng = np.random.default_rng(0)
    polys = np.array([[[30, 40], [150, 40], [150, 70], [30, 70]]], dtype=np.float64)
    for _ in range(50):
        x, y, w, h = random_text_crop(polys, (180, 320), rng=rng)
        inside_x = (x <= 30 and 150 <= x + w) or x + w <= 30 or x >= 150
        inside_y = (y <= 40 and 70 <= y + h) or y + h <= 40 or y >= 70
        assert inside_x and inside_y, (x, y, w, h)


def test_ignore_flags_follow_their_polygons():
    image, target = _sample()
    target = TextDetectionTarget(target.polygons, target.texts, torch.tensor([True, True]))
    torch.manual_seed(0)
    _, targets = DBTrainPreset(size=128)(image, target)
    assert targets.probability.sum() == 0  # everything ignored


@pytest.mark.parametrize("size", [0, 100])
def test_size_must_be_a_positive_multiple_of_32(size):
    with pytest.raises(ValueError, match="multiple of 32"):
        DBTrainPreset(size=size)
