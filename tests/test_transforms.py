"""Inference presets bound to pretrained weights (``weights.transforms()``)."""

import pytest
import torch

from torchocr.transforms import DetectionPreset


def test_output_is_normalized_float_with_32_multiple_sides():
    image = torch.randint(0, 256, (3, 720, 1280), dtype=torch.uint8)
    out = DetectionPreset(max_side=960)(image)
    assert out.dtype == torch.float32
    assert out.shape == (3, 544, 960)


def test_batched_input_keeps_batch_dimension():
    out = DetectionPreset(max_side=None)(torch.zeros(2, 3, 50, 70, dtype=torch.uint8))
    assert out.shape == (2, 3, 64, 64)


def test_tiny_images_are_never_resized_to_zero():
    assert DetectionPreset(max_side=960)(torch.zeros(3, 5, 4000, dtype=torch.uint8)).shape == (3, 32, 960)


def test_normalization_and_channel_order():
    red = torch.zeros(3, 32, 32, dtype=torch.uint8)
    red[0] = 255
    mean, std = (0.1, 0.2, 0.3), (0.5, 0.5, 0.5)

    rgb = DetectionPreset(max_side=None, mean=mean, std=std)(red)
    bgr = DetectionPreset(max_side=None, mean=mean, std=std, channel_order="bgr")(red)

    assert rgb[:, 0, 0].tolist() == pytest.approx([(1 - 0.1) / 0.5, -0.2 / 0.5, -0.3 / 0.5])
    # Paddle-style: swap to BGR *before* applying the (unchanged) statistics.
    assert bgr[:, 0, 0].tolist() == pytest.approx([-0.1 / 0.5, -0.2 / 0.5, (1 - 0.3) / 0.5])


def test_rejects_non_uint8_and_wrong_channels():
    with pytest.raises(ValueError, match="uint8"):
        DetectionPreset()(torch.zeros(3, 32, 32))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, H, W\)"):
        DetectionPreset()(torch.zeros(1, 32, 32, dtype=torch.uint8))


def test_invalid_channel_order():
    with pytest.raises(ValueError, match="channel_order"):
        DetectionPreset(channel_order="rbg")


def test_recognition_preset_keeps_aspect_ratio_then_pads():
    from torchocr.transforms import RecognitionPreset

    crop = torch.full((3, 20, 50), 255, dtype=torch.uint8)
    out = RecognitionPreset(height=32, max_width=320)(crop)
    assert out.shape == (3, 32, 320)
    assert out[:, :, :80].allclose(torch.ones(1))  # 50 * 32/20 = 80 columns of (1-0.5)/0.5
    assert out[:, :, 80:].eq(0).all()


def test_recognition_preset_clamps_wide_crops():
    from torchocr.transforms import RecognitionPreset

    out = RecognitionPreset(height=32, max_width=100)(torch.zeros(3, 10, 900, dtype=torch.uint8))
    assert out.shape == (3, 32, 100)
