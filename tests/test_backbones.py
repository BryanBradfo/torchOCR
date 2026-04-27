"""Shape contracts for the ResNet-VD backbone."""

import pytest
import torch

from torchocr.models.backbones import ResNetVd


def test_resnet_vd_returns_four_level_pyramid():
    backbone = ResNetVd(depth=18).train(False)
    with torch.no_grad():
        out = backbone(torch.randn(2, 3, 64, 64))
    assert set(out) == {"c2", "c3", "c4", "c5"}


def test_resnet_vd_output_strides():
    """c2..c5 are at strides 4, 8, 16, 32 with channels (64, 128, 256, 512)."""
    backbone = ResNetVd(depth=18).train(False)
    with torch.no_grad():
        out = backbone(torch.randn(1, 3, 128, 128))
    assert out["c2"].shape == (1, 64, 32, 32)   # /4
    assert out["c3"].shape == (1, 128, 16, 16)  # /8
    assert out["c4"].shape == (1, 256, 8, 8)    # /16
    assert out["c5"].shape == (1, 512, 4, 4)    # /32


def test_resnet_vd_out_channels_property():
    backbone = ResNetVd(depth=18)
    assert backbone.out_channels == (64, 128, 256, 512)


def test_resnet_vd_rejects_unsupported_depth():
    with pytest.raises(NotImplementedError):
        ResNetVd(depth=50)  # type: ignore[arg-type]


def test_resnet_vd_rejects_non_4d_input():
    backbone = ResNetVd(depth=18)
    with pytest.raises(ValueError):
        backbone(torch.randn(3, 64, 64))


def test_resnet_vd_paddle_compatible_state_dict_keys():
    """Sanity-check that keys follow the PaddleOCR-aligned naming.

    The converter assumes specific submodule names (conv1_1, stages.N,
    bb_<i>_<j>, _conv, _batch_norm). If these drift, the converter
    breaks silently. This test pins the contract.
    """
    backbone = ResNetVd(depth=18)
    keys = set(backbone.state_dict().keys())
    expected = {
        "conv1_1._conv.weight",
        "conv1_1._batch_norm.weight",
        "conv1_1._batch_norm.running_mean",
        "conv1_2._conv.weight",
        "conv1_3._conv.weight",
        "stages.0.bb_0_0.conv0._conv.weight",
        "stages.0.bb_0_0.conv1._conv.weight",
        "stages.1.bb_1_0.short._conv.weight",  # VD shortcut on stride-2 block
    }
    missing = expected - keys
    assert not missing, f"Missing expected keys: {missing}"
