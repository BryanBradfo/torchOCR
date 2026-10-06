"""Shape contracts and architectural pinning for the MobileNetV3 backbone."""

import pytest
import torch

from torchocr.models.backbones import MobileNetV3


def test_large_05_disable_se_pp_ocrv3_config():
    """The PP-OCRv3 detector config: large + 0.5 + disable_se=True."""
    backbone = MobileNetV3(model_name="large", scale=0.5, disable_se=True).train(False)
    # Channels at strides 4, 8, 16, 32 must match PaddleOCR's published numbers.
    assert backbone.out_channels == (16, 24, 56, 480)


def test_large_05_forward_strides():
    backbone = MobileNetV3(model_name="large", scale=0.5).train(False)
    with torch.no_grad():
        out = backbone(torch.randn(1, 3, 320, 320))
    assert out["c2"].shape == (1, 16, 80, 80)   # /4
    assert out["c3"].shape == (1, 24, 40, 40)   # /8
    assert out["c4"].shape == (1, 56, 20, 20)   # /16
    assert out["c5"].shape == (1, 480, 10, 10)  # /32


def test_pp_ocrv3_param_count_matches_published():
    """PaddleOCR ships PP-OCRv3 detector backbone at ~0.4M params; pin it
    so refactors that drift this raise alarms."""
    backbone = MobileNetV3(model_name="large", scale=0.5, disable_se=True)
    n = sum(p.numel() for p in backbone.parameters())
    # Allow 5% variance for any minor structural choices that don't affect
    # the converter (e.g. BatchNorm tracking buffers vs params).
    assert 380_000 <= n <= 430_000, f"unexpected param count {n}"


def test_disable_se_drops_attention_blocks():
    with_se = MobileNetV3(model_name="large", scale=0.5, disable_se=False)
    without_se = MobileNetV3(model_name="large", scale=0.5, disable_se=True)
    assert sum(p.numel() for p in with_se.parameters()) > sum(
        p.numel() for p in without_se.parameters()
    )
    # SE submodules must be entirely absent when disabled.
    se_keys_without = [k for k in without_se.state_dict() if "mid_se" in k]
    assert se_keys_without == []
    se_keys_with = [k for k in with_se.state_dict() if "mid_se" in k]
    assert len(se_keys_with) > 0


def test_state_dict_keys_match_paddle_layout():
    """The converter assumes specific submodule names. Pin them."""
    keys = MobileNetV3(model_name="large", scale=0.5, disable_se=True).state_dict().keys()
    expected_samples = {
        "conv.conv.weight",
        "conv.bn.weight",
        "stages.0.0.expand_conv.conv.weight",
        "stages.0.0.bottleneck_conv.conv.weight",
        "stages.0.0.linear_conv.conv.weight",
    }
    missing = expected_samples - keys
    assert not missing, f"missing expected keys: {missing}"


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.0])
def test_make_divisible_channel_counts_are_multiples_of_eight(scale):
    backbone = MobileNetV3(model_name="large", scale=scale)
    for c in backbone.out_channels:
        assert c % 8 == 0, f"channel {c} not divisible by 8 at scale={scale}"


def test_rejects_unsupported_scale():
    with pytest.raises(ValueError):
        MobileNetV3(scale=0.6)


def test_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        MobileNetV3(model_name="medium")  # type: ignore[arg-type]
