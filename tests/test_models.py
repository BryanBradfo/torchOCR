"""Shape contracts and input guards for DBNet and CRNN."""

import pytest
import torch

from torchocr.models import CRNN, DBNet, DBNetOutput


# === DBNet ===

def test_dbnet_returns_output_dataclass():
    model = DBNet().train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert isinstance(out, DBNetOutput)


def test_dbnet_output_shape_matches_input():
    model = DBNet().train(False)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 96, 64))
    assert out.probability.shape == (2, 1, 96, 64)
    assert out.threshold.shape == (2, 1, 96, 64)


def test_dbnet_outputs_in_unit_range():
    model = DBNet().train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert 0.0 <= out.probability.min().item() <= out.probability.max().item() <= 1.0
    assert 0.0 <= out.threshold.min().item() <= out.threshold.max().item() <= 1.0


@pytest.mark.parametrize(
    "shape,reason",
    [
        ((1, 3, 65, 64), "H not divisible by 32"),
        ((1, 3, 64, 100), "W not divisible by 32"),
        ((1, 1, 64, 64), "wrong channel count"),
        ((3, 64, 64), "wrong ndim"),
    ],
)
def test_dbnet_rejects_bad_shape(shape, reason):
    model = DBNet().train(False)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_dbnet_pretrained_backbone_offline_safe():
    """pretrained_backbone=False must not hit the network."""
    model = DBNet(pretrained_backbone=False).train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.probability.shape == (1, 1, 64, 64)


# === DBNet ResNet-VD path ===

def test_dbnet_resnet18_vd_returns_output_dataclass():
    model = DBNet(backbone="resnet18_vd").train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert isinstance(out, DBNetOutput)


def test_dbnet_resnet18_vd_output_shape_matches_input():
    model = DBNet(backbone="resnet18_vd").train(False)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 96, 64))
    assert out.probability.shape == (2, 1, 96, 64)
    assert out.threshold.shape == (2, 1, 96, 64)


def test_dbnet_resnet18_vd_outputs_in_unit_range():
    model = DBNet(backbone="resnet18_vd").train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert 0.0 <= out.probability.min().item() <= out.probability.max().item() <= 1.0


def test_dbnet_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        DBNet(backbone="totally-not-real")  # type: ignore[arg-type]


def test_dbnet_resnet18_vd_state_dict_keys_have_expected_prefixes():
    """The converter assumes top-level prefixes (backbone, fpn, binarize, thresh)."""
    keys = list(DBNet(backbone="resnet18_vd").state_dict().keys())
    prefixes = {k.split(".")[0] for k in keys}
    assert prefixes == {"backbone", "fpn", "binarize", "thresh"}


# === CRNN ===

def test_crnn_forward_shape():
    model = CRNN(num_classes=96).train(False)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 128))
    assert out.shape == (32, 2, 96)  # T = W // 4


def test_crnn_t_equals_w_over_four():
    model = CRNN(num_classes=10).train(False)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 256))
    assert out.shape == (64, 1, 10)


@pytest.mark.parametrize(
    "shape,reason",
    [
        ((1, 3, 64, 128), "H != 32"),
        ((3, 32, 128), "wrong ndim"),
    ],
)
def test_crnn_rejects_bad_shape(shape, reason):
    model = CRNN(num_classes=96).train(False)
    with pytest.raises(ValueError):
        model(torch.randn(*shape))


def test_crnn_requires_num_classes():
    with pytest.raises(TypeError):
        CRNN()  # type: ignore[call-arg]
