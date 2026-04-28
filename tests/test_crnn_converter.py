"""Unit tests for the Paddle->torchocr CRNN name-mapping logic."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_paddle_crnn import detect_stage_format, paddle_name_for  # noqa: E402


# === Format detection ===

def test_detect_flat_format():
    state = {"backbone.bb_0_0.conv0._conv.weight": None}
    assert detect_stage_format(state) == "flat"


def test_detect_prefixed_format():
    state = {"backbone.stage0.bb_0_0.conv0._conv.weight": None}
    assert detect_stage_format(state) == "prefixed"


# === Mapping rules shared with DBNet converter ===

@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_skips_num_batches_tracked(fmt):
    assert paddle_name_for("backbone.conv1_1._batch_norm.num_batches_tracked", fmt) is None


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_running_stats_renamed(fmt):
    assert (
        paddle_name_for("backbone.conv1_1._batch_norm.running_mean", fmt)
        == "backbone.conv1_1._batch_norm._mean"
    )


def test_flat_format_strips_stages_prefix():
    assert (
        paddle_name_for("backbone.stages.0.bb_0_0.conv0._conv.weight", "flat")
        == "backbone.bb_0_0.conv0._conv.weight"
    )


def test_prefixed_format_collapses_stages_to_stage_n():
    assert (
        paddle_name_for("backbone.stages.2.bb_2_1.conv0._conv.weight", "prefixed")
        == "backbone.stage2.bb_2_1.conv0._conv.weight"
    )


# === CRNN-specific: neck and head names ===

@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_lstm_names_pass_through_unchanged(fmt):
    """PyTorch's nn.LSTM weight naming matches PaddleOCR's flat LSTM export
    one-to-one, so no remap is needed for the neck."""
    for k in [
        "neck.encoder.lstm.weight_ih_l0",
        "neck.encoder.lstm.weight_hh_l0_reverse",
        "neck.encoder.lstm.bias_ih_l1",
    ]:
        assert paddle_name_for(k, fmt) == k


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_head_names_pass_through_unchanged(fmt):
    """The head's nn.Linear sits at ``head.fc.*`` in both frameworks; only the
    weight tensor itself needs a transpose at copy time (handled in
    convert_paddle_crnn.convert), not a name remap."""
    assert paddle_name_for("head.fc.weight", fmt) == "head.fc.weight"
    assert paddle_name_for("head.fc.bias", fmt) == "head.fc.bias"


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        paddle_name_for("head.fc.weight", "weird-format")


# === End-to-end coverage of the live model ===

@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_full_state_dict_maps_bijectively(fmt):
    from torchocr.models import CRNN

    model = CRNN(num_classes=6625, backbone="resnet34_vd")
    keys = list(model.state_dict().keys())

    mapped: set[str] = set()
    skipped = 0
    for k in keys:
        result = paddle_name_for(k, fmt)
        if result is None:
            skipped += 1
        else:
            assert result not in mapped, f"Two torch keys mapped to {result!r}"
            mapped.add(result)

    n_bn = sum(1 for k in keys if k.endswith("running_mean"))
    assert skipped == n_bn
    assert len(mapped) == len(keys) - skipped
