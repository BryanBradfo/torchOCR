"""Unit tests for the Paddle->torchocr name-mapping logic.

The converter script lives in ``scripts/`` rather than the package, so
the import path goes through the repo root. These tests pin the two
naming conventions PaddleOCR ships:
  - ``flat``: ``backbone.bb_0_0...`` (no stage prefix)
  - ``prefixed``: ``backbone.stage0.bb_0_0...``
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_paddle_dbnet import detect_stage_format, paddle_name_for  # noqa: E402


# === Format detection ===

def test_detect_flat_format():
    state = {"backbone.bb_0_0.conv0._conv.weight": None, "neck.in2_conv.weight": None}
    assert detect_stage_format(state) == "flat"


def test_detect_prefixed_format():
    state = {"backbone.stage0.bb_0_0.conv0._conv.weight": None}
    assert detect_stage_format(state) == "prefixed"


# === Mapping: shared rules across formats ===

@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_skips_num_batches_tracked(fmt):
    assert paddle_name_for("backbone.conv1_1._batch_norm.num_batches_tracked", fmt) is None


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_running_stats_renamed(fmt):
    assert (
        paddle_name_for("backbone.conv1_1._batch_norm.running_mean", fmt)
        == "backbone.conv1_1._batch_norm._mean"
    )
    assert (
        paddle_name_for("backbone.conv1_1._batch_norm.running_var", fmt)
        == "backbone.conv1_1._batch_norm._variance"
    )


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_fpn_renamed_to_neck(fmt):
    assert paddle_name_for("fpn.in2_conv.weight", fmt) == "neck.in2_conv.weight"
    assert paddle_name_for("fpn.p5_conv.weight", fmt) == "neck.p5_conv.weight"


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_heads_nest_under_head_prefix(fmt):
    assert paddle_name_for("binarize.conv1.weight", fmt) == "head.binarize.conv1.weight"
    assert paddle_name_for("thresh.conv3.bias", fmt) == "head.thresh.conv3.bias"


@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_stem_unchanged(fmt):
    """conv1_1 / conv1_2 / conv1_3 are top-level under backbone in both formats."""
    assert (
        paddle_name_for("backbone.conv1_1._conv.weight", fmt)
        == "backbone.conv1_1._conv.weight"
    )


# === Mapping: format-specific stage rules ===

def test_flat_format_strips_stages_prefix():
    """``stages.N.`` is removed entirely; blocks live directly under backbone."""
    assert (
        paddle_name_for("backbone.stages.0.bb_0_0.conv0._conv.weight", "flat")
        == "backbone.bb_0_0.conv0._conv.weight"
    )
    assert (
        paddle_name_for("backbone.stages.3.bb_3_1.conv1._batch_norm.running_mean", "flat")
        == "backbone.bb_3_1.conv1._batch_norm._mean"
    )


def test_prefixed_format_collapses_stages_to_stage_n():
    """``stages.N.`` becomes ``stageN.``."""
    assert (
        paddle_name_for("backbone.stages.0.bb_0_0.conv0._conv.weight", "prefixed")
        == "backbone.stage0.bb_0_0.conv0._conv.weight"
    )
    assert (
        paddle_name_for("backbone.stages.2.bb_2_1.short._batch_norm.running_var", "prefixed")
        == "backbone.stage2.bb_2_1.short._batch_norm._variance"
    )


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        paddle_name_for("backbone.stages.0.bb_0_0.conv0._conv.weight", "weird-format")


# === End-to-end coverage of the model state_dict ===

@pytest.mark.parametrize("fmt", ["flat", "prefixed"])
def test_full_model_state_dict_maps_bijectively(fmt):
    """Every torchocr param either maps cleanly or is explicitly skipped."""
    from torchocr.models import DBNet

    model = DBNet(backbone="resnet18_vd")
    state_keys = list(model.state_dict().keys())

    mapped: set[str] = set()
    skipped = 0
    for key in state_keys:
        result = paddle_name_for(key, fmt)
        if result is None:
            skipped += 1
        else:
            assert result not in mapped, f"Two torch keys mapped to {result!r}"
            mapped.add(result)

    # Every BatchNorm contributes one num_batches_tracked that gets skipped.
    n_bn = sum(1 for k in state_keys if k.endswith("running_mean"))
    assert skipped == n_bn
    assert len(mapped) == len(state_keys) - skipped
