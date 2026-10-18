"""Unit tests for the PP-OCRv3 detector name-mapping logic."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_paddle_dbnet_v3 import (  # noqa: E402
    detect_distill_prefix,
    detect_stage_format,
    paddle_name_for,
)


# === Format detection ===

def test_detect_distill_train_prefix_picks_the_exported_student():
    """distill_train holds Teacher, Student and Student2; PaddleOCR exports
    ``Student`` (it is byte-identical to the shipped ``student.pdparams``).
    Picking ``Student2`` costs ~10 hmean points on ICDAR-2015."""
    state = {
        f"{prefix}.backbone.stage0.0.expand_conv.weight": None
        for prefix in ("Teacher", "Student", "Student2")
    }
    assert detect_distill_prefix(state) == "Student."


def test_detect_inference_no_prefix():
    state = {"backbone.stage0.0.expand_conv.weight": None}
    assert detect_distill_prefix(state) == ""


def test_detect_stage_format_returns_prefixed_when_stage_token_present():
    state = {"backbone.stage0.0.expand_conv.weight": None}
    assert detect_stage_format(state) == "prefixed"


def test_detect_stage_format_rejects_flat_with_clear_error():
    """v2-style flat keys would collide on MobileNetV3 inner blocks; the
    v3 converter raises rather than silently producing garbage."""
    state = {"backbone.bb_0_0.conv0.weight": None}  # would-be ResNet-VD flat
    with pytest.raises(ValueError, match="prefixed"):
        detect_stage_format(state)


# === Mapping rules ===

@pytest.mark.parametrize("distill", ["", "Student."])
def test_skips_num_batches_tracked(distill):
    assert paddle_name_for("backbone.conv.bn.num_batches_tracked", "prefixed", distill) is None


def test_distill_prefix_wraps_after_other_transforms():
    """``Student.`` is prepended LAST so the inner transformations
    (running_mean -> _mean, fpn -> neck, etc.) work uniformly."""
    assert (
        paddle_name_for("backbone.conv.bn.running_mean", "prefixed", "Student.")
        == "Student.backbone.conv.bn._mean"
    )


def test_fpn_to_neck_under_distill():
    assert (
        paddle_name_for("fpn.ins_conv.0.in_conv.weight", "prefixed", "Student.")
        == "Student.neck.ins_conv.0.in_conv.weight"
    )


def test_head_nesting_under_distill():
    assert (
        paddle_name_for("binarize.conv1.weight", "prefixed", "Student.")
        == "Student.head.binarize.conv1.weight"
    )


def test_stages_to_stage_under_distill():
    assert (
        paddle_name_for(
            "backbone.stages.2.5.expand_conv.conv.weight",
            "prefixed",
            "Student.",
        )
        == "Student.backbone.stage2.5.expand_conv.conv.weight"
    )


def test_inference_checkpoint_no_student2():
    """When distill_prefix='' (inference checkpoint), names match v2 logic."""
    assert (
        paddle_name_for("backbone.stages.0.0.expand_conv.conv.weight", "prefixed")
        == "backbone.stage0.0.expand_conv.conv.weight"
    )


def test_unknown_stage_format_raises():
    with pytest.raises(ValueError):
        paddle_name_for("backbone.conv.bn.weight", "flat", "")


# === End-to-end coverage ===

@pytest.mark.parametrize("distill", ["", "Student."])
def test_full_state_dict_maps_bijectively(distill):
    from torchocr.models import DBNet

    model = DBNet(backbone="mobilenet_v3_large_05")
    keys = list(model.state_dict().keys())

    mapped: set[str] = set()
    skipped = 0
    for k in keys:
        result = paddle_name_for(k, "prefixed", distill)
        if result is None:
            skipped += 1
        else:
            assert result not in mapped, f"two torch keys mapped to {result!r}"
            mapped.add(result)

    n_bn_running = sum(1 for k in keys if k.endswith("running_mean"))
    assert skipped == n_bn_running
    assert len(mapped) == len(keys) - skipped
