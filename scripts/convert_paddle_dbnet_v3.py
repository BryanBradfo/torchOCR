"""Convert a PaddleOCR PP-OCRv3 detector (.pdparams) to a torchocr .pth.

Targets ``ch_PP-OCRv3_det_distill_train`` and ``ch_PP-OCRv3_det_infer``
checkpoints whose architecture is MobileNetV3-large + RSEFPN(out=96) + DBHead.

The distillation training checkpoint (``*_distill_train``) holds three
networks (``Teacher.*``, ``Student.*``, ``Student2.*``) because PP-OCRv3
was trained with collaborative mutual learning. PaddleOCR exports
``Student`` for inference -- it is byte-identical to the ``student.pdparams``
shipped in the same archive -- so that is the one we convert. Inference
checkpoints and ``student.pdparams`` have no prefix. We auto-detect which
form we're given.

Usage:
    pip install torchocr[convert]
    python scripts/convert_paddle_dbnet_v3.py \\
        --paddle-weights /path/to/ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams \\
        --output /tmp/dbnet_mobilenet_v3_large_05.pth

Conversion recipes derived from PaddleOCR2Pytorch (Apache-2.0); see
``CREDITS.md``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch

from torchocr.models import DBNet


# ---------------------------------------------------------------------------
# Parameter-name mapping
# ---------------------------------------------------------------------------
# Two transformations on top of the v2 converter's logic:
#   1. Distillation prefix: PP-OCRv3 distill_train wraps every backbone /
#      neck / head key under ``Student.`` (plus a ``Student2.`` peer and a
#      ``Teacher.``). Inference checkpoints don't.
#   2. RSEFPN replaces DBFPN; the FPN's submodule names changed from
#      ``in2_conv`` / ``p2_conv`` to ``ins_conv.0.in_conv`` /
#      ``inp_conv.0.in_conv`` -- but our torchocr model uses the same
#      ``ins_conv`` / ``inp_conv`` ModuleList layout, so no extra remap
#      is needed beyond the standard ``fpn.* -> neck.*`` rename.
#
# Paddle's stage segmentation is the same as v2 ResNet-VD: ``stages.N.`` ->
# ``stageN.`` (prefixed) or ``stages.N.`` -> `` `` (flat), depending on
# how the .pdparams was saved.


StageFormat = str  # "flat" | "prefixed"


def detect_stage_format(paddle_state: dict[str, object]) -> StageFormat:
    """PP-OCRv3's MobileNetV3 backbone always saves the stage prefix
    (``stageN``); flat format would collide because inner blocks are
    numbered, not named with the unique ``bb_<i>_<j>`` pattern that
    ResNet-VD uses. We surface a clear error if a flat checkpoint shows
    up so the user knows the v3 converter is the wrong tool."""
    if not any("stage" in k for k in paddle_state):
        raise ValueError(
            "PP-OCRv3 detector checkpoints must use the stage-prefixed naming. "
            "If your .pdparams uses flat naming you likely have a v2 checkpoint -- "
            "use scripts/convert_paddle_dbnet.py instead."
        )
    return "prefixed"


def detect_distill_prefix(paddle_state: dict[str, object]) -> str:
    """Return ``"Student."`` if the checkpoint nests params under the
    exported distillation student, else the empty string.

    Converting ``Student2.`` instead loads a weaker peer network: on
    ICDAR-2015 the Chinese model drops from 0.422 to 0.325 hmean.
    """
    return "Student." if any(k.startswith("Student.") for k in paddle_state) else ""


def paddle_name_for(
    torch_name: str,
    stage_format: StageFormat = "prefixed",
    distill_prefix: str = "",
) -> str | None:
    """Translate a torchocr DBNet(mobilenet_v3_large_05) parameter name
    to its Paddle counterpart. Returns ``None`` for params with no
    Paddle equivalent (``num_batches_tracked``).
    """
    if torch_name.endswith("num_batches_tracked"):
        return None

    name = torch_name
    name = name.replace(".running_mean", "._mean")
    name = name.replace(".running_var", "._variance")

    if name.startswith("fpn."):
        name = "neck." + name[len("fpn.") :]
    elif name.startswith("binarize.") or name.startswith("thresh."):
        name = "head." + name

    if stage_format != "prefixed":
        raise ValueError(
            f"v3 converter only supports prefixed stage format; got {stage_format!r}."
        )
    name = re.sub(r"^backbone\.stages\.(\d+)\.", r"backbone.stage\1.", name)

    if distill_prefix:
        name = distill_prefix + name
    return name


# ---------------------------------------------------------------------------
# Paddle weight loading
# ---------------------------------------------------------------------------


def load_paddle_state(weights_path: Path) -> dict[str, Any]:
    """Load a ``.pdparams`` Paddle checkpoint (Paddle >= 2.5)."""
    try:
        import paddle  # type: ignore[import-not-found]
    except ImportError as exc:
        sys.exit(
            "ERROR: paddlepaddle is required to read .pdparams files. "
            "Install conversion extras: pip install torchocr[convert]\n"
            f"  ({type(exc).__name__}: {exc})"
        )
    if not weights_path.is_file():
        sys.exit(f"ERROR: {weights_path} does not exist.")
    return paddle.load(str(weights_path))


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(weights_path: Path, output_path: Path) -> None:
    paddle_state = load_paddle_state(weights_path)
    stage_format = detect_stage_format(paddle_state)
    distill_prefix = detect_distill_prefix(paddle_state)
    print(
        f"Loaded {len(paddle_state)} tensors from {weights_path}\n"
        f"  stage format: {stage_format}\n"
        f"  distillation prefix: {distill_prefix or '(none)'}"
    )

    model = DBNet(backbone="mobilenet_v3_large_05")
    model.train(False)
    torch_state = model.state_dict()

    matched = 0
    skipped: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []

    for torch_key in torch_state:
        paddle_key = paddle_name_for(torch_key, stage_format, distill_prefix)
        if paddle_key is None:
            skipped.append(torch_key)
            continue
        if paddle_key not in paddle_state:
            missing.append(f"{torch_key} (-> {paddle_key})")
            continue

        paddle_tensor = paddle_state[paddle_key]
        if hasattr(paddle_tensor, "numpy"):
            paddle_tensor = paddle_tensor.numpy()
        new_value = torch.as_tensor(paddle_tensor)

        if new_value.shape != torch_state[torch_key].shape:
            shape_mismatch.append(
                (torch_key, paddle_key, tuple(new_value.shape), tuple(torch_state[torch_key].shape))
            )
            continue

        torch_state[torch_key] = new_value
        matched += 1

    print(f"Matched {matched} / {len(torch_state)} parameters; skipped {len(skipped)}.")

    if missing:
        print("\nERROR: Paddle state_dict is missing the following keys:")
        for entry in missing[:20]:
            print(f"  - {entry}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)

    if shape_mismatch:
        print("\nERROR: shape mismatches between torchocr and Paddle:")
        for torch_key, paddle_key, paddle_shape, torch_shape in shape_mismatch[:20]:
            print(f"  - {torch_key} (torch {torch_shape}) vs {paddle_key} (paddle {paddle_shape})")
        sys.exit(1)

    model.load_state_dict(torch_state, strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_state, output_path)
    print(f"\nWrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paddle-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.paddle_weights, args.output)


if __name__ == "__main__":
    main()
