"""Convert a PaddleOCR DBNet (.pdparams) checkpoint to a torchocr .pth.

This is a *build-time* utility, intentionally outside ``src/torchocr/``
because it needs ``paddlepaddle`` to read ``.pdparams`` files. End
users running torchocr at inference time should never invoke this.

Conversion targets PP-OCRv2 English-style DBNet checkpoints whose
architecture is ResNet-18-VD + DBFPN(out_channels=256) + DBHead.
The naming conventions of the underlying layers were chosen in
``src/torchocr/models/backbones/resnet_vd.py`` and
``src/torchocr/models/detection.py`` specifically to make this script
short and mechanical.

Usage:
    pip install torchocr[convert]
    python scripts/convert_paddle_dbnet.py \\
        --paddle-weights /path/to/en_PP-OCRv2_det_infer/inference \\
        --output /tmp/dbnet_resnet18_vd.pth

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
# Given a torchocr parameter name like
# ``backbone.stages.0.bb_0_0.conv0._conv.weight``, translate it to the
# equivalent Paddle name. PaddleOCR ships two on-disk naming conventions
# for the same architecture (see ``PaddleOCR2Pytorch/converter/det_converter.py:21-36``):
#
#   "prefixed":  ``backbone.stage0.bb_0_0.conv0._conv.weight``
#   "flat":      ``backbone.bb_0_0.conv0._conv.weight``
#
# We auto-detect which one is in the loaded ``.pdparams`` (presence of any
# ``stage`` substring in the keys) and pick the right rule.
#
# Other torchocr <-> Paddle name differences:
#   1. BatchNorm running stats: ``running_mean`` <-> ``_mean``,
#      ``running_var`` <-> ``_variance``. Paddle has no equivalent of
#      ``num_batches_tracked``; we skip those keys entirely.
#   2. FPN module name: torchocr uses ``fpn.*``, Paddle uses ``neck.*``.
#   3. Detection-head wrapper: torchocr exposes ``binarize.*`` / ``thresh.*``
#      directly on DBNet; Paddle nests them under ``head.binarize.*`` /
#      ``head.thresh.*``.
#
# This default implementation uses a regex chain. Other reasonable styles:
#   - explicit dict + prefix replace (easier to extend to v3/v4
#     distillation prefixes like ``Student2.*``)
#   - token-based traversal: split on ``.`` then transform tokens
#     (most flexible for variants but more code)
#
# Choose the style that fits the variants you plan to add next.


StageFormat = str  # "flat" | "prefixed"


def detect_stage_format(paddle_state: dict[str, object]) -> StageFormat:
    """Return ``"prefixed"`` if any Paddle key contains a ``stage`` token,
    else ``"flat"``. Mirrors PaddleOCR2Pytorch's heuristic."""
    return "prefixed" if any("stage" in k for k in paddle_state) else "flat"


def paddle_name_for(torch_name: str, stage_format: StageFormat = "prefixed") -> str | None:
    """Translate a torchocr parameter name to its Paddle counterpart.

    Returns ``None`` for parameters that have no Paddle equivalent
    (e.g. ``num_batches_tracked``) so the caller can skip them.
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

    if stage_format == "prefixed":
        name = re.sub(r"^backbone\.stages\.(\d+)\.", r"backbone.stage\1.", name)
    elif stage_format == "flat":
        name = re.sub(r"^backbone\.stages\.\d+\.", "backbone.", name)
    else:
        raise ValueError(f"Unknown stage_format {stage_format!r}; expected 'flat' or 'prefixed'.")
    return name


# ---------------------------------------------------------------------------
# Paddle weight loading
# ---------------------------------------------------------------------------


def load_paddle_state(weights_path: Path) -> dict[str, Any]:
    """Load a ``.pdparams`` (or ``best_accuracy``) Paddle checkpoint.

    Handles both Paddle 1.x dygraph and Paddle 2.x formats, mirroring
    PaddleOCR2Pytorch's strategy at
    ``pytorchocr/base_ocr_v20.py:94-103``.
    """
    try:
        import paddle  # type: ignore[import-not-found]
    except ImportError as exc:
        sys.exit(
            "ERROR: paddlepaddle is required to read .pdparams files. "
            "Install conversion extras: pip install torchocr[convert]\n"
            f"  ({type(exc).__name__}: {exc})"
        )

    try:
        # Paddle 2.x path
        return paddle.load(str(weights_path))
    except Exception:
        # Paddle 1.x dygraph fallback
        import paddle.fluid as fluid  # type: ignore[import-not-found]

        with fluid.dygraph.guard():
            params, _ = fluid.load_dygraph(str(weights_path))
        return params


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(weights_path: Path, output_path: Path) -> None:
    paddle_state = load_paddle_state(weights_path)
    stage_format = detect_stage_format(paddle_state)
    print(
        f"Loaded {len(paddle_state)} tensors from {weights_path} "
        f"(stage format: {stage_format})"
    )

    model = DBNet(backbone="resnet18_vd")
    model.train(False)
    torch_state = model.state_dict()

    matched = 0
    skipped: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []

    for torch_key in torch_state:
        paddle_key = paddle_name_for(torch_key, stage_format)
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

    # Verify the model accepts the new state cleanly before writing.
    model.load_state_dict(torch_state, strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_state, output_path)
    print(f"\nWrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--paddle-weights",
        type=Path,
        required=True,
        help="Path prefix to the .pdparams file (e.g. .../inference, no extension).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the converted .pth file.",
    )
    args = parser.parse_args()
    convert(args.paddle_weights, args.output)


if __name__ == "__main__":
    main()
