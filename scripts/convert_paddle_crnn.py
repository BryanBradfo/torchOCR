"""Convert a PaddleOCR CRNN (.pdparams) checkpoint to a torchocr .pth.

Build-time utility, not part of the runtime package. Targets the
PP-OCRv2 server-style recognizer:
ResNet-34-VD (with recognition strides ``(2, 1)``) +
SequenceEncoder (Im2Seq + 2-layer BiLSTM, hidden=256) +
CTC head (single ``Linear(512, 6625)``).

The naming inside ``src/torchocr/models/recognition.py`` and
``src/torchocr/models/backbones/resnet_vd.py`` was chosen so the
remap below is short and mechanical.

Usage:
    pip install torchocr[convert]
    python scripts/convert_paddle_crnn.py \\
        --paddle-weights /path/to/ch_ppocr_server_v2.0_rec_train/best_accuracy.pdparams \\
        --output /tmp/crnn_resnet34_vd.pth

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

from torchocr.models import CRNN


# ---------------------------------------------------------------------------
# Parameter-name mapping
# ---------------------------------------------------------------------------
# Paddle CRNN params live under three top-level prefixes: ``backbone.``,
# ``neck.``, ``head.``. Two complications versus a vanilla shape-equal
# port:
#   1. Backbone keys may be stored either ``flat`` (``backbone.bb_0_0...``)
#      or ``prefixed`` (``backbone.stage0.bb_0_0...``); we auto-detect
#      via :func:`detect_stage_format`, same as the DBNet converter.
#   2. ``head.fc.weight`` is a 2-D Linear weight. Paddle stores Linear
#      weights as ``(in, out)`` while PyTorch stores them as ``(out, in)``;
#      these tensors must be transposed before copy. Conv2d weights
#      keep ``(out, in, kH, kW)`` in both frameworks so they don't
#      need this treatment.
#
# A bonus simplification: the LSTM weight names PyTorch's ``nn.LSTM``
# emits (``weight_ih_l0``, ``weight_hh_l0_reverse``, ...) match
# PaddleOCR's flat LSTM-export naming exactly, so the entire neck
# ports over with no remap or transpose.


StageFormat = str  # "flat" | "prefixed"

# Linear weights (2-D) that need a transpose during copy. The CRNN here
# has just one; expand if you add more Linear layers (e.g. mid_channels
# in CTCHead, attention heads, etc.).
_TRANSPOSE_KEYS: frozenset[str] = frozenset({"head.fc.weight"})


def detect_stage_format(paddle_state: dict[str, object]) -> StageFormat:
    """Mirror PaddleOCR2Pytorch's heuristic: ``stageN`` substring => prefixed."""
    return "prefixed" if any("stage" in k for k in paddle_state) else "flat"


def paddle_name_for(torch_name: str, stage_format: StageFormat = "prefixed") -> str | None:
    """Translate a torchocr CRNN parameter name to its Paddle counterpart.

    Returns ``None`` for params with no Paddle equivalent
    (``num_batches_tracked``).
    """
    if torch_name.endswith("num_batches_tracked"):
        return None

    name = torch_name
    name = name.replace(".running_mean", "._mean")
    name = name.replace(".running_var", "._variance")

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
    """Load a ``.pdparams`` Paddle checkpoint (Paddle 1.x or 2.x)."""
    try:
        import paddle  # type: ignore[import-not-found]
    except ImportError as exc:
        sys.exit(
            "ERROR: paddlepaddle is required to read .pdparams files. "
            "Install conversion extras: pip install torchocr[convert]\n"
            f"  ({type(exc).__name__}: {exc})"
        )
    try:
        return paddle.load(str(weights_path))
    except Exception:
        import paddle.fluid as fluid  # type: ignore[import-not-found]

        with fluid.dygraph.guard():
            params, _ = fluid.load_dygraph(str(weights_path))
        return params


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(weights_path: Path, output_path: Path, num_classes: int = 6625) -> None:
    paddle_state = load_paddle_state(weights_path)
    stage_format = detect_stage_format(paddle_state)
    print(
        f"Loaded {len(paddle_state)} tensors from {weights_path} "
        f"(stage format: {stage_format})"
    )

    model = CRNN(num_classes=num_classes, backbone="resnet34_vd")
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
        if torch_key in _TRANSPOSE_KEYS:
            new_value = new_value.T.contiguous()

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
    parser.add_argument(
        "--paddle-weights",
        type=Path,
        required=True,
        help="Path to the Paddle .pdparams (or best_accuracy prefix without extension).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the converted .pth file.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=6625,
        help="Output classes for the CTC head. Default 6625 matches PaddleOCR's "
        "ch_ppocr_v2.0_rec (Chinese full charset + blank).",
    )
    args = parser.parse_args()
    convert(args.paddle_weights, args.output, args.num_classes)


if __name__ == "__main__":
    main()
