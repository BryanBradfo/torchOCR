"""Convert PaddleOCR's PP-LCNet page-orientation classifier (.pdparams) to a torchocr .pth.

Build-time utility, not part of the runtime package. Targets
``PP-LCNet_x1_0_doc_ori`` (4 classes: 0, 90, 180, 270 degrees), released by
PaddlePaddle under Apache-2.0:
https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams

Paddle is not needed: a ``.pdparams`` file is a pickle of numpy arrays, read
here with an unpickler that only resolves numpy's array constructors, so a
tampered file cannot run code.

Usage:
    python scripts/convert_paddle_pplcnet.py \\
        --paddle-weights PP-LCNet_x1_0_doc_ori_pretrained.pdparams \\
        --output pplcnet_x1_0_doc_ori.pth
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch

from torchocr.models.hub import zero_subnormals_
from torchocr.models.orientation import PPLCNet


# The only globals a .pdparams pickle references.
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a .pdparams file.")
        return super().find_class(module, name)


def load_paddle_state(weights_path: Path) -> dict[str, np.ndarray]:
    """The arrays of a ``.pdparams`` checkpoint, keyed by Paddle parameter name."""
    with open(weights_path, "rb") as f:
        state = _NumpyOnlyUnpickler(f).load()
    # Paddle also stores a name-mapping dict ("StructuredToParameterName@@"); keep arrays only.
    return {k: v for k, v in state.items() if isinstance(v, np.ndarray)}


def remap(paddle_state: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Paddle names -> :class:`PPLCNet` names (BN statistics renamed, Linear weight transposed)."""
    state = {}
    for key, array in paddle_state.items():
        key = key.replace("._mean", ".running_mean").replace("._variance", ".running_var")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        # Paddle stores Linear weights as (in, out), PyTorch as (out, in).
        state[key] = tensor.T.contiguous() if key == "fc.weight" else tensor
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paddle-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = PPLCNet(num_classes=4)
    state = remap(load_paddle_state(args.paddle_weights))
    state.update({k: v for k, v in model.state_dict().items() if k.endswith("num_batches_tracked")})
    model.load_state_dict(state)  # strict: every name and shape must match
    zeroed = zero_subnormals_(model)

    torch.save(model.state_dict(), args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    final = args.output.with_name(f"{args.output.stem}-{digest[:8]}{args.output.suffix}")
    args.output.rename(final)
    print(f"Wrote {final} ({sum(p.numel() for p in model.parameters()):,} parameters, {zeroed} subnormals zeroed)")


if __name__ == "__main__":
    main()
