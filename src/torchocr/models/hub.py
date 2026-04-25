"""Internal weight registry and downloader for torchocr's pretrained models.

This module is intentionally *not* re-exported. Public access is via the
``weights`` constructor argument on :class:`DBNet` and :class:`CRNN`,
mirroring the ergonomics of ``torchvision.models.resnet18(weights=...)``.

Each model_key maps to a dict of named aliases (currently just
``"DEFAULT"``); future revisions can register dataset-specific
checkpoints (``"ICDAR2015"``, ``"TOTALTEXT"``, ...) without changing the
public API.
"""

from torch.hub import load_state_dict_from_url


_BASE_URL = "https://huggingface.co/BryanBradfo/torchocr-weights/resolve/main"

_WEIGHTS_REGISTRY: dict[str, dict[str, str]] = {
    "dbnet": {
        "DEFAULT": f"{_BASE_URL}/dbnet_resnet18.pth",
    },
    "crnn": {
        "DEFAULT": f"{_BASE_URL}/crnn_vgg.pth",
    },
}


def load_pretrained_state_dict(model_key: str, weights: str) -> dict | None:
    """Resolve ``weights`` to a URL and download the state_dict.

    Returns the state_dict on success or ``None`` on download failure.
    A descriptive warning is printed in the failure case so callers can
    fall back to random initialization without crashing the user's code.

    Args:
        model_key: Name of the model in the registry (``"dbnet"`` or
            ``"crnn"``).
        weights: Alias inside the model's registry. ``"DEFAULT"`` is
            the only registered alias for the v0.1.0 MVP.
    """
    if model_key not in _WEIGHTS_REGISTRY:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Known keys: {sorted(_WEIGHTS_REGISTRY)}."
        )
    aliases = _WEIGHTS_REGISTRY[model_key]
    if weights not in aliases:
        raise ValueError(
            f"Unknown weights alias '{weights}' for {model_key}. "
            f"Known aliases: {sorted(aliases)}."
        )

    url = aliases[weights]
    try:
        return load_state_dict_from_url(url, map_location="cpu", progress=False)
    except Exception as exc:
        print(
            "WARNING: Pre-trained weights not yet uploaded to Hugging Face. "
            "Falling back to random initialization."
        )
        print(f"  (model={model_key}, alias={weights}, url={url}, error={type(exc).__name__})")
        return None
