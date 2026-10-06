"""Pretrained-weight API and model registry, mirroring ``torchvision.models``.

Each architecture with published checkpoints exposes a :class:`WeightsEnum`
subclass (e.g. ``DBNet_ResNet18_VD_Weights``) whose members bundle the
checkpoint URL, the preprocessing it was trained with, and metadata
including benchmark numbers::

    weights = DBNet_ResNet18_VD_Weights.DEFAULT
    model = dbnet_resnet18_vd(weights=weights).eval()
    preprocess = weights.transforms()
    weights.meta["_metrics"]  # reproducible with references/detection/evaluate.py

Checkpoint file names embed the first 8 hex digits of their SHA-256
(``name-<hash>.pth``); downloads are verified against it, so a corrected
checkpoint always gets a new name and stale ``torch.hub`` caches cannot
shadow it.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from torch import nn
from torch.hub import HASH_REGEX, load_state_dict_from_url


BASE_URL = "https://huggingface.co/BryanBradfo/torchocr-weights/resolve/main"

__all__ = [
    "BASE_URL",
    "HASH_REGEX",
    "Weights",
    "WeightsEnum",
    "get_model",
    "get_model_weights",
    "get_weight",
    "list_models",
    "register_model",
]


@dataclass(frozen=True)
class Weights:
    """One published checkpoint.

    Attributes:
        url: Download URL; the file name carries a SHA-256 prefix.
        transforms: Zero-argument factory for the inference preset the
            checkpoint expects (call it: ``weights.transforms()``).
        meta: Free-form metadata. Keys shared by every torchocr checkpoint:
            ``task``, ``backbone``, ``num_params``, ``source``, ``license``,
            ``_metrics`` (benchmark -> scores) and ``_docs``.
    """

    url: str
    transforms: Callable[..., nn.Module]
    meta: dict[str, Any]


class WeightsEnum(Enum):
    """Base class for the per-architecture weights enums.

    A ``DEFAULT`` member aliases the recommended checkpoint.
    """

    @classmethod
    def verify(cls, obj: "WeightsEnum | str | None") -> "WeightsEnum | None":
        """Resolve ``obj`` (member, member name, or ``"Cls.NAME"``) to a member of ``cls``."""
        if obj is None or isinstance(obj, cls):
            return obj
        if isinstance(obj, str):
            name = obj.removeprefix(f"{cls.__name__}.")
            if name in cls.__members__:
                return cls[name]
            raise ValueError(
                f"Unknown weights '{obj}' for {cls.__name__}. Available: {sorted(cls.__members__)}."
            )
        raise ValueError(f"Expected {cls.__name__} or one of its member names; got {obj!r}.")

    def get_state_dict(self, progress: bool = True) -> dict[str, Any]:
        return load_state_dict_from_url(
            self.url, map_location="cpu", progress=progress, check_hash=True, weights_only=True
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self._name_}"

    @property
    def url(self) -> str:
        return self.value.url

    @property
    def transforms(self) -> Callable[..., nn.Module]:
        return self.value.transforms

    @property
    def meta(self) -> dict[str, Any]:
        return self.value.meta


def load_weights(model: nn.Module, weights: WeightsEnum, progress: bool = True) -> None:
    """Load ``weights`` into ``model``; on network failure, warn and keep random init.

    Only connectivity errors (``OSError``, which covers ``URLError`` and
    HTTP errors) fall back. Hash mismatches and incompatible checkpoints
    raise, because silently running random weights would hide them.
    """
    try:
        state_dict = weights.get_state_dict(progress=progress)
    except OSError as exc:
        warnings.warn(
            f"Could not download {weights!r} from {weights.url} ({type(exc).__name__}: {exc}). "
            "Falling back to random initialization.",
            stacklevel=3,
        )
        return
    model.load_state_dict(state_dict)


# === Registry ===

M = TypeVar("M", bound=nn.Module)

_MODELS: dict[str, Callable[..., nn.Module]] = {}
_MODEL_WEIGHTS: dict[str, type[WeightsEnum] | None] = {}


def register_model(
    name: str, weights: type[WeightsEnum] | None = None
) -> Callable[[Callable[..., M]], Callable[..., M]]:
    """Decorator registering a model builder under ``name``."""

    def wrapper(builder: Callable[..., M]) -> Callable[..., M]:
        if name in _MODELS:
            raise ValueError(f"Model '{name}' is already registered.")
        _MODELS[name] = builder
        _MODEL_WEIGHTS[name] = weights
        return builder

    return wrapper


def list_models() -> list[str]:
    """Sorted names of every registered model builder."""
    return sorted(_MODELS)


def _check_name(name: str) -> None:
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}.")


def get_model(name: str, **config: Any) -> nn.Module:
    """Build a registered model, e.g. ``get_model("dbnet_resnet18_vd", weights="DEFAULT")``."""
    _check_name(name)
    return _MODELS[name](**config)


def get_model_weights(name: str) -> type[WeightsEnum] | None:
    """The weights enum of a registered model, or ``None`` if it has no published weights."""
    _check_name(name)
    return _MODEL_WEIGHTS[name]


def get_weight(name: str) -> WeightsEnum:
    """Resolve a fully-qualified ``"Cls.NAME"`` string, e.g. ``"DBNet_ResNet18_VD_Weights.DEFAULT"``."""
    enum_name, _, member = name.partition(".")
    if not member:
        raise ValueError(f"Expected a fully-qualified 'Cls.NAME' weights name; got '{name}'.")
    enums = {e.__name__: e for e in _MODEL_WEIGHTS.values() if e is not None}
    if enum_name not in enums:
        raise ValueError(f"Unknown weights enum '{enum_name}'. Available: {sorted(enums)}.")
    return enums[enum_name].verify(member)
