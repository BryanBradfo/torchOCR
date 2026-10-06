"""Image transforms for OCR, mirroring ``torchvision.transforms``."""

from ._presets import IMAGENET_MEAN, IMAGENET_STD, DetectionPreset, RecognitionPreset
from .db_targets import DBTargets, MakeDBTargets

__all__ = [
    "DBTargets",
    "DetectionPreset",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MakeDBTargets",
    "RecognitionPreset",
]
