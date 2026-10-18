"""OCR datasets with typed targets, mirroring ``torchvision.datasets``."""

from ..core.structures import TextDetectionTarget
from .icdar import ICDAR2015
from .totaltext import TotalText

__all__ = ["ICDAR2015", "TextDetectionTarget", "TotalText"]
