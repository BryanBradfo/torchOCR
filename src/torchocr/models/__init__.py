"""Model components for text detection and recognition."""

from .detection import DBNet, DBNetOutput
from .recognition import CRNN

__all__ = ["CRNN", "DBNet", "DBNetOutput"]
