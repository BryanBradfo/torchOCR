"""Evaluation metrics for OCR tasks."""

from .detection import DetectionHmean, HmeanResult
from .recognition import RecognitionAccuracy, RecognitionResult

__all__ = ["DetectionHmean", "HmeanResult", "RecognitionAccuracy", "RecognitionResult"]
