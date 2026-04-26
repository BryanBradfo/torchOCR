"""End-to-end OCRPipeline composition test."""

import pytest
import torch

from torchocr import (
    CTCGreedyDecoder,
    DBPostProcessor,
    DocumentTensor,
    OCRPipeline,
)
from torchocr.models import CRNN, DBNet


def _build_pipeline(charset: list[str]) -> OCRPipeline:
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(charset)).train(False)
    return OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(threshold=0.3),
        CTCGreedyDecoder(charset),
    )


def test_pipeline_returns_document_tensor(ascii_charset):
    pipeline = _build_pipeline(ascii_charset)
    doc = pipeline(torch.randn(3, 64, 64))
    assert isinstance(doc, DocumentTensor)
    assert doc.pixels.shape == (3, 64, 64)


def test_pipeline_box_count_matches_text(ascii_charset):
    pipeline = _build_pipeline(ascii_charset)
    doc = pipeline(torch.randn(3, 64, 64))
    box_count = 0 if doc.bounding_boxes is None else doc.bounding_boxes.shape[0]
    assert len(doc.text) == box_count


@pytest.mark.parametrize("shape", [(3, 64), (64, 64), (1, 3, 64, 64)])
def test_pipeline_rejects_bad_input_ndim(ascii_charset, shape):
    pipeline = _build_pipeline(ascii_charset)
    with pytest.raises(ValueError):
        pipeline(torch.randn(*shape))


def test_pipeline_rejects_bad_crop_size(ascii_charset):
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(ascii_charset)).train(False)
    with pytest.raises(ValueError):
        OCRPipeline(
            detector,
            recognizer,
            DBPostProcessor(),
            CTCGreedyDecoder(ascii_charset),
            crop_size=(64, 128),  # height must be 32
        )
