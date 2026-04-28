"""Charset loader for the PaddleOCR Chinese full charset."""

import pytest

from torchocr import load_ppocr_keys_v1
from torchocr.charsets import BLANK_TOKEN


def test_load_at_default_paddleocr_size():
    """6625 = blank + 6622 vendored chars + space + (any required padding)."""
    chars = load_ppocr_keys_v1(6625)
    assert len(chars) == 6625
    assert chars[0] == BLANK_TOKEN  # PaddleOCR convention: blank at index 0
    assert " " in chars  # space char must be present


def test_load_supports_larger_num_classes():
    """Extra slots are padded with the blank token (matches CTC ignore behavior)."""
    chars = load_ppocr_keys_v1(7000)
    assert len(chars) == 7000
    # The 'real' content occupies up to 6624, the rest is blank padding.
    assert chars[-1] == BLANK_TOKEN


def test_load_rejects_undersized_num_classes():
    with pytest.raises(ValueError):
        load_ppocr_keys_v1(100)


def test_chars_are_unicode_chinese():
    chars = load_ppocr_keys_v1(6625)
    # The first vendored entries are punctuation (apostrophe, etc.). Sample
    # well into the file where Chinese characters dominate.
    assert ord(chars[1000]) > 0x4E00  # CJK Unified Ideographs start at U+4E00
