"""CTCGreedyDecoder: collapse-repeats + remove-blanks semantics."""

import pytest
import torch

from torchocr import CTCGreedyDecoder


def _onehot(seq: list[int], num_classes: int = 4) -> torch.Tensor:
    """Build (T, B=1, num_classes) logits whose argmax equals ``seq``."""
    logits = torch.full((len(seq), 1, num_classes), -10.0)
    for t, idx in enumerate(seq):
        logits[t, 0, idx] = 10.0
    return logits


def test_collapse_consecutive_duplicates_then_remove_blanks():
    # [a, a, blank, a, b, b, c] -> 'aabc'
    # consecutive a's collapse to a, blank lets next a survive,
    # consecutive b's collapse to b, c alone.
    dec = CTCGreedyDecoder(["-", "a", "b", "c"])
    assert dec(_onehot([1, 1, 0, 1, 2, 2, 3])) == ["aabc"]


def test_blank_lets_repeated_character_survive():
    # [a, blank, a, blank, a] -> 'aaa'
    dec = CTCGreedyDecoder(["-", "a"])
    assert dec(_onehot([1, 0, 1, 0, 1], num_classes=2)) == ["aaa"]


def test_all_blank_decodes_to_empty_string():
    dec = CTCGreedyDecoder(["-", "a"])
    assert dec(_onehot([0, 0, 0, 0], num_classes=2)) == [""]


def test_batched_decoding():
    dec = CTCGreedyDecoder(["-", "a", "b", "c"])
    logits = torch.full((4, 2, 4), -10.0)
    # Batch 0: [a, b, b, c] -> 'abc'
    for t, idx in enumerate([1, 2, 2, 3]):
        logits[t, 0, idx] = 10
    # Batch 1: [a, blank, a, blank] -> 'aa'
    for t, idx in enumerate([1, 0, 1, 0]):
        logits[t, 1, idx] = 10
    assert dec(logits) == ["abc", "aa"]


def test_rejects_bad_logits_ndim():
    dec = CTCGreedyDecoder(["-", "a"])
    with pytest.raises(ValueError):
        dec(torch.randn(4, 2))


def test_rejects_class_count_mismatch():
    dec = CTCGreedyDecoder(["-", "a", "b"])  # 3 classes
    with pytest.raises(ValueError):
        dec(torch.randn(4, 1, 5))


def test_rejects_blank_index_outside_charset():
    with pytest.raises(ValueError):
        CTCGreedyDecoder(["-", "a"], blank_index=5)
