"""RecognitionAccuracy: word accuracy and character error rate."""

import pytest

from torchocr.metrics import RecognitionAccuracy, RecognitionResult


def test_exact_matches():
    metric = RecognitionAccuracy()
    metric.update(["hello", "world"], ["hello", "world"])
    result = metric.compute()
    assert isinstance(result, RecognitionResult)
    assert (result.word_accuracy, result.char_error_rate, result.num_words) == (1.0, 0.0, 2)


def test_character_error_rate_is_edit_distance_over_target_length():
    metric = RecognitionAccuracy()
    # kitten -> sitting: 3 edits over 7 target characters.
    metric.update(["kitten"], ["sitting"])
    assert metric.compute().char_error_rate == pytest.approx(3 / 7)


def test_counts_accumulate_over_updates():
    metric = RecognitionAccuracy()
    metric.update(["a"], ["a"])
    metric.update(["b", "x"], ["b", "cd"])
    result = metric.compute()
    assert result.word_accuracy == pytest.approx(2 / 3)
    assert result.char_error_rate == pytest.approx(2 / 4)  # "x" vs "cd": 2 edits; 4 target chars


def test_icdar_normalization_is_case_and_punctuation_insensitive():
    strict = RecognitionAccuracy()
    icdar = RecognitionAccuracy(case_sensitive=False, alphanumeric_only=True)
    for metric in (strict, icdar):
        metric.update(["Hello!", "U.S.A"], ["hello", "USA"])
    assert strict.compute().word_accuracy == 0.0
    assert icdar.compute().word_accuracy == 1.0


def test_empty_and_reset():
    metric = RecognitionAccuracy()
    assert metric.compute().word_accuracy == 0.0
    metric.update(["a"], ["b"])
    metric.reset()
    assert metric.compute().num_words == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="2 predictions"):
        RecognitionAccuracy().update(["a", "b"], ["a"])
