"""CTC prefix beam search with an optional character n-gram language model."""

import math

import pytest
import torch

from torchocr import CTCGreedyDecoder
from torchocr.decoders import CharNGramLM, CTCBeamDecoder

CHARSET = ["<blank>", "a", "b", "c", "l", "o", "k", "x", "B", "L"]


def _logits(frames: list[dict[str, float]]) -> torch.Tensor:
    """(T, 1, C) logits from per-frame probabilities (unlisted classes get ~0)."""
    probs = torch.full((len(frames), 1, len(CHARSET)), 1e-6)
    for t, frame in enumerate(frames):
        for char, p in frame.items():
            probs[t, 0, CHARSET.index(char)] = p
    return (probs / probs.sum(-1, keepdim=True)).log()


# === Language model ===


def test_lm_probabilities_are_normalized():
    lm = CharNGramLM.from_texts(["block", "black", "lock"], order=3)
    for history in ("", "b", "bl", "zz"):
        total = sum(math.exp(lm.log_prob(history, c)) for c in lm.vocabulary)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_lm_prefers_seen_continuations_and_ignores_case():
    lm = CharNGramLM.from_texts(["block", "block", "black"], order=3)
    assert lm.log_prob("blo", "c") > lm.log_prob("blo", "x")
    assert lm.log_prob("BLO", "C") == lm.log_prob("blo", "c")
    assert lm.score("BLOCK") > lm.score("BLOCX")


def test_lm_round_trips_through_state_dict(tmp_path):
    lm = CharNGramLM.from_texts(["block", "lock"], order=4)
    path = tmp_path / "lm.json"
    lm.save(path)
    assert CharNGramLM.load(path).score("block") == pytest.approx(lm.score("block"))


# === Beam search ===


def test_beam_without_lm_matches_greedy_on_peaked_logits():
    logits = _logits([{"a": 0.9}, {"<blank>": 0.9}, {"b": 0.9}, {"b": 0.9}, {"<blank>": 0.9}, {"c": 0.9}])
    assert CTCBeamDecoder(CHARSET, beam_width=5)(logits) == CTCGreedyDecoder(CHARSET)(logits) == ["abc"]


def test_blank_separates_repeated_characters():
    logits = _logits([{"l": 0.95}, {"<blank>": 0.95}, {"l": 0.95}])
    assert CTCBeamDecoder(CHARSET)(logits) == ["ll"]


def test_beam_sums_alignments_that_greedy_misses():
    """Greedy picks the best *path*; beam search picks the best *label sequence*.

    Blank wins every frame, yet "a" has more total probability over its
    alignments ("a-", "-a", "aa") than "" has over its single one ("--").
    """
    logits = _logits([{"a": 0.4, "<blank>": 0.6}, {"a": 0.4, "<blank>": 0.6}])
    assert CTCGreedyDecoder(CHARSET)(logits) == [""]
    # P("") = 0.36 ; P("a") = 0.4*0.6 + 0.6*0.4 + 0.4*0.4 = 0.64
    assert CTCBeamDecoder(CHARSET, beam_width=4)(logits) == ["a"]


def test_language_model_fixes_a_one_character_confusion():
    # The optical model slightly prefers the misspelling "blocx".
    frames = [{"b": 0.95}, {"l": 0.95}, {"o": 0.95}, {"c": 0.95}, {"x": 0.55, "k": 0.45}]
    logits = _logits(frames)
    lm = CharNGramLM.from_texts(["block", "lock", "clock", "blocks"] * 5, order=4)
    assert CTCBeamDecoder(CHARSET, beam_width=8)(logits) == ["blocx"]
    assert CTCBeamDecoder(CHARSET, beam_width=8, lm=lm, lm_weight=0.5)(logits) == ["block"]


def test_lm_is_case_insensitive_but_keeps_the_optical_case():
    frames = [{"B": 0.95}, {"L": 0.95}, {"o": 0.95}, {"c": 0.95}, {"x": 0.55, "k": 0.45}]
    lm = CharNGramLM.from_texts(["block"] * 5, order=4)
    assert CTCBeamDecoder(CHARSET, lm=lm, lm_weight=0.5)(_logits(frames)) == ["BLock"]


def test_batch_and_shape_contract():
    logits = torch.randn(12, 3, len(CHARSET))
    assert len(CTCBeamDecoder(CHARSET)(logits)) == 3
    with pytest.raises(ValueError, match=r"\(T, B, num_classes\)"):
        CTCBeamDecoder(CHARSET)(torch.randn(12, len(CHARSET)))
    with pytest.raises(ValueError, match="charset size"):
        CTCBeamDecoder(CHARSET)(torch.randn(12, 1, 4))
