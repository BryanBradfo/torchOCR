"""Text-recognition metrics."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionResult:
    """Dataset-level recognition scores.

    ``char_error_rate`` is the summed edit distance divided by the summed
    target length, so long words weigh more than short ones.
    """

    word_accuracy: float
    char_error_rate: float
    num_words: int


class RecognitionAccuracy:
    """Word accuracy and character error rate over (prediction, target) string pairs.

    ``case_sensitive=False, alphanumeric_only=True`` is the usual
    scene-text protocol (ICDAR / IIIT5K / SVT report "case-insensitive
    word accuracy" over letters and digits).

    Args:
        case_sensitive: Compare strings as-is. Default True.
        alphanumeric_only: Drop every character that is not a letter or
            a digit before comparing. Default False.
    """

    def __init__(self, case_sensitive: bool = True, alphanumeric_only: bool = False) -> None:
        self.case_sensitive = case_sensitive
        self.alphanumeric_only = alphanumeric_only
        self.reset()

    def reset(self) -> None:
        self._correct = 0
        self._words = 0
        self._edits = 0
        self._target_chars = 0

    def _normalize(self, text: str) -> str:
        if self.alphanumeric_only:
            text = "".join(c for c in text if c.isalnum())
        return text if self.case_sensitive else text.lower()

    def update(self, preds: Sequence[str], targets: Sequence[str]) -> None:
        if len(preds) != len(targets):
            raise ValueError(f"Got {len(preds)} predictions for {len(targets)} targets.")
        for pred, target in zip(preds, targets):
            pred, target = self._normalize(pred), self._normalize(target)
            self._correct += pred == target
            self._edits += _edit_distance(pred, target)
            self._target_chars += len(target)
        self._words += len(targets)

    def compute(self) -> RecognitionResult:
        return RecognitionResult(
            word_accuracy=self._correct / self._words if self._words else 0.0,
            char_error_rate=self._edits / self._target_chars if self._target_chars else 0.0,
            num_words=self._words,
        )


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (unit-cost insert / delete / substitute)."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
