"""Decoders that map model logits to human-readable strings."""

import torch
from torch import Tensor


class CTCGreedyDecoder:
    """Greedy CTC decoder for CRNN-style logits.

    Implements the canonical CTC decoding rule: take the per-step argmax,
    collapse consecutive duplicate indices (treating blanks as ordinary
    symbols during the collapse), then remove the remaining blank
    indices. Padding the recognizer's input with blanks therefore
    encodes "same character twice" correctly.

    Args:
        charset: Character strings indexed by class id. The string at
            ``charset[blank_index]`` is unused; only the index matters.
        blank_index: Index of the CTC blank class. Default 0, matching
            the convention used by ``torch.nn.CTCLoss``.
    """

    def __init__(self, charset: list[str], blank_index: int = 0) -> None:
        if not 0 <= blank_index < len(charset):
            raise ValueError(
                f"blank_index {blank_index} is outside charset of size {len(charset)}."
            )
        self.charset = list(charset)
        self.blank_index = blank_index

    @property
    def num_classes(self) -> int:
        return len(self.charset)

    def __call__(self, logits: Tensor) -> list[str]:
        """Decode ``(T, B, num_classes)`` logits into a list of B strings."""
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits of shape (T, B, num_classes); got {tuple(logits.shape)}."
            )
        if logits.shape[2] != self.num_classes:
            raise ValueError(
                f"Logits' num_classes={logits.shape[2]} does not match "
                f"charset size {self.num_classes}."
            )

        indices = logits.argmax(dim=-1)
        time_steps, batch_size = indices.shape
        if time_steps == 0:
            return [""] * batch_size

        prev_equal = torch.zeros_like(indices, dtype=torch.bool)
        prev_equal[1:] = indices[1:] == indices[:-1]
        keep = (~prev_equal) & (indices != self.blank_index)

        results: list[str] = []
        for b in range(batch_size):
            kept = indices[:, b][keep[:, b]].tolist()
            results.append("".join(self.charset[i] for i in kept))
        return results
