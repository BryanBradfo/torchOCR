"""Decoders that map model logits to human-readable strings."""

import json
import math
from pathlib import Path

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


_BOS = "\x02"  # start-of-word context symbol
_EOS = "\x03"  # end-of-word event


class CharNGramLM:
    """Case-insensitive character n-gram language model (interpolated Witten-Bell).

    ``P(c | h) = (C(h, c) + T(h) * P(c | h')) / (C(h) + T(h))`` where ``h'``
    drops the oldest character of ``h``, ``T(h)`` counts the distinct
    continuations of ``h`` and the recursion ends at a uniform distribution
    over :attr:`vocabulary` (which includes an end-of-word event). Every
    distribution therefore sums to one and unseen characters keep non-zero
    mass. Text is lowercased, so the model judges spelling while the optical
    model keeps choosing case.

    Args:
        counts: ``{context: {char: count}}`` for contexts of length
            ``0 .. order - 1``.
        order: n-gram order (context length + 1).
    """

    def __init__(self, counts: dict[str, dict[str, int]], order: int) -> None:
        if order < 1:
            raise ValueError(f"order must be >= 1; got {order}.")
        self.order = order
        self.counts = counts
        self.vocabulary = sorted({c for continuations in counts.values() for c in continuations} | {_EOS})
        self._totals = {h: sum(cs.values()) for h, cs in counts.items()}
        self._cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_texts(cls, texts: list[str], order: int = 5) -> "CharNGramLM":
        """Count n-grams over ``texts`` (one word or line each)."""
        counts: dict[str, dict[str, int]] = {}
        for text in texts:
            chars = [_BOS] * (order - 1) + list(text.lower()) + [_EOS]
            for i in range(order - 1, len(chars)):
                for n in range(order):  # contexts of length 0 .. order-1
                    context = "".join(chars[i - n : i])
                    bucket = counts.setdefault(context, {})
                    bucket[chars[i]] = bucket.get(chars[i], 0) + 1
        return cls(counts, order)

    def log_prob(self, history: str, char: str) -> float:
        """``log P(char | history)``; ``char=""`` scores the end of the word."""
        context = (_BOS * (self.order - 1) + history.lower())[-(self.order - 1) :] if self.order > 1 else ""
        key = (context, char.lower() or _EOS)
        if key not in self._cache:
            self._cache[key] = math.log(self._prob(*key))
        return self._cache[key]

    def score(self, text: str) -> float:
        """Total log-probability of ``text`` as a complete word."""
        return sum(self.log_prob(text[:i], text[i]) for i in range(len(text))) + self.log_prob(text, "")

    def _prob(self, context: str, char: str) -> float:
        lower = self._prob(context[1:], char) if context else 1.0 / len(self.vocabulary)
        continuations = self.counts.get(context)
        if not continuations:
            return lower
        distinct, total = len(continuations), self._totals[context]
        return (continuations.get(char, 0) + distinct * lower) / (total + distinct)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"order": self.order, "counts": self.counts}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharNGramLM":
        state = json.loads(Path(path).read_text())
        return cls(state["counts"], state["order"])


class CTCBeamDecoder:
    """CTC prefix beam search with optional character-LM shallow fusion.

    Implements the prefix beam search of Hannun et al. (2014): each prefix
    tracks the probability of ending in blank and in its last character, so
    "ll" (separated by a blank) and "l" (one repeated character) stay
    distinct. When a character is appended its LM log-probability, scaled by
    ``lm_weight``, is added; at the end every beam also pays the LM's
    end-of-word probability. Frames whose blank probability exceeds
    ``blank_skip`` are treated as pure blank and only the ``top_k`` most
    likely characters are expanded per frame, which keeps a pure-Python
    search fast on short words.

    Args:
        charset: Character strings indexed by class id.
        beam_width: Prefixes kept per frame. Default 10.
        lm: Optional :class:`CharNGramLM`.
        lm_weight: LM scale in log space. Default 0.3.
        blank_index: CTC blank class. Default 0.
        top_k: Characters expanded per frame. Default 8.
        blank_skip: Blank probability above which a frame is skipped. Default 0.999.
    """

    def __init__(
        self,
        charset: list[str],
        beam_width: int = 10,
        lm: CharNGramLM | None = None,
        lm_weight: float = 0.3,
        blank_index: int = 0,
        top_k: int = 8,
        blank_skip: float = 0.999,
    ) -> None:
        if not 0 <= blank_index < len(charset):
            raise ValueError(f"blank_index {blank_index} is outside charset of size {len(charset)}.")
        if beam_width < 1 or top_k < 1:
            raise ValueError(f"beam_width and top_k must be >= 1; got {beam_width}, {top_k}.")
        self.charset = list(charset)
        self.beam_width = beam_width
        self.lm = lm
        self.lm_weight = lm_weight
        self.blank_index = blank_index
        self.top_k = top_k
        self.blank_skip = blank_skip

    @property
    def num_classes(self) -> int:
        return len(self.charset)

    def __call__(self, logits: Tensor) -> list[str]:
        """Decode ``(T, B, num_classes)`` logits into a list of B strings."""
        if logits.ndim != 3:
            raise ValueError(f"Expected logits of shape (T, B, num_classes); got {tuple(logits.shape)}.")
        if logits.shape[2] != self.num_classes:
            raise ValueError(
                f"Logits' num_classes={logits.shape[2]} does not match charset size {self.num_classes}."
            )
        log_probs = logits.float().log_softmax(-1).detach().cpu()
        k = min(self.top_k, self.num_classes - 1)
        top = log_probs.topk(k + 1, dim=-1)  # +1: the blank may be among them
        return [self._decode(log_probs[:, b, self.blank_index], top.values[:, b], top.indices[:, b])
                for b in range(log_probs.shape[1])]

    def _decode(self, blank_lp: Tensor, top_lp: Tensor, top_ids: Tensor) -> str:
        neg_inf = -math.inf
        # prefix (tuple of class ids) -> (log P ending in blank, log P ending in last char)
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}
        skip = math.log(self.blank_skip)
        for t in range(blank_lp.shape[0]):
            p_blank = float(blank_lp[t])
            if p_blank > skip:  # (almost) pure blank: every prefix just absorbs it
                beams = {prefix: (_logsumexp(pb, pnb) + p_blank, neg_inf) for prefix, (pb, pnb) in beams.items()}
                continue
            candidates = [(int(c), float(lp)) for c, lp in zip(top_ids[t], top_lp[t]) if int(c) != self.blank_index]
            nxt: dict[tuple[int, ...], list[float]] = {}

            def add(prefix, pb=neg_inf, pnb=neg_inf):
                entry = nxt.setdefault(prefix, [neg_inf, neg_inf])
                entry[0] = _logsumexp(entry[0], pb)
                entry[1] = _logsumexp(entry[1], pnb)

            for prefix, (pb, pnb) in beams.items():
                total = _logsumexp(pb, pnb)
                add(prefix, pb=total + p_blank)  # emit blank
                last = prefix[-1] if prefix else None
                for c, p in candidates:
                    if c == last:
                        add(prefix, pnb=pnb + p)  # repeat collapses into the same prefix
                        if pb > neg_inf:  # blank in between: a genuinely new character
                            add(prefix + (c,), pnb=pb + p + self._lm(prefix, c))
                    else:
                        add(prefix + (c,), pnb=total + p + self._lm(prefix, c))
            beams = dict(sorted(((k, tuple(v)) for k, v in nxt.items()),
                                key=lambda kv: _logsumexp(*kv[1]), reverse=True)[: self.beam_width])

        def final(item):
            prefix, (pb, pnb) = item
            end = self.lm_weight * self.lm.log_prob(self._text(prefix), "") if self.lm else 0.0
            return _logsumexp(pb, pnb) + end

        return self._text(max(beams.items(), key=final)[0])

    def _lm(self, prefix: tuple[int, ...], c: int) -> float:
        if self.lm is None:
            return 0.0
        return self.lm_weight * self.lm.log_prob(self._text(prefix), self.charset[c])

    def _text(self, prefix: tuple[int, ...]) -> str:
        return "".join(self.charset[i] for i in prefix)


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
