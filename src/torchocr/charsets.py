"""Character set loaders for OCR recognizers.

Currently exposes the PaddleOCR Chinese full-charset (``ppocr_keys_v1.txt``)
matched to PaddleOCR's CTC head shape: ``blank`` at index 0, followed by
6622 characters from the dictionary, a space, and any padding the model's
``num_classes`` requires (PaddleOCR hardcodes ``out_channels=6625`` even
though the active chars only fill 6624 slots).

The loader is sized against ``num_classes`` so callers can pass it
straight into :class:`CTCGreedyDecoder` without size mismatches.
"""

from __future__ import annotations

from importlib.resources import files


_DATA_PACKAGE = "torchocr.data"
_PPOCR_KEYS_FILENAME = "ppocr_keys_v1.txt"

# PaddleOCR's CTC blank lives at index 0 by convention; matches the default
# ``blank_index`` in :class:`torchocr.CTCGreedyDecoder`.
BLANK_TOKEN = "<blank>"


def load_ppocr_keys_v1(num_classes: int) -> list[str]:
    """Return PaddleOCR's Chinese charset sized to match ``num_classes``.

    Layout is ``[<blank>] + 6622 chars + [" "]`` followed by padding
    ``<blank>`` entries up to ``num_classes``. The CTC blank at index 0
    is what the decoder collapses, so any padding entries (also blank)
    are harmless even if the model never emits them.

    For PaddleOCR's standard ch_ppocr_v2.0_rec checkpoint pass
    ``num_classes=6625``.
    """
    if num_classes < 6624:
        raise ValueError(
            f"num_classes={num_classes} is too small for ppocr_keys_v1 "
            "(needs >= 6624 to fit blank + 6622 chars + space)."
        )
    chars = (
        files(_DATA_PACKAGE)
        .joinpath(_PPOCR_KEYS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    full = [BLANK_TOKEN, *chars, " "]
    while len(full) < num_classes:
        full.append(BLANK_TOKEN)
    return full
