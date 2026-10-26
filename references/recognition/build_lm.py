"""Build the character n-gram language model used by the recognizers' beam search.

The model only sees *training* transcriptions: the care words of the ICDAR-2015
and Total-Text training splits, plus any LMDB word corpora (e.g. MLT-2019
Latin crops). Running it with the published inputs reproduces the published
file byte for byte. Example::

    python references/recognition/build_lm.py --ic15 ~/data/icdar2015 \\
        --totaltext ~/data/totaltext --lmdb ~/data/mlt2019/Latin --output charlm6.json
"""

import argparse
import hashlib
from pathlib import Path

from torchocr.decoders import CharNGramLM
from torchocr.datasets import ICDAR2015, LMDBWords, TotalText


def training_texts(ic15: Path | None, totaltext: Path | None, lmdb: list[Path]) -> list[str]:
    """Training transcriptions, in a fixed order: LMDB corpora, ICDAR-2015, Total-Text."""
    texts: list[str] = []
    for path in lmdb:
        texts += LMDBWords(path).labels
    for root, cls in ((ic15, ICDAR2015), (totaltext, TotalText)):
        if root is None:
            continue
        dataset = cls(root, split="train")
        for index in range(len(dataset)):
            target = dataset.load_target(index)
            texts += [text for text, ignored in zip(target.texts, target.ignore.tolist()) if not ignored]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ic15", type=Path, help="ICDAR-2015 root (training split only is read).")
    parser.add_argument("--totaltext", type=Path, help="Total-Text root (training split only is read).")
    parser.add_argument("--lmdb", type=Path, action="append", default=[], help="LMDB word corpus (repeatable).")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    texts = training_texts(args.ic15, args.totaltext, args.lmdb)
    lm = CharNGramLM.from_texts(texts, order=args.order)
    lm.save(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{len(texts)} transcriptions, {len(lm.counts)} contexts -> {args.output} (sha256 {digest[:8]})")


if __name__ == "__main__":
    main()
