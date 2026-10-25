"""Convert Hugging Face Parquet word corpora into the LMDB layout read by ``LMDBWords``.

Hugging Face hosts MJSynth, SynthText and similar corpora as Parquet files
with an ``image`` column (``{"bytes", "path"}``) and a text column. LMDB is
the format recognition corpora usually ship in and the one ``LMDBWords``
reads, with fast random access for shuffled training. Files are streamed one
row group at a time, so memory stays flat whatever the corpus size. Every
image is test-decoded and undecodable ones are skipped (MJSynth ships a few
truncated files that would otherwise crash training hours in).
Example::

    python references/recognition/parquet_to_lmdb.py \\
        --parquet "~/data/mjsynth/data/train-*.parquet" --text-column label \\
        --output ~/data/mjsynth/lmdb_train
"""

import argparse
import glob
import os
from collections.abc import Sequence
from pathlib import Path

import lmdb
import pyarrow.parquet as pq
import torch
from torchvision.io import decode_image


def convert(
    files: Sequence[Path],
    output: Path,
    text_column: str = "label",
    image_column: str = "image",
    limit: int | None = None,
    map_size: int = 1 << 40,
    commit_every: int = 10_000,
) -> int:
    """Write every (image bytes, text) row of ``files`` to a new LMDB at ``output``; return the count."""
    output = Path(output)
    if (output / "data.mdb").exists():
        raise FileExistsError(f"{output} already holds an LMDB database.")
    output.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output), map_size=map_size)
    count = skipped = 0
    txn = env.begin(write=True)
    try:
        for path in files:
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=[image_column, text_column])
                for image, text in zip(table.column(image_column).to_pylist(), table.column(text_column).to_pylist()):
                    data = image["bytes"] if isinstance(image, dict) else image
                    if not _decodes(data):
                        skipped += 1
                        continue
                    count += 1
                    txn.put(f"image-{count:09d}".encode(), data)
                    txn.put(f"label-{count:09d}".encode(), text.encode("utf-8"))
                    if count % commit_every == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                    if limit is not None and count >= limit:
                        return count
        return count
    finally:
        txn.put(b"num-samples", str(count).encode())
        txn.commit()
        env.close()
        print(f"Kept {count} samples; skipped {skipped} undecodable images.")


def _decodes(data: bytes) -> bool:
    try:
        decode_image(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    except RuntimeError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, help="Glob of input Parquet files (quote it).")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", default="label")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--limit", type=int, help="Stop after this many samples.")
    args = parser.parse_args()
    files = sorted(Path(p) for p in glob.glob(os.path.expanduser(args.parquet)))
    if not files:
        parser.error(f"no file matches {args.parquet}")
    count = convert(files, args.output.expanduser(), args.text_column, args.image_column, args.limit)
    print(f"Wrote {count} samples from {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
