"""references/recognition/parquet_to_lmdb.py: Hugging Face word corpora -> LMDBWords layout."""

import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("lmdb")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "references" / "recognition"))

from parquet_to_lmdb import convert  # noqa: E402

from torchocr.datasets import LMDBWords  # noqa: E402


def _png(width: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, 32), (90, 90, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _write(path: Path, words: list[str], text_column: str = "label") -> None:
    images = pa.array([{"bytes": _png(10 + i), "path": None} for i in range(len(words))],
                      type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    pq.write_table(pa.table({"image": images, text_column: words}), path, row_group_size=2)


def test_round_trip_across_files_and_row_groups(tmp_path):
    _write(tmp_path / "a.parquet", ["alpha", "beta", "gamma"])
    _write(tmp_path / "b.parquet", ["delta", "epsilon"])
    count = convert(sorted(tmp_path.glob("*.parquet")), tmp_path / "lmdb", text_column="label")

    dataset = LMDBWords(tmp_path / "lmdb")
    assert count == len(dataset) == 5
    assert dataset.labels == ["alpha", "beta", "gamma", "delta", "epsilon"]
    image, text = dataset[4]
    assert text == "epsilon" and image.shape == (3, 32, 11) and image.dtype == torch.uint8  # 2nd row of b.parquet


def test_limit_and_text_column(tmp_path):
    _write(tmp_path / "a.parquet", ["one", "two", "three"], text_column="text")
    assert convert([tmp_path / "a.parquet"], tmp_path / "lmdb", text_column="text", limit=2) == 2
    assert LMDBWords(tmp_path / "lmdb").labels == ["one", "two"]


def test_refuses_to_overwrite(tmp_path):
    _write(tmp_path / "a.parquet", ["one"])
    convert([tmp_path / "a.parquet"], tmp_path / "lmdb")
    with pytest.raises(FileExistsError):
        convert([tmp_path / "a.parquet"], tmp_path / "lmdb")
