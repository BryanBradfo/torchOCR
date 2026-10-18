"""LMDBWords: word crops in the deep-text-recognition-benchmark LMDB layout."""

import io

import pytest
import torch
from PIL import Image

lmdb = pytest.importorskip("lmdb")

from torchocr.datasets import LMDBWords  # noqa: E402


def _jpeg(width: int, value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, 32), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def root(tmp_path):
    words = ["Hello", "straße", "", "123"]
    env = lmdb.open(str(tmp_path), map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, word in enumerate(words, start=1):
            txn.put(f"image-{i:09d}".encode(), _jpeg(10 * (i + 1), 40 * i))
            txn.put(f"label-{i:09d}".encode(), word.encode("utf-8"))
        txn.put(b"num-samples", str(len(words)).encode())
    env.close()
    return tmp_path


def test_reads_images_and_labels(root):
    dataset = LMDBWords(root)
    assert len(dataset) == 4
    image, text = dataset[0]
    assert text == "Hello"
    assert image.dtype == torch.uint8 and image.shape == (3, 32, 20)
    assert int(image[0, 0, 0]) == 40


def test_label_filter_selects_samples_without_decoding_images(root):
    dataset = LMDBWords(root, label_filter=lambda t: t != "" and all(c.isascii() for c in t))
    assert dataset.labels == ["Hello", "123"]
    assert dataset[1][1] == "123"
    assert dataset[1][0].shape == (3, 32, 50)


def test_transform_is_applied(root):
    dataset = LMDBWords(root, transform=lambda image: image.float())
    assert dataset[0][0].dtype == torch.float32


def test_missing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.mdb"):
        LMDBWords(tmp_path / "nope")
