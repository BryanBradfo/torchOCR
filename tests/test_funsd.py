"""FUNSD dataset: word boxes of scanned forms."""

import json
from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from torchocr.datasets import FUNSD


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for split in ("training_data", "testing_data"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "annotations").mkdir(parents=True)
    form = {"form": [
        {"id": 0, "label": "question", "text": "DATE: 12", "box": [0, 0, 1, 1], "linking": [],
         "words": [{"box": [10, 20, 50, 30], "text": "DATE:"}, {"box": [60, 20, 80, 30], "text": "12"}]},
        {"id": 1, "label": "other", "text": "", "box": [0, 0, 1, 1], "linking": [],
         "words": [{"box": [5, 5, 9, 9], "text": ""}]},
    ]}
    for stem in ("doc_10", "doc_9"):
        write_png(torch.full((1, 40, 90), 200, dtype=torch.uint8), str(tmp_path / "testing_data" / "images" / f"{stem}.png"))
        (tmp_path / "testing_data" / "annotations" / f"{stem}.json").write_text(json.dumps(form))
    return tmp_path


def test_sample_contract(root):
    dataset = FUNSD(root, split="test")
    assert [p.stem for p in dataset.image_paths] == ["doc_9", "doc_10"]
    image, target = dataset[0]
    assert image.shape == (3, 40, 90) and image.dtype == torch.uint8  # grayscale -> RGB
    assert target.polygons.shape == (3, 4, 2)
    assert target.polygons[0].tolist() == [[10, 20], [50, 20], [50, 30], [10, 30]]
    assert target.texts == ["DATE:", "12", ""]
    assert target.ignore.tolist() == [False, False, True]


def test_load_target_matches_getitem(root):
    dataset = FUNSD(root, split="test")
    assert torch.equal(dataset.load_target(1).polygons, dataset[1][1].polygons)


def test_missing_split(root):
    with pytest.raises(FileNotFoundError, match="dataset.zip"):
        FUNSD(root / "nope", split="test")
