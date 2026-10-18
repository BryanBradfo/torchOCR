"""TotalText dataset: curved word polygons with variable vertex counts."""

from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from torchocr.datasets import TextDetectionTarget, TotalText

# Real quirks seen in the released ground truth.
_GT = {
    "img11": (
        "x: [[283 286 304 367 338 332]], y: [[543 480 408 411 490 541]], ornt: [u'c'], transcriptions: [u'FOR']\n"
        "x: [[485 650 607 437]], y: [[250 373 409 291]], ornt: [u'm'], transcriptions: [u\"it's\"]\n"
        "x: [[333 373 368 329]], y: [[222 215 241 246]], ornt: [u'#'], transcriptions: [u'#']\n"
    ),
    "img2": (
        # Empty orientation, negative coordinate, missing closing bracket, wrapped line.
        "x: [[ -1 140 203  19]], y: [[167  92 119 232]], ornt: [], transcriptions: [u'UBBA'\n"
        "x: [[18 42 86\n 84 20 -2]], y: [[429 463 479 502 484 437]], ornt: [u'c'], transcriptions: [u'KINA']\n"
        # Empty transcription and a 1-vertex "polygon": both unusable, so ignored.
        "x: [[10 20 20 10]], y: [[10 10 20 20]], ornt: [u'#'], transcriptions: []\n"
        "x: [[5]], y: [[5]], ornt: [u'h'], transcriptions: [u'X']\n"
    ),
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for split in ("Train", "Test"):
        (tmp_path / "Images" / split).mkdir(parents=True)
        (tmp_path / "txt_format" / split).mkdir(parents=True)
    for stem, gt in _GT.items():
        write_png(torch.zeros(3, 8, 8, dtype=torch.uint8), str(tmp_path / "Images" / "Train" / f"{stem}.jpg"))
        (tmp_path / "txt_format" / "Train" / f"poly_gt_{stem}.txt").write_text(gt)
    write_png(torch.zeros(3, 8, 8, dtype=torch.uint8), str(tmp_path / "Images" / "Test" / "img1.JPG"))
    (tmp_path / "txt_format" / "Test" / "poly_gt_img1.txt").write_text("")
    return tmp_path


def test_numeric_order_and_extensions(root):
    assert [p.name for p in TotalText(root, split="train").image_paths] == ["img2.jpg", "img11.jpg"]
    assert [p.name for p in TotalText(root, split="test").image_paths] == ["img1.JPG"]


def test_variable_vertex_counts_are_padded_by_repeating_the_last_vertex(root):
    _, target = TotalText(root, split="train")[1]  # img11
    assert isinstance(target, TextDetectionTarget)
    assert target.polygons.shape == (3, 6, 2)
    assert target.polygons[1].tolist() == [[485, 250], [650, 373], [607, 409], [437, 291]] + [[437, 291]] * 2
    assert target.texts == ["FOR", "it's", "#"]
    assert target.ignore.tolist() == [False, False, True]


def test_malformed_lines_from_the_release_are_parsed(root):
    _, target = TotalText(root, split="train")[0]  # img2
    assert target.texts == ["UBBA", "KINA", "", "X"]
    assert target.polygons[0, 0].tolist() == [-1, 167]
    assert target.polygons[1, :, 0].tolist() == [18, 42, 86, 84, 20, -2]
    # Empty transcription and degenerate polygon are don't-care.
    assert target.ignore.tolist() == [False, False, True, True]


def test_image_without_text(root):
    image, target = TotalText(root, split="test")[0]
    assert image.shape == (3, 8, 8)
    assert target.polygons.shape == (0, 4, 2) and target.texts == []


def test_missing_folders_explain_the_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="txt_format"):
        TotalText(tmp_path, split="train")


def test_load_target_matches_getitem(root):
    dataset = TotalText(root, split="train")
    assert torch.equal(dataset.load_target(1).polygons, dataset[1][1].polygons)
