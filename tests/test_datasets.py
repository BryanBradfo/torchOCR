"""ICDAR2015 dataset: official RRC directory layout -> (image, TextDetectionTarget)."""

from pathlib import Path

import pytest
import torch
from torchvision.io import write_png

from torchocr.datasets import ICDAR2015, TextDetectionTarget


def _write_split(root: Path, images_dir: str, gt_dir: str, gts: dict[int, str]) -> None:
    (root / images_dir).mkdir(parents=True)
    (root / gt_dir).mkdir(parents=True)
    for idx, content in gts.items():
        image = torch.full((3, 24, 40), idx, dtype=torch.uint8)
        # ICDAR ships .jpg; PNG bytes under a .jpg name decode identically
        # and keep pixel values exact for the assertion below.
        write_png(image, str(root / images_dir / f"img_{idx}.jpg"))
        (root / gt_dir / f"gt_img_{idx}.txt").write_bytes(content.encode("utf-8-sig"))


@pytest.fixture
def icdar_root(tmp_path: Path) -> Path:
    _write_split(
        tmp_path,
        "ch4_test_images",
        "Challenge4_Test_Task1_GT",
        {
            # Index 10 sorts before 2 lexically; the dataset must sort numerically.
            10: "1,2,11,2,11,8,1,8,Hello, world\r\n377,117,463,117,465,130,378,130,###\r\n",
            2: "0,0,5,0,5,5,0,5,A\r\n",
            1: "",
        },
    )
    _write_split(
        tmp_path,
        "ch4_training_images",
        "ch4_training_localization_transcription_gt",
        {1: "0,0,5,0,5,5,0,5,train\r\n"},
    )
    return tmp_path


def test_test_split_length_and_numeric_order(icdar_root):
    dataset = ICDAR2015(icdar_root, split="test")
    assert len(dataset) == 3
    assert [p.name for p in dataset.image_paths] == ["img_1.jpg", "img_2.jpg", "img_10.jpg"]


def test_sample_contract(icdar_root):
    image, target = ICDAR2015(icdar_root, split="test")[2]

    assert image.dtype == torch.uint8 and image.shape == (3, 24, 40)
    assert int(image[0, 0, 0]) == 10
    assert isinstance(target, TextDetectionTarget)
    assert target.polygons.shape == (2, 4, 2) and target.polygons.dtype == torch.float32
    assert target.polygons[0].tolist() == [[1, 2], [11, 2], [11, 8], [1, 8]]
    assert target.texts == ["Hello, world", "###"]
    assert target.ignore.tolist() == [False, True]


def test_image_without_text_has_empty_target(icdar_root):
    _, target = ICDAR2015(icdar_root, split="test")[0]
    assert target.polygons.shape == (0, 4, 2)
    assert target.texts == []
    assert target.ignore.shape == (0,)


def test_train_split(icdar_root):
    dataset = ICDAR2015(icdar_root, split="train")
    assert len(dataset) == 1
    assert dataset[0][1].texts == ["train"]


def test_transforms_receive_image_and_target(icdar_root):
    def flip_text(image, target):
        return image.float(), TextDetectionTarget(target.polygons, target.texts[::-1], target.ignore)

    image, target = ICDAR2015(icdar_root, split="test", transforms=flip_text)[2]
    assert image.dtype == torch.float32
    assert target.texts == ["###", "Hello, world"]


def test_missing_root_explains_manual_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="rrc.cvc.uab.es"):
        ICDAR2015(tmp_path, split="test")


def test_unknown_split_raises(icdar_root):
    with pytest.raises(ValueError, match="split"):
        ICDAR2015(icdar_root, split="val")


def test_malformed_line_reports_file_and_line(icdar_root):
    gt = icdar_root / "Challenge4_Test_Task1_GT" / "gt_img_2.txt"
    gt.write_text("0,0,5,0,5,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"gt_img_2\.txt:1"):
        ICDAR2015(icdar_root, split="test")[1]
