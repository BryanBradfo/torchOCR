"""ICDAR Robust Reading datasets."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from ..core.structures import TextDetectionTarget


_SPLITS = {
    "train": ("ch4_training_images", "ch4_training_localization_transcription_gt"),
    "test": ("ch4_test_images", "Challenge4_Test_Task1_GT"),
}
_DONT_CARE = "###"
_IMAGE_INDEX = re.compile(r"img_(\d+)\.jpg$")


class ICDAR2015(Dataset[tuple[Tensor, TextDetectionTarget]]):
    """ICDAR 2015 Incidental Scene Text, Task 4.1 (text localization).

    1000 training and 500 test images of street-level scenes captured
    with wearable cameras; text regions are word-level quadrilaterals.

    The dataset requires registration, so it cannot be downloaded
    automatically. Get the four archives from
    https://rrc.cvc.uab.es/?ch=4&com=downloads and extract each into a
    folder of the same name under ``root``::

        root/
            ch4_training_images/                          img_1.jpg ...
            ch4_training_localization_transcription_gt/   gt_img_1.txt ...
            ch4_test_images/                              img_1.jpg ...
            Challenge4_Test_Task1_GT/                     gt_img_1.txt ...

    Args:
        root: Directory holding the extracted archives.
        split: ``"train"`` or ``"test"``. Default ``"test"``.
        transforms: Optional callable ``(image, target) -> (image, target)``
            applied to every sample.

    Returns:
        ``(image, target)`` where ``image`` is a ``(3, H, W)`` ``uint8``
        RGB tensor and ``target`` a :class:`TextDetectionTarget` with
        ``(N, 4, 2)`` polygons in original-image pixels.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"] = "test",
        transforms: Callable[[Tensor, TextDetectionTarget], tuple[Tensor, TextDetectionTarget]]
        | None = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}; got '{split}'.")
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        images_dir, gt_dir = (self.root / name for name in _SPLITS[split])
        missing = [d.name for d in (images_dir, gt_dir) if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"ICDAR2015 '{split}' split not found under {self.root} (missing: {', '.join(missing)}). "
                "Download the archives from https://rrc.cvc.uab.es/?ch=4&com=downloads "
                "(registration required) and extract each into a folder of the same name."
            )
        self.gt_dir = gt_dir
        self.image_paths = sorted(
            (p for p in images_dir.iterdir() if _IMAGE_INDEX.search(p.name)),
            key=lambda p: int(_IMAGE_INDEX.search(p.name).group(1)),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, TextDetectionTarget]:
        image_path = self.image_paths[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        target = _parse_ground_truth(self.gt_dir / f"gt_{image_path.stem}.txt")
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def _parse_ground_truth(path: Path) -> TextDetectionTarget:
    """Parse ``x1,y1,...,x4,y4,transcription`` lines (UTF-8 with BOM, CRLF)."""
    polygons: list[list[float]] = []
    texts: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(",")
        try:
            if len(fields) < 9:
                raise ValueError("expected 8 coordinates and a transcription")
            coords = [float(v) for v in fields[:8]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: malformed ICDAR line {line!r}.") from exc
        polygons.append(coords)
        # Transcriptions may themselves contain commas.
        texts.append(",".join(fields[8:]))
    return TextDetectionTarget(
        polygons=torch.tensor(polygons, dtype=torch.float32).reshape(-1, 4, 2),
        texts=texts,
        ignore=torch.tensor([t == _DONT_CARE for t in texts], dtype=torch.bool),
    )
