"""FUNSD: word-level text in noisy scanned forms."""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from ..core.structures import TextDetectionTarget


_SPLITS = {"train": "training_data", "test": "testing_data"}


class FUNSD(Dataset[tuple[Tensor, TextDetectionTarget]]):
    """FUNSD (Jaume et al., ICDAR-OST 2019): 149 train / 50 test noisy scanned
    business forms with word-level boxes and transcriptions.

    A *document* benchmark, unlike ICDAR-2015 and Total-Text: use it to check
    how a scene-text model transfers to scans. Words with an empty
    transcription (check boxes, rules) are marked ``ignore``.

    Download and unzip https://guillaumejaume.github.io/FUNSD/dataset.zip
    (no registration); ``root`` is the extracted ``dataset`` folder::

        root/training_data/{images,annotations}/  root/testing_data/{images,annotations}/

    Args:
        root: The extracted ``dataset`` directory.
        split: ``"train"`` or ``"test"``. Default ``"test"``.
        transforms: Optional callable ``(image, target) -> (image, target)``.

    Returns:
        ``(image, target)``: ``(3, H, W)`` ``uint8`` RGB (scans are
        grayscale, replicated to three channels) and a
        :class:`TextDetectionTarget` of axis-aligned ``(N, 4, 2)`` word quads.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "test"] = "test",
        transforms: Callable[[Tensor, TextDetectionTarget], tuple[Tensor, TextDetectionTarget]] | None = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}; got '{split}'.")
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        base = self.root / _SPLITS[split]
        self.annotations_dir = base / "annotations"
        images_dir = base / "images"
        if not images_dir.is_dir() or not self.annotations_dir.is_dir():
            raise FileNotFoundError(
                f"FUNSD '{split}' split not found under {self.root}: expected {_SPLITS[split]}/images and "
                f"{_SPLITS[split]}/annotations. Unzip https://guillaumejaume.github.io/FUNSD/dataset.zip and "
                "pass its 'dataset' folder."
            )
        self.image_paths = sorted(images_dir.glob("*.png"), key=_natural_key)

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_target(self, index: int) -> TextDetectionTarget:
        """The annotations of sample ``index``, without decoding its image."""
        form = json.loads((self.annotations_dir / f"{self.image_paths[index].stem}.json").read_text())["form"]
        words = [word for entity in form for word in entity["words"]]
        if not words:
            return TextDetectionTarget(torch.zeros(0, 4, 2), [], torch.zeros(0, dtype=torch.bool))
        boxes = torch.tensor([word["box"] for word in words], dtype=torch.float32)
        x1, y1, x2, y2 = boxes.unbind(-1)
        quads = torch.stack([torch.stack(p, -1) for p in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))], dim=1)
        texts = [word["text"] for word in words]
        return TextDetectionTarget(quads, texts, torch.tensor([not t.strip() for t in texts]))

    def __getitem__(self, index: int) -> tuple[Tensor, TextDetectionTarget]:
        image = read_image(str(self.image_paths[index]), mode=ImageReadMode.RGB)
        target = self.load_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]
