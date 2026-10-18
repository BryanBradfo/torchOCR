"""Total-Text: curved scene text with word-level polygons."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from ..core.structures import TextDetectionTarget


_SPLITS = {"train": "Train", "test": "Test"}
_IMAGE = re.compile(r"img(\d+)\.(jpe?g|png)$", re.IGNORECASE)
_COORDS = re.compile(r"([xy]): \[\[([-\d\s]+)\]\]")
_TEXT = re.compile(r"transcriptions: \[\s*u?(['\"])(.*)\1")
_DONT_CARE = "#"


class TotalText(Dataset[tuple[Tensor, TextDetectionTarget]]):
    """Total-Text (Ch'ng & Chan, ICDAR 2017): 1255 train / 300 test images of
    horizontal, multi-oriented and *curved* scene text, annotated with
    word-level polygons of 3 to 19 vertices.

    Expected layout (the released images plus the ``txt_format`` ground truth)::

        root/
            Images/Train/img11.jpg ...          Images/Test/img1.jpg ...
            txt_format/Train/poly_gt_img11.txt  txt_format/Test/poly_gt_img1.txt

    Polygons are padded to the largest vertex count of their image by
    repeating their last vertex, which changes neither area, IoU nor
    rasterization, so they fit one ``(N, P, 2)`` tensor. Words transcribed
    ``#`` (illegible) or with an empty transcription, and polygons with fewer
    than 3 vertices, are marked ``ignore``.

    Images are read *without* applying EXIF orientation: the annotations
    follow the stored pixels (checked on the six EXIF-rotated images).

    Args:
        root: Directory holding ``Images/`` and ``txt_format/``.
        split: ``"train"`` or ``"test"``. Default ``"test"``.
        transforms: Optional callable ``(image, target) -> (image, target)``.

    Returns:
        ``(image, target)``: ``(3, H, W)`` ``uint8`` RGB and a
        :class:`TextDetectionTarget` with ``(N, P, 2)`` polygons.
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
        images_dir = self.root / "Images" / _SPLITS[split]
        self.gt_dir = self.root / "txt_format" / _SPLITS[split]
        missing = [str(d.relative_to(self.root)) for d in (images_dir, self.gt_dir) if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Total-Text '{split}' split not found under {self.root} (missing: {', '.join(missing)}). "
                "Get the images and the txt_format ground truth from "
                "https://github.com/cs-chan/Total-Text-Dataset and arrange them as Images/{Train,Test} "
                "and txt_format/{Train,Test}."
            )
        self.image_paths = sorted(
            (p for p in images_dir.iterdir() if _IMAGE.search(p.name)),
            key=lambda p: int(_IMAGE.search(p.name).group(1)),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_target(self, index: int) -> TextDetectionTarget:
        """The annotations of sample ``index``, without decoding its image."""
        return parse_totaltext_ground_truth(self.gt_dir / f"poly_gt_{self.image_paths[index].stem}.txt")

    def __getitem__(self, index: int) -> tuple[Tensor, TextDetectionTarget]:
        image = read_image(str(self.image_paths[index]), mode=ImageReadMode.RGB)
        target = self.load_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def parse_totaltext_ground_truth(path: Path) -> TextDetectionTarget:
    """Parse a ``poly_gt_*.txt`` file, tolerating the release's malformed lines."""
    # An entry starts with "x:"; a few entries wrap onto the next physical line.
    entries = [e for e in re.split(r"\n(?=x:)", path.read_text(encoding="utf-8")) if e.strip()]
    polygons: list[list[list[float]]] = []
    texts: list[str] = []
    for entry in entries:
        coords = {axis: [float(v) for v in values.split()] for axis, values in _COORDS.findall(entry)}
        xs, ys = coords.get("x", []), coords.get("y", [])
        if not xs or len(xs) != len(ys):
            raise ValueError(f"{path}: malformed Total-Text entry {entry.strip()!r}.")
        match = _TEXT.search(entry)
        texts.append(match.group(2) if match else "")
        polygons.append([[x, y] for x, y in zip(xs, ys)])

    if not polygons:
        return TextDetectionTarget(torch.zeros(0, 4, 2), [], torch.zeros(0, dtype=torch.bool))
    vertices = max(4, max(len(p) for p in polygons))
    padded = [p + [p[-1]] * (vertices - len(p)) for p in polygons]
    ignore = [t in (_DONT_CARE, "") or len(p) < 3 for t, p in zip(texts, polygons)]
    return TextDetectionTarget(
        polygons=torch.tensor(padded, dtype=torch.float32),
        texts=texts,
        ignore=torch.tensor(ignore, dtype=torch.bool),
    )
