"""DB training augmentation: PaddleOCR's ICDAR-2015 recipe, polygon-aware.

Flip (p=0.5) -> rotate (+-10 deg) -> rescale (x0.5..x3) -> text-preserving
random crop letterboxed into ``size x size`` -> DB targets -> the same
normalization as the inference preset.
"""

import cv2
import numpy as np
import torch
from torch import Tensor

from torchocr.datasets import TextDetectionTarget
from torchocr.transforms import IMAGENET_MEAN, IMAGENET_STD, DBTargets, DetectionPreset, MakeDBTargets


class DBTrainPreset:
    """``(uint8 RGB image, TextDetectionTarget) -> (normalized float image, DBTargets)``.

    Randomness comes from torch's RNG, which ``DataLoader`` reseeds in
    every worker (NumPy's global RNG is not, so workers would otherwise
    replay identical augmentations).

    Args:
        size: Square output side, a multiple of 32. Default 640.
        flip_prob: Horizontal-flip probability. Default 0.5.
        max_rotation: Rotation is uniform in ``[-max_rotation, max_rotation]`` degrees.
        scale_range: Uniform rescale factor range before cropping.
        mean, std, channel_order: Normalization; must match the inference
            preset of the weights being fine-tuned.
        targets: DB target builder. Default :class:`MakeDBTargets` ().
    """

    def __init__(
        self,
        size: int = 640,
        flip_prob: float = 0.5,
        max_rotation: float = 10.0,
        scale_range: tuple[float, float] = (0.5, 3.0),
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        channel_order: str = "bgr",
        targets: MakeDBTargets | None = None,
    ) -> None:
        if size <= 0 or size % 32:
            raise ValueError(f"size must be a positive multiple of 32; got {size}.")
        self.size = size
        self.flip_prob = flip_prob
        self.max_rotation = max_rotation
        self.scale_range = scale_range
        self.normalize = DetectionPreset(max_side=None, mean=mean, std=std, channel_order=channel_order)
        self.targets = targets or MakeDBTargets()

    def __call__(self, image: Tensor, target: TextDetectionTarget) -> tuple[Tensor, DBTargets]:
        rng = np.random.default_rng(int(torch.randint(2**31, (1,))))
        img = np.ascontiguousarray(image.permute(1, 2, 0).numpy())
        polys = target.polygons.numpy().astype(np.float64).copy()
        ignore = target.ignore.numpy().copy()
        height, width = img.shape[:2]

        if rng.random() < self.flip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            polys[..., 0] = width - polys[..., 0]

        angle = rng.uniform(-self.max_rotation, self.max_rotation)
        rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, rotation, (width, height), flags=cv2.INTER_LINEAR)
        polys = polys @ rotation[:, :2].T + rotation[:, 2]

        scale = rng.uniform(*self.scale_range)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        polys *= (new_w / width, new_h / height)

        x, y, crop_w, crop_h = random_text_crop(polys[~ignore], img.shape[:2], rng=rng)
        fit = min(self.size / crop_w, self.size / crop_h)
        out_w, out_h = max(1, int(crop_w * fit)), max(1, int(crop_h * fit))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:out_h, :out_w] = cv2.resize(img[y : y + crop_h, x : x + crop_w], (out_w, out_h))
        polys = (polys - (x, y)) * fit

        keep = ~_outside(polys, out_w, out_h)
        targets = self.targets(polys[keep], ignore[keep], (self.size, self.size))
        return self.normalize(torch.from_numpy(canvas).permute(2, 0, 1)), targets


def random_text_crop(
    polys: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
    min_crop_side_ratio: float = 0.1,
    max_tries: int = 50,
) -> tuple[int, int, int, int]:
    """Pick an ``(x, y, w, h)`` crop whose borders avoid every text polygon.

    Crop edges are drawn from rows/columns not covered by any polygon's
    axis-aligned extent (PaddleOCR's ``EastRandomCropData``), so text is
    either fully inside or fully outside the crop on each axis. A crop
    must contain at least one polygon; after ``max_tries`` failures the
    whole image is returned.
    """
    height, width = shape
    rows = np.zeros(height, dtype=bool)
    cols = np.zeros(width, dtype=bool)
    for poly in np.round(polys).astype(np.int64):
        x0, x1 = np.clip([poly[:, 0].min(), poly[:, 0].max()], 0, width)
        y0, y1 = np.clip([poly[:, 1].min(), poly[:, 1].max()], 0, height)
        cols[x0:x1] = True
        rows[y0:y1] = True
    free_rows, free_cols = np.flatnonzero(~rows), np.flatnonzero(~cols)
    if len(free_rows) == 0 or len(free_cols) == 0:
        return 0, 0, width, height
    row_regions, col_regions = _split_runs(free_rows), _split_runs(free_cols)

    for _ in range(max_tries):
        x0, x1 = _pick_span(col_regions, free_cols, rng)
        y0, y1 = _pick_span(row_regions, free_rows, rng)
        if x1 - x0 < min_crop_side_ratio * width or y1 - y0 < min_crop_side_ratio * height:
            continue
        if len(polys) and not _outside(polys, x1 - x0, y1 - y0, origin=(x0, y0)).all():
            return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
    return 0, 0, width, height


def _split_runs(axis: np.ndarray) -> list[np.ndarray]:
    """Split sorted indices into runs of consecutive values."""
    return np.split(axis, np.flatnonzero(np.diff(axis) != 1) + 1)


def _pick_span(regions: list[np.ndarray], axis: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    # Two points from two (possibly equal) free runs, so a crop can span text.
    if len(regions) > 1:
        picks = [rng.choice(regions[i]) for i in rng.choice(len(regions), size=2)]
    else:
        picks = rng.choice(axis, size=2)
    return int(min(picks)), int(max(picks))


def _outside(polys: np.ndarray, w: float, h: float, origin: tuple[float, float] = (0, 0)) -> np.ndarray:
    """``(N,)`` mask of polygons entirely outside the ``w x h`` rectangle at ``origin``."""
    if len(polys) == 0:
        return np.zeros(0, dtype=bool)
    x, y = polys[..., 0] - origin[0], polys[..., 1] - origin[1]
    return (x.max(1) < 0) | (x.min(1) > w) | (y.max(1) < 0) | (y.min(1) > h)
