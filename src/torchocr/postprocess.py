"""Post-processing utilities for detector outputs."""

import numpy as np
import scipy.ndimage as ndi
import torch
from torch import Tensor

from .models.detection import DBNetOutput


class DBPostProcessor:
    """Extract per-region bounding boxes from DBNet probability maps.

    Thresholds the probability map into a boolean mask, runs
    connected-component labeling on each batch image via
    :func:`scipy.ndimage.label`, and emits one axis-aligned bounding
    box per disconnected component using
    :func:`scipy.ndimage.find_objects`.

    The output is shaped ``(K, 5)`` with columns
    ``[batch_idx, x1, y1, x2, y2]`` so it composes directly with
    :func:`torchvision.ops.roi_align`. ``K`` is the *total* number of
    components across the batch (not capped at one per image), so a
    multi-line document yields one box per text line. Coordinates use
    inclusive max (``x2``, ``y2`` are the *last* foreground pixels)
    matching the convention of :func:`torchvision.ops.masks_to_boxes`.

    Note:
        ``scipy.ndimage`` runs on CPU; GPU probability maps are moved
        to CPU for labeling, then the resulting box tensor is placed
        back on the original device. The sync cost is sub-millisecond
        for typical OCR-page resolutions.

    Args:
        threshold: Probability threshold for binarization. Default 0.3,
            matching the DBNet paper's recommended value.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1]; got {threshold}.")
        self.threshold = threshold

    def __call__(self, output: DBNetOutput) -> Tensor:
        probability = output.probability
        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError(
                f"Expected probability of shape (B, 1, H, W); got {tuple(probability.shape)}."
            )

        masks_np: np.ndarray = (probability[:, 0] > self.threshold).cpu().numpy()
        rows: list[list[float]] = []
        for batch_idx in range(masks_np.shape[0]):
            page_mask = masks_np[batch_idx]
            if not page_mask.any():
                continue
            labeled, _ = ndi.label(page_mask)
            for region in ndi.find_objects(labeled):
                if region is None:
                    continue
                y_slice, x_slice = region
                rows.append(
                    [
                        float(batch_idx),
                        float(x_slice.start),
                        float(y_slice.start),
                        float(x_slice.stop - 1),
                        float(y_slice.stop - 1),
                    ]
                )

        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32, device=probability.device)
        return torch.tensor(rows, dtype=torch.float32, device=probability.device)
