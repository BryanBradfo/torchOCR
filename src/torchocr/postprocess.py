"""Post-processing utilities for detector outputs."""

import torch
from torch import Tensor
from torchvision.ops import masks_to_boxes

from .models.detection import DBNetOutput


class DBPostProcessor:
    """Extract axis-aligned bounding boxes from DBNet probability maps.

    The probability map is binarized at ``threshold`` and the resulting
    boolean mask is fed to :func:`torchvision.ops.masks_to_boxes`,
    yielding one tight bounding box per non-empty image in the batch.
    Images whose masks have no foreground pixels are silently dropped.

    The output is shaped ``(K, 5)`` with columns
    ``[batch_idx, x1, y1, x2, y2]`` so it composes directly with
    :func:`torchvision.ops.roi_align`. ``K`` is the number of batch
    elements that actually contain text — never larger than ``B``.

    This MVP collapses every detected pixel into a single AABB per
    image. Multi-region detection requires per-instance masks (via
    connected-component labeling or contour finding); see the roadmap.

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

        masks = probability[:, 0] > self.threshold
        present = masks.flatten(1).any(dim=1).nonzero(as_tuple=True)[0]
        if present.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=probability.device)

        boxes = masks_to_boxes(masks[present])
        batch_indices = present.to(boxes.dtype).unsqueeze(1)
        return torch.cat([batch_indices, boxes], dim=1)
