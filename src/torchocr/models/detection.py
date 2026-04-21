"""Text detection model definitions."""

import torch
from torch import Tensor, nn


class TextDetectionModel(nn.Module):
    """Placeholder model for predicting text bounding boxes."""

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("Expected images with shape (B, C, H, W).")
        batch_size, _, height, width = images.shape
        polygon = torch.tensor(
            [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]],
            device=images.device,
            dtype=torch.float32,
        )
        return polygon.unsqueeze(0).repeat(batch_size, 1, 1)
