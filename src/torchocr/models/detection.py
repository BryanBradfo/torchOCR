"""Text detection model definitions."""

from torch import Tensor, nn


class TextDetectionModel(nn.Module):
    """Placeholder model for predicting text bounding boxes."""

    def forward(self, images: Tensor) -> Tensor:
        return images
