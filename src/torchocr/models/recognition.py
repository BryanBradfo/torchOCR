"""Text recognition model definitions."""

from torch import Tensor, nn


class TextRecognitionModel(nn.Module):
    """Placeholder model for reading text from cropped image regions."""

    def forward(self, features: Tensor) -> Tensor:
        return features
