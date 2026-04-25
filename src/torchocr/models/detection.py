"""Text detection model definitions."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from .hub import load_pretrained_state_dict


_BACKBONE_TAPS = {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}
_BACKBONE_CHANNELS = (64, 128, 256, 512)


@dataclass
class DBNetOutput:
    """Probability and threshold maps from a DBNet forward pass."""

    probability: Tensor
    threshold: Tensor


class _FPN(nn.Module):
    """Top-down Feature Pyramid Network with lateral connections."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2.0, mode="nearest"
            )
        return [smooth(lat) for smooth, lat in zip(self.smooth, laterals)]


class DBNet(nn.Module):
    """DBNet text detector with a ResNet-18 backbone and FPN neck.

    The model produces a probability map and a threshold map at the input
    resolution. The differentiable-binarization combination of the two
    maps is a training-time loss helper and lives outside this module.

    Args:
        weights: Optional named preset of pretrained OCR weights. Pass
            ``"DEFAULT"`` to download torchocr's published checkpoint
            from the model hub. If the download fails (e.g. the file
            is not yet published) a warning is printed and the model
            is left initialized from architecture defaults --
            ``pretrained_backbone`` is the right knob to combine with
            ``weights="DEFAULT"`` so the ResNet-18 backbone keeps
            ImageNet weights when the OCR checkpoint is unavailable.
            Default ``None`` (random init).
        pretrained_backbone: If True, load ImageNet weights for the
            ResNet-18 backbone. Default False to keep instantiation
            offline-safe.
        fpn_out_channels: Channels per FPN level. Default 256.
        head_inner_channels: Inner channels of the probability and
            threshold heads. Default 64.

    Note:
        The backbone is wrapped via ``create_feature_extractor``, which
        FX-traces the module. Custom backbones with data-dependent
        control flow may fail to trace.
    """

    def __init__(
        self,
        weights: str | None = None,
        pretrained_backbone: bool = False,
        fpn_out_channels: int = 256,
        head_inner_channels: int = 64,
    ) -> None:
        super().__init__()
        backbone_weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=backbone_weights)
        self.backbone = create_feature_extractor(backbone, return_nodes=_BACKBONE_TAPS)
        self.fpn = _FPN(_BACKBONE_CHANNELS, fpn_out_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(
                fpn_out_channels * 4, fpn_out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU(inplace=True),
        )
        self.probability_head = self._make_head(fpn_out_channels, head_inner_channels)
        self.threshold_head = self._make_head(fpn_out_channels, head_inner_channels)

        if weights is not None:
            state_dict = load_pretrained_state_dict("dbnet", weights)
            if state_dict is not None:
                self.load_state_dict(state_dict)

    @staticmethod
    def _make_head(in_channels: int, inner_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, inner_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> DBNetOutput:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W) images; got {tuple(images.shape)}.")
        height, width = images.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError(f"Height and width must be divisible by 32; got {height}x{width}.")

        feature_maps = self.backbone(images)
        pyramid = self.fpn([feature_maps[name] for name in ("c2", "c3", "c4", "c5")])
        target_size = pyramid[0].shape[-2:]
        upsampled = [pyramid[0]] + [
            F.interpolate(level, size=target_size, mode="nearest") for level in pyramid[1:]
        ]
        fused = self.fuse(torch.cat(upsampled, dim=1))
        return DBNetOutput(
            probability=self.probability_head(fused),
            threshold=self.threshold_head(fused),
        )
