"""Text detection model definitions."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from .backbones import ResNetVd
from .hub import load_pretrained_state_dict


_BACKBONE_TAPS = {"layer1": "c2", "layer2": "c3", "layer3": "c4", "layer4": "c5"}
_BACKBONE_CHANNELS = (64, 128, 256, 512)
_BACKBONE_KEYS = ("c2", "c3", "c4", "c5")


@dataclass
class DBNetOutput:
    """Probability and threshold maps from a DBNet forward pass."""

    probability: Tensor
    threshold: Tensor


class _FPN(nn.Module):
    """Top-down Feature Pyramid Network with lateral connections.

    This is the original torchocr FPN, paired with the torchvision
    ``resnet18`` backbone path. The PaddleOCR-compatible
    :class:`_DBFPN` is used with the ``resnet18_vd`` path instead.
    """

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


class _DBFPN(nn.Module):
    """PaddleOCR-compatible DBFPN with cascaded top-down adds.

    Produces a single fused tensor at 1/4 resolution shaped
    ``(B, out_channels, H/4, W/4)``. Each pyramid level is reduced to
    ``out_channels // 4`` and concatenated to recover the full
    ``out_channels`` count -- this matches PaddleOCR's
    ``ppocr.modeling.necks.db_fpn.DBFPN`` exactly so weights port over
    with a small mechanical name remap.

    Submodule names (``in2_conv``, ``p2_conv``, ...) match PaddleOCR.
    """

    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"DBFPN expects 4 input scales; got {len(in_channels)}.")
        c2, c3, c4, c5 = in_channels
        self.in2_conv = nn.Conv2d(c2, out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(c3, out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(c4, out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(c5, out_channels, kernel_size=1, bias=False)
        smooth = out_channels // 4
        self.p2_conv = nn.Conv2d(out_channels, smooth, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(out_channels, smooth, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(out_channels, smooth, kernel_size=3, padding=1, bias=False)
        self.p5_conv = nn.Conv2d(out_channels, smooth, kernel_size=3, padding=1, bias=False)

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        c2, c3, c4, c5 = (features[name] for name in _BACKBONE_KEYS)
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2.0, mode="nearest")
        out3 = in3 + F.interpolate(out4, scale_factor=2.0, mode="nearest")
        out2 = in2 + F.interpolate(out3, scale_factor=2.0, mode="nearest")

        p5 = F.interpolate(self.p5_conv(in5), scale_factor=8.0, mode="nearest")
        p4 = F.interpolate(self.p4_conv(out4), scale_factor=4.0, mode="nearest")
        p3 = F.interpolate(self.p3_conv(out3), scale_factor=2.0, mode="nearest")
        p2 = self.p2_conv(out2)
        return torch.cat([p5, p4, p3, p2], dim=1)


class _DBHead(nn.Module):
    """One head (binarize or threshold) of a DBNet.

    Module layout matches PaddleOCR's ``ppocr.modeling.heads.det_db_head.Head``:
    ``conv1`` -> ``conv_bn1`` -> ReLU -> ``conv2`` (transpose, x2) ->
    ``conv_bn2`` -> ReLU -> ``conv3`` (transpose, x2) -> sigmoid.
    Sigmoid is applied in forward and is not a registered module, so
    the state_dict has exactly the same parameter names as PaddleOCR's
    head once the converter strips its ``head.binarize.`` /
    ``head.thresh.`` prefixes.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        inner = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, inner, kernel_size=3, padding=1, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.ConvTranspose2d(inner, inner, kernel_size=2, stride=2)
        self.conv_bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.ConvTranspose2d(inner, 1, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv_bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.conv_bn2(self.conv2(x)), inplace=True)
        return torch.sigmoid(self.conv3(x))


BackboneName = Literal["resnet18", "resnet18_vd"]


class DBNet(nn.Module):
    """DBNet text detector.

    The model produces a probability map and a threshold map at the
    input resolution. The differentiable-binarization combination of
    the two maps is a training-time loss helper and lives outside this
    module.

    Args:
        backbone: Which backbone to instantiate. ``"resnet18"`` (default)
            uses torchvision's ResNet-18 with its 7x7 stem and is what
            torchocr trains from scratch. ``"resnet18_vd"`` uses the
            PaddleOCR-compatible ResNet-18-VD with a 3-conv stem and
            avg-pool shortcuts on stride-2 blocks; its parameter shapes
            and naming are aligned with PaddleOCR weights so the
            ``scripts/convert_paddle_dbnet.py`` build-time converter
            can produce drop-in checkpoints.
        weights: Optional named preset of pretrained OCR weights. Pass
            ``"DEFAULT"`` to download torchocr's published checkpoint
            for the chosen backbone from the model hub. If the download
            fails (e.g. the file is not yet published) a warning is
            printed and the model is left initialized from architecture
            defaults -- ``pretrained_backbone`` is the right knob to
            combine with ``weights="DEFAULT"`` so the ResNet-18 backbone
            keeps ImageNet weights when the OCR checkpoint is
            unavailable. Default ``None`` (random init).
            ``pretrained_backbone`` is ignored for ``backbone="resnet18_vd"``
            since no canonical ImageNet ResNet-VD-18 weights ship with
            torchvision; use ``weights="DEFAULT"`` instead.
        pretrained_backbone: If True and ``backbone="resnet18"``, load
            ImageNet weights for the torchvision ResNet-18 backbone.
            Default False to keep instantiation offline-safe.
        fpn_out_channels: Channels per FPN level. Default 256.
        head_inner_channels: Inner channels of the probability and
            threshold heads. Used only on the ``resnet18`` path; the
            ``resnet18_vd`` path uses ``fpn_out_channels // 4`` to
            match PaddleOCR.

    Note:
        The ``resnet18`` backbone is wrapped via
        ``create_feature_extractor`` (FX-traced). Custom backbones with
        data-dependent control flow may fail to trace. The
        ``resnet18_vd`` path bypasses FX tracing.
    """

    def __init__(
        self,
        backbone: BackboneName = "resnet18",
        weights: str | None = None,
        pretrained_backbone: bool = False,
        fpn_out_channels: int = 256,
        head_inner_channels: int = 64,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            backbone_weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            torchvision_backbone = resnet18(weights=backbone_weights)
            self.backbone = create_feature_extractor(
                torchvision_backbone, return_nodes=_BACKBONE_TAPS
            )
            self.fpn = _FPN(_BACKBONE_CHANNELS, fpn_out_channels)
            self.fuse = nn.Sequential(
                nn.Conv2d(
                    fpn_out_channels * 4,
                    fpn_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(fpn_out_channels),
                nn.ReLU(inplace=True),
            )
            self.probability_head = self._make_legacy_head(
                fpn_out_channels, head_inner_channels
            )
            self.threshold_head = self._make_legacy_head(
                fpn_out_channels, head_inner_channels
            )
        elif backbone == "resnet18_vd":
            self.backbone = ResNetVd(depth=18)
            self.fpn = _DBFPN(self.backbone.out_channels, fpn_out_channels)
            self.binarize = _DBHead(fpn_out_channels)
            self.thresh = _DBHead(fpn_out_channels)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Known: 'resnet18', 'resnet18_vd'."
            )

        if weights is not None:
            registry_key = "dbnet_resnet18_vd" if backbone == "resnet18_vd" else "dbnet"
            state_dict = load_pretrained_state_dict(registry_key, weights)
            if state_dict is not None:
                self.load_state_dict(state_dict)

    @staticmethod
    def _make_legacy_head(in_channels: int, inner_channels: int) -> nn.Sequential:
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

        if self.backbone_name == "resnet18":
            pyramid = self.fpn([feature_maps[name] for name in _BACKBONE_KEYS])
            target_size = pyramid[0].shape[-2:]
            upsampled = [pyramid[0]] + [
                F.interpolate(level, size=target_size, mode="nearest") for level in pyramid[1:]
            ]
            fused = self.fuse(torch.cat(upsampled, dim=1))
            return DBNetOutput(
                probability=self.probability_head(fused),
                threshold=self.threshold_head(fused),
            )

        fused = self.fpn(feature_maps)
        return DBNetOutput(
            probability=self.binarize(fused),
            threshold=self.thresh(fused),
        )
