"""Text detection model definitions."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor

from ..transforms import DetectionPreset
from .backbones import MobileNetV3, ResNetVd
from .backbones.mobilenet_v3 import _SEModule
from .hub import BASE_URL, Weights, WeightsEnum, load_weights, register_model


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


class _RSELayer(nn.Module):
    """1x1 (or 3x3) conv -> SE block, with optional shortcut.

    Used by :class:`_RSEFPN` to inject channel attention at every FPN
    convolution. Submodule names match PaddleOCR (``in_conv``,
    ``se_block``).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, shortcut: bool = True) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.se_block = _SEModule(out_channels)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        return x + self.se_block(x) if self.shortcut else self.se_block(x)


class _RSEFPN(nn.Module):
    """PP-OCRv3 RSEFPN: DBFPN cascade with SE attention at every conv.

    Same top-down + concat structure as :class:`_DBFPN` but every
    1x1 lateral and 3x3 smooth conv is replaced by an :class:`_RSELayer`.
    Default ``out_channels`` is 96, the value PP-OCRv3 ships with.
    """

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 96, shortcut: bool = True) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"RSEFPN expects 4 input scales; got {len(in_channels)}.")
        smooth = out_channels // 4
        # PaddleOCR stores the 4 scales as ModuleLists indexed c2..c5 (i=0..3).
        self.ins_conv = nn.ModuleList(
            _RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            for i in range(4)
        )
        self.inp_conv = nn.ModuleList(
            _RSELayer(out_channels, smooth, kernel_size=3, shortcut=shortcut)
            for _ in range(4)
        )

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        c2, c3, c4, c5 = (features[name] for name in _BACKBONE_KEYS)
        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2.0, mode="nearest")
        out3 = in3 + F.interpolate(out4, scale_factor=2.0, mode="nearest")
        out2 = in2 + F.interpolate(out3, scale_factor=2.0, mode="nearest")

        p5 = F.interpolate(self.inp_conv[3](in5), scale_factor=8.0, mode="nearest")
        p4 = F.interpolate(self.inp_conv[2](out4), scale_factor=4.0, mode="nearest")
        p3 = F.interpolate(self.inp_conv[1](out3), scale_factor=2.0, mode="nearest")
        p2 = self.inp_conv[0](out2)
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


BackboneName = Literal["resnet18", "resnet18_vd", "mobilenet_v3_large_05"]


# PaddleOCR detectors saw cv2-decoded BGR pixels normalized with RGB-ordered
# ImageNet statistics; on ICDAR-2015 "bgr" beats "rgb" for every checkpoint.
_PADDLE_DET_PRESET = partial(DetectionPreset, max_side=960, channel_order="bgr")

# PaddleOCR's DB post-processing defaults, under which the converted checkpoints were scored.
_PADDLE_DB_POSTPROCESS = {"threshold": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5}

_ICDAR2015_PROTOCOL = (
    "ICDAR-2015 test (500 images, 2077 care words), official IoU protocol on rotated quads; "
    "transforms() preset, DBPostProcessor(**meta['postprocess']). "
    "Reproduce with references/detection/evaluate.py --weights <this enum>."
)
_LINE_LEVEL_CAVEAT = (
    "Trained by PaddleOCR on line-level annotations of mostly Chinese/English document and "
    "scene text, then converted to PyTorch. ICDAR-2015 scores *word* boxes in low-resolution "
    "street scenes, so adjacent words merged into one line count as misses; use this number "
    "to compare checkpoints, not as the accuracy you will see on documents."
)


class DBNet_ResNet18_VD_Weights(WeightsEnum):
    PPOCR_SERVER_V2 = Weights(
        url=f"{BASE_URL}/dbnet_resnet18_vd_ppocr_server_v2-59d99b11.pth",
        transforms=_PADDLE_DET_PRESET,
        meta={
            "task": "detection",
            "backbone": "resnet18_vd",
            "num_params": 12_364_386,
            "source": "PaddleOCR ch_ppocr_server_v2.0_det_train, via scripts/convert_paddle_dbnet.py",
            "license": "Apache-2.0",
            "languages": ["ch", "en"],
            "postprocess": _PADDLE_DB_POSTPROCESS,
            "_metrics": {"ICDAR2015-test": {"precision": 0.5814, "recall": 0.3216, "hmean": 0.4141}},
            "_docs": f"{_LINE_LEVEL_CAVEAT} {_ICDAR2015_PROTOCOL}",
        },
    )
    DEFAULT = PPOCR_SERVER_V2


class DBNet_MobileNetV3_Large_05_Weights(WeightsEnum):
    PPOCR_V3_CH = Weights(
        url=f"{BASE_URL}/dbnet_mobilenet_v3_large_05_ppocr_v3_ch-fc000d1e.pth",
        transforms=_PADDLE_DET_PRESET,
        meta={
            "task": "detection",
            "backbone": "mobilenet_v3_large_05",
            "num_params": 603_418,
            "source": "PaddleOCR ch_PP-OCRv3_det_distill_train (Student), via scripts/convert_paddle_dbnet_v3.py",
            "license": "Apache-2.0",
            "languages": ["ch", "en"],
            "postprocess": _PADDLE_DB_POSTPROCESS,
            "_metrics": {"ICDAR2015-test": {"precision": 0.5641, "recall": 0.3370, "hmean": 0.4219}},
            "_docs": f"{_LINE_LEVEL_CAVEAT} {_ICDAR2015_PROTOCOL}",
        },
    )
    PPOCR_V3_EN = Weights(
        url=f"{BASE_URL}/dbnet_mobilenet_v3_large_05_ppocr_v3_en-9bfd3b59.pth",
        transforms=_PADDLE_DET_PRESET,
        meta={
            "task": "detection",
            "backbone": "mobilenet_v3_large_05",
            "num_params": 603_418,
            "source": "PaddleOCR en_PP-OCRv3_det_distill_train (Student), via scripts/convert_paddle_dbnet_v3.py",
            "license": "Apache-2.0",
            "languages": ["en"],
            "postprocess": _PADDLE_DB_POSTPROCESS,
            "_metrics": {"ICDAR2015-test": {"precision": 0.5342, "recall": 0.3755, "hmean": 0.4411}},
            "_docs": f"{_LINE_LEVEL_CAVEAT} {_ICDAR2015_PROTOCOL}",
        },
    )
    ICDAR2015 = Weights(
        url=f"{BASE_URL}/dbnet_mobilenet_v3_large_05_ic15-d9d17ae7.pth",
        # Trained and scored at PaddleOCR's ICDAR-2015 test size (736 x 1280).
        transforms=partial(DetectionPreset, max_side=1280, channel_order="bgr"),
        meta={
            "task": "detection",
            "backbone": "mobilenet_v3_large_05",
            "num_params": 603_418,
            "source": (
                "torchocr: PPOCR_V3_EN fine-tuned 300 epochs on the 1000 ICDAR-2015 training images "
                "with references/detection/train.py"
            ),
            "license": "Apache-2.0 (base weights); fine-tuned on ICDAR 2015, released for research use",
            "languages": ["en"],
            "postprocess": {"threshold": 0.3, "box_thresh": 0.45, "unclip_ratio": 1.5},
            "_metrics": {"ICDAR2015-test": {"precision": 0.7845, "recall": 0.6890, "hmean": 0.7337}},
            "_docs": (
                "Word-level detector for incidental scene text (street-level photos, Latin script). "
                "box_thresh=0.45 maximizes hmean on the *training* split "
                "(references/detection/calibrate.py) and was applied once to the test split; with "
                "PaddleOCR's box_thresh=0.6 this checkpoint scores 0.688. Weights are the last "
                "epoch -- no checkpoint was selected on the test set. Expect lower recall on dense "
                f"documents and on non-Latin scripts. {_ICDAR2015_PROTOCOL}"
            ),
        },
    )
    DEFAULT = PPOCR_V3_CH


_WEIGHTS_BY_BACKBONE: dict[str, type[WeightsEnum]] = {
    "resnet18_vd": DBNet_ResNet18_VD_Weights,
    "mobilenet_v3_large_05": DBNet_MobileNetV3_Large_05_Weights,
}


class DBNet(nn.Module):
    """DBNet text detector.

    The model produces a probability map and a threshold map at the
    input resolution. The differentiable-binarization combination of
    the two maps is a training-time loss helper and lives outside this
    module.

    Args:
        backbone: Which backbone to instantiate. ``"resnet18"`` uses
            torchvision's ResNet-18 with its 7x7 stem and is what
            torchocr trains from scratch (no published OCR weights).
            ``"resnet18_vd"`` and ``"mobilenet_v3_large_05"`` mirror
            PaddleOCR's detectors so converted checkpoints load as-is.
            Default ``None``: taken from ``weights`` when it is an enum
            member, else ``"resnet18"``.
        weights: Pretrained OCR weights -- a member of
            :class:`DBNet_ResNet18_VD_Weights` or
            :class:`DBNet_MobileNetV3_Large_05_Weights`, or a member name
            such as ``"DEFAULT"`` resolved against ``backbone``. If the
            download fails a ``UserWarning`` is emitted and the model
            keeps its random initialization. Pair with
            ``weights.transforms()`` for the matching preprocessing.
            Default ``None`` (random init).
            ``pretrained_backbone`` is ignored for the PaddleOCR-style
            backbones since no canonical ImageNet weights ship with
            torchvision for them.
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
        backbone: BackboneName | None = None,
        weights: WeightsEnum | str | None = None,
        pretrained_backbone: bool = False,
        fpn_out_channels: int = 256,
        head_inner_channels: int = 64,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = weights.meta["backbone"] if isinstance(weights, WeightsEnum) else "resnet18"
        if weights is not None:
            if backbone not in _WEIGHTS_BY_BACKBONE:
                raise ValueError(
                    f"No pretrained weights are published for backbone '{backbone}'. "
                    f"Backbones with weights: {sorted(_WEIGHTS_BY_BACKBONE)}."
                )
            weights = _WEIGHTS_BY_BACKBONE[backbone].verify(weights)
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
        elif backbone == "mobilenet_v3_large_05":
            # PP-OCRv3 detector: MobileNetV3-large scale=0.5 with SE *disabled*
            # in the backbone (SE only lives in the RSEFPN). Default neck width
            # is 96 for v3 instead of the 256 used by the ResNet-VD detector.
            self.backbone = MobileNetV3(model_name="large", scale=0.5, disable_se=True)
            v3_fpn_channels = 96 if fpn_out_channels == 256 else fpn_out_channels
            self.fpn = _RSEFPN(self.backbone.out_channels, out_channels=v3_fpn_channels)
            self.binarize = _DBHead(v3_fpn_channels)
            self.thresh = _DBHead(v3_fpn_channels)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                "Known: 'resnet18', 'resnet18_vd', 'mobilenet_v3_large_05'."
            )

        if weights is not None:
            load_weights(self, weights)

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

        # ResNet-VD and MobileNetV3 paths share the same forward shape:
        # backbone -> FPN/RSEFPN (single fused tensor) -> two DBHead modules.
        fused = self.fpn(feature_maps)
        return DBNetOutput(
            probability=self.binarize(fused),
            threshold=self.thresh(fused),
        )


@register_model("dbnet_resnet18")
def dbnet_resnet18(**kwargs: object) -> DBNet:
    """DBNet with torchvision's ResNet-18 backbone (no published OCR weights)."""
    return DBNet(backbone="resnet18", **kwargs)


@register_model("dbnet_resnet18_vd", weights=DBNet_ResNet18_VD_Weights)
def dbnet_resnet18_vd(*, weights: DBNet_ResNet18_VD_Weights | str | None = None, **kwargs: object) -> DBNet:
    """DBNet with PaddleOCR's ResNet-18-VD backbone; see :class:`DBNet_ResNet18_VD_Weights`."""
    return DBNet(backbone="resnet18_vd", weights=weights, **kwargs)


@register_model("dbnet_mobilenet_v3_large_05", weights=DBNet_MobileNetV3_Large_05_Weights)
def dbnet_mobilenet_v3_large_05(
    *, weights: DBNet_MobileNetV3_Large_05_Weights | str | None = None, **kwargs: object
) -> DBNet:
    """PP-OCRv3 detector (MobileNetV3-large x0.5 + RSEFPN); see :class:`DBNet_MobileNetV3_Large_05_Weights`."""
    return DBNet(backbone="mobilenet_v3_large_05", weights=weights, **kwargs)
