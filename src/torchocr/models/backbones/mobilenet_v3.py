"""MobileNetV3 backbone matching PaddleOCR's det_mobilenet_v3 structure.

Provides the backbone for PP-OCRv2/v3 lightweight detectors and
recognizers. Compared to ResNet-VD this network uses inverted-residual
blocks (1x1 expand -> depthwise k×k -> 1x1 project) with optional SE
attention, ``hard_swish`` and ``hard_sigmoid`` activations, and a
``make_divisible`` rounding rule for channel counts.

Internal module names mirror PaddleOCR
(``conv``, ``stages``, ``expand_conv``, ``bottleneck_conv``,
``linear_conv``, ``mid_se``, ``conv_last``) so checkpoints port over
with a small mechanical name remap. The public forward returns the
same ``c2``/``c3``/``c4``/``c5`` dict the existing detector necks
already consume.
"""

from typing import Literal

from torch import Tensor, nn
from torch.nn import functional as F


def _hard_swish(x: Tensor, inplace: bool = True) -> Tensor:
    """``x * relu6(x + 3) / 6`` -- HardSwish activation."""
    return x * F.relu6(x + 3.0, inplace=inplace) / 6.0


def _hard_sigmoid(x: Tensor, inplace: bool = True) -> Tensor:
    """``relu6(x + 3) / 6`` -- HardSigmoid for SE gates."""
    return F.relu6(x + 3.0, inplace=inplace) / 6.0


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round ``v`` to the nearest multiple of ``divisor`` (>= min_value).

    Mirrors PaddleOCR's ``make_divisible`` so scaled channel counts agree
    bit-for-bit with the trained weights. Without this exact function,
    rounding differences would produce shape mismatches at scale != 1.0.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class _ConvBNLayer(nn.Module):
    """Conv -> BN -> optional activation, with submodule names ``conv`` / ``bn``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int = 1,
        if_act: bool = True,
        act: str | None = None,
    ) -> None:
        super().__init__()
        self.if_act = if_act
        self.act_kind = act
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if not self.if_act:
            return x
        if self.act_kind == "relu":
            return F.relu(x, inplace=True)
        if self.act_kind == "hard_swish":
            return _hard_swish(x)
        raise ValueError(f"Unknown activation {self.act_kind!r}; expected 'relu' or 'hard_swish'.")


class _SEModule(nn.Module):
    """Channel-attention block: GAP -> 1x1 -> ReLU -> 1x1 -> HardSigmoid -> mul.

    Internal submodule names ``conv1`` / ``conv2`` match PaddleOCR's
    ``SEModule`` so weights port over without a remap.
    """

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        s = self.avg_pool(x)
        s = F.relu(self.conv1(s), inplace=True)
        s = self.conv2(s)
        s = _hard_sigmoid(s)
        return x * s


class _ResidualUnit(nn.Module):
    """Inverted-residual block: 1x1 expand -> depthwise k×k -> optional SE -> 1x1 project."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        act: str,
    ) -> None:
        super().__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se
        self.expand_conv = _ConvBNLayer(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, act=act
        )
        self.bottleneck_conv = _ConvBNLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=mid_channels,
            act=act,
        )
        if use_se:
            self.mid_se = _SEModule(mid_channels)
        self.linear_conv = _ConvBNLayer(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, if_act=False
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = residual + x
        return x


# --------------------------------------------------------------------------------------
# Architecture configuration tables (k, exp, c, se, nl, s)
# Source: PaddleOCR2Pytorch det_mobilenet_v3.MobileNetV3, identical so weights port over.
# --------------------------------------------------------------------------------------
_CFG_LARGE: tuple[tuple[int, int, int, bool, str, int], ...] = (
    (3, 16, 16, False, "relu", 1),
    (3, 64, 24, False, "relu", 2),
    (3, 72, 24, False, "relu", 1),
    (5, 72, 40, True, "relu", 2),
    (5, 120, 40, True, "relu", 1),
    (5, 120, 40, True, "relu", 1),
    (3, 240, 80, False, "hard_swish", 2),
    (3, 200, 80, False, "hard_swish", 1),
    (3, 184, 80, False, "hard_swish", 1),
    (3, 184, 80, False, "hard_swish", 1),
    (3, 480, 112, True, "hard_swish", 1),
    (3, 672, 112, True, "hard_swish", 1),
    (5, 672, 160, True, "hard_swish", 2),
    (5, 960, 160, True, "hard_swish", 1),
    (5, 960, 160, True, "hard_swish", 1),
)
_CLS_CH_SQUEEZE_LARGE = 960

_CFG_SMALL: tuple[tuple[int, int, int, bool, str, int], ...] = (
    (3, 16, 16, True, "relu", 2),
    (3, 72, 24, False, "relu", 2),
    (3, 88, 24, False, "relu", 1),
    (5, 96, 40, True, "hard_swish", 2),
    (5, 240, 40, True, "hard_swish", 1),
    (5, 240, 40, True, "hard_swish", 1),
    (5, 120, 48, True, "hard_swish", 1),
    (5, 144, 48, True, "hard_swish", 1),
    (5, 288, 96, True, "hard_swish", 2),
    (5, 576, 96, True, "hard_swish", 1),
    (5, 576, 96, True, "hard_swish", 1),
)
_CLS_CH_SQUEEZE_SMALL = 576


_SUPPORTED_SCALES = (0.35, 0.5, 0.75, 1.0, 1.25)


class MobileNetV3(nn.Module):
    """MobileNetV3 backbone returning a four-level feature pyramid.

    Args:
        in_channels: Number of input image channels. Default 3.
        model_name: ``"large"`` (default, used by PP-OCRv3 detector) or
            ``"small"``.
        scale: Channel-width multiplier. Must be one of
            ``{0.35, 0.5, 0.75, 1.0, 1.25}``. PP-OCRv3 detector uses
            ``0.5``.
        disable_se: If True, ignore the per-block ``use_se`` flags. The
            PP-OCRv3 *detector* uses this; recognizer keeps SE blocks.

    Forward signature:
        Input ``(B, in_channels, H, W)`` with H, W divisible by 32.
        Output ``dict[str, Tensor]`` with keys ``c2``, ``c3``, ``c4``,
        ``c5`` at strides 4, 8, 16, 32.
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_name: Literal["large", "small"] = "large",
        scale: float = 0.5,
        disable_se: bool = False,
    ) -> None:
        super().__init__()
        if scale not in _SUPPORTED_SCALES:
            raise ValueError(
                f"scale must be one of {_SUPPORTED_SCALES}; got {scale}."
            )
        if model_name == "large":
            cfg, cls_ch_squeeze = _CFG_LARGE, _CLS_CH_SQUEEZE_LARGE
        elif model_name == "small":
            cfg, cls_ch_squeeze = _CFG_SMALL, _CLS_CH_SQUEEZE_SMALL
        else:
            raise ValueError(f"model_name must be 'large' or 'small'; got {model_name!r}.")

        self.disable_se = disable_se
        self.model_name = model_name
        self.scale = scale

        inplanes = _make_divisible(16 * scale)
        self.conv = _ConvBNLayer(
            in_channels=in_channels,
            out_channels=inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            act="hard_swish",
        )

        # Walk the cfg table, breaking into stages whenever a stride-2 unit
        # appears AFTER index 2 (matches PaddleOCR's exact stage segmentation).
        self.stages = nn.ModuleList()
        out_channels: list[int] = []
        block_list: list[nn.Module] = []
        for i, (k, exp, c, use_se, nl, s) in enumerate(cfg):
            if s == 2 and i > 2:
                out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                _ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=_make_divisible(scale * exp),
                    out_channels=_make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=use_se and not disable_se,
                    act=nl,
                )
            )
            inplanes = _make_divisible(scale * c)

        # The final 1x1 ConvBNLayer (``conv_last``) goes at the tail of the
        # last stage in PaddleOCR's checkpoint layout.
        block_list.append(
            _ConvBNLayer(
                in_channels=inplanes,
                out_channels=_make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                act="hard_swish",
            )
        )
        out_channels.append(_make_divisible(scale * cls_ch_squeeze))
        self.stages.append(nn.Sequential(*block_list))
        self._out_channels = tuple(out_channels)

    @property
    def out_channels(self) -> tuple[int, ...]:
        """Channel counts at strides 4, 8, 16, 32."""
        return self._out_channels

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) input; got shape {tuple(x.shape)}.")
        x = self.conv(x)
        feats: list[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        if len(feats) != 4:
            raise RuntimeError(
                f"MobileNetV3 produced {len(feats)} stages; expected 4. "
                "This indicates an architectural drift from PaddleOCR's reference."
            )
        return {"c2": feats[0], "c3": feats[1], "c4": feats[2], "c5": feats[3]}
