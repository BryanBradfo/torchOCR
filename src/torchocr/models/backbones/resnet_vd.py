"""ResNet-VD backbone matching PaddleOCR's det_resnet_vd / rec_resnet_vd.

The "VD" variant differs from torchvision's ResNet in two places:

1. The stem replaces the single 7x7 conv with three stacked 3x3 convs
   (32 -> 32 -> 64), giving a richer low-level representation at the
   same downsampling factor.
2. Stride-2 shortcuts on residual blocks use a 2x2 average-pool
   followed by a 1x1 conv instead of a 1x1 strided conv. This avoids
   discarding 3 of every 4 input pixels at the shortcut.

The same module powers both detection (depth=18, downsample_stride=2)
and recognition (depth=34, downsample_stride=(2, 1) so width is
preserved for CTC decoding).

Internal module naming intentionally mirrors PaddleOCR
(``conv1_1``, ``stages``, ``_conv``, ``_batch_norm``) so PaddleOCR
checkpoints translate to torchocr state_dicts with a small mechanical
name remap. This is the only place in torchocr we accept Paddle-style
names; the public forward contract returns idiomatic PyTorch outputs.
"""

from typing import Literal

from torch import Tensor, nn


_BACKBONE_OUT_CHANNELS = (64, 128, 256, 512)

_BLOCK_COUNTS_BY_DEPTH: dict[int, tuple[int, int, int, int]] = {
    18: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
}


StrideArg = int | tuple[int, int]


class _ConvBNLayer(nn.Module):
    """Conv -> BatchNorm -> optional ReLU, with the VD avg-pool trick.

    When ``is_vd_mode`` is True a ``stride``-sized average pool is
    applied before the (now-stride-1) convolution. This is the VD
    shortcut trick on stride-2 blocks. The pool's kernel and stride
    both equal the constructor's ``stride`` argument so the same
    layer works for stride 2 (detection) and stride (2, 1)
    (recognition). The internal submodule names ``_conv`` and
    ``_batch_norm`` match PaddleOCR's checkpoints.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: StrideArg = 1,
        is_vd_mode: bool = False,
        act: bool = False,
    ) -> None:
        super().__init__()
        self.is_vd_mode = is_vd_mode
        self.act = act
        if is_vd_mode:
            self._pool2d_avg = nn.AvgPool2d(
                kernel_size=stride, stride=stride, padding=0, ceil_mode=True
            )
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1 if is_vd_mode else stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self._batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_vd_mode:
            x = self._pool2d_avg(x)
        x = self._conv(x)
        x = self._batch_norm(x)
        if self.act:
            x = nn.functional.relu(x, inplace=True)
        return x


class _BasicBlock(nn.Module):
    """ResNet-VD basic block (two 3x3 convs + identity-or-VD shortcut).

    Both conv0 and the shortcut use the same downsample stride so that
    feature-map shapes line up at the residual addition.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: StrideArg,
        shortcut: bool,
        if_first: bool,
    ) -> None:
        super().__init__()
        self.conv0 = _ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act=True,
        )
        self.conv1 = _ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=False,
        )
        self.shortcut = shortcut
        if not shortcut:
            self.short = _ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                is_vd_mode=not if_first,
                act=False,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv0(x)
        y = self.conv1(y)
        identity = x if self.shortcut else self.short(x)
        return nn.functional.relu(identity + y, inplace=True)


class ResNetVd(nn.Module):
    """ResNet-VD backbone returning a four-level feature pyramid.

    Args:
        depth: 18 (default, used by DBNet detector) or 34 (used by the
            CRNN recognizer). Both are BasicBlock-based; bottleneck
            depths 50+ are deferred.
        in_channels: Number of input image channels. Default 3.
        downsample_stride: Stride applied to the first block of stages
            1, 2, 3 (stage 0 is always stride 1). ``2`` for detection
            (default), ``(2, 1)`` for recognition so width is preserved
            for CTC decoding. The VD avg-pool stride matches.
        stem_stride: Stride of the first stem conv (``conv1_1``).
            ``2`` for detection (default), ``1`` for recognition --
            PaddleOCR's ``rec_resnet_vd`` keeps the stem at full
            resolution so the recognizer's feature maps line up with
            the trained weights.
        final_pool: If True, append a ``MaxPool(2, 2)`` after the last
            stage. Required for recognition (matches PaddleOCR's
            ``out_pool``); detection leaves it off.

    Forward signature:
        Input ``(B, in_channels, H, W)``. With ``downsample_stride=2``,
        H and W must be divisible by 32 and the four output strides
        are 4, 8, 16, 32. With ``downsample_stride=(2, 1)`` the width
        is preserved through stages 1-3 (only the stem and pool reduce
        it), so output width is W/4 and final height is H/32.

        Output ``dict[str, Tensor]`` with keys ``c2``, ``c3``, ``c4``,
        ``c5`` and channels (64, 128, 256, 512).

    Note:
        Each stage's blocks are exposed as ``stages[i].bb_<i>_<j>``
        matching PaddleOCR's checkpoint layout. This is what enables
        ``scripts/convert_paddle_dbnet.py`` and
        ``scripts/convert_paddle_crnn.py`` to do small mechanical name
        remaps rather than full re-architecture.
    """

    def __init__(
        self,
        depth: Literal[18, 34] = 18,
        in_channels: int = 3,
        downsample_stride: StrideArg = 2,
        stem_stride: int = 2,
        final_pool: bool = False,
    ) -> None:
        super().__init__()
        if depth not in _BLOCK_COUNTS_BY_DEPTH:
            supported = sorted(_BLOCK_COUNTS_BY_DEPTH)
            raise NotImplementedError(
                f"Supported depths: {supported}. Got depth={depth}. "
                "Bottleneck depths (50, 101, ...) are planned for a follow-up."
            )

        self.depth = depth
        self.downsample_stride = downsample_stride
        block_counts = _BLOCK_COUNTS_BY_DEPTH[depth]
        in_per_stage = (64, 64, 128, 256)
        out_per_stage = _BACKBONE_OUT_CHANNELS

        self.conv1_1 = _ConvBNLayer(in_channels, 32, kernel_size=3, stride=stem_stride, act=True)
        self.conv1_2 = _ConvBNLayer(32, 32, kernel_size=3, stride=1, act=True)
        self.conv1_3 = _ConvBNLayer(32, 64, kernel_size=3, stride=1, act=True)
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Some PaddleOCR recipes (rec_resnet_vd) append a final pool to halve
        # the remaining height/width after the last stage. Detection skips it.
        self.out_pool = nn.MaxPool2d(kernel_size=2, stride=2) if final_pool else None

        self.stages = nn.ModuleList()
        for stage_idx, (n_blocks, c_in, c_out) in enumerate(
            zip(block_counts, in_per_stage, out_per_stage)
        ):
            stage = nn.Sequential()
            for block_idx in range(n_blocks):
                first_in_stage = block_idx == 0
                first_overall = stage_idx == 0 and first_in_stage
                # stage 0 is stride 1; later stages downsample on their first block.
                stride: StrideArg = 1 if stage_idx == 0 or not first_in_stage else downsample_stride
                block = _BasicBlock(
                    in_channels=c_in if first_in_stage else c_out,
                    out_channels=c_out,
                    stride=stride,
                    shortcut=not first_in_stage,
                    if_first=first_overall,
                )
                stage.add_module(f"bb_{stage_idx}_{block_idx}", block)
            self.stages.append(stage)

    @property
    def out_channels(self) -> tuple[int, ...]:
        return _BACKBONE_OUT_CHANNELS

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) input; got shape {tuple(x.shape)}.")
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.pool2d_max(x)
        c2 = self.stages[0](x)
        c3 = self.stages[1](c2)
        c4 = self.stages[2](c3)
        c5 = self.stages[3](c4)
        if self.out_pool is not None:
            c5 = self.out_pool(c5)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
