"""ResNet-VD backbone matching PaddleOCR's det_resnet_vd structure.

The "VD" variant differs from torchvision's ResNet-18 in two places:

1. The stem replaces the single 7x7 conv with three stacked 3x3 convs
   (32 -> 32 -> 64), giving a richer low-level representation at the
   same downsampling factor.
2. Stride-2 shortcuts on residual blocks use a 2x2 average-pool
   followed by a 1x1 conv instead of a 1x1 strided conv. This avoids
   discarding 3 of every 4 input pixels at the shortcut.

The internal module naming intentionally mirrors PaddleOCR's
``det_resnet_vd.py`` (``conv1_1``, ``stages``, ``_conv``,
``_batch_norm``) so PaddleOCR-trained weights translate to torchocr
state_dicts with a small, mechanical name remap. This is the only
place in torchocr we accept Paddle-style names; the public forward
contract returns idiomatic PyTorch outputs.
"""

from typing import Literal

from torch import Tensor, nn


_BACKBONE_OUT_CHANNELS = (64, 128, 256, 512)


class _ConvBNLayer(nn.Module):
    """3x3 (or 1x1) Conv -> BatchNorm -> optional ReLU.

    When ``is_vd_mode`` is True an upstream 2x2 avg-pool is applied
    before the convolution; this is the VD shortcut trick on stride-2
    blocks. The internal submodule names ``_conv`` / ``_batch_norm``
    match PaddleOCR's checkpoints.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        is_vd_mode: bool = False,
        act: bool = False,
    ) -> None:
        super().__init__()
        self.is_vd_mode = is_vd_mode
        self.act = act
        if is_vd_mode:
            self._pool2d_avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
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
    """ResNet-VD basic block (two 3x3 convs + identity-or-VD shortcut)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
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
                stride=1,
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
        depth: 18 (BasicBlock-based, default) is the only currently
            supported depth. ResNet-50-VD support requires bottleneck
            blocks and is left to a follow-up.
        in_channels: Number of input image channels. Default 3.

    Forward signature:
        Input ``(B, in_channels, H, W)`` with ``H, W`` divisible by 32.
        Output ``dict[str, Tensor]`` with keys ``c2``, ``c3``, ``c4``,
        ``c5`` at strides 4, 8, 16, 32 with channels (64, 128, 256, 512).

    Note:
        Each stage's blocks are exposed as ``stages[i].bb_<i>_<j>``
        matching PaddleOCR's checkpoint layout. This is what enables
        the converter in ``scripts/convert_paddle_dbnet.py`` to do a
        small mechanical name remap rather than a full re-architecture.
    """

    def __init__(
        self,
        depth: Literal[18] = 18,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if depth != 18:
            raise NotImplementedError(
                f"Only depth=18 is currently supported; got depth={depth}. "
                "ResNet-50-VD requires bottleneck blocks and is planned for a follow-up."
            )

        self.depth = depth
        block_counts = (2, 2, 2, 2)
        in_per_stage = (64, 64, 128, 256)
        out_per_stage = _BACKBONE_OUT_CHANNELS

        self.conv1_1 = _ConvBNLayer(in_channels, 32, kernel_size=3, stride=2, act=True)
        self.conv1_2 = _ConvBNLayer(32, 32, kernel_size=3, stride=1, act=True)
        self.conv1_3 = _ConvBNLayer(32, 64, kernel_size=3, stride=1, act=True)
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        for stage_idx, (n_blocks, c_in, c_out) in enumerate(
            zip(block_counts, in_per_stage, out_per_stage)
        ):
            stage = nn.Sequential()
            for block_idx in range(n_blocks):
                block = _BasicBlock(
                    in_channels=c_in if block_idx == 0 else c_out,
                    out_channels=c_out,
                    stride=2 if block_idx == 0 and stage_idx != 0 else 1,
                    shortcut=block_idx != 0,
                    if_first=stage_idx == 0 and block_idx == 0,
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
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}
