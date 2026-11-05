"""Page orientation classification: is the page upright, or turned by 90, 180 or 270 degrees?"""

import torch
from torch import Tensor, nn

from ..transforms import ClassificationPreset
from .backbones.mobilenet_v3 import _hard_swish, _SEModule
from .hub import BASE_URL, Weights, WeightsEnum, load_weights, register_model


# (kernel, in, out, stride, SE) per depthwise-separable block: PP-LCNet x1.0 (PaddleClas NET_CONFIG).
_PP_LCNET_X1_0 = {
    "blocks2": [(3, 16, 32, 1, False)],
    "blocks3": [(3, 32, 64, 2, False), (3, 64, 64, 1, False)],
    "blocks4": [(3, 64, 128, 2, False), (3, 128, 128, 1, False)],
    "blocks5": [(3, 128, 256, 2, False)] + [(5, 256, 256, 1, False)] * 5,
    "blocks6": [(5, 256, 512, 2, True), (5, 512, 512, 1, True)],
}


@torch.fx.wrap
def _check_classifier_input(images: Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W) images; got {tuple(images.shape)}.")


class PPLCNet_x1_0_Weights(WeightsEnum):
    PPOCR_DOC_ORIENTATION = Weights(
        url=f"{BASE_URL}/pplcnet_x1_0_doc_ori-ba9045e4.pth",
        transforms=ClassificationPreset,
        meta={
            "task": "page-orientation",
            "backbone": "pp_lcnet_x1_0",
            "num_params": 1_676_644,
            # Class c: the page content is turned c * 90 degrees clockwise;
            # torch.rot90(image, c, dims=(-2, -1)) sets it upright.
            "categories": ["0", "90", "180", "270"],
            "source": "PaddleOCR PP-LCNet_x1_0_doc_ori, via scripts/convert_paddle_pplcnet.py",
            "license": "Apache-2.0",
            # Chosen on DocLayNet validation (references/orientation/evaluate.py --sweep).
            "min_confidence": 0.7,
            "_metrics": {
                "DocLayNet-test": {
                    "accuracy": 0.9848,
                    "upright_wrongly_turned": 0.0024,
                    "turned_set_upright": 0.9753,
                }
            },
            "_docs": (
                "Trained by PaddleOCR on document pages (mostly Chinese and English), converted to "
                "PyTorch. Scored on the 4999 DocLayNet test pages (financial reports, manuals, laws, "
                "patents, scientific articles, tenders), each fed upright and turned 90, 180 and 270 "
                "degrees: accuracy is the argmax; the other two numbers are PageOrientation's decisions "
                "at min_confidence=0.7, chosen on DocLayNet validation and applied once to test. "
                "Weakest on manuals (0.973) and laws (0.979), where most errors send upright pages to "
                "180 degrees. Not meant for scene photos: on a street photo it is unsure (p < 0.8) and can "
                "still turn it. Reproduce with references/orientation/evaluate.py."
            ),
        },
    )
    DEFAULT = PPOCR_DOC_ORIENTATION


class _ConvBNHardSwish(nn.Module):
    """Conv (no bias) -> BN -> HardSwish, with submodule names ``conv`` / ``bn``."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return _hard_swish(self.bn(self.conv(x)))


class _DepthwiseSeparable(nn.Module):
    """Depthwise k x k -> optional SE -> pointwise 1 x 1 (PP-LCNet's only block type)."""

    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, stride: int, use_se: bool) -> None:
        super().__init__()
        self.dw_conv = _ConvBNHardSwish(in_channels, in_channels, kernel_size, stride, groups=in_channels)
        self.se = _SEModule(in_channels) if use_se else nn.Identity()
        self.pw_conv = _ConvBNHardSwish(in_channels, out_channels, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw_conv(self.se(self.dw_conv(x)))


class PPLCNet(nn.Module):
    """PP-LCNet x1.0 image classifier (Cui et al., 2021), as used by PaddleOCR for page orientation.

    A 1.7M-parameter CPU-oriented network: depthwise-separable blocks with
    HardSwish, squeeze-excitation only in the last two blocks, then global
    pooling and a 1280-wide 1x1 convolution before the classifier.
    Submodule names mirror PaddleClas so checkpoints port with a short
    remap (``scripts/convert_paddle_pplcnet.py``).

    Args:
        num_classes: Number of output logits. Default: from ``weights``, else 4.
        dropout: Dropout probability before the classifier.
        weights: Pretrained :class:`PPLCNet_x1_0_Weights` (member or name).
        progress: Show a download progress bar.

    Forward:
        ``(B, 3, H, W)`` normalized images -> ``(B, num_classes)`` logits.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        dropout: float = 0.2,
        weights: "PPLCNet_x1_0_Weights | str | None" = None,
        progress: bool = True,
    ) -> None:
        super().__init__()
        weights = PPLCNet_x1_0_Weights.verify(weights)
        if weights is not None:
            categories = len(weights.meta["categories"])
            if num_classes not in (None, categories):
                raise ValueError(f"{weights!r} has {categories} classes; got num_classes={num_classes}.")
            num_classes = categories
        num_classes = 4 if num_classes is None else num_classes
        self.conv1 = _ConvBNHardSwish(3, 16, 3, 2)
        for name, config in _PP_LCNET_X1_0.items():
            setattr(self, name, nn.Sequential(*(_DepthwiseSeparable(*block) for block in config)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(512, 1280, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)
        if weights is not None:
            load_weights(self, weights, progress=progress)

    def forward(self, images: Tensor) -> Tensor:
        _check_classifier_input(images)
        x = self.conv1(images)
        x = self.blocks6(self.blocks5(self.blocks4(self.blocks3(self.blocks2(x)))))
        x = _hard_swish(self.last_conv(self.avg_pool(x)))
        return self.fc(self.dropout(torch.flatten(x, 1)))


@register_model("pp_lcnet_x1_0", weights=PPLCNet_x1_0_Weights)
def pp_lcnet_x1_0(*, weights: PPLCNet_x1_0_Weights | str | None = None, **kwargs: object) -> PPLCNet:
    """PP-LCNet x1.0 classifier; see :class:`PPLCNet_x1_0_Weights`."""
    return PPLCNet(weights=weights, **kwargs)
