"""Text recognition model definitions."""

from functools import partial
from typing import Literal

from torch import Tensor, nn

from ..transforms import RecognitionPreset
from .backbones import ResNetVd
from .hub import BASE_URL, Weights, WeightsEnum, load_weights, register_model


BackboneName = Literal["vgg", "resnet34_vd"]


class CRNN_ResNet34_VD_Weights(WeightsEnum):
    PPOCR_SERVER_V2 = Weights(
        url=f"{BASE_URL}/crnn_resnet34_vd_ppocr_server_v2-803ba4c9.pth",
        # PaddleOCR rec: BGR pixels, (x - 127.5) / 127.5, height 32.
        transforms=partial(RecognitionPreset, height=32, max_width=320, channel_order="bgr"),
        meta={
            "task": "recognition",
            "backbone": "resnet34_vd",
            "num_params": 27_860_673,
            "num_classes": 6625,
            "charset": "ppocr_keys_v1",  # torchocr.charsets.load_charset
            "source": "PaddleOCR ch_ppocr_server_v2.0_rec_train, via scripts/convert_paddle_crnn.py",
            "license": "Apache-2.0",
            "languages": ["ch", "en"],
            "_metrics": {
                "ICDAR2015-test-gt-quads": {"word_accuracy": 0.6649, "char_error_rate": 0.1333},
            },
            "_docs": (
                "Chinese + English line recognizer (CTC, 6623 characters + blank) trained by "
                "PaddleOCR mostly on document and street-sign text lines. Benchmark: the 2077 "
                "alphanumeric care words of ICDAR-2015 test, cropped from the full images with "
                "torchocr.ops.crop_quads on their ground-truth quads, scored case-insensitively "
                "over letters and digits (not the official Task 4.3 crops). Axis-aligned crops of "
                "the same words score 0.5787 -- rotated scene text needs quad rectification. "
                "Reproduce with references/recognition/evaluate.py."
            ),
        },
    )
    DEFAULT = PPOCR_SERVER_V2


_WEIGHTS_BY_BACKBONE: dict[str, type[WeightsEnum]] = {"resnet34_vd": CRNN_ResNet34_VD_Weights}


class _Im2Seq(nn.Module):
    """Collapse a height-1 feature map into a sequence.

    Input ``(B, C, 1, T)`` -> output ``(B, T, C)``. PaddleOCR's
    ``ppocr.modeling.necks.SequenceEncoder.encoder_reshape`` does the
    same thing; we mirror its module name so weights port over without
    a remap.
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[2] != 1:
            raise ValueError(
                f"Im2Seq expects (B, C, 1, T); got shape {tuple(x.shape)}. "
                "The recognizer backbone must collapse height to 1 before this layer."
            )
        return x.squeeze(2).permute(0, 2, 1)


class _RNNEncoder(nn.Module):
    """2-layer bidirectional LSTM (batch_first), matching PaddleOCR's EncoderWithRNN."""

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return out


class _SequenceEncoder(nn.Module):
    """Im2Seq + RNN encoder, structured to match PaddleOCR exactly.

    State-dict layout: ``encoder_reshape.*`` (no params, just the
    reshape) and ``encoder.lstm.*`` (the LSTM). PyTorch's
    ``nn.LSTM`` parameter names (``weight_ih_l0``,
    ``weight_hh_l0_reverse``, ...) line up 1:1 with PaddleOCR's flat
    LSTM-export naming, so no transposes are needed.
    """

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.encoder_reshape = _Im2Seq()
        self.encoder = _RNNEncoder(in_channels, hidden)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_reshape(x)
        return self.encoder(x)


class _CTCHead(nn.Module):
    """Single Linear classifier, matching PaddleOCR's CTCHead (no mid_channels)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class CRNN(nn.Module):
    """CRNN recognizer.

    Two backbone variants share a single forward contract:

    - ``backbone="vgg"`` (default): the original Shi et al. 2015
      VGG-style CNN with one deviation -- the final ``conv7`` uses
      ``kernel_size=(2, 1)`` so the output width matches ``W // 4``
      exactly. This is what torchocr trains from scratch.
    - ``backbone="resnet34_vd"``: the PaddleOCR-compatible recognizer
      stack -- ResNet-34-VD backbone (with recognizer-style ``(2, 1)``
      strides so width is preserved), 2-layer BiLSTM, single Linear
      head. Parameter shapes and naming align with PaddleOCR weights
      so ``scripts/convert_paddle_crnn.py`` produces drop-in
      checkpoints.

    Args:
        num_classes: Number of output classes including the CTC blank.
            Required without ``weights`` -- the caller must commit to a
            charset. With ``weights`` it defaults to
            ``weights.meta["num_classes"]`` and must equal it if given.
        backbone: ``"vgg"`` or ``"resnet34_vd"``. Default ``None``:
            taken from ``weights`` when it is an enum member, else
            ``"vgg"``.
        weights: Pretrained weights -- a :class:`CRNN_ResNet34_VD_Weights`
            member or a member name such as ``"DEFAULT"``. If the download
            fails a ``UserWarning`` is emitted and the model keeps its
            random initialization. Default ``None``.
        input_channels: Channels in the input crops. Default 3.
        rnn_hidden: Hidden size of each BiLSTM direction. Default 256.

    Forward:
        Inputs are crops of shape ``(B, input_channels, 32, W)`` with
        ``W >= 16``. Output is ``(T, B, num_classes)`` logits where
        ``T = W // 4``. ``log_softmax`` is not applied -- pass logits
        directly to ``nn.CTCLoss(zero_infinity=True)`` after a
        ``log_softmax(dim=-1)`` at training time.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        backbone: BackboneName | None = None,
        weights: WeightsEnum | str | None = None,
        input_channels: int = 3,
        rnn_hidden: int = 256,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = weights.meta["backbone"] if isinstance(weights, WeightsEnum) else "vgg"
        if weights is not None:
            if backbone not in _WEIGHTS_BY_BACKBONE:
                raise ValueError(
                    f"No pretrained weights are published for backbone '{backbone}'. "
                    f"Backbones with weights: {sorted(_WEIGHTS_BY_BACKBONE)}."
                )
            weights = _WEIGHTS_BY_BACKBONE[backbone].verify(weights)
            expected = weights.meta["num_classes"]
            if num_classes is None:
                num_classes = expected
            elif num_classes != expected:
                raise ValueError(f"{weights!r} has num_classes={expected}; got num_classes={num_classes}.")
        if num_classes is None:
            raise ValueError("num_classes is required when no pretrained weights are given.")
        self.backbone_name = backbone

        if backbone == "vgg":
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(512, 512, kernel_size=(2, 1)),
                nn.ReLU(inplace=True),
            )
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=rnn_hidden,
                num_layers=2,
                bidirectional=True,
            )
            self.classifier = nn.Linear(rnn_hidden * 2, num_classes)
        elif backbone == "resnet34_vd":
            self.backbone = ResNetVd(
                depth=34,
                in_channels=input_channels,
                downsample_stride=(2, 1),
                stem_stride=1,    # rec keeps stem at full resolution
                final_pool=True,  # rec adds an extra MaxPool(2, 2) at the end
            )
            self.neck = _SequenceEncoder(in_channels=512, hidden=rnn_hidden)
            self.head = _CTCHead(in_channels=rnn_hidden * 2, num_classes=num_classes)
        else:
            raise ValueError(
                f"Unknown backbone '{backbone}'. Known: 'vgg', 'resnet34_vd'."
            )

        if weights is not None:
            load_weights(self, weights)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[2] != 32:
            raise ValueError(f"CRNN expects (B, C, 32, W); got {tuple(images.shape)}.")

        if self.backbone_name == "vgg":
            features = self.cnn(images)
            sequence = features.squeeze(2).permute(2, 0, 1)
            contextual, _ = self.rnn(sequence)
            return self.classifier(contextual)

        # resnet34_vd path
        pyramid = self.backbone(images)
        c5 = pyramid["c5"]  # (B, 512, 1, T)
        contextual = self.neck(c5)  # (B, T, 2*rnn_hidden)
        logits = self.head(contextual)  # (B, T, num_classes)
        return logits.permute(1, 0, 2)  # (T, B, num_classes) for CTC decoder


@register_model("crnn_vgg")
def crnn_vgg(*, num_classes: int, **kwargs: object) -> CRNN:
    """CRNN with the original VGG-style CNN (no published weights)."""
    return CRNN(num_classes=num_classes, backbone="vgg", **kwargs)


@register_model("crnn_resnet34_vd", weights=CRNN_ResNet34_VD_Weights)
def crnn_resnet34_vd(
    *, weights: CRNN_ResNet34_VD_Weights | str | None = None, num_classes: int | None = None, **kwargs: object
) -> CRNN:
    """PaddleOCR-compatible CRNN (ResNet-34-VD + BiLSTM + CTC); see :class:`CRNN_ResNet34_VD_Weights`."""
    return CRNN(num_classes=num_classes, backbone="resnet34_vd", weights=weights, **kwargs)
