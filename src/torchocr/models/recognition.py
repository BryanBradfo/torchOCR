"""Text recognition model definitions."""

from typing import Literal

import torch
from torch import Tensor, nn

from .backbones import ResNetVd
from .hub import load_pretrained_state_dict


BackboneName = Literal["vgg", "resnet34_vd"]


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
            Required -- caller must commit to a charset. PaddleOCR's
            Chinese full charset is 6625 (6624 chars + blank).
        backbone: ``"vgg"`` (default) or ``"resnet34_vd"``.
        weights: Optional named preset of pretrained recognition
            weights. Pass ``"DEFAULT"`` to download torchocr's
            published checkpoint for the chosen backbone. If the
            download fails the model falls back to random init with a
            printed warning. Default ``None``. The downloaded
            checkpoint must match the ``num_classes`` you instantiate
            with.
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
        num_classes: int,
        backbone: BackboneName = "vgg",
        weights: str | None = None,
        input_channels: int = 3,
        rnn_hidden: int = 256,
    ) -> None:
        super().__init__()
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
            registry_key = "crnn_resnet34_vd" if backbone == "resnet34_vd" else "crnn"
            state_dict = load_pretrained_state_dict(registry_key, weights)
            if state_dict is not None:
                self.load_state_dict(state_dict)

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
