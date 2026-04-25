"""Text recognition model definitions."""

from torch import Tensor, nn

from .hub import load_pretrained_state_dict


class CRNN(nn.Module):
    """CRNN recognizer with a VGG-style CNN, BiLSTM, and CTC-ready head.

    The architecture follows Shi et al. 2015 with one deviation: the
    final ``conv7`` uses ``kernel_size=(2, 1)`` instead of ``(2, 2)``
    so the output width matches ``W // 4`` exactly, giving callers a
    closed-form sequence length. The earlier conv layers already mix
    horizontal context, so this is a benign simplification.

    Args:
        num_classes: Number of output classes including the CTC blank.
            Required — caller must commit to a charset.
        weights: Optional named preset of pretrained recognition
            weights. Pass ``"DEFAULT"`` to download torchocr's
            published CRNN checkpoint from the model hub. If the
            download fails the model falls back to random init with
            a printed warning -- caller's code keeps running. Default
            ``None``. The downloaded checkpoint must match the
            ``num_classes`` you instantiate with; otherwise the
            ``load_state_dict`` call raises a size-mismatch error.
        input_channels: Number of channels in the input crops. Default 3.
        rnn_hidden: Hidden size of each BiLSTM direction. Default 256.

    Forward:
        Inputs are crops of shape ``(B, input_channels, 32, W)`` with
        ``W >= 16``. Output is ``(T, B, num_classes)`` logits where
        ``T = W // 4``. ``log_softmax`` is not applied — pass the logits
        directly to ``nn.CTCLoss(zero_infinity=True)`` after a
        ``log_softmax(dim=-1)`` at training time.
    """

    def __init__(
        self,
        num_classes: int,
        weights: str | None = None,
        input_channels: int = 3,
        rnn_hidden: int = 256,
    ) -> None:
        super().__init__()
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

        if weights is not None:
            state_dict = load_pretrained_state_dict("crnn", weights)
            if state_dict is not None:
                self.load_state_dict(state_dict)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[2] != 32:
            raise ValueError(f"CRNN expects (B, C, 32, W); got {tuple(images.shape)}.")
        features = self.cnn(images)
        sequence = features.squeeze(2).permute(2, 0, 1)
        contextual, _ = self.rnn(sequence)
        return self.classifier(contextual)
