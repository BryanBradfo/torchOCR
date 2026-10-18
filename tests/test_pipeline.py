"""End-to-end OCRPipeline composition test."""

import pytest
import torch

from torchocr import (
    CTCGreedyDecoder,
    DBPostProcessor,
    DocumentTensor,
    OCRPipeline,
)
from torchocr.models import CRNN, DBNet


def _build_pipeline(charset: list[str]) -> OCRPipeline:
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(charset)).train(False)
    return OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(threshold=0.3),
        CTCGreedyDecoder(charset),
    )


def test_pipeline_returns_document_tensor(ascii_charset):
    pipeline = _build_pipeline(ascii_charset)
    doc = pipeline(torch.randn(3, 64, 64))
    assert isinstance(doc, DocumentTensor)
    assert doc.pixels.shape == (3, 64, 64)


def test_pipeline_box_count_matches_text(ascii_charset):
    pipeline = _build_pipeline(ascii_charset)
    doc = pipeline(torch.randn(3, 64, 64))
    box_count = 0 if doc.bounding_boxes is None else doc.bounding_boxes.shape[0]
    assert len(doc.text) == box_count


@pytest.mark.parametrize("shape", [(3, 64), (64, 64), (1, 3, 64, 64)])
def test_pipeline_rejects_bad_input_ndim(ascii_charset, shape):
    pipeline = _build_pipeline(ascii_charset)
    with pytest.raises(ValueError):
        pipeline(torch.randn(*shape))


def test_pipeline_rejects_bad_crop_size(ascii_charset):
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=len(ascii_charset)).train(False)
    with pytest.raises(ValueError):
        OCRPipeline(
            detector,
            recognizer,
            DBPostProcessor(),
            CTCGreedyDecoder(ascii_charset),
            crop_size=(64, 128),  # height must be 32
        )


# === Pretrained, preset-driven path ===

import os  # noqa: E402

from torch import nn  # noqa: E402

from torchocr.models.detection import DBNetOutput  # noqa: E402
from torchocr.transforms import DetectionPreset, RecognitionPreset  # noqa: E402


class _BlobDetector(nn.Module):
    """Emits one bright rectangle at fixed detector-space coordinates."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_shape = None

    def forward(self, images):
        self.seen_shape = tuple(images.shape)
        prob = torch.zeros(images.shape[0], 1, *images.shape[-2:])
        prob[:, :, 16:48, 24:72] = 0.95
        return DBNetOutput(prob, torch.zeros_like(prob))


class _ConstantRecognizer(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seen_shape = None

    def forward(self, crops):
        self.seen_shape = tuple(crops.shape)
        logits = torch.zeros(crops.shape[-1] // 4, crops.shape[0], self.num_classes)
        logits[..., 34] = 1.0  # "A" in the ASCII fixture charset
        return logits


def test_presets_resize_for_detection_and_map_quads_back(ascii_charset):
    detector, recognizer = _BlobDetector(), _ConstantRecognizer(len(ascii_charset))
    pipeline = OCRPipeline(
        detector,
        recognizer,
        DBPostProcessor(),
        CTCGreedyDecoder(ascii_charset),
        detector_transforms=DetectionPreset(max_side=100),
        recognizer_transforms=RecognitionPreset(height=32, max_width=96),
    )
    image = torch.randint(0, 256, (3, 100, 200), dtype=torch.uint8)
    doc = pipeline(image)

    assert detector.seen_shape == (1, 3, 64, 96)  # resized by the preset
    assert recognizer.seen_shape == (1, 3, 32, 96)  # aspect-preserving crops, padded
    assert doc.pixels is image
    assert doc.text == ["A"]
    assert doc.polygons.shape == (1, 4, 2) and doc.bounding_boxes.shape == (1, 4)
    # Blob spans x 24..72 / y 16..48 of a 96x64 map -> x 50..150 / y 25..75 in the image,
    # centred on (100, 50); unclip then grows it on every side. cv2's pixel-index
    # convention and pyclipper's integer rounding shift it by ~1.5 detector pixels.
    x1, y1, x2, y2 = doc.bounding_boxes[0].tolist()
    assert (x1 + x2) / 2 == pytest.approx(100, abs=4) and (y1 + y2) / 2 == pytest.approx(50, abs=4)
    assert x1 < 50 and x2 > 150 and y1 < 25 and y2 > 75


def test_from_pretrained_builds_everything_from_weights(monkeypatch):
    from torchocr.models import CRNN, DBNet, hub
    from torchocr.models import CRNN_ResNet34_VD_Weights, DBNet_MobileNetV3_Large_05_Weights

    def random_state(url, **kwargs):
        if "crnn" in url:
            return CRNN(weights=None, num_classes=6625, backbone="resnet34_vd").state_dict()
        return DBNet(backbone="mobilenet_v3_large_05").state_dict()

    monkeypatch.setattr(hub, "load_state_dict_from_url", random_state)
    pipeline = OCRPipeline.from_pretrained(
        detector_weights=DBNet_MobileNetV3_Large_05_Weights.ICDAR2015,
        recognizer_weights="CRNN_ResNet34_VD_Weights.DEFAULT",
    )
    assert pipeline.post_processor.box_thresh == 0.45  # from meta["postprocess"]
    assert pipeline.decoder.num_classes == CRNN_ResNet34_VD_Weights.DEFAULT.meta["num_classes"]
    assert not pipeline.detector.training and not pipeline.recognizer.training
    doc = pipeline(torch.randint(0, 256, (3, 120, 200), dtype=torch.uint8))
    assert len(doc.text) == (0 if doc.polygons is None else doc.polygons.shape[0])


def _cached(weights) -> bool:
    return os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", weights.url.rsplit("/", 1)[-1]))


def test_pretrained_pipeline_reads_a_real_receipt():
    """Integration: real weights, real image. Skipped unless the checkpoints are cached."""
    from torchocr import load_image
    from torchocr.models import CRNN_ResNet34_VD_Weights, DBNet_MobileNetV3_Large_05_Weights

    if not (_cached(DBNet_MobileNetV3_Large_05_Weights.DEFAULT) and _cached(CRNN_ResNet34_VD_Weights.DEFAULT)):
        pytest.skip("pretrained checkpoints not in the torch.hub cache")
    pipeline = OCRPipeline.from_pretrained()
    doc = pipeline(load_image("examples/chinese_receipt.jpg")[:3])
    assert "乳酸脱氢酶" in doc.text  # "lactate dehydrogenase" on the lab report
