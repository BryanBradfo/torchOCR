"""torchvision-style pretrained-weights API: WeightsEnum, builders, registry.

All tests are offline: the download function is monkeypatched.
"""

import urllib.error

import pytest
import torch
from torch import nn

from torchocr.models import (
    CRNN,
    CRNN_ResNet34_VD_Weights,
    DBNet,
    DBNet_MobileNetV3_Large_05_Weights,
    DBNet_ResNet18_VD_Weights,
    WeightsEnum,
    get_model,
    get_model_weights,
    get_weight,
    list_models,
)
from torchocr.models import hub


ALL_WEIGHTS = [
    w
    for enum in (DBNet_ResNet18_VD_Weights, DBNet_MobileNetV3_Large_05_Weights, CRNN_ResNet34_VD_Weights)
    for w in enum
]


def _random_state_dict(weights: WeightsEnum) -> dict[str, torch.Tensor]:
    """A state_dict with the right keys/shapes, built without downloading."""
    if weights.meta["task"] == "detection":
        return DBNet(backbone=weights.meta["backbone"]).state_dict()
    return CRNN(num_classes=weights.meta["num_classes"], backbone=weights.meta["backbone"]).state_dict()


@pytest.fixture
def fake_download(monkeypatch):
    """Serve every weights URL from a randomly-initialised model."""
    by_url = {w.url: w for w in ALL_WEIGHTS}
    calls = []

    def load(url, **kwargs):
        calls.append((url, kwargs))
        return _random_state_dict(by_url[url])

    monkeypatch.setattr(hub, "load_state_dict_from_url", load)
    return calls


# === Enum contract ===


@pytest.mark.parametrize("weights", ALL_WEIGHTS, ids=repr)
def test_meta_contract(weights):
    meta = weights.meta
    for key in ("task", "backbone", "num_params", "source", "license", "_docs", "_metrics"):
        assert key in meta, f"{weights!r} meta lacks {key!r}"
    assert weights.url.startswith("https://")
    # torch.hub verifies the sha256 prefix embedded in the file name.
    assert hub.HASH_REGEX.search(weights.url.rsplit("/", 1)[-1]) is not None
    assert isinstance(weights.transforms(), nn.Module)


@pytest.mark.parametrize("weights", ALL_WEIGHTS, ids=repr)
def test_num_params_matches_architecture(weights):
    model = (
        DBNet(backbone=weights.meta["backbone"])
        if weights.meta["task"] == "detection"
        else CRNN(num_classes=weights.meta["num_classes"], backbone=weights.meta["backbone"])
    )
    assert sum(p.numel() for p in model.parameters()) == weights.meta["num_params"]


def test_default_is_an_alias():
    assert DBNet_ResNet18_VD_Weights.DEFAULT is DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2
    assert repr(DBNet_ResNet18_VD_Weights.DEFAULT) == "DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2"


@pytest.mark.parametrize(
    "value",
    ["DEFAULT", "PPOCR_SERVER_V2", "DBNet_ResNet18_VD_Weights.DEFAULT", DBNet_ResNet18_VD_Weights.DEFAULT],
)
def test_verify_accepts_names_and_members(value):
    assert DBNet_ResNet18_VD_Weights.verify(value) is DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2


def test_verify_rejects_unknown_name():
    with pytest.raises(ValueError, match="PPOCR_SERVER_V2"):
        DBNet_ResNet18_VD_Weights.verify("IMAGENET1K_V1")


def test_verify_rejects_weights_of_another_model():
    with pytest.raises(ValueError, match="DBNet_ResNet18_VD_Weights"):
        DBNet_ResNet18_VD_Weights.verify(DBNet_MobileNetV3_Large_05_Weights.DEFAULT)


# === Registry ===


def test_list_models():
    assert list_models() == [
        "crnn_resnet34_vd",
        "crnn_vgg",
        "dbnet_mobilenet_v3_large_05",
        "dbnet_resnet18",
        "dbnet_resnet18_vd",
    ]


def test_get_model_weights():
    assert get_model_weights("dbnet_resnet18_vd") is DBNet_ResNet18_VD_Weights
    assert get_model_weights("dbnet_resnet18") is None
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_weights("resnet50")


def test_get_weight():
    assert get_weight("DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN") is DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN
    with pytest.raises(ValueError, match="Cls.NAME"):
        get_weight("DEFAULT")
    with pytest.raises(ValueError, match="Unknown weights enum"):
        get_weight("Nope_Weights.DEFAULT")


def test_get_model_builds_without_weights():
    model = get_model("dbnet_mobilenet_v3_large_05")
    assert isinstance(model, DBNet) and model.backbone_name == "mobilenet_v3_large_05"


# === Loading ===


def test_builder_loads_state_dict_with_hash_check(fake_download):
    model = get_model("dbnet_resnet18_vd", weights="DEFAULT")
    assert isinstance(model, DBNet)
    (url, kwargs), = fake_download
    assert url == DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2.url
    assert kwargs["check_hash"] is True and kwargs["weights_only"] is True


def test_dbnet_infers_backbone_from_weights(fake_download):
    model = DBNet(weights=DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN)
    assert model.backbone_name == "mobilenet_v3_large_05"


def test_dbnet_rejects_weights_for_a_different_backbone():
    with pytest.raises(ValueError, match="DBNet_ResNet18_VD_Weights"):
        DBNet(backbone="resnet18_vd", weights=DBNet_MobileNetV3_Large_05_Weights.DEFAULT)


def test_backbone_without_published_weights_raises():
    with pytest.raises(ValueError, match="resnet18_vd"):
        DBNet(backbone="resnet18", weights="DEFAULT")


def test_crnn_infers_num_classes_and_backbone(fake_download):
    model = CRNN(weights=CRNN_ResNet34_VD_Weights.DEFAULT)
    assert model.backbone_name == "resnet34_vd"
    assert model.head.fc.out_features == 6625


def test_crnn_num_classes_must_match_weights():
    with pytest.raises(ValueError, match="6625"):
        CRNN(num_classes=97, weights=CRNN_ResNet34_VD_Weights.DEFAULT)


def test_crnn_requires_num_classes_without_weights():
    with pytest.raises(ValueError, match="num_classes"):
        CRNN()


def test_download_failure_warns_and_keeps_random_init(monkeypatch):
    def offline(url, **kwargs):
        raise urllib.error.URLError("network unreachable")

    monkeypatch.setattr(hub, "load_state_dict_from_url", offline)
    with pytest.warns(UserWarning, match="random initialization"):
        model = DBNet(weights=DBNet_ResNet18_VD_Weights.DEFAULT)
    assert model.backbone_name == "resnet18_vd"


def test_corrupt_checkpoint_is_not_silenced(monkeypatch):
    """Only network failures fall back; a bad checkpoint must raise."""

    def corrupt(url, **kwargs):
        raise RuntimeError("invalid hash value")

    monkeypatch.setattr(hub, "load_state_dict_from_url", corrupt)
    with pytest.raises(RuntimeError, match="invalid hash"):
        DBNet(weights=DBNet_ResNet18_VD_Weights.DEFAULT)


@pytest.mark.parametrize("weights", [w for w in ALL_WEIGHTS if w.meta["task"] == "detection"], ids=repr)
def test_detection_weights_record_their_postprocessing(weights):
    """meta["postprocess"] holds the DBPostProcessor settings _metrics was measured with."""
    from torchocr import DBPostProcessor

    processor = DBPostProcessor(**weights.meta["postprocess"])
    assert 0 < processor.box_thresh < 1
