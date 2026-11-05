"""Page orientation: preset, rotation helper, PageOrientation and the PP-LCNet converter."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from torchocr import PageOrientation, rotate_page
from torchocr.models import PPLCNet
from torchocr.transforms import ClassificationPreset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_paddle_pplcnet import load_paddle_state, remap  # noqa: E402


def _page(height: int = 40, width: int = 30) -> torch.Tensor:
    return torch.randint(0, 256, (3, height, width), dtype=torch.uint8)


class _FixedClassifier(nn.Module):
    """Returns the same logits for every page, whatever its size."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits), requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(images.shape[0], -1)


# === ClassificationPreset ===


@pytest.mark.parametrize("size", [(300, 200), (200, 300), (224, 224)])
def test_classification_preset_outputs_a_square_crop(size):
    out = ClassificationPreset()(_page(*size))
    assert out.shape == (3, 224, 224) and out.dtype == torch.float32


def test_classification_preset_rejects_float_input():
    with pytest.raises(ValueError, match="uint8"):
        ClassificationPreset()(torch.rand(3, 64, 64))


# === rotate_page ===


def test_rotate_page_undoes_a_clockwise_turn():
    page = _page()
    turned_clockwise = torch.rot90(page, -1, dims=(-2, -1))
    assert torch.equal(rotate_page(turned_clockwise, 90), page)
    assert torch.equal(rotate_page(torch.rot90(page, 2, dims=(-2, -1)), 180), page)
    assert torch.equal(rotate_page(page, 0), page)


def test_rotate_page_rejects_other_angles():
    with pytest.raises(ValueError, match="multiple of 90"):
        rotate_page(_page(), 45)


# === PageOrientation ===


def test_page_orientation_accepts_pages_of_different_sizes():
    orientation = PageOrientation(_FixedClassifier([0.0, 0.0, 0.0, 9.0]), ClassificationPreset())
    probabilities = orientation.probabilities([_page(40, 30), _page(30, 50)])
    assert probabilities.shape == (2, 4)
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))


def test_page_orientation_rejects_an_invalid_threshold():
    with pytest.raises(ValueError, match="min_confidence"):
        PageOrientation(_FixedClassifier([1.0, 0.0, 0.0, 0.0]), ClassificationPreset(), min_confidence=1.5)


def test_pplcnet_output_shape_and_input_contract():
    model = PPLCNet().eval()
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 4)
    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        model(torch.randn(3, 224, 224))


# === Converter ===


def _pdparams(tmp_path, state: dict) -> Path:
    path = tmp_path / "model.pdparams"
    path.write_bytes(pickle.dumps(state, protocol=4))
    return path


def test_converter_renames_bn_statistics_and_transposes_the_classifier(tmp_path):
    fc = np.arange(1280 * 4, dtype=np.float32).reshape(1280, 4)  # Paddle Linear: (in, out)
    state = {
        "conv1.bn._mean": np.zeros(16, np.float32),
        "conv1.bn._variance": np.ones(16, np.float32),
        "fc.weight": fc,
        "StructuredToParameterName@@": {"fc.weight": "linear_0.w_0"},
    }
    converted = remap(load_paddle_state(_pdparams(tmp_path, state)))
    assert set(converted) == {"conv1.bn.running_mean", "conv1.bn.running_var", "fc.weight"}
    assert torch.equal(converted["fc.weight"], torch.from_numpy(fc.T.copy()))


def test_converter_refuses_pickles_that_reference_other_globals(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ("this must never run",))

    with pytest.raises(pickle.UnpicklingError, match="print"):
        load_paddle_state(_pdparams(tmp_path, {"x": Payload()}))


def test_page_is_turned_only_above_the_threshold():
    turned_270 = _FixedClassifier([0.0, 0.0, 0.0, 2.0])  # p(270) = 0.71
    page = _page()
    assert PageOrientation(turned_270, ClassificationPreset(), min_confidence=0.7)(page).tolist() == [270]
    assert PageOrientation(turned_270, ClassificationPreset(), min_confidence=0.8)(page).tolist() == [0]


def test_upright_returns_the_page_set_upright():
    orientation = PageOrientation(_FixedClassifier([0.0, 9.0, 0.0, 0.0]), ClassificationPreset())
    page = _page(40, 30)
    upright, angle = orientation.upright(torch.rot90(page, -1, dims=(-2, -1)))  # turned 90 clockwise
    assert angle == 90 and torch.equal(upright, page)


def test_pretrained_classifier_sets_a_real_page_upright():
    """Integration: real weights, real page. Skipped unless the checkpoint is cached."""
    import os

    from torchocr import load_image
    from torchocr.models import PPLCNet_x1_0_Weights

    name = PPLCNet_x1_0_Weights.DEFAULT.url.rsplit("/", 1)[-1]
    if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", name)):
        pytest.skip("pretrained checkpoint not in the torch.hub cache")
    orientation = PageOrientation.from_pretrained()
    page = load_image(REPO_ROOT / "examples" / "chinese_typeset.jpg")  # a printed page, not a scene photo
    for quarter_turns_clockwise in range(4):
        turned = torch.rot90(page, -quarter_turns_clockwise, dims=(-2, -1))
        upright, angle = orientation.upright(turned)
        assert angle == 90 * quarter_turns_clockwise and torch.equal(upright, page)
