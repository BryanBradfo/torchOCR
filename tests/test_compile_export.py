"""Graph capture: every model must FX-trace, torch.export (with dynamic sizes) and torch.compile.

These back the CLAUDE.md promise "Models are FX-traceable and torch.compile-friendly".
Compilation uses the ``aot_eager`` backend: it exercises Dynamo/AOTAutograd graph
capture without needing a C++ toolchain for Inductor code generation.
"""

import pytest
import torch
from torch.export import Dim, export
from torch.fx import symbolic_trace

from torchocr.models import CRNN, DBNet
from torchocr.models.detection import DBNetOutput

DETECTORS = ["resnet18", "resnet18_vd", "mobilenet_v3_large_05"]
RECOGNIZERS = ["vgg", "resnet34_vd"]


def _flat(output) -> torch.Tensor:
    if isinstance(output, DBNetOutput):
        return torch.cat([output.probability.flatten(), output.threshold.flatten()])
    return output.flatten()


def _detector(backbone: str) -> DBNet:
    torch.manual_seed(0)
    return DBNet(backbone=backbone).eval()


def _recognizer(backbone: str) -> CRNN:
    torch.manual_seed(0)
    return CRNN(num_classes=97, backbone=backbone).eval()


# === FX ===


@pytest.mark.parametrize("backbone", DETECTORS)
def test_dbnet_fx_trace(backbone):
    model, images = _detector(backbone), torch.randn(1, 3, 64, 96)
    traced = symbolic_trace(model)
    assert torch.allclose(_flat(traced(images)), _flat(model(images)), atol=1e-5)


@pytest.mark.parametrize("backbone", RECOGNIZERS)
def test_crnn_fx_trace(backbone):
    model, crops = _recognizer(backbone), torch.randn(2, 3, 32, 64)
    traced = symbolic_trace(model)
    assert torch.allclose(traced(crops), model(crops), atol=1e-5)


def test_fx_traced_model_keeps_its_shape_contract():
    traced = symbolic_trace(_detector("resnet18_vd"))
    with pytest.raises(ValueError, match="divisible by 32"):
        traced(torch.randn(1, 3, 64, 90))


# === torch.export ===


@pytest.mark.parametrize("backbone", DETECTORS)
def test_dbnet_export_with_dynamic_size(backbone):
    model = _detector(backbone)
    height, width = 32 * Dim("h32", min=1, max=64), 32 * Dim("w32", min=1, max=64)
    program = export(model, (torch.randn(1, 3, 64, 96),), dynamic_shapes=({2: height, 3: width},))
    images = torch.randn(1, 3, 96, 128)  # a size not seen at export time
    out = program.module()(images)
    assert isinstance(out, DBNetOutput)
    assert torch.allclose(_flat(out), _flat(model(images)), atol=1e-5)


@pytest.mark.parametrize("backbone", RECOGNIZERS)
def test_crnn_export_with_dynamic_batch(backbone):
    """The number of words varies per page; crop width is fixed (crop_quads pads to max_width).

    torch.export specializes nn.LSTM's sequence length, so width cannot be dynamic.
    """
    model = _recognizer(backbone)
    program = export(model, (torch.randn(2, 3, 32, 320),), dynamic_shapes=({0: Dim("batch", min=1, max=512)},))
    crops = torch.randn(7, 3, 32, 320)
    assert torch.allclose(program.module()(crops), model(crops), atol=1e-5)


# === torch.compile ===


@pytest.mark.parametrize("backbone", DETECTORS)
def test_dbnet_compiles_as_a_single_graph(backbone):
    torch._dynamo.reset()
    model, images = _detector(backbone), torch.randn(1, 3, 64, 96)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    assert torch.allclose(_flat(compiled(images)), _flat(model(images)), atol=1e-5)


@pytest.mark.parametrize("backbone", RECOGNIZERS)
def test_crnn_compiles(backbone):
    torch._dynamo.reset()
    model, crops = _recognizer(backbone), torch.randn(2, 3, 32, 64)
    compiled = torch.compile(model, backend="aot_eager")
    assert torch.allclose(compiled(crops), model(crops), atol=1e-5)
