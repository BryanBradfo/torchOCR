# torchOCR

**A PyTorch-native, end-to-end OCR library — batchable, GPU-first, and free of clunky C++ runtimes.**

[ build · passing ] · [ pypi · pre-release ] · [ python · 3.10+ ] · [ pytorch · 2.0+ ] · [ license · MIT ]

---

## Why torchOCR

Every modern deep learning stack lives in PyTorch — except OCR. The dominant tools (Tesseract, PaddleOCR, EasyOCR) wrap C++ binaries, ship custom runtimes, or stitch together TensorFlow and ONNX detours that fight against PyTorch's batching, autograd, and `torch.compile`. The result is an ecosystem where reading text from an image is the only thing you can't do natively.

**torchOCR** is a clean-slate, pure-PyTorch OCR pipeline modeled after `torchvision`. Detection, recognition, and document-level data structures are first-class `nn.Module`s and dataclasses. Tensors stay on-device, batches stay batches, and the public API is small enough to read in an afternoon.

## Philosophy

- **Pure PyTorch.** No Tesseract, no Leptonica, no ONNX intermediaries. Every layer is `nn.Module` you can subclass, fine-tune, or `state_dict`-load.
- **PyTorch 2.0 first.** Models are `torch.compile`-friendly out of the box. No data-dependent control flow in the hot path; FX-traceable backbones via `torchvision.models.feature_extraction`.
- **Explicit shape contracts.** Outputs are typed dataclasses (`DBNetOutput`), not `Dict[str, Tensor]`. Every model documents its tensor shapes — and asserts them at runtime.
- **Modular, torchvision-style API.** `from torchocr.models import DBNet, CRNN`. Models compose, swap, and re-export the way you'd expect from `torchvision.models`.
- **Zero magic at import time.** No global state, no monkey-patching, no network calls unless you ask for them.

## Installation

```bash
pip install torchocr      # once published — currently install from source
```

```bash
git clone https://github.com/BryanBradfo/torchOCR.git
cd torchOCR
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, and torchvision.

## Quickstart

```python
import torch
from torchocr.models import DBNet, CRNN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Detection: ResNet-18 + FPN -> probability and threshold maps.
# .train(False) is the explicit form of the standard PyTorch eval-mode idiom.
detector = DBNet(pretrained_backbone=True).train(False).to(device)

# Recognition: 7-block CNN + 2-layer BiLSTM -> CTC-ready logits.
recognizer = CRNN(num_classes=96).train(False).to(device)

# Standard NCHW batch. H and W must be multiples of 32.
images = torch.randn(4, 3, 512, 512, device=device)

with torch.no_grad():
    detection = detector(images)

# Typed output -- no string keys, no positional unpacking.
print(detection.probability.shape)   # torch.Size([4, 1, 512, 512])
print(detection.threshold.shape)     # torch.Size([4, 1, 512, 512])

# Recognizer expects 32-tall crops; sequence length T = W // 4.
crops = torch.randn(4, 3, 32, 128, device=device)
logits = recognizer(crops)           # torch.Size([32, 4, 96])

# Logits flow directly into nn.CTCLoss after a log_softmax.
log_probs = logits.log_softmax(dim=-1)
```

### `torch.compile` opt-in

```python
detector = torch.compile(DBNet().train(False))
recognizer = torch.compile(CRNN(num_classes=96).train(False))
```

Both modules are FX-friendly; compilation is a single line and the shape contracts above are unchanged.

### Document-level container

```python
from torchocr import DocumentTensor

doc = DocumentTensor(
    pixels=images[0],            # (3, H, W)
    text=[],                     # filled by the pipeline
    bounding_boxes=None,         # filled by the detector
)
doc = doc.to(device)             # value-semantics, returns a new instance
```

## Architecture

| Component | Module | What it is |
| --- | --- | --- |
| `DBNet` | `torchocr.models.detection` | Differentiable Binarization detector with a ResNet-18 backbone, top-down FPN neck (256 channels per level), and twin probability/threshold heads at full input resolution. Outputs a typed `DBNetOutput` dataclass. |
| `CRNN` | `torchocr.models.recognition` | Convolutional Recurrent Network: VGG-style 7-block CNN, two stacked bidirectional LSTMs (hidden=256), linear classifier. Returns CTC-ready logits of shape `(T, B, num_classes)` with closed-form `T = W // 4`. |
| `DocumentTensor` | `torchocr.core.structures` | A dataclass bundling pixels, recognized text, and bounding boxes, with a `.to(device)` that mirrors `nn.Module` semantics. |

Both models enforce their input contracts at runtime:

- `DBNet` requires `(B, 3, H, W)` with `H, W ≡ 0 (mod 32)`.
- `CRNN` requires `(B, C, 32, W)` — exactly 32 pixels tall.

Mismatched inputs raise `ValueError` with the offending shape — a debugging aid worth its weight.

## Status

| Capability | State |
| --- | --- |
| `DBNet` architecture + shape contract | ✅ Implemented |
| `CRNN` architecture + shape contract | ✅ Implemented |
| `DocumentTensor` data structure | ✅ Implemented |
| Smoke-test inference (`scripts/test_inference.py`) | ✅ Implemented |
| Pretrained OCR weights | 🚧 In progress |
| CTC greedy + beam decoding | 🚧 In progress |
| DB post-processing (polygon extraction) | 🚧 In progress |
| Training loops & loss helpers | 📋 Planned |

The current release establishes the public API and shape contracts. Heads are randomly initialized; only the ResNet-18 backbone has ImageNet-pretrained weights when `pretrained_backbone=True`. Treat the API as stable and the weights as forthcoming.

## Roadmap

- **Pretrained weights**: Detection weights trained on ICDAR-2015 + Total-Text + SynthText; recognition weights covering ASCII printable + a multilingual variant.
- **CTC decoding**: Greedy and prefix-beam decoders with optional KenLM/n-gram fusion, distributed as a `torchocr.decoders` submodule.
- **Polygon extraction**: Batched, vectorized DB post-processing — Vatti clipping in pure PyTorch, no OpenCV runtime dependency.
- **Pipeline orchestration**: A `torchocr.pipelines.OCRPipeline` that chains detector, polygon extractor, crop normalizer, and recognizer behind a single `__call__` — Hugging Face-style ergonomics.
- **Training utilities**: `DBLoss` (BCE + L1 + dice), CTC training helpers, ICDAR/Total-Text dataset wrappers.
- **Quantization & export**: `torch.ao.quantization` integration and clean ONNX export for both models.

## Verifying the install

```bash
python scripts/test_inference.py
# probability: (1, 1, 512, 512)
# threshold:   (1, 1, 512, 512)
# logits:      (32, 1, 96)
# OK
```

Exit code 0 means both architectures are wired correctly on your machine.

## Contributing

Issues and PRs welcome. Each model in `src/torchocr/models/` documents its shape contract in the docstring; please match the existing style (`from torch import Tensor, nn`, Google-style docstrings, `Tensor | None` unions). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full coding standards and PR process.

## License

MIT — see `LICENSE`.
