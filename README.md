# torchOCR

**A PyTorch-native, end-to-end OCR library — batchable, GPU-first, and free of clunky C++ runtimes.**

[![CI](https://github.com/BryanBradfo/torchOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/BryanBradfo/torchOCR/actions/workflows/ci.yml) · [![PyPI](https://img.shields.io/pypi/v/torchocr)](https://pypi.org/project/torchocr/) · [ python · 3.10+ ] · [ pytorch · 2.0+ ] · [docs](https://bryanbradfo.me/torchOCR/) · [ license · BSD-3-Clause ]

---

## Why torchOCR

Every modern deep learning stack lives in PyTorch — except OCR. The dominant tools (Tesseract, PaddleOCR, EasyOCR) wrap C++ binaries, ship custom runtimes, or stitch together TensorFlow and ONNX detours that fight against PyTorch's batching, autograd, and `torch.compile`. The result is an ecosystem where reading text from an image is the only thing you can't do natively.

**torchOCR** is a clean-slate, pure-PyTorch OCR pipeline modeled after `torchvision`. Detection, recognition, and document-level data structures are first-class `nn.Module`s and dataclasses. Tensors stay on-device, batches stay batches, and the public API is small enough to read in an afternoon.

## Philosophy

- **Pure PyTorch.** No Tesseract, no Leptonica, no ONNX intermediaries. Every layer is `nn.Module` you can subclass, fine-tune, or `state_dict`-load.
- **PyTorch 2.0 first.** Every model FX-traces, `torch.export`s with dynamic sizes and `torch.compile`s — checked in CI, shape contracts included (see *Graph capture* below).
- **Explicit shape contracts.** Outputs are typed dataclasses (`DBNetOutput`), not `Dict[str, Tensor]`. Every model documents its tensor shapes — and asserts them at runtime.
- **Modular, torchvision-style API.** `from torchocr.models import DBNet, CRNN`. Models compose, swap, and re-export the way you'd expect from `torchvision.models`.
- **Zero magic at import time.** No global state, no monkey-patching, no network calls unless you ask for them.

## Installation

```bash
pip install torchocr
```

From source:

```bash
git clone https://github.com/BryanBradfo/torchOCR.git
cd torchOCR
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, and torchvision 0.23+.

## Quickstart

### Image to text in three lines

```python
from torchocr import OCRPipeline, load_image

pipeline = OCRPipeline.from_pretrained(
    detector_weights="DBNet_ResNet18_VD_Weights.ICDAR2015",           # scene text, English
    recognizer_weights="CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19",  # Latin scene text
    device="cuda",                                                      # or "cpu" (~4 s/page)
    min_text_score=0.8,                                                 # drop unreadable regions
)
doc = pipeline(load_image("photo.jpg"))
doc.text        # list[str], one per region, in reading order
doc.polygons    # (K, 4, 2) rotated quads, reading order, in input pixels
doc.scores      # (K,) recognition confidence
```

Each stage brings its own preprocessing (`weights.transforms()`), post-processing
(`weights.meta["postprocess"]`), charset (`weights.meta["charset"]`) and decoder
(`weights.meta["decoder"]`: here CTC beam search with a character language model, downloaded on first
use); words are rectified out of the original-resolution image with `torchocr.ops.crop_quads`. The defaults
(`from_pretrained()` with no arguments) pair the multilingual PP-OCRv3 detector with the
Chinese + English recognizer, suited to documents.

### Building blocks

Every stage is an ordinary module or callable you can use, replace or fine-tune on its own:

```python
import torch
from torchocr import DBPostProcessor, load_charset
from torchocr.decoders import build_decoder
from torchocr.io import load_image
from torchocr.models import CRNN, DBNet, CRNN_ResNet34_VD_Weights, DBNet_ResNet18_VD_Weights
from torchocr.ops import crop_quads

image = load_image("photo.jpg")                                   # (3, H, W) uint8 RGB

det_weights = DBNet_ResNet18_VD_Weights.ICDAR2015
detector = DBNet(weights=det_weights).eval()
batch = det_weights.transforms()(image)[None]                     # (1, 3, H', W'), multiples of 32
with torch.no_grad():
    detection = detector(batch)                                   # DBNetOutput(probability, threshold)
quads = DBPostProcessor(**det_weights.meta["postprocess"]).quadrilaterals(detection)[0]  # (K, 4, 2)
quads = quads * torch.tensor([image.shape[2] / batch.shape[3], image.shape[1] / batch.shape[2]])

rec_weights = CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19
recognizer = CRNN(weights=rec_weights).eval()
page = rec_weights.transforms().normalize(image)[None].float()
crops, _ = crop_quads(page, quads, torch.zeros(len(quads), dtype=torch.long), height=32, max_width=320)
with torch.no_grad():
    logits = recognizer(crops)                                    # (T, K, num_classes), CTC-ready
charset = load_charset(rec_weights.meta["charset"], rec_weights.meta["num_classes"])
texts = build_decoder(rec_weights, charset)(logits)               # beam search + character LM
```

### Augmenting text polygons with `torchvision.transforms.v2`

Text regions travel through torchvision's own v2 transforms as `tv_tensors.KeyPoints`; torchocr adds
the two OCR-specific pieces, a crop that never cuts a word and a sanitizer that keeps `texts` and
`ignore` aligned with the polygons that survive.

```python
from torchvision.transforms import v2
from torchocr.datasets import TotalText
from torchocr.transforms import MakeDBTargets
from torchocr.transforms.v2 import RandomTextCrop, SanitizeTextPolygons, wrap_sample

augment = v2.Compose([
    v2.RandomRotation(10), v2.RandomAffine(0, scale=(0.5, 3.0)),
    RandomTextCrop(), v2.Resize(size=None, max_size=640),
    SanitizeTextPolygons(),            # off-canvas words dropped, clipped ones marked don't-care
])
image, target = augment(*wrap_sample(*TotalText("~/data/totaltext", split="train")[0]))
db_targets = MakeDBTargets()(target.polygons, target.ignore, image.shape[-2:])
```

### Graph capture: `torch.compile`, `torch.export`, FX

Every model is tested (`tests/test_compile_export.py`) to FX-trace, `torch.export` and `torch.compile`,
with its runtime shape contract intact:

```python
from torch.export import Dim, export

# Detector: one fused graph, ~1.2x faster on GPU (ResNet18-VD at 736x1280: 25.0 -> 20.3 ms).
detector = torch.compile(DBNet(weights="DBNet_ResNet18_VD_Weights.ICDAR2015").eval(), fullgraph=True)

# Ahead-of-time export with any image size that is a multiple of 32.
side = lambda name: 32 * Dim(name, min=1, max=128)
program = export(DBNet(backbone="resnet18_vd").eval(), (torch.randn(1, 3, 736, 1280),),
                 dynamic_shapes=({2: side("h"), 3: side("w")},))
```

`CRNN` exports with a dynamic batch but a static crop width, because `torch.export` specializes
`nn.LSTM`'s sequence length; inference already pads every crop to `max_width`. Under
`torch.compile` it graph-breaks at the LSTM (a Dynamo limitation), so `fullgraph=True` applies to
the detector only. Compiled outputs match eager to ~1e-3 (Inductor reorders floating-point
operations); published metrics are measured in eager mode.

## Architecture

| Component | Module | What it is |
| --- | --- | --- |
| `DBNet` | `torchocr.models` | Differentiable Binarization detector: ResNet-18-VD + FPN or MobileNetV3-large ×0.5 + RSEFPN backbone, probability and threshold heads. Returns a typed `DBNetOutput`. |
| `CRNN` | `torchocr.models` | Convolutional recurrent recognizer: ResNet-34-VD (or VGG-style) CNN, two bidirectional LSTMs, linear classifier. Returns CTC logits `(T, B, num_classes)`. |
| `DBPostProcessor` | `torchocr` | Probability map → rotated word quads in reading order. |
| `CTCGreedyDecoder`, `CTCBeamDecoder`, `CharNGramLM` | `torchocr.decoders` | CTC decoding, optionally fused with a character language model. |
| `OCRPipeline`, `DocumentTensor` | `torchocr` | Composition of the stages; a dataclass holding the input image, texts, quads, boxes and confidences. |

Both models enforce their input contracts at runtime (`DBNet`: `(B, 3, H, W)` with `H, W ≡ 0 (mod 32)`;
`CRNN`: `(B, 3, 32, W)`) and raise `ValueError` with the offending shape.

## Status

| Capability | State |
| --- | --- |
| `DBNet` / `CRNN` architectures + shape contracts | ✅ Implemented |
| Pretrained weights (`*_Weights` enums, hash-checked, on [Hugging Face](https://huggingface.co/BryanBradfo/torchocr-weights)) | ✅ 4 PaddleOCR conversions + 5 torchocr-trained checkpoints |
| DB post-processing (contours → rotated quads) | ✅ Implemented |
| `torchocr.ops.crop_quads` (batched perspective rectification) | ✅ Implemented |
| `torchocr.metrics` (ICDAR hmean, word accuracy / CER) and `torchocr.datasets.ICDAR2015` | ✅ Implemented |
| DB training (`references/detection/train.py`) | ✅ Implemented |
| CTC greedy and beam-search decoding (character n-gram LM) | ✅ Implemented |
| Datasets: ICDAR-2015, Total-Text, FUNSD, LMDB word crops | ✅ Implemented |
| End-to-end `OCRPipeline.from_pretrained` (image → text, reading order) | ✅ Implemented |
| Polygon-aware augmentation on `torchvision.transforms.v2` (`torchocr.transforms.v2`) | ✅ Implemented |
| Synthetic pre-training data (MJSynth via Parquet → LMDB) | ✅ Implemented |
| Sphinx documentation site with example gallery (`docs/`, `gallery/`) | ✅ Implemented |

Benchmarks (official IoU protocol, reproducible with `references/`; fine-tuned weights are the last epoch,
thresholds chosen on training splits only):

| Detection (test hmean) | ICDAR-2015 | Total-Text | FUNSD |
| --- | --- | --- | --- |
| `DBNet_ResNet18_VD_Weights.ICDAR2015` (trained by torchocr, 12.4 M params) | **0.809** | 0.696 | **0.826** |
| `DBNet_MobileNetV3_Large_05_Weights.ICDAR2015` (trained by torchocr, 0.6 M params) | 0.734 | | |
| `DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_EN` (converted, line-level) | 0.441 | | 0.180 |

| Recognition (word accuracy on ground-truth words) | ICDAR-2015 | Total-Text |
| --- | --- | --- |
| `CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19` + beam search & char LM | **0.799** | **0.819** |
| `CRNN_ResNet34_VD_Weights.ICDAR2015_TOTALTEXT_MLT19` + beam search & char LM | 0.791 | 0.802 |
| `CRNN_ResNet34_VD_Weights.ICDAR2015` (greedy) | 0.738 | |
| `CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2` (converted, greedy) | 0.665 | 0.701 |

End to end (`OCRPipeline`, image → text; a word counts when detected *and* read correctly,
case-insensitive, generic vocabulary; `references/end2end/evaluate.py`), with the two bold models and
`min_text_score=0.8`: **0.665** on ICDAR-2015 (0.219 with the converted PaddleOCR weights), **0.585** on
Total-Text and **0.722** on FUNSD scanned forms; ~60 ms per street image on a laptop GPU.

`weights.meta` documents each checkpoint's provenance, license, preprocessing and known limits.

## Roadmap

- **Distribution**: PyPI release (`pip install torchocr`) and a hosted documentation site.
- **Detection**: detectors pre-trained on SynthText and fine-tuned on curved text (Total-Text is the weakest benchmark).
- **Recognition**: an ASCII-only recognizer head and a multilingual variant; an attention/transformer recognizer.
- **Deployment**: `torch.ao` quantization and ONNX / `torch.export` packaging of both models.

## Documentation

Online at **https://bryanbradfo.me/torchOCR/** (API reference, pretrained weights, benchmarks, example gallery). To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html   # TORCHOCR_DOCS_GALLERY=0 skips running the examples
```

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

## Contributing

Issues and PRs welcome. Each model in `src/torchocr/models/` documents its shape contract in the docstring; please match the existing style (`from torch import Tensor, nn`, Google-style docstrings, `Tensor | None` unions). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full coding standards and PR process.

## License

BSD 3-Clause, like torchvision — see `LICENSE`.
