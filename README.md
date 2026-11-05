# torchOCR

**Text detection and recognition for PyTorch, built the way torchvision is: pretrained weights,
typed outputs, and every stage an `nn.Module` you can replace, fine-tune or export.**

[![CI](https://github.com/BryanBradfo/torchOCR/actions/workflows/ci.yml/badge.svg)](https://github.com/BryanBradfo/torchOCR/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torchocr)](https://pypi.org/project/torchocr/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://bryanbradfo.github.io/torchOCR/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

```python
from torchocr import OCRPipeline, load_image

doc = OCRPipeline.from_pretrained()(load_image("page.jpg"))
doc.text        # list[str], one per text region, in reading order
```

## Results

Every published checkpoint is scored on public test sets, with the script that reproduces the number
(`references/`). Scores below are for the scene-text detector and recognizer of the
[Quickstart](#quickstart). End to end, a word counts only when it is both found and read correctly:

| Test set | What it contains | End to end | Detection (hmean) | Recognition (word accuracy) |
| --- | --- | --- | --- | --- |
| [ICDAR-2015](https://rrc.cvc.uab.es/?ch=4) | street photos, small blurry words | 0.665 | 0.809 | 0.799 |
| [Total-Text](https://github.com/cs-chan/Total-Text-Dataset) | curved text | 0.585 | 0.696 | 0.819 |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | scanned forms | 0.722 | 0.826 | |

About 60 ms per street image on a laptop GPU. Per-checkpoint numbers, protocols and known limits are in
[Benchmarks](#benchmarks) and in each checkpoint's `weights.meta`.

## When to use torchOCR

torchOCR is for OCR inside a PyTorch system: batching pages on the GPU, fine-tuning one stage on your
own documents, swapping in your own detector, or exporting with `torch.export`. It is a library, not a
turnkey product. If you only need the most accurate text out of an image, use
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (especially for Chinese) or
[EasyOCR](https://github.com/JaidedAI/EasyOCR) (many languages): their models are larger and better
trained, and several of ours are converted from PaddleOCR.

- **Only PyTorch at inference.** No Tesseract, no Paddle runtime, no ONNX Runtime. PaddleOCR weights
  are converted once, at build time, by the scripts in `scripts/`.
- **Typed, checked outputs.** Models return dataclasses (`DBNetOutput`, `DocumentTensor`), document
  their tensor shapes and raise `ValueError` with the offending shape when the input is wrong.
- **Ready for PyTorch 2.** Every model FX-traces, `torch.export`s with dynamic sizes and `torch.compile`s,
  checked in CI.
- **torchvision conventions.** `*_Weights` enums with hash-checked downloads from
  [Hugging Face](https://huggingface.co/BryanBradfo/torchocr-weights), `weights.transforms()`, and
  augmentation on `torchvision.transforms.v2`.
- **Nothing happens at import.** No network calls, no global state, no monkey-patching.

Current scope: printed text in Latin script and Chinese, on photos and document pages. Not covered:
handwriting, layout analysis and table structure.

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

### Pages scanned sideways or upside down

```python
from torchocr import OCRPipeline, PageOrientation, load_image

orientation = PageOrientation.from_pretrained()
page, angle = orientation.upright(load_image("scan.jpg"))   # angle: 0, 90, 180 or 270 (clockwise)

pipeline = OCRPipeline.from_pretrained(correct_orientation=True)  # or let the pipeline do it
doc = pipeline(load_image("scan.jpg"))
doc.rotation    # how far the page was turned; doc.pixels and the regions are upright
```

A 1.7 M-parameter classifier converted from PaddleOCR (~3 ms per page on a laptop GPU). On the 4999
DocLayNet test pages, each fed in all four orientations, it is right 98.5% of the time; it only turns a
page when it is at least 70% sure, which leaves 0.24% of upright pages wrongly turned and sets 97.5% of
turned pages upright. Under the same protocol and as cautious with upright pages, docTR sets 31% of
turned pages upright and Tesseract OSD 45% ([comparison](https://bryanbradfo.github.io/torchOCR/benchmarks.html)).
It is meant for document pages, not street photos.

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
| `PPLCNet`, `PageOrientation` | `torchocr.models`, `torchocr` | 4-way page-orientation classifier (PP-LCNet x1.0) and the step that sets pages upright before detection. |
| `OCRPipeline`, `DocumentTensor` | `torchocr` | Composition of the stages; a dataclass holding the input image, texts, quads, boxes and confidences. |

Both models enforce their input contracts at runtime (`DBNet`: `(B, 3, H, W)` with `H, W ≡ 0 (mod 32)`;
`CRNN`: `(B, 3, 32, W)`) and raise `ValueError` with the offending shape.

## Benchmarks

Official IoU protocol, reproducible with `references/`. Fine-tuned weights are the last epoch, and
thresholds are chosen on training splits only.

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

- **Recognition**: a stronger Latin recognizer (PP-OCR SVTR conversion), accented characters, an attention/transformer recognizer.
- **Detection**: detectors pre-trained on SynthText and fine-tuned on curved text (Total-Text is the weakest benchmark).
- **Deployment**: `torch.ao` quantization and ONNX / `torch.export` packaging of both models.
- **1.0**: a frozen public API with a deprecation policy.

## Documentation

Online at **https://bryanbradfo.github.io/torchOCR/** (API reference, pretrained weights, benchmarks, example gallery). To build it locally:

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

The code is BSD 3-Clause, see [`LICENSE`](LICENSE). Pretrained weights carry their own license, given in
`weights.meta["license"]`: Apache-2.0 for the PaddleOCR conversions, research use only for the checkpoints
torchocr fine-tuned on academic datasets (ICDAR-2015, Total-Text, MLT-2019, MJSynth). Third-party work is acknowledged in [`CREDITS.md`](CREDITS.md).
