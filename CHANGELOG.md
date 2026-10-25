# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `torchocr.transforms.v2`: text polygons through `torchvision.transforms.v2` as `tv_tensors.KeyPoints`.
  `TextDetectionTarget` is now a pytree node, so `texts` and `ignore` stay aligned through any v2
  transform; `wrap_sample` wraps dataset samples, `RandomTextCrop` crops without cutting text
  (PaddleOCR's `EastRandomCropData`), `SanitizeTextPolygons` drops off-canvas regions and marks
  clipped ones don't-care.

### Changed
- Requires `torchvision>=0.23` (`tv_tensors.KeyPoints`).

## [0.3.0] - 2026-10-19

More data, a language model, and a document benchmark. Every number is reproducible with `references/`
and published with its checkpoint at https://huggingface.co/BryanBradfo/torchocr-weights.

| Test set (end-to-end hmean, image → text) | v0.2.0 | v0.3.0 |
| --- | --- | --- |
| ICDAR-2015 (street scenes) | 0.578 | **0.649** |
| Total-Text (curved text) | — | **0.585** |
| FUNSD (scanned forms) | — | **0.715** |

v0.3.0 pairs `DBNet_ResNet18_VD_Weights.ICDAR2015` with the new
`CRNN_ResNet34_VD_Weights.ICDAR2015_TOTALTEXT_MLT19`, its beam-search decoder and `min_text_score=0.8`.
The roadmap target of 0.65 on ICDAR-2015 is missed by 0.001; no setting was tuned on a test set to close it.

### Added

**Data**
- `torchocr.datasets.TotalText`: curved word polygons (3–19 vertices, padded by repeating the last
  vertex), tolerant of the release's malformed annotation lines; images read without EXIF rotation,
  which is what the annotations follow.
- `torchocr.datasets.FUNSD`: word boxes of 199 noisy scanned forms, the first *document* benchmark.
- `torchocr.datasets.LMDBWords`: word crops in the deep-text-recognition-benchmark LMDB layout (MJSynth,
  SynthText, IIIT5K, SVT, MLT crops…), with a label filter that never decodes images. Optional
  dependency: `pip install torchocr[lmdb]`.
- `load_target(index)` on the detection datasets: annotations without decoding the image.

**Decoding**
- `CTCBeamDecoder`: CTC prefix beam search with shallow fusion of an optional language model
  (~2 ms per word in pure Python).
- `CharNGramLM`: case-insensitive interpolated Witten-Bell character n-gram model with an end-of-word
  event; JSON save / load.
- `weights.meta["decoder"]` declares the decoder a recognizer was benchmarked with;
  `build_decoder(weights, charset, kind)` builds it and `hub.download_asset` fetches its language model
  (hash-checked).

**Weights**
- `CRNN_ResNet34_VD_Weights.ICDAR2015_TOTALTEXT_MLT19`: fine-tuned on 66k Latin scene words
  (ICDAR-2015, Total-Text, MLT-2019). Word accuracy 0.791 on ICDAR-2015 and 0.802 on Total-Text with
  its beam search and 6-gram character LM (0.750 / 0.775 greedy).

**Pipeline**
- `DocumentTensor.scores`: per-word recognition confidence.
- `OCRPipeline(min_text_score=...)` / `from_pretrained(..., decoder=, min_text_score=)`: drop regions
  read with low confidence (0.8 maximizes ICDAR-2015 *training*-split end-to-end hmean; +2.8 points on
  test).

**Training**
- `references/recognition/train.py` mixes full-image word sources (`--ic15`, `--totaltext`) with LMDB
  corpora (`--lmdb`), and gains `--amp` (bf16; an epoch went from 21 to 5 minutes) and `--resume`.
- `references/detection/train.py --totaltext`; `evaluate.py --dataset`;
  `references/end2end/evaluate.py --dataset {icdar2015,totaltext,funsd} --decoder --min-text-score`.

### Fixed
- `OCRPipeline` crops and recognizes words in bounded chunks (`recognizer_batch_size=64`): a dense form
  sent every word to the GPU at once and ran out of memory.

### Notes
- Training the ResNet18-VD detector on ICDAR-2015 + Total-Text for the same number of steps raised
  Total-Text hmean (0.696 → 0.758) but lowered ICDAR-2015 (0.809 → 0.784) and FUNSD (0.826 → 0.802);
  it was not published.
- No usable full-image ICDAR-2013 or MLT detection mirror was found; MLT-2019 contributes word crops.

## [0.2.0] - 2026-10-18

The first release with pretrained weights, benchmarks and an image-to-text pipeline. Every number
below is reproducible with `references/` and published with its checkpoint at
https://huggingface.co/BryanBradfo/torchocr-weights.

| ICDAR-2015 test | Converted PaddleOCR | Fine-tuned by torchocr |
| --- | --- | --- |
| Detection hmean (`DBNet` ResNet18-VD) | 0.414 | **0.809** |
| Recognition word accuracy (`CRNN` ResNet34-VD) | 0.665 | **0.738** |
| End-to-end hmean (`OCRPipeline`, image → text) | 0.219 | **0.578** |

Fine-tuned checkpoints are the last epoch; their post-processing threshold was calibrated on the
training split. Nothing was selected on the test set.

### Added

**Models and weights**
- torchvision-style weights API: `WeightsEnum` / `Weights` with `.url`, `.transforms()` and `.meta`
  (`task`, `backbone`, `num_params`, `source`, `license`, `postprocess`, `charset`, `_metrics`,
  `_docs`); per-architecture builders and `get_model`, `list_models`, `get_model_weights`,
  `get_weight`. Checkpoint names embed a SHA-256 prefix checked on download.
- Seven checkpoints: `DBNet_ResNet18_VD_Weights.{PPOCR_SERVER_V2, ICDAR2015}`,
  `DBNet_MobileNetV3_Large_05_Weights.{PPOCR_V3_CH, PPOCR_V3_EN, ICDAR2015}`,
  `CRNN_ResNet34_VD_Weights.{PPOCR_SERVER_V2, ICDAR2015}`.
- PaddleOCR-compatible architectures: `ResNetVd` and `MobileNetV3` backbones, DBFPN and RSEFPN
  necks, `DBNet(backbone="resnet18_vd" | "mobilenet_v3_large_05")`,
  `CRNN(backbone="resnet34_vd")`.
- Build-time converters `scripts/convert_paddle_{dbnet,dbnet_v3,crnn}.py` (opt-in `convert`
  extra). The PP-OCRv3 converter takes the exported `Student` network, byte-identical to PaddleOCR's
  `student.pdparams`; PaddleOCR2Pytorch's `Student2` choice costs about 10 hmean points.
- `torchocr.load_charset` / `load_ppocr_keys_v1` with the vendored PaddleOCR charset.

**Pipeline**
- `OCRPipeline.from_pretrained(detector_weights, recognizer_weights, device)`: per-stage presets,
  `DBPostProcessor(**meta["postprocess"])`, charset decoder; rotated regions in
  `DocumentTensor.polygons`, mapped back to input pixels.
- `DBPostProcessor`: PaddleOCR's contour flow (threshold → contours → rotated rectangle → score →
  `pyclipper` unclip) and `quadrilaterals()` for rotated quads in reading order.

**Ops and transforms**
- `torchocr.ops.crop_quads`: batched, differentiable perspective rectification of text quads, the
  OCR counterpart of `roi_align` (+8.6 word-accuracy points over axis-aligned crops). Also
  `order_quad_vertices`, `polygon_area`, `quad_to_box`.
- `torchocr.transforms`: `DetectionPreset`, `RecognitionPreset` (with `normalize()`),
  `MakeDBTargets` / `DBTargets`.

**Evaluation and data**
- `torchocr.metrics`: `DetectionHmean` (identical to PaddleOCR's evaluator to 4 decimals),
  `RecognitionAccuracy` (word accuracy, CER), `EndToEndHmean`.
- `torchocr.datasets.ICDAR2015` with the typed `TextDetectionTarget`.
- `references/`: `detection/{evaluate,calibrate,train,presets}.py`,
  `recognition/{evaluate,train}.py`, `end2end/evaluate.py`. Training resumes from atomic
  checkpoints with `--resume`.

**Graph capture**
- Every model FX-traces, `torch.export`s (DBNet with dynamic height/width in multiples of 32,
  CRNN with a dynamic batch) and `torch.compile`s (`fullgraph=True` for DBNet), checked by
  `tests/test_compile_export.py`. Compiled ResNet18-VD at 736x1280 on GPU: 25.0 → 20.3 ms.

**Project**
- CI on Python 3.10–3.13, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CREDITS.md`; 282 tests.

### Changed
- **Breaking:** `DBNet(weights=...)` / `CRNN(weights=...)` take a weights enum member, a member
  name, or a fully-qualified `"Cls.NAME"` string, from which the backbone is inferred. Requesting
  weights for a backbone without checkpoints (`resnet18`, `vgg`) raises `ValueError`;
  `CRNN.num_classes` is optional with weights and validated against them.
- **Breaking:** `OCRPipeline` crops with `crop_quads` (aspect-preserving, padded) instead of
  stretched axis-aligned `roi_align` crops; `crop_size` is `(height, max_width)`.
- Download failures emit a `UserWarning` and fall back to random init only on connectivity errors;
  hash mismatches and incompatible checkpoints raise.
- Shape checks moved to module-level `torch.fx.wrap` functions (still `ValueError`);
  `DBNetOutput` is registered with `torch.export.register_dataclass`.
- New runtime dependencies: `opencv-python-headless`, `pyclipper`, `shapely>=2.0`, `scipy`.

### Fixed
- Subnormal floats are zeroed on every weights load, conversion and training checkpoint:
  PaddleOCR's CRNN stores 6.3 M of them, which made CPU inference ~14x slower (54 s → 3.9 s per
  page) with bit-identical outputs.
- Paddle converters no longer hide load errors behind the removed `paddle.fluid` fallback.
- `scripts/infer.py` no longer requests weights that do not exist.

### Removed
- Synthetic `examples/sample_doc.jpg` / `demo_output.jpg`, replaced by real PaddleOCR demo images.
- `CLAUDE.md` from the public tree, superseded by `CONTRIBUTING.md`.

## [0.1.0] - 2026-04-25
### Added
- DBNet text detection architecture with ResNet-18 backbone.
- CRNN text recognition architecture with BiLSTM.
- `OCRPipeline` for end-to-end inference using `torchvision.ops.roi_align`.
- `DBLoss` and `CRNNLoss` modules for model training.
- PDF ingestion via PyMuPDF (`load_pdf`).
- Weight Hub for downloading pre-trained weights (with fallback to random init).
- Production inference CLI (`scripts/infer.py`) and demos.

[Unreleased]: https://github.com/BryanBradfo/torchOCR/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/BryanBradfo/torchOCR/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/BryanBradfo/torchOCR/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/BryanBradfo/torchOCR/releases/tag/v0.1.0