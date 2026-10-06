# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `torchocr.metrics.DetectionHmean` / `HmeanResult`: ICDAR-2015 IoU protocol (invalid polygons dropped, don't-care filtering at 50% of prediction area, greedy one-to-one matching at IoU > 0.5, dataset-level aggregation). Vectorized with shapely 2. Verified identical to PaddleOCR's `DetectionIoUEvaluator` to 4 decimals on the 500 ICDAR-2015 test images.
- `torchocr.metrics.RecognitionAccuracy` / `RecognitionResult`: word accuracy and character error rate, with the case-insensitive alphanumeric scene-text protocol as an option.
- `torchocr.datasets.ICDAR2015` (official RRC folder layout, train/test, numeric image order, BOM/CRLF/comma-in-transcription parsing, `###` → `ignore`) and the typed `TextDetectionTarget(polygons, texts, ignore)` dataclass. Manual download (registration required); a missing folder raises with instructions.
- `torchocr.transforms.DetectionPreset` and `RecognitionPreset`: the preprocessing bound to a checkpoint (resize to multiples of 32 / height-32 keep-ratio + pad, normalization, `channel_order="bgr"` for PaddleOCR weights).
- torchvision-style weights API: `WeightsEnum` / `Weights` with `.url`, `.transforms()`, `.meta` (`task`, `backbone`, `num_params`, `source`, `license`, `_metrics`, `_docs`); enums `DBNet_ResNet18_VD_Weights.PPOCR_SERVER_V2`, `DBNet_MobileNetV3_Large_05_Weights.PPOCR_V3_CH` / `PPOCR_V3_EN` (new), `CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2`, each with a `DEFAULT` alias. Checkpoint names embed a SHA-256 prefix and downloads use `check_hash=True`.
- Model builders `dbnet_resnet18`, `dbnet_resnet18_vd`, `dbnet_mobilenet_v3_large_05`, `crnn_vgg`, `crnn_resnet34_vd` and registry helpers `list_models`, `get_model`, `get_model_weights`, `get_weight`.
- `torchocr.ops`: `crop_quads` (batched perspective rectification of text quads via homography + `grid_sample`; GPU, differentiable, vertical-text rotation, matches PaddleOCR's cv2 recipe), `order_quad_vertices`, `polygon_area`, `quad_to_box`.
- `DBPostProcessor.quadrilaterals()`: rotated text quads per image, in reading order (`__call__` keeps the `(K, 5)` contract).
- `references/detection/evaluate.py` and `references/recognition/evaluate.py`: the scripts behind every `meta["_metrics"]` number.
- Measured on ICDAR-2015 test: detection hmean 0.414 (ResNet-18-VD server v2), 0.422 (PP-OCRv3 ch), 0.441 (PP-OCRv3 en) — line-level detectors on a word-level benchmark, documented in `_docs`; recognition word accuracy 0.665 with `crop_quads` vs 0.579 with axis-aligned crops of the same 2077 words.
- Tests: `test_metrics.py`, `test_recognition_metrics.py`, `test_datasets.py`, `test_transforms.py`, `test_weights.py`, `test_ops.py`.
- `torchocr.transforms.MakeDBTargets` / `DBTargets`: DB training targets (pyclipper-shrunk probability map with the r / 2r fallback, exact point-to-segment border map in `[thresh_min, thresh_max]`, mask over `###`, tiny and unshrinkable text), following PaddleOCR's `MakeShrinkMap` + `MakeBorderMap`.
- `references/detection/presets.py` (`DBTrainPreset`, `random_text_crop`): polygon-aware flip / rotation / rescale and the text-preserving random crop of PaddleOCR's ICDAR-2015 recipe, seeded from torch's RNG so `DataLoader` workers draw different augmentations.
- `references/detection/train.py`: DBNet fine-tuning / training on ICDAR-2015 (AdamW, warmup + cosine, optional bf16 forward with an fp32 loss, PaddleOCR's 5:10:1 loss ratios). The test split is monitored, never used for checkpoint selection. `evaluate.py` exposes the reusable `evaluate_detector`.
- `DBNet_MobileNetV3_Large_05_Weights.ICDAR2015`: first torchocr-trained checkpoint (`PPOCR_V3_EN` fine-tuned 300 epochs on ICDAR-2015 train). ICDAR-2015 test hmean 0.734 (P 0.785, R 0.689) at the train-calibrated `box_thresh=0.45`; 0.688 at PaddleOCR's 0.6. Last epoch, no test-set selection.
- `meta["postprocess"]` on every detection checkpoint: the `DBPostProcessor` settings its `_metrics` were measured with (`DBPostProcessor(**weights.meta["postprocess"])`); `references/detection/evaluate.py --weights` uses it unless overridden.
- All five checkpoints published at https://huggingface.co/BryanBradfo/torchocr-weights with a model card.
- `references/detection/calibrate.py`: picks `box_thresh` by train-split hmean (one forward pass per image for all candidate thresholds) and scores the test split once with it.
- Tests: `test_db_targets.py`, `test_detection_presets.py`.
- `OCRPipeline.from_pretrained(detector_weights, recognizer_weights, device)`: assembles both models, their presets, `DBPostProcessor(**meta["postprocess"])` and the charset decoder from published weights. ICDAR-2015 end-to-end hmean 0.472 (`ICDAR2015` + `CRNN_ResNet34_VD_Weights.PPOCR_SERVER_V2`), 27 ms/image on GPU.
- `OCRPipeline(detector_transforms=..., recognizer_transforms=...)`: per-stage presets; regions mapped back to input pixels; `DocumentTensor.polygons` exposes the rotated quads.
- `torchocr.metrics.EndToEndHmean` (detection matching + transcription check) and `references/end2end/evaluate.py`.
- `torchocr.load_charset(name, num_classes)`; `CRNN_ResNet34_VD_Weights.meta["charset"]` is now the machine-readable `"ppocr_keys_v1"`.
- `RecognitionPreset.normalize()`: the per-pixel part of the preset, for normalizing a whole page before `crop_quads`.
- `torchocr.models.hub.zero_subnormals_`, applied by every weights load, converter and training checkpoint. PaddleOCR's CRNN stores 6.3 M subnormal floats (dead channels) that made CPU inference ~14x slower (receipt: 54 s -> 3.9 s), with bit-identical outputs.
- `MobileNetV3` backbone (`src/torchocr/models/backbones/mobilenet_v3.py`) matching PaddleOCR's `det_mobilenet_v3` structure: HardSwish/HardSigmoid activations, SE attention blocks, inverted-residual `_ResidualUnit` with 1x1 expand → depthwise k×k → optional SE → 1x1 project, `make_divisible` channel rounding. Supports `model_name ∈ {"large", "small"}`, `scale ∈ {0.35, 0.5, 0.75, 1.0, 1.25}`, optional `disable_se`.
- `_RSEFPN` neck and `_RSELayer` (1x1/3x3 conv + SEModule + optional shortcut) in `src/torchocr/models/detection.py`. Replaces DBFPN's plain Conv2d layers with channel-attention-augmented variants. Default `out_channels=96` matches PP-OCRv3.
- `DBNet(backbone="mobilenet_v3_large_05")` constructor option that wires MobileNetV3-large@0.5 (with `disable_se=True` per PaddleOCR) through RSEFPN(96) + DBHead. ~0.6M params total (vs ~12M for ResNet-18-VD).
- `scripts/convert_paddle_dbnet_v3.py`: build-time CLI for PP-OCRv3 detector `.pdparams`. Auto-detects the `Student.*` distillation prefix of `*_distill_train` checkpoints — the network PaddleOCR exports, byte-identical to the bundled `student.pdparams` (converting `Student2.*` as PaddleOCR2Pytorch does costs ~10 hmean points) — and falls back to no prefix for `*_infer` / `student.pdparams`. Stage format is locked to `prefixed` because MobileNetV3's numeric inner-block names would collide under flat-format mapping.
- `tests/test_mobilenet_v3.py` (8 shape-contract and architectural-pinning tests).
- `tests/test_v3_converter.py` (14 tests covering distillation-prefix detection, mapping rules, and bijective full-state-dict check).
- `.gitignore` patterns extended to cover PaddleOCR's mixed naming (`ch_PP-OCR*/`, `*_distill_train/`, etc.).
- `CRNN(backbone="resnet34_vd")`: PaddleOCR-compatible recognizer path. Wires `ResNetVd(depth=34, downsample_stride=(2, 1), stem_stride=1, final_pool=True)` through a `_SequenceEncoder` (Im2Seq + 2-layer BiLSTM) and a `_CTCHead` (single Linear). Output contract is unchanged — `(T, B, num_classes)` for direct CTC-decoder reuse.
- `scripts/convert_paddle_crnn.py`: build-time CLI converting PaddleOCR recognizer `.pdparams` (e.g. `ch_ppocr_server_v2.0_rec_train`) to a torchocr `.pth`. Auto-detects flat vs. stage-prefixed naming. Includes the FC-weight transpose Paddle Linear layers require.
- `scripts/test_full_ocr.py`: standalone end-to-end OCR demo that handles per-stage preprocessing (ImageNet normalization for the detector, `(x-127.5)/127.5` for the recognizer) and decodes Chinese results via the vendored charset. Bypasses `OCRPipeline` because the latter assumes a single normalization across stages — proper integration is Phase C scope.
- `torchocr.charsets.load_ppocr_keys_v1`: loader for the vendored Chinese full-charset (6622 chars + blank + space, sized to 6625 to match PaddleOCR's hardcoded `out_channels`).
- `src/torchocr/data/ppocr_keys_v1.txt`: vendored from PaddleOCR2Pytorch (Apache-2.0); package data wired via `[tool.setuptools.package-data]`.
- `crnn_resnet34_vd` weight-hub registry entry pointing at `huggingface.co/BryanBradfo/torchocr-weights/.../crnn_resnet34_vd.pth`.
- `tests/test_charsets.py` and `tests/test_crnn_converter.py` (parametrized for both `flat` and `prefixed` formats, including a bijective full-state-dict mapping check).
- New backbone parameters `stem_stride` (default 2; recognizer needs 1) and `final_pool` (default False; recognizer adds `MaxPool(2, 2)` after the last stage). Detector default behavior is byte-identical to before.
- `ResNetVd` backbone (`src/torchocr/models/backbones/resnet_vd.py`) matching PaddleOCR's `det_resnet_vd` structure: 3-conv VD stem, avg-pool shortcuts on stride-2 blocks. Internal submodule names mirror PaddleOCR (`conv1_1`, `stages.N.bb_<i>_<j>`, `_conv`, `_batch_norm`) so PaddleOCR-trained weights translate via a small mechanical name remap.
- `DBNet(backbone="resnet18_vd")` constructor option that wires the new backbone through a Paddle-compatible DBFPN neck (cascaded top-down adds + concat) and DBHead modules (`binarize`, `thresh`).
- `scripts/convert_paddle_dbnet.py`: build-time CLI that reads a PaddleOCR `.pdparams` checkpoint and writes a torchocr-compatible `.pth`. Lazy-imports `paddle`; the rest of torchocr has no Paddle dependency.
- `[project.optional-dependencies] convert = ["paddlepaddle>=2.5"]` extras group so conversion is opt-in.
- Runtime deps: `opencv-python-headless`, `pyclipper`, `shapely` for the contour-based post-processor.
- `dbnet_resnet18_vd` entry in the weight hub registry pointing at `huggingface.co/BryanBradfo/torchocr-weights/.../dbnet_resnet18_vd.pth`.
- `CREDITS.md` attributing PaddleOCR2Pytorch (Apache-2.0) for conversion recipes and architecture references.
- `tests/test_backbones.py` shape-contract tests for `ResNetVd`, plus extended `test_models.py` and `test_postprocess.py` for the new paths.
- `tests/test_converter.py` covering the dual-format (`flat` / `prefixed`) Paddle param-name mapping.
- `scripts/test_converted_dbnet.py` CLI: load converted weights, run detection on an image, save annotated visualization.
- Curated example images under `examples/` vendored from PaddleOCR2Pytorch's demo gallery: `chinese_receipt.jpg`, `chinese_typeset.jpg`, `english_doc.jpg`, `japanese.jpg`. Attribution in `CREDITS.md`.
- `scipy` runtime dependency for connected-component labeling in `DBPostProcessor`.
- `tests/` directory with 42 pytest tests covering all public APIs (models, pipeline, losses, decoders, post-processing, I/O).
- GitHub Actions CI workflow running on a Python 3.10 / 3.11 / 3.12 / 3.13 matrix; live build-status badge in `README.md`.
- `[project.optional-dependencies] test` group and `[tool.pytest.ini_options]` in `pyproject.toml`.
- `CONTRIBUTING.md` onboarding doc covering dev setup, style invariants, architecture invariants, branch + commit conventions.
- `CODE_OF_CONDUCT.md` with the full Contributor Covenant 2.1 text; reports route to bryan.chen@polytechnique.edu.

### Removed
- `examples/sample_doc.jpg` and `examples/demo_output.jpg`. The synthetic-bar demo image and its annotated output were replaced by real PaddleOCR demo images (see `examples/chinese_receipt.jpg` and friends). `examples/demo_inference.py` no longer synthesizes an input — it consumes a real image and accepts an optional `--weights` flag for converted PaddleOCR checkpoints.

### Fixed
- Paddle converters no longer fall back to the removed `paddle.fluid` API, which turned any load error (e.g. a wrong path) into `No module named 'paddle.fluid'`; a missing file now exits with a clear message.

### Changed
- `OCRPipeline` crops regions with `torchocr.ops.crop_quads` (aspect-preserving, padded to `crop_size` width) instead of stretching axis-aligned `roi_align` crops; `crop_size` is now `(height, max_width)`.
- **Breaking:** `DBNet(weights=...)` / `CRNN(weights=...)` take a `WeightsEnum` member or member name. `backbone` now defaults to `None` (inferred from an enum member, else `"resnet18"` / `"vgg"`); `CRNN.num_classes` is optional when weights are given and validated against them. Requesting weights for a backbone without published checkpoints (`resnet18`, `vgg`) raises `ValueError` instead of printing a warning, and `CRNN()` without `num_classes` raises `ValueError` instead of `TypeError`.
- Download failures now emit a `UserWarning` (was `print`) and only connectivity errors (`OSError`) fall back to random init; hash mismatches and incompatible checkpoints raise.
- `scripts/infer.py` builds randomly-initialized models explicitly (it previously requested non-existent `DEFAULT` weights); its docstring points to `scripts/test_full_ocr.py` for real OCR until `OCRPipeline` supports per-stage presets.
- `shapely>=2.0` (vectorized API used by `DetectionHmean`).
- `examples/demo_inference.py` rewritten around real document images: defaults to `examples/chinese_receipt.jpg`, accepts `--image` and `--weights` flags. Without weights it still verifies pipeline composition end-to-end; with converted weights it produces real bounding boxes.
- `DBPostProcessor` rewritten around PaddleOCR's contour-based flow: `cv2.findContours` → rotated rect via `cv2.minAreaRect` → mean-probability score under the rectangle → `pyclipper` polygon offset by `unclip_ratio` → axis-aligned projection. New parameters `box_thresh` (default 0.7), `max_candidates` (default 1000), `unclip_ratio` (default 1.5), `min_size` (default 3). The output contract `(K, 5)` with `[batch_idx, x1, y1, x2, y2]` is unchanged.
- `DBPostProcessor` now emits one bounding box per disconnected text region via `scipy.ndimage.label` + `find_objects`, instead of one AABB per image. The output format `(K, 5)` with columns `[batch_idx, x1, y1, x2, y2]` is unchanged; only `K` semantics broaden — a multi-line document yields one box per text line.
- `README.md` Contributing section now points at `CONTRIBUTING.md` for coding standards and PR process.

### Removed
- `CLAUDE.md` — superseded by `CONTRIBUTING.md`. Project conventions for human contributors are now in one canonical place; AI-agent tooling docs are not carried in the public tree (matches torchvision/pytorch/torchgeo).

## [0.1.0] - 2026-04-25
### Added
- DBNet text detection architecture with ResNet-18 backbone.
- CRNN text recognition architecture with BiLSTM.
- `OCRPipeline` for end-to-end inference using `torchvision.ops.roi_align`.
- `DBLoss` and `CRNNLoss` modules for model training.
- PDF ingestion via PyMuPDF (`load_pdf`).
- Weight Hub for downloading pre-trained weights (with fallback to random init).
- Production inference CLI (`scripts/infer.py`) and demos.

[Unreleased]: https://github.com/BryanBradfo/torchOCR/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BryanBradfo/torchOCR/releases/tag/v0.1.0