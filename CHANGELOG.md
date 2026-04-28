# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
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