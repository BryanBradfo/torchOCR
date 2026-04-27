# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scipy` runtime dependency for connected-component labeling in `DBPostProcessor`.
- `tests/` directory with 42 pytest tests covering all public APIs (models, pipeline, losses, decoders, post-processing, I/O).
- GitHub Actions CI workflow running on a Python 3.10 / 3.11 / 3.12 / 3.13 matrix; live build-status badge in `README.md`.
- `[project.optional-dependencies] test` group and `[tool.pytest.ini_options]` in `pyproject.toml`.
- `CONTRIBUTING.md` onboarding doc covering dev setup, style invariants, architecture invariants, branch + commit conventions.
- `CODE_OF_CONDUCT.md` with the full Contributor Covenant 2.1 text; reports route to bryan.chen@polytechnique.edu.

### Changed
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