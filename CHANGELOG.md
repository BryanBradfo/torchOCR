# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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