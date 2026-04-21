# torchocr

**torchocr** is a PyTorch-native, end-to-end Optical Character Recognition (OCR) pipeline.

It provides a modern `src/` package layout with clear modules for:

- **Detection**: locating text regions and bounding boxes
- **Recognition**: reading text from detected regions
- **Core structures**: strongly typed containers such as `DocumentTensor`
- **I/O utilities**: image loading helpers for tensor-first workflows

The project is designed as a clean foundation for building and scaling OCR systems fully within the PyTorch ecosystem.
