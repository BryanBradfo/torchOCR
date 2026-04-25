"""torchocr: a PyTorch-native end-to-end OCR library."""

from .core.structures import DocumentTensor
from .decoders import CTCGreedyDecoder
from .io import load_image, load_pdf
from .losses import CRNNLoss, DBLoss
from .pipelines import OCRPipeline
from .postprocess import DBPostProcessor

__all__ = [
    "CRNNLoss",
    "CTCGreedyDecoder",
    "DBLoss",
    "DBPostProcessor",
    "DocumentTensor",
    "OCRPipeline",
    "load_image",
    "load_pdf",
]
