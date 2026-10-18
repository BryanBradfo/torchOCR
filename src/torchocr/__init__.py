"""torchocr: a PyTorch-native end-to-end OCR library."""

__version__ = "0.2.0"

from .charsets import load_charset, load_ppocr_keys_v1
from .core.structures import DocumentTensor
from .decoders import CharNGramLM, CTCBeamDecoder, CTCGreedyDecoder
from .io import load_image, load_pdf
from .losses import CRNNLoss, DBLoss
from .pipelines import OCRPipeline
from .postprocess import DBPostProcessor

__all__ = [
    "CRNNLoss",
    "CTCBeamDecoder",
    "CharNGramLM",
    "CTCGreedyDecoder",
    "DBLoss",
    "DBPostProcessor",
    "DocumentTensor",
    "OCRPipeline",
    "load_image",
    "load_charset",
    "load_pdf",
    "load_ppocr_keys_v1",
]
