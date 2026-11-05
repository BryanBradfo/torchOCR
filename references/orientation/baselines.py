"""Page-orientation engines compared by ``evaluate.py``: torchocr, docTR and Tesseract OSD.

Every engine maps a list of ``(3, H, W)`` ``uint8`` RGB pages to ``(classes, confidences)``:
``classes[i]`` is how far page ``i`` is turned clockwise, in quarter turns (torchocr's
convention), and ``confidences[i]`` is the engine's own score for that answer. Scores are not
comparable across engines, so each engine's threshold is chosen on validation separately.

docTR and Tesseract are only needed to run the comparison:
``pip install python-doctr==1.1.0 pytesseract`` and ``apt install tesseract-ocr`` (with ``osd``).
"""

import multiprocessing

import numpy as np
import torch
from torch import Tensor

from torchocr import PageOrientation
from torchocr.models import PPLCNet, PPLCNet_x1_0_Weights

# Confidence of a page an engine gave no answer for; below every threshold, so the page is kept.
NO_ANSWER = -1.0


class TorchOCR:
    """torchocr's ``PageOrientation`` classifier; the confidence is the softmax probability."""

    def __init__(self, weights: str, checkpoint: str | None, device: str) -> None:
        weights = PPLCNet_x1_0_Weights.verify(weights)
        if checkpoint:
            classifier = PPLCNet(num_classes=len(weights.meta["categories"]))
            classifier.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        else:
            classifier = PPLCNet(weights=weights)
        self.orientation = PageOrientation(
            classifier.eval().to(device), weights.transforms()
        )
        self.name = f"torchocr ({weights!r})"

    @staticmethod
    def default_min_confidence(weights: str) -> float:
        return PPLCNet_x1_0_Weights.verify(weights).meta["min_confidence"]

    def __call__(self, pages: list[Tensor]) -> tuple[Tensor, Tensor]:
        confidences, classes = self.orientation.probabilities(pages).cpu().max(-1)
        return classes, confidences


class DocTR:
    """docTR's page-orientation classifier (MobileNetV3-small); the confidence is its probability."""

    # docTR answers with the angle the content is turned clockwise (-90 = 270 clockwise).
    _CLASS_OF_ANGLE = {0: 0, 90: 1, 180: 2, -90: 3}

    def __init__(self, device: str) -> None:
        from doctr.models import page_orientation_predictor

        self.predictor = page_orientation_predictor(pretrained=True).to(device)
        self.name = "docTR 1.1 (mobilenet_v3_small_page_orientation)"

    @torch.inference_mode()
    def __call__(self, pages: list[Tensor]) -> tuple[Tensor, Tensor]:
        _, angles, confidences = self.predictor(
            [page.permute(1, 2, 0).cpu().numpy() for page in pages]
        )
        classes = torch.tensor([self._CLASS_OF_ANGLE[a] for a in angles])
        return classes, torch.tensor(confidences, dtype=torch.float32)


def _tesseract_osd(image: np.ndarray, scale: float) -> tuple[int, float]:
    import pytesseract
    from PIL import Image

    page = Image.fromarray(image)
    if scale != 1:
        page = page.resize(
            (round(page.width * scale), round(page.height * scale)), Image.BICUBIC
        )
    try:
        osd = pytesseract.image_to_osd(page, output_type=pytesseract.Output.DICT)
    except (
        pytesseract.TesseractError
    ):  # too little text to decide: the page is kept as it is
        return 0, NO_ANSWER
    # "rotate" is the clockwise turn that sets the page upright: 270 for a page turned 90 clockwise.
    return (4 - osd["rotate"] // 90) % 4, float(osd["orientation_conf"])


class Tesseract:
    """Tesseract's orientation and script detection (``--psm 0``), as used by ``ocrmypdf --rotate-pages``.

    The confidence is Tesseract's ``orientation_conf``. Pages are upscaled by ``scale`` first:
    Tesseract expects ~300 dpi text, and DocLayNet pages are ~100 dpi.
    """

    def __init__(self, scale: float, workers: int) -> None:
        self.scale = scale
        self.pool = (
            multiprocessing.get_context("spawn").Pool(workers) if workers > 1 else None
        )
        self.name = f"Tesseract OSD (scale {scale:g})"

    def __call__(self, pages: list[Tensor]) -> tuple[Tensor, Tensor]:
        # Upscaling happens in the workers: done here, it would serialize the whole run.
        jobs = [
            (np.ascontiguousarray(page.permute(1, 2, 0).cpu().numpy()), self.scale)
            for page in pages
        ]
        results = (
            self.pool.starmap(_tesseract_osd, jobs)
            if self.pool
            else [_tesseract_osd(*j) for j in jobs]
        )
        classes, confidences = zip(*results)
        return torch.tensor(classes), torch.tensor(confidences, dtype=torch.float32)
