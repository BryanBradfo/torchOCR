"""Shape-validation smoke test for DBNet and CRNN.

Run from the repo root:
    python scripts/test_inference.py
"""

import torch

from torchocr.models import CRNN, DBNet


def main() -> None:
    detector = DBNet().train(False)
    recognizer = CRNN(num_classes=96).train(False)

    images = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        detection = detector(images)
    print(f"probability: {tuple(detection.probability.shape)}")
    print(f"threshold:   {tuple(detection.threshold.shape)}")
    assert detection.probability.shape == (1, 1, 512, 512)
    assert detection.threshold.shape == (1, 1, 512, 512)

    # detector -> recognizer bridge
    # A real pipeline would threshold detection.probability, extract
    # polygons, crop `images` to each polygon, and resize each crop
    # to H=32. For shape validation we substitute a dummy crop.
    crop = torch.randn(1, 3, 32, 128)

    with torch.no_grad():
        logits = recognizer(crop)
    print(f"logits:      {tuple(logits.shape)}")
    assert logits.shape == (32, 1, 96)

    print("OK")


if __name__ == "__main__":
    main()
