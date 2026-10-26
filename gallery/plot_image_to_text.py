"""
Image to text in three lines
============================

:meth:`OCRPipeline.from_pretrained <torchocr.OCRPipeline.from_pretrained>` assembles a text
detector and a word recognizer from published weights. Each model brings its own preprocessing,
post-processing and decoder, read from ``weights.meta``.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon

from torchocr import OCRPipeline
from torchocr.io import load_image

ASSETS = Path("../examples")  # run from the gallery/ directory, as sphinx-gallery does

pipeline = OCRPipeline.from_pretrained(
    "DBNet_ResNet18_VD_Weights.ICDAR2015",
    "CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19",
    device="cuda" if torch.cuda.is_available() else "cpu",
    min_text_score=0.8,
)
image = load_image(ASSETS / "english_doc.jpg")  # (3, H, W) uint8 RGB
document = pipeline(image)

# %%
# The result is a :class:`~torchocr.DocumentTensor`: the untouched input, one string per region
# in reading order, the rotated quadrilaterals they were read from, axis-aligned boxes and a
# recognition confidence per region.

for text, score in zip(document.text, document.scores.tolist()):
    print(f"{score:.2f}  {text}")

# %%
# Drawing the regions on the image:

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(document.pixels.permute(1, 2, 0))
for quad, text in zip(document.polygons.cpu(), document.text):
    ax.add_patch(Polygon(quad.numpy(), fill=False, edgecolor="#f5c400", linewidth=2))
    ax.text(*quad[0].tolist(), text, color="black", backgroundcolor="#f5c400", fontsize=8, va="bottom")
ax.set_axis_off()
plt.tight_layout()

# %%
# The models are trained on Latin scene text (ICDAR-2015, Total-Text, MLT-2019). Words read with
# a confidence below ``min_text_score`` are dropped rather than returned as noise; pass
# ``min_text_score=None`` to keep every detected region.
