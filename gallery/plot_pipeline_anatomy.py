"""
Anatomy of the pipeline
=======================

:class:`~torchocr.OCRPipeline` is four swappable stages:

``detector → post-processor → crop_quads → recognizer → decoder``

This example runs them one by one, so each can be inspected or replaced.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchocr import CTCGreedyDecoder, DBPostProcessor, load_charset
from torchocr.decoders import build_decoder
from torchocr.io import load_image
from torchocr.models import CRNN, DBNet, CRNN_ResNet34_VD_Weights, DBNet_ResNet18_VD_Weights
from torchocr.ops import crop_quads

ASSETS = Path("../examples")  # run from the gallery/ directory, as sphinx-gallery does
image = load_image(ASSETS / "english_doc.jpg")

# %%
# 1. Detection
# ------------
# The detector predicts a per-pixel text probability map. Its weights entry provides the
# preprocessing it was trained with (resize to multiples of 32, normalization).

det_weights = DBNet_ResNet18_VD_Weights.ICDAR2015
detector = DBNet(weights=det_weights).eval()
detector_input = det_weights.transforms()(image)[None]
with torch.inference_mode():
    detection = detector(detector_input)  # DBNetOutput(probability, threshold)

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.5))
left.imshow(image.permute(1, 2, 0))
right.imshow(detection.probability[0, 0], cmap="magma")
for ax, title in ((left, "input"), (right, "detection.probability")):
    ax.set_title(title)
    ax.set_axis_off()
plt.tight_layout()

# %%
# 2. Post-processing
# ------------------
# :class:`~torchocr.DBPostProcessor` turns the probability map into rotated quadrilaterals, sorted
# in reading order (:func:`torchocr.ops.reading_order`).
# They are in the resized image's pixels; scale them back to the original.

postprocess = DBPostProcessor(**det_weights.meta["postprocess"])
quads = postprocess.quadrilaterals(detection)[0]
scale = torch.tensor([image.shape[-1] / detector_input.shape[-1], image.shape[-2] / detector_input.shape[-2]])
quads = quads * scale
print(f"{len(quads)} regions; first quad:\n{quads[0]}")

# %%
# 3. Rectified crops
# ------------------
# :func:`torchocr.ops.crop_quads` warps every quad into an axis-aligned strip of fixed height with
# one batched ``grid_sample`` call, undoing rotation and perspective.

rec_weights = CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19
preset = rec_weights.transforms()
crops, widths = crop_quads(preset.normalize(image)[None].float(), quads, torch.zeros(len(quads), dtype=torch.long),
                           height=32, max_width=320)
print(crops.shape)  # (K, 3, 32, W)

raw, _ = crop_quads(image[None].float(), quads, torch.zeros(len(quads), dtype=torch.long), height=32, max_width=320)
fig, axes = plt.subplots(len(raw), 1, figsize=(6, 0.5 * len(raw)))
for ax, crop in zip(axes, raw):
    ax.imshow(crop.permute(1, 2, 0).clamp(0, 255).byte())
    ax.set_axis_off()

# %%
# 4. Recognition and decoding
# ---------------------------
# The recognizer emits per-column character logits for CTC. The weights entry declares its
# decoder: beam search with a character language model. Greedy decoding is the cheap baseline.

recognizer = CRNN(weights=rec_weights).eval()
with torch.inference_mode():
    logits = recognizer(crops)  # (T, K, num_classes)
charset = load_charset(rec_weights.meta["charset"], rec_weights.meta["num_classes"])
greedy = CTCGreedyDecoder(charset)(logits)
beam = build_decoder(rec_weights, charset)(logits)  # what the pipeline uses
for g, b in zip(greedy, beam):
    print(f"{g:>14}  {b:>14}" + ("   <- the language model changed this word" if g != b else ""))
