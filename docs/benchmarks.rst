Benchmarks and limits
=====================

Every number here is reproducible with the scripts in ``references/`` and tied to a specific
hash-checked weights file. Fine-tuned checkpoints are the last training epoch; thresholds, beam
width, language-model weight and ``min_text_score`` were chosen on *training* splits, and each
test set was scored once.

End to end (image → text)
-------------------------

A word counts when its detected region matches a ground-truth word (ICDAR IoU protocol, > 0.5)
*and* is read correctly (case-insensitive, letters and digits, generic vocabulary). Pipeline:
``DBNet_ResNet18_VD_Weights.ICDAR2015`` + ``CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19``
with beam search and ``min_text_score=0.8``; ``references/end2end/evaluate.py``.

.. list-table::
   :header-rows: 1

   * - Test set
     - v0.3 recognizer
     - MJSynth recognizer (current)
   * - ICDAR-2015 (incidental street scenes)
     - 0.649
     - **0.665**
   * - Total-Text (curved text)
     - 0.585
     - **0.585**
   * - FUNSD (scanned forms)
     - 0.715
     - **0.722**

The v0.3 recognizer is ``CRNN_ResNet34_VD_Weights.ICDAR2015_TOTALTEXT_MLT19``; the current one is the
same model trained further with MJSynth synthetic words mixed in.

Per-model detection and recognition results are listed with each weights entry in
:doc:`models`.

What the models are good at
---------------------------

- **Latin-script scene text** (street signs, shop fronts, product labels) and **scanned
  documents** in English and other Latin-script languages.
- Word-level output: the detector finds words, not lines; the recognizer reads one word per crop.

Known limits
------------

- **Chinese and other non-Latin scripts**: the torchocr fine-tunes were trained on Latin words
  only. The converted ``PPOCR_*`` weights keep PaddleOCR's Chinese coverage but score far lower on
  Latin benchmarks.
- **Curved text** is read through a quadrilateral crop: end-to-end hmean is 0.585 on Total-Text
  against 0.665 on the mostly straight ICDAR-2015.
- **Documents**: the detector was trained on street scenes only. It still reaches 0.826
  detection hmean on FUNSD forms, but no checkpoint has seen a document during training.
- **Handwriting, vertical text, other domains** have not been benchmarked. ``min_text_score``
  drops low-confidence reads, so such inputs tend to give fewer words rather than wrong ones;
  measure on your own data before relying on it.
