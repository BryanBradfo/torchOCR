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

Page orientation against other tools
------------------------------------

Each of the 4999 DocLayNet test pages is fed upright and turned 90, 180 and 270 degrees. Every tool
gets a threshold chosen on DocLayNet *validation*: the lowest one at which it turns at most 0.25% of
upright pages. Each tool then runs once on test with that threshold. "Without a threshold" is the
tool's raw answer. Reproduce with ``references/orientation/evaluate.py --engine {torchocr,doctr,tesseract}``.

.. list-table::
   :header-rows: 1

   * - Tool
     - Right angle (no threshold)
     - Upright pages wrongly turned / turned pages set upright, no threshold
     - Same, at the validation threshold
     - CPU, one core, per page
   * - torchocr ``PPLCNet_x1_0_Weights.PPOCR_DOC_ORIENTATION``
     - **98.5%**
     - 1.5% / 98.5%
     - 0.40% / **97.8%**
     - **14 ms**
   * - docTR 1.1 ``mobilenet_v3_small_page_orientation``
     - 90.8%
     - 10.0% / 91.1%
     - 0.68% / 31.2%
     - 32 ms
   * - Tesseract 5.3.4 OSD (``ocrmypdf --rotate-pages``), pages upscaled 2x
     - 90.0%
     - 7.4% / 89.1%
     - 0.18% / 45.3%
     - 890 ms

To stay as cautious with upright pages as torchocr, docTR and Tesseract must be so sure of
themselves that they leave most turned pages as they are. Tesseract needs ~300 dpi text, and DocLayNet pages
are ~100 dpi: on a 12-page validation sample it was right for 10 of 48 images at native size and 41
of 48 upscaled 2x, so it gets 2x. It also gives no answer for 2.6% of pages (too little text), which are kept as they
are. torchocr's published threshold (0.7, a little stricter than the 0.62 picked by this rule) gives
0.24% / 97.5%.

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
