torchocr
========

**What an OCR library would look like if torchvision built it.**

torchocr is a PyTorch-native OCR library: text detectors, recognizers, decoders, datasets,
metrics and transforms, each an ordinary ``nn.Module`` or callable with explicit shape
contracts, composed into an :class:`~torchocr.OCRPipeline`. Pretrained weights follow
torchvision's API (``DBNet_ResNet18_VD_Weights.ICDAR2015``), are hash-checked, and ship with
the benchmark that justifies them.

.. code-block:: python

   from torchocr import OCRPipeline
   from torchocr.io import load_image

   pipeline = OCRPipeline.from_pretrained(
       "DBNet_ResNet18_VD_Weights.ICDAR2015",
       "CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19",
       min_text_score=0.8,
   )
   document = pipeline(load_image("street.jpg"))
   print(document.text)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   models
   benchmarks
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/torchocr
   api/models
   api/datasets
   api/transforms
   api/ops
   api/metrics
   api/io
