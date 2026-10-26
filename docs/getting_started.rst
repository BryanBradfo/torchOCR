Getting started
===============

Installation
------------

.. code-block:: bash

   pip install torchocr

or, from source, ``git clone https://github.com/BryanBradfo/torchOCR.git && pip install -e torchOCR``.

torchocr needs PyTorch 2 and torchvision 0.23 or later. ``pip install torchocr[lmdb]`` adds the
LMDB word-corpus reader; from source, ``.[test]`` adds the test suite and ``.[docs]`` this site.

Image to text
-------------

:meth:`OCRPipeline.from_pretrained <torchocr.OCRPipeline.from_pretrained>` takes a detector and a
recognizer weights entry, downloads both (SHA-256 checked, cached by ``torch.hub``), and wires
each model's preprocessing, post-processing and decoder from its ``weights.meta``:

.. code-block:: python

   import torch
   from torchocr import OCRPipeline
   from torchocr.io import load_image

   device = "cuda" if torch.cuda.is_available() else "cpu"
   pipeline = OCRPipeline.from_pretrained(
       "DBNet_ResNet18_VD_Weights.ICDAR2015",
       "CRNN_ResNet34_VD_Weights.MJSYNTH_ICDAR2015_TOTALTEXT_MLT19",
       device=device,
       min_text_score=0.8,  # drop regions read with low confidence
   )
   document = pipeline(load_image("street.jpg"))

   document.text            # list[str], one per region, in reading order
   document.polygons        # (K, 4, 2) rotated quads
   document.bounding_boxes  # (K, 4) xyxy
   document.scores          # (K,) recognition confidence

Building blocks
---------------

Every stage can be used, replaced or fine-tuned on its own. The detection half of the pipeline:

.. code-block:: python

   import torch
   from torchocr import DBPostProcessor
   from torchocr.io import load_image
   from torchocr.models import DBNet, DBNet_ResNet18_VD_Weights

   weights = DBNet_ResNet18_VD_Weights.ICDAR2015
   detector = DBNet(weights=weights).eval()
   images = weights.transforms()(load_image("street.jpg"))[None]  # (1, 3, H, W), H and W multiples of 32
   with torch.no_grad():
       output = detector(images)                                  # DBNetOutput(probability, threshold)
   postprocess = DBPostProcessor(**weights.meta["postprocess"])
   quads = postprocess.quadrilaterals(output)[0]                  # (K, 4, 2), in the resized image's pixels

Models are FX-traceable and work with ``torch.compile`` and ``torch.export``.

Training
--------

The ``references/`` directory holds the scripts that trained and evaluated every published
checkpoint, in the spirit of torchvision's reference scripts: ``references/detection``,
``references/recognition`` and ``references/end2end``. Each script's ``--help`` documents a
complete example command.
