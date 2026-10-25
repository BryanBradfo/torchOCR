Models and pretrained weights
=============================

Weights follow torchvision's API: each architecture has a ``*_Weights`` enum whose entries carry
a hash-checked URL, the matching ``transforms()`` and a ``meta`` dict (task, backbone, parameter
count, training source, license, post-processing and decoder settings, benchmark results).

.. code-block:: python

   from torchocr.models import DBNet, DBNet_ResNet18_VD_Weights, get_model, list_models

   weights = DBNet_ResNet18_VD_Weights.ICDAR2015
   model = DBNet(weights=weights).eval()
   preprocess = weights.transforms()
   weights.meta["_metrics"]

   list_models()                                    # registered builders
   get_model("dbnet_resnet18_vd", weights="DEFAULT")

The tables below are generated from ``weights.meta`` at build time.

Text detection
--------------

Detection hmean on the ICDAR-2015 test set, IoU protocol with don't-care regions.

.. weights-table:: detection

Text recognition
----------------

Case-insensitive word accuracy on the test words, cropped from the full images along their
ground-truth quadrilaterals (:func:`torchocr.ops.crop_quads`).

.. weights-table:: recognition
