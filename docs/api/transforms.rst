torchocr.transforms
===================

.. currentmodule:: torchocr.transforms

Inference presets
-----------------

Each weights entry's ``transforms()`` returns the preset it was trained with.

.. autoclass:: DetectionPreset
.. autoclass:: RecognitionPreset

Training targets
----------------

.. autoclass:: MakeDBTargets
   :members: __call__
.. autoclass:: DBTargets

Augmentation with ``torchvision.transforms.v2``
-----------------------------------------------

.. automodule:: torchocr.transforms.v2
   :no-members:

.. autofunction:: torchocr.transforms.v2.wrap_sample
.. autoclass:: torchocr.transforms.v2.RandomTextCrop
.. autoclass:: torchocr.transforms.v2.SanitizeTextPolygons
