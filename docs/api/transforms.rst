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
