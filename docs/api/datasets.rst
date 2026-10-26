torchocr.datasets
=================

.. currentmodule:: torchocr.datasets

Datasets return ``(uint8 image, TextDetectionTarget)`` pairs (detection) or ``(image, text)`` pairs
(word crops). None of them downloads anything: point them at the extracted archives.

.. autoclass:: TextDetectionTarget
.. autoclass:: ICDAR2015
.. autoclass:: TotalText
.. autoclass:: FUNSD
.. autoclass:: LMDBWords
