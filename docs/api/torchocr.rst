torchocr
========

.. currentmodule:: torchocr

Pipeline
--------

.. autoclass:: OCRPipeline
   :members: from_pretrained, __call__

.. autoclass:: DocumentTensor

Page orientation
----------------

.. autoclass:: PageOrientation
   :members: from_pretrained, probabilities, forward, upright
.. autofunction:: rotate_page

Post-processing
---------------

.. autoclass:: DBPostProcessor
   :members: __call__, quadrilaterals

Decoders
--------

.. autoclass:: CTCGreedyDecoder
.. autoclass:: CTCBeamDecoder
.. autoclass:: CharNGramLM
   :members: from_texts, log_prob, score, save, load
.. autofunction:: torchocr.decoders.build_decoder

Losses
------

.. autoclass:: DBLoss
.. autoclass:: CRNNLoss

Character sets
--------------

.. autofunction:: load_charset
