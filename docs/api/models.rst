torchocr.models
===============

.. currentmodule:: torchocr.models

See :doc:`../models` for every pretrained weights entry and its benchmark.

Detection
---------

.. autoclass:: DBNet
.. autoclass:: DBNetOutput
.. autofunction:: dbnet_resnet18_vd
.. autofunction:: dbnet_resnet18
.. autofunction:: dbnet_mobilenet_v3_large_05
.. autoclass:: DBNet_ResNet18_VD_Weights
   :members:
   :undoc-members:
.. autoclass:: DBNet_MobileNetV3_Large_05_Weights
   :members:
   :undoc-members:

Recognition
-----------

.. autoclass:: CRNN
.. autofunction:: crnn_resnet34_vd
.. autofunction:: crnn_vgg
.. autoclass:: CRNN_ResNet34_VD_Weights
   :members:
   :undoc-members:

Weights API
-----------

.. autoclass:: Weights
.. autoclass:: WeightsEnum
   :members: verify, get_state_dict, transforms, url, meta
.. autofunction:: list_models
.. autofunction:: get_model
.. autofunction:: get_model_weights
.. autofunction:: get_weight
