"""Model components for text detection and recognition.

Mirrors ``torchvision.models``: classes, per-architecture builders,
``*_Weights`` enums, and the ``get_model`` / ``list_models`` registry.
"""

from .detection import (
    DBNet,
    DBNet_MobileNetV3_Large_05_Weights,
    DBNet_ResNet18_VD_Weights,
    DBNetOutput,
    dbnet_mobilenet_v3_large_05,
    dbnet_resnet18,
    dbnet_resnet18_vd,
)
from .hub import Weights, WeightsEnum, get_model, get_model_weights, get_weight, list_models
from .recognition import CRNN, CRNN_ResNet34_VD_Weights, crnn_resnet34_vd, crnn_vgg

__all__ = [
    "CRNN",
    "CRNN_ResNet34_VD_Weights",
    "DBNet",
    "DBNetOutput",
    "DBNet_MobileNetV3_Large_05_Weights",
    "DBNet_ResNet18_VD_Weights",
    "Weights",
    "WeightsEnum",
    "crnn_resnet34_vd",
    "crnn_vgg",
    "dbnet_mobilenet_v3_large_05",
    "dbnet_resnet18",
    "dbnet_resnet18_vd",
    "get_model",
    "get_model_weights",
    "get_weight",
    "list_models",
]
