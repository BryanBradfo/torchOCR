"""Backbone networks for torchocr detectors and recognizers."""

from .mobilenet_v3 import MobileNetV3
from .resnet_vd import ResNetVd

__all__ = ["MobileNetV3", "ResNetVd"]
