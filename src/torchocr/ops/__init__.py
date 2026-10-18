"""Batched, device-agnostic geometry ops for OCR, mirroring ``torchvision.ops``."""

from .crop import crop_quads
from .geometry import order_quad_vertices, polygon_area, quad_to_box

__all__ = ["crop_quads", "order_quad_vertices", "polygon_area", "quad_to_box"]
