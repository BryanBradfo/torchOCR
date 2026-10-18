"""torchocr.ops: batched, device-agnostic geometry primitives for OCR."""

import math

import pytest
import torch

from torchocr.ops import crop_quads, order_quad_vertices, polygon_area, quad_to_box


def _rotated_quad(cx: float, cy: float, w: float, h: float, degrees: float) -> torch.Tensor:
    """TL, TR, BR, BL corners of a w x h rectangle rotated about its centre."""
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    local = torch.tensor([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = torch.tensor([[c, -s], [s, c]])
    return local @ rot.T + torch.tensor([cx, cy])


# === polygon_area / quad_to_box ===


def test_polygon_area_is_orientation_independent_and_batched():
    square = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
    polys = torch.stack([square, square.flip(0), _rotated_quad(10, 10, 6, 2, 30)])
    assert polygon_area(polys).tolist() == pytest.approx([12.0, 12.0, 12.0], abs=1e-4)


def test_quad_to_box():
    quads = _rotated_quad(10, 20, 4, 2, 90)[None]
    assert quad_to_box(quads)[0].tolist() == pytest.approx([9.0, 18.0, 11.0, 22.0], abs=1e-5)


@pytest.mark.parametrize("fn", [polygon_area, quad_to_box, order_quad_vertices])
def test_shape_contract(fn):
    with pytest.raises(ValueError, match=r"\(\.\.\., "):
        fn(torch.zeros(3, 4, 3))


# === order_quad_vertices ===


@pytest.mark.parametrize("degrees", [-30.0, -5.0, 0.0, 12.0, 40.0])
@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_order_quad_vertices_recovers_reading_order(degrees, shift):
    """cv2.boxPoints starts anywhere; reading order is TL, TR, BR, BL."""
    quad = _rotated_quad(50, 50, 40, 10, degrees)
    scrambled = quad.roll(shift, dims=0)
    assert torch.allclose(order_quad_vertices(scrambled[None])[0], quad, atol=1e-4)


def test_order_quad_vertices_handles_counter_clockwise_input():
    quad = _rotated_quad(50, 50, 40, 10, 10)
    assert torch.allclose(order_quad_vertices(quad.flip(0)[None])[0], quad, atol=1e-4)


# === crop_quads ===


def _striped_page() -> torch.Tensor:
    """(1, 1, 64, 128) page with a bright 40x10 horizontal bar."""
    page = torch.zeros(1, 1, 64, 128)
    page[0, 0, 27:37, 44:84] = 1.0
    return page


def test_axis_aligned_crop_extracts_the_region():
    page = _striped_page()
    quad = torch.tensor([[[44.0, 27.0], [84.0, 27.0], [84.0, 37.0], [44.0, 37.0]]])
    crops, widths = crop_quads(page, quad, torch.tensor([0]), height=10, max_width=60)

    assert crops.shape == (1, 1, 10, 60)
    assert widths.tolist() == [40]
    assert crops[0, 0, :, :40].mean() > 0.95
    assert crops[0, 0, :, 40:].abs().max() == 0  # right padding


def test_rotated_text_is_rectified():
    """A bar rotated by 30 degrees comes out as a horizontal, filled crop."""
    image = torch.zeros(1, 1, 200, 200)
    ys, xs = torch.meshgrid(torch.arange(200.0) + 0.5, torch.arange(200.0) + 0.5, indexing="ij")
    t = math.radians(30)
    u = (xs - 100) * math.cos(t) + (ys - 100) * math.sin(t)
    v = -(xs - 100) * math.sin(t) + (ys - 100) * math.cos(t)
    image[0, 0] = ((u.abs() < 40) & (v.abs() < 8)).float()

    quad = _rotated_quad(100, 100, 80, 16, 30)[None]
    crops, widths = crop_quads(image, quad, torch.tensor([0]), height=16, max_width=100)
    valid = crops[0, 0, :, : widths[0]]
    assert widths.tolist() == [80]
    # Interior is solid (edges blur by bilinear sampling).
    assert valid[2:-2, 2:-2].min() > 0.9


def test_vertical_quads_are_rotated_to_horizontal():
    image = torch.zeros(1, 1, 100, 100)
    image[0, 0, 10:90, 45:55] = 1.0  # tall 10 x 80 bar
    quad = torch.tensor([[[45.0, 10.0], [55.0, 10.0], [55.0, 90.0], [45.0, 90.0]]])
    crops, widths = crop_quads(image, quad, torch.tensor([0]), height=10, max_width=100)
    assert widths.tolist() == [80]
    assert crops[0, 0, 1:-1, 1:79].min() > 0.9


def test_batch_index_selects_the_page_and_fill_pads():
    pages = torch.stack([torch.zeros(3, 32, 32), torch.ones(3, 32, 32)])
    quads = torch.tensor([[[0.0, 0.0], [16.0, 0.0], [16.0, 8.0], [0.0, 8.0]]]).repeat(2, 1, 1)
    crops, widths = crop_quads(pages, quads, torch.tensor([1, 0]), height=8, max_width=32, fill=-1.0)
    assert crops.shape == (2, 3, 8, 32)
    assert crops[0, :, :, :16].eq(1).all() and crops[1, :, :, :16].eq(0).all()
    assert crops[:, :, :, 16:].eq(-1).all()


def test_width_is_clamped():
    page = torch.rand(1, 3, 64, 512)
    quad = torch.tensor([[[0.0, 0.0], [500.0, 0.0], [500.0, 10.0], [0.0, 10.0]]])
    crops, widths = crop_quads(page, quad, torch.tensor([0]), height=32, max_width=320)
    assert crops.shape == (1, 3, 32, 320) and widths.tolist() == [320]


def test_empty_input():
    crops, widths = crop_quads(torch.rand(1, 3, 32, 32), torch.zeros(0, 4, 2), torch.zeros(0, dtype=torch.long), 32, 100)
    assert crops.shape == (0, 3, 32, 100) and widths.shape == (0,)


def test_crop_is_differentiable_wrt_image():
    page = torch.rand(1, 1, 32, 64, requires_grad=True)
    quad = torch.tensor([[[4.0, 4.0], [40.0, 6.0], [40.0, 20.0], [4.0, 18.0]]])
    crops, _ = crop_quads(page, quad, torch.tensor([0]), height=8, max_width=48)
    crops.sum().backward()
    assert page.grad is not None and page.grad.abs().sum() > 0


def test_crop_contract_errors():
    page = torch.rand(1, 3, 32, 32)
    quad = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        crop_quads(page[0], quad, torch.tensor([0]), 8, 16)
    with pytest.raises(ValueError, match=r"\(K, 4, 2\)"):
        crop_quads(page, torch.zeros(1, 5, 2), torch.tensor([0]), 8, 16)
    with pytest.raises(ValueError, match=r"batch_idx"):
        crop_quads(page, quad, torch.tensor([0, 0]), 8, 16)
