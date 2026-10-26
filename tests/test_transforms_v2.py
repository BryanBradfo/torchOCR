"""torchocr.transforms.v2: text polygons through torchvision.transforms.v2."""

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from torchocr.datasets import TextDetectionTarget
from torchocr.transforms import MakeDBTargets
from torchocr.transforms.v2 import RandomTextCrop, SanitizeTextPolygons, wrap_sample


def _sample(rects: list[tuple[int, int, int, int]], size: tuple[int, int] = (120, 200)):
    """White filled rectangles ``(x0, y0, x1, y1)`` on black, with their quads as targets."""
    image = torch.zeros(3, *size, dtype=torch.uint8)
    polys = []
    for x0, y0, x1, y1 in rects:
        image[:, y0:y1, x0:x1] = 255
        polys.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    texts = [f"w{i}" for i in range(len(rects))]
    ignore = torch.tensor([i == len(rects) - 1 for i in range(len(rects))])
    return wrap_sample(image, TextDetectionTarget(torch.tensor(polys, dtype=torch.float32), texts, ignore))


def test_polygons_follow_the_pixels_through_torchvision_transforms():
    torch.manual_seed(0)
    augment = v2.Compose([v2.RandomHorizontalFlip(1.0), v2.RandomAffine(20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                          v2.Resize(96)])
    for _ in range(10):
        image, target = augment(*_sample([(40, 30, 110, 70)]))
        assert isinstance(target, TextDetectionTarget) and isinstance(target.polygons, tv_tensors.KeyPoints)
        assert target.polygons.canvas_size == tuple(image.shape[-2:])
        cx, cy = target.polygons[0].mean(0).round().long().tolist()
        assert image[0, cy, cx] > 200  # the quad's centre is still on the white rectangle
        assert target.texts == ["w0"] and target.ignore.tolist() == [True]


def test_sanitize_drops_offcanvas_and_flags_clipped_regions():
    image, target = _sample([(10, 10, 50, 40), (150, 80, 199, 110), (50, 20, 90, 50)])
    target.ignore[:] = False
    shifted = v2.functional.affine(target.polygons, angle=0.0, translate=[-60, 0], scale=1.0, shear=[0.0, 0.0])
    moved = TextDetectionTarget(shifted, target.texts, target.ignore)
    # After shifting left by 60 px: w0 is gone, w1 is fully inside, w2 straddles x = 0.
    out = SanitizeTextPolygons()(moved)
    assert out.texts == ["w1", "w2"] and out.ignore.tolist() == [False, True]
    assert isinstance(out.polygons, tv_tensors.KeyPoints) and out.polygons.canvas_size == (120, 200)
    assert SanitizeTextPolygons(partial="drop")(moved).texts == ["w1"]
    assert SanitizeTextPolygons(partial="keep")(moved).ignore.tolist() == [False, False]


def test_sanitize_inside_compose_keeps_the_image():
    image, target = _sample([(60, 20, 120, 60), (10, 10, 40, 40)])  # the crop keeps x in [50, 150)
    out_image, out_target = v2.Compose([v2.CenterCrop(100), SanitizeTextPolygons()])(image, target)
    assert out_image.shape[-2:] == (100, 100) and out_target.texts == ["w0"]


def test_sanitize_requires_keypoints():
    target = TextDetectionTarget(torch.zeros(1, 4, 2), ["a"], torch.zeros(1, dtype=torch.bool))
    with pytest.raises(TypeError, match="wrap_sample"):
        SanitizeTextPolygons()(target)


def test_random_text_crop_never_cuts_text():
    torch.manual_seed(0)
    rects = [(10, 10, 50, 30), (70, 50, 130, 70), (150, 90, 190, 110)]
    crop = RandomTextCrop()
    sizes = set()
    for _ in range(50):
        image, target = crop(*_sample(rects))
        height, width = image.shape[-2:]
        sizes.add((height, width))
        p = target.polygons.as_subclass(torch.Tensor)
        x0, x1 = p[..., 0].min(1).values, p[..., 0].max(1).values
        y0, y1 = p[..., 1].min(1).values, p[..., 1].max(1).values
        inside = (x0 >= 0) & (x1 <= width) & (y0 >= 0) & (y1 <= height)
        outside = (x1 <= 0) | (x0 >= width) | (y1 <= 0) | (y0 >= height)
        assert (inside | outside).all() and inside.any()
    assert len(sizes) > 5  # it does crop, and randomly


def test_random_text_crop_keeps_the_whole_image_when_text_is_everywhere():
    image, target = RandomTextCrop()(*_sample([(0, 0, 200, 120)]))
    assert image.shape[-2:] == (120, 200) and torch.equal(target.polygons[0, 2], torch.tensor([200.0, 120.0]))


def test_db_targets_from_an_augmented_sample():
    torch.manual_seed(0)
    augment = v2.Compose([v2.RandomRotation(10), RandomTextCrop(), v2.Resize((160, 160)), SanitizeTextPolygons()])
    image, target = augment(*_sample([(10, 10, 80, 40), (100, 60, 180, 100)]))
    db = MakeDBTargets()(target.polygons.as_subclass(torch.Tensor), target.ignore, tuple(image.shape[-2:]))
    assert db.probability.shape[-2:] == (160, 160) and db.probability.sum() > 0


def test_wrap_sample_checks_the_image_shape():
    target = TextDetectionTarget(torch.zeros(0, 4, 2), [], torch.zeros(0, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        wrap_sample(torch.zeros(1, 3, 8, 8), target)
