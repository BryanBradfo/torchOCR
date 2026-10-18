"""MakeDBTargets: polygons -> (shrunk probability map, border threshold map, mask)."""

import pytest
import torch

from torchocr.transforms import DBTargets, MakeDBTargets


def _box(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_shapes_and_dtypes():
    targets = MakeDBTargets()(torch.tensor([_box(10, 10, 90, 40)]), torch.tensor([False]), (64, 128))
    assert isinstance(targets, DBTargets)
    assert targets.probability.shape == targets.threshold.shape == targets.mask.shape == (1, 64, 128)
    assert targets.probability.dtype == targets.threshold.dtype == torch.float32
    assert targets.mask.dtype == torch.bool


def test_probability_is_the_shrunk_polygon():
    """80x30 box, r=0.4: D = A(1-r^2)/L = 2400*0.84/220 ~= 9.16 px inwards."""
    prob = MakeDBTargets()(torch.tensor([_box(10, 10, 90, 40)]), torch.tensor([False]), (64, 128)).probability[0]
    ys, xs = prob.nonzero(as_tuple=True)
    assert xs.min() == pytest.approx(19, abs=1) and xs.max() == pytest.approx(81, abs=1)
    assert ys.min() == pytest.approx(19, abs=1) and ys.max() == pytest.approx(31, abs=1)
    assert prob.unique().tolist() == [0.0, 1.0]


def test_threshold_band_peaks_on_the_border():
    targets = MakeDBTargets()(torch.tensor([_box(10, 10, 90, 40)]), torch.tensor([False]), (64, 128))
    thresh = targets.threshold[0]
    assert thresh[10, 50] == pytest.approx(0.7, abs=0.02)  # on the top edge
    assert thresh[25, 50] == pytest.approx(0.3, abs=1e-6)  # deep inside: >= D from every edge
    assert thresh[60, 120] == 0.0  # outside the dilated band: unsupervised
    band = thresh[thresh > 0]
    assert band.min() >= 0.3 - 1e-6 and band.max() <= 0.7 + 1e-6


def test_ignored_and_tiny_regions_are_masked_out():
    polys = torch.tensor([_box(10, 10, 60, 30), _box(70, 10, 120, 30), _box(10, 50, 60, 54)])
    ignore = torch.tensor([False, True, False])  # second: "###"; third: only 4 px tall
    targets = MakeDBTargets(min_text_size=8)(polys, ignore, (64, 128))

    assert targets.probability[0, 20, 35] == 1  # kept
    assert not targets.mask[0, 20, 95] and targets.probability[0, 20, 95] == 0  # ignored
    assert not targets.mask[0, 52, 35] and targets.probability[0, 52, 35] == 0  # too small
    assert targets.threshold[0, 10, 95] == 0  # no border supervision for ignored text
    assert targets.mask[0, 60, 100]  # background stays valid


def test_polygons_outside_the_canvas_are_clipped():
    targets = MakeDBTargets()(torch.tensor([_box(-50, 10, 60, 40)]), torch.tensor([False]), (64, 128))
    assert targets.probability.sum() > 0 and torch.isfinite(targets.threshold).all()


def test_no_text():
    targets = MakeDBTargets()(torch.zeros(0, 4, 2), torch.zeros(0, dtype=torch.bool), (32, 32))
    assert targets.probability.sum() == 0 and targets.threshold.sum() == 0 and targets.mask.all()


def test_contract_errors():
    with pytest.raises(ValueError, match=r"\(N, P, 2\)"):
        MakeDBTargets()(torch.zeros(1, 4), torch.zeros(1, dtype=torch.bool), (32, 32))
    with pytest.raises(ValueError, match="ignore"):
        MakeDBTargets()(torch.zeros(1, 4, 2), torch.zeros(2, dtype=torch.bool), (32, 32))
    with pytest.raises(ValueError, match="thresh_min"):
        MakeDBTargets(thresh_min=0.8, thresh_max=0.7)
