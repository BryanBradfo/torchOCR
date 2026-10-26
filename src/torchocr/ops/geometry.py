"""Polygon and quadrilateral geometry on batched tensors."""

import torch
from torch import Tensor


def _check_points(polys: Tensor, name: str, vertices: int | None = None) -> None:
    if polys.ndim < 2 or polys.shape[-1] != 2 or (vertices is not None and polys.shape[-2] != vertices):
        expected = f"(..., {vertices}, 2)" if vertices else "(..., P, 2)"
        raise ValueError(f"Expected {name} of shape {expected}; got {tuple(polys.shape)}.")


def polygon_area(polys: Tensor) -> Tensor:
    """Unsigned area of simple polygons via the shoelace formula.

    Args:
        polys: ``(..., P, 2)`` vertices in either winding order.

    Returns:
        ``(...)`` areas.
    """
    _check_points(polys, "polys")
    x, y = polys.unbind(-1)
    cross = x * y.roll(-1, dims=-1) - y * x.roll(-1, dims=-1)
    return 0.5 * cross.sum(-1).abs()


def quad_to_box(quads: Tensor) -> Tensor:
    """Axis-aligned ``(x1, y1, x2, y2)`` hull of ``(..., 4, 2)`` quads."""
    _check_points(quads, "quads", vertices=4)
    return torch.cat([quads.amin(-2), quads.amax(-2)], dim=-1)


def order_quad_vertices(quads: Tensor) -> Tensor:
    """Reorder quad vertices to reading order: top-left, top-right, bottom-right, bottom-left.

    Accepts any starting vertex and either winding (e.g. the output of
    :func:`cv2.boxPoints`). Vertices are first sorted clockwise around the
    centroid; the start is the upper of the two leftmost vertices, which is
    the reading-order top-left for text rotated by less than 45 degrees.

    Args:
        quads: ``(..., 4, 2)`` vertices in ``(x, y)`` image coordinates.

    Returns:
        ``(..., 4, 2)`` reordered vertices.
    """
    _check_points(quads, "quads", vertices=4)
    centered = quads - quads.mean(-2, keepdim=True)
    # With y pointing down, increasing atan2 sweeps clockwise on screen.
    clockwise = torch.atan2(centered[..., 1], centered[..., 0]).argsort(-1)
    quads = quads.gather(-2, clockwise.unsqueeze(-1).expand_as(quads))

    leftmost = quads[..., 0].argsort(-1)[..., :2]
    left_y = quads[..., 1].gather(-1, leftmost)
    start = leftmost.gather(-1, left_y.argmin(-1, keepdim=True))
    order = (torch.arange(4, device=quads.device) + start) % 4
    return quads.gather(-2, order.unsqueeze(-1).expand_as(quads))


def reading_order(quads: Tensor, min_overlap: float = 0.5) -> Tensor:
    """Permutation that sorts text regions in reading order: lines top to bottom, words left to right.

    Regions are taken by increasing vertical centre; each joins the line
    whose most recently added region overlaps it vertically by at least
    ``min_overlap`` of the smaller height, else starts a new line.
    Chaining through the latest region follows slightly slanted lines.
    Multi-column layouts are read across columns, line by line.

    Args:
        quads: ``(K, P, 2)`` region vertices (any ``P``) in image coordinates.
        min_overlap: Vertical overlap, as a fraction of the smaller region
            height, needed to put two regions on the same line.

    Returns:
        ``(K,)`` long indices: ``quads[reading_order(quads)]`` is sorted.
    """
    _check_points(quads, "quads")
    if quads.ndim != 3:
        raise ValueError(f"Expected quads of shape (K, P, 2); got {tuple(quads.shape)}.")
    top, bottom = quads[..., 1].amin(-1).tolist(), quads[..., 1].amax(-1).tolist()
    left = quads[..., 0].amin(-1).tolist()
    centre = [(t + b) / 2 for t, b in zip(top, bottom)]
    lines: list[list[int]] = []
    for i in sorted(range(len(centre)), key=centre.__getitem__):
        best, best_overlap = None, min_overlap
        for line in lines:
            j = line[-1]
            shared = min(bottom[i], bottom[j]) - max(top[i], top[j])
            overlap = shared / max(min(bottom[i] - top[i], bottom[j] - top[j]), 1e-6)
            if overlap >= best_overlap:
                best, best_overlap = line, overlap
        if best is None:
            lines.append([i])
        else:
            best.append(i)
    lines.sort(key=lambda line: sum(centre[i] for i in line) / len(line))
    order = [i for line in lines for i in sorted(line, key=left.__getitem__)]
    return torch.tensor(order, dtype=torch.long, device=quads.device)
