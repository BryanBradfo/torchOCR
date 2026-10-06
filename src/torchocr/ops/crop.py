"""Perspective rectification of text quadrilaterals: the OCR analog of ``roi_align``."""

import torch
from torch import Tensor
from torch.nn import functional as F


def crop_quads(
    images: Tensor,
    quads: Tensor,
    batch_idx: Tensor,
    height: int,
    max_width: int,
    fill: float = 0.0,
    vertical_ratio: float = 1.5,
) -> tuple[Tensor, Tensor]:
    """Rectify text quadrilaterals into fixed-height, left-aligned crops.

    Each quad is mapped onto a ``height x w_k`` rectangle by the
    homography sending its corners (in reading order: top-left,
    top-right, bottom-right, bottom-left) to the rectangle's corners, and
    sampled bilinearly with :func:`torch.nn.functional.grid_sample`.
    ``w_k`` keeps the quad's aspect ratio (clamped to ``max_width``);
    columns past ``w_k`` are set to ``fill``. Quads whose height is at
    least ``vertical_ratio`` times their width are treated as vertical text
    and rotated 90 degrees counter-clockwise, as PaddleOCR does.

    Everything runs on the input device, is batched across all quads of
    a page, and is differentiable with respect to ``images``. Because
    per-pixel normalization commutes with bilinear sampling, normalizing
    the whole page once and then cropping equals cropping then
    normalizing each crop.

    Args:
        images: ``(B, C, H, W)`` float pages.
        quads: ``(K, 4, 2)`` vertices in continuous pixel coordinates
            (pixel ``j`` spans ``[j, j + 1)``), in reading order -- use
            :func:`order_quad_vertices` for detector output.
        batch_idx: ``(K,)`` page index of each quad.
        height: Output height (e.g. 32 for CRNN).
        max_width: Output width; every crop is padded to it.
        fill: Value of the padding columns.
        vertical_ratio: Height / width ratio from which a quad is rotated.

    Returns:
        ``(crops, widths)``: ``(K, C, height, max_width)`` crops and the
        ``(K,)`` number of valid (non-padding) columns of each.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images of shape (B, C, H, W); got {tuple(images.shape)}.")
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError(f"Expected quads of shape (K, 4, 2); got {tuple(quads.shape)}.")
    if batch_idx.shape != (quads.shape[0],):
        raise ValueError(
            f"batch_idx must have shape ({quads.shape[0]},) to match quads; got {tuple(batch_idx.shape)}."
        )

    num_quads = quads.shape[0]
    channels, page_h, page_w = images.shape[1:]
    crops = images.new_full((num_quads, channels, height, max_width), fill)
    widths = torch.zeros(num_quads, dtype=torch.long, device=images.device)
    if num_quads == 0:
        return crops, widths

    quads = quads.to(device=images.device, dtype=torch.float64)
    edge = lambda a, b: (quads[:, a] - quads[:, b]).norm(dim=-1)  # noqa: E731
    quad_w = torch.maximum(edge(0, 1), edge(3, 2))
    quad_h = torch.maximum(edge(0, 3), edge(1, 2))
    vertical = quad_h >= vertical_ratio * quad_w
    # Rotating the crop 90 degrees CCW makes the old top-right the new top-left.
    quads = torch.where(vertical[:, None, None], quads.roll(-1, dims=1), quads)
    quad_w, quad_h = torch.where(vertical, quad_h, quad_w), torch.where(vertical, quad_w, quad_h)
    widths = (height * quad_w / quad_h.clamp_min(1e-6)).round().clamp(1, max_width).long()

    homographies = _unit_square_to_quads(quads)  # (K, 3, 3)
    # Pixel-centre sampling positions, in unit-square coordinates per crop.
    cols = torch.arange(max_width, device=images.device, dtype=torch.float64) + 0.5
    rows = torch.arange(height, device=images.device, dtype=torch.float64) + 0.5
    u = cols[None, None, :] / widths[:, None, None]
    v = (rows / height)[None, :, None].expand(num_quads, height, max_width)
    uv1 = torch.stack([u.expand_as(v), v, torch.ones_like(v)], dim=-1)  # (K, h, W, 3)
    xyw = torch.einsum("kij,khwj->khwi", homographies, uv1)
    xy = xyw[..., :2] / xyw[..., 2:]
    # align_corners=False: continuous coordinate x maps to 2x/W - 1.
    scale = torch.tensor([2.0 / page_w, 2.0 / page_h], device=images.device, dtype=torch.float64)
    grid = (xy * scale - 1).to(images.dtype)

    valid = (cols[None, :] < widths[:, None])[:, None, None, :]  # (K, 1, 1, W)
    for page in batch_idx.unique().tolist():
        members = (batch_idx == page).nonzero().squeeze(1)
        # One grid_sample per page: stack its crops' grids vertically.
        page_grid = grid[members].reshape(1, -1, max_width, 2)
        sampled = F.grid_sample(
            images[page : page + 1], page_grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        sampled = sampled.reshape(channels, len(members), height, max_width).transpose(0, 1)
        crops[members] = torch.where(valid[members], sampled, crops[members])
    return crops, widths


def _unit_square_to_quads(quads: Tensor) -> Tensor:
    """Homographies mapping (0,0), (1,0), (1,1), (0,1) to each quad's corners."""
    u = quads.new_tensor([0.0, 1.0, 1.0, 0.0]).expand(quads.shape[0], 4)
    v = quads.new_tensor([0.0, 0.0, 1.0, 1.0]).expand(quads.shape[0], 4)
    x, y = quads[..., 0], quads[..., 1]
    zeros, ones = torch.zeros_like(u), torch.ones_like(u)
    rows_x = torch.stack([u, v, ones, zeros, zeros, zeros, -u * x, -v * x], dim=-1)
    rows_y = torch.stack([zeros, zeros, zeros, u, v, ones, -u * y, -v * y], dim=-1)
    system = torch.cat([rows_x, rows_y], dim=1)  # (K, 8, 8)
    target = torch.cat([x, y], dim=1)  # (K, 8)
    # pinv rather than solve: degenerate (zero-area) quads must not raise.
    h = (torch.linalg.pinv(system) @ target.unsqueeze(-1)).squeeze(-1)
    return torch.cat([h, torch.ones_like(h[:, :1])], dim=1).reshape(-1, 3, 3)
