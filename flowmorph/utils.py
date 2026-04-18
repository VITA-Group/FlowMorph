"""Small helpers shared by FlowMorph entry points and baselines."""

from __future__ import annotations

import torch
from PIL import Image


def make_image_grid(images, row_width: int = 6) -> Image.Image | None:
    """Tile a list of PIL images into a single grid image."""
    if not images:
        return None

    w, h = images[0].size
    n = len(images)
    per_row = row_width
    rows = (n + per_row - 1) // per_row

    grid = Image.new("RGB", (w * per_row, h * rows), color="black")
    for idx, img in enumerate(images):
        r, c = idx // per_row, idx % per_row
        grid.paste(img, (c * w, r * h))
    return grid


def slerp(a: torch.Tensor, b: torch.Tensor, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation along tensor direction.

    Follows the conventional SLERP formula but falls back to linear
    interpolation when the vectors are close to collinear (small angle),
    which keeps the SLERP numerically stable in the low-dimensional regime.
    """
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()

    a_norm = torch.linalg.norm(a_flat) + eps
    b_norm = torch.linalg.norm(b_flat) + eps

    dot = torch.clamp((a_flat / a_norm) @ (b_flat / b_norm), -1.0, 1.0)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega)

    if sin_omega.abs() < eps:
        return (1 - alpha) * a + alpha * b

    w_a = torch.sin((1 - alpha) * omega) / sin_omega
    w_b = torch.sin(alpha * omega) / sin_omega
    return (w_a * a.float() + w_b * b.float()).to(a.dtype)


def slerp_direction(u_src: torch.Tensor, u_tgt: torch.Tensor, alpha: float) -> torch.Tensor:
    """Paper Eq. (13)-(14): SLERP the direction, then re-scale with a linear
    combination of the source and target norms so magnitude is interpolated
    explicitly rather than being determined by the SLERP formula.
    """
    src_norm = torch.linalg.norm(u_src).clamp_min(1e-8)
    tgt_norm = torch.linalg.norm(u_tgt).clamp_min(1e-8)
    d_hat = slerp(u_src / src_norm, u_tgt / tgt_norm, alpha)
    scale = (1 - alpha) * src_norm + alpha * tgt_norm
    return scale * d_hat
