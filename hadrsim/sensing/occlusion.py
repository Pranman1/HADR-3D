"""Line of sight by ray marching: buildings block, smoke dims by optical depth."""

from __future__ import annotations

import torch

STEPS = 32

EPSILON = 0.02


def _sample(field: torch.Tensor, xy: torch.Tensor, cell: float) -> torch.Tensor:
    """Read a (W, H) or (B, W, H) field at world positions (..., 2), clamped."""
    w, h = field.shape[-2:]
    x = (xy[..., 0] / cell).long().clamp(0, w - 1)
    y = (xy[..., 1] / cell).long().clamp(0, h - 1)
    if field.dim() == 2:
        return field[x, y]
    b = torch.arange(field.shape[0], device=field.device)
    shape = [field.shape[0]] + [1] * (x.dim() - 1)
    return field[b.view(shape).expand_as(x), x, y]


def trace(terrain, origin: torch.Tensor, target: torch.Tensor,
          smoke=None, env: torch.Tensor | None = None):
    """March from `origin` (B, 3) to each `target` (B, N, 3): (blocked, smoke optical depth)."""
    cell = terrain.cell_size_m
    n = target.shape[1]
    start = origin[:, None, :].expand(-1, n, -1)

    t = torch.linspace(EPSILON, 1.0 - EPSILON, STEPS, device=target.device)
    points = start[..., None, :] + (target - start)[..., None, :] * t.view(1, 1, -1, 1)

    solid = _sample(terrain.solid_top, points[..., :2], cell)
    hit_wall = points[..., 2] < solid

    z0 = _sample(terrain.canopy_z0, points[..., :2], cell)
    z1 = _sample(terrain.canopy_z1, points[..., :2], cell)
    hit_roof = (z1 > 0) & (points[..., 2] >= z0) & (points[..., 2] <= z1)

    blocked = (hit_wall | hit_roof).any(-1)

    depth = torch.zeros_like(blocked, dtype=torch.float32)
    if smoke is not None:
        depth = _smoke_depth(smoke, points, cell, env)
    return blocked, depth


def _smoke_depth(smoke, points: torch.Tensor, cell: float, env) -> torch.Tensor:
    """Optical depth along each ray: mean density times length times extinction."""
    alt = smoke.layer_altitudes
    layer = torch.bucketize(points[..., 2].contiguous(), alt).clamp(0, len(alt) - 1)

    w, h = smoke.density.shape[-2:]
    x = (points[..., 0] / cell).long().clamp(0, w - 1)
    y = (points[..., 1] / cell).long().clamp(0, h - 1)
    idx = torch.zeros_like(x) if env is None else env.view(-1, 1, 1).expand_as(x)

    density = smoke.density[idx, layer, x, y]
    density = torch.where(points[..., 2] < 0, torch.zeros_like(density), density)

    length = (points[..., -1, :] - points[..., 0, :]).norm(dim=-1)
    return density.mean(-1) * length * smoke.p.extinction


def visible(terrain, origin, target, smoke=None, env=None, threshold: float = 2.0):
    """(can_see, clarity in 0..1)."""
    blocked, depth = trace(terrain, origin, target, smoke, env)
    clarity = torch.exp(-depth) * (~blocked).float()
    return clarity > torch.exp(torch.tensor(-threshold)), clarity
