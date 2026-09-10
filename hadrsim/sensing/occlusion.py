"""What is in the way.

Two kinds of obstruction, and the difference between them is the paper's central
claim made physical rather than asserted:

    buildings   hard.  A wall blocks a sight line completely.
    smoke       soft.  A plume dims it, by an amount that depends on how much of it
                the ray passes through.

Which is why the high rescuer and the low rescuer see different things.  The high
one looks down through the smoke column above a fire; the low one looks sideways
*under* the canopies and awnings that stop the high one seeing anything at all.  In
the 2D simulator that asymmetry is a lookup table (`see_inside_hlp`); here it falls
out of a ray and some geometry.

Ray marching rather than exact intersection.  Exact is tempting and wrong for this:
smoke has no surface to intersect, so any ray through it has to be integrated
anyway, and once you are stepping along the ray the buildings come free.
"""

from __future__ import annotations

import torch

# Samples along each ray.  More is more faithful and linearly more expensive; at 32
# an eighty-metre sight line is sampled every 2.5 m, which resolves a building but
# would step over a fence.  Nothing in this simulator is fence-sized.
STEPS = 32

# A small nudge off each end, so a ray does not report the ground the observer is
# standing on, or the target's own body, as an obstruction.  It is a *fraction* of
# the ray, so it must stay small: at 0.35 it sampled only the middle third, and a
# drone at twelve metres cheerfully saw through a three-metre building.
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
    """March from `origin` to each `target`, returning what got in the way.

    `origin` is (B, 3), `target` is (B, N, 3).  Returns
    `(blocked, optical_depth)`: whether a solid obstruction stopped the ray, and how
    much smoke it passed through if it did not.

    Buildings are tested against their *height*, not merely their footprint, which is
    the whole reason for doing this in 3D: a ray from eighty metres up clears a
    six-metre building that a ray from three metres up does not.
    """
    cell = terrain.cell_size_m
    n = target.shape[1]
    start = origin[:, None, :].expand(-1, n, -1)

    t = torch.linspace(EPSILON, 1.0 - EPSILON, STEPS, device=target.device)
    points = start[..., None, :] + (target - start)[..., None, :] * t.view(1, 1, -1, 1)

    solid = _sample(terrain.solid_top, points[..., :2], cell)
    hit_wall = points[..., 2] < solid

    # A canopy is a slab with air under it: a ray passing beneath one is unobstructed,
    # which is exactly how the low drone sees under what the high drone cannot.
    z0 = _sample(terrain.canopy_z0, points[..., :2], cell)
    z1 = _sample(terrain.canopy_z1, points[..., :2], cell)
    hit_roof = (z1 > 0) & (points[..., 2] >= z0) & (points[..., 2] <= z1)

    blocked = (hit_wall | hit_roof).any(-1)

    depth = torch.zeros_like(blocked, dtype=torch.float32)
    if smoke is not None:
        depth = _smoke_depth(smoke, points, cell, env)
    return blocked, depth


def _smoke_depth(smoke, points: torch.Tensor, cell: float, env) -> torch.Tensor:
    """Optical depth along each ray: how much plume it passed through.

    Soft rather than binary, because smoke is: a thin veil costs a detection a little
    confidence and a thick column costs it everything, and the policy should be able
    to tell those apart.
    """
    alt = smoke.layer_altitudes
    layer = torch.bucketize(points[..., 2].contiguous(), alt).clamp(0, len(alt) - 1)

    w, h = smoke.density.shape[-2:]
    x = (points[..., 0] / cell).long().clamp(0, w - 1)
    y = (points[..., 1] / cell).long().clamp(0, h - 1)
    idx = torch.zeros_like(x) if env is None else env.view(-1, 1, 1).expand_as(x)

    density = smoke.density[idx, layer, x, y]
    density = torch.where(points[..., 2] < 0, torch.zeros_like(density), density)

    # Mean density along the ray times its length: a proper line integral, so a long
    # slant through a plume costs more than a short one through the same smoke.
    length = (points[..., -1, :] - points[..., 0, :]).norm(dim=-1)
    return density.mean(-1) * length * smoke.p.extinction


def visible(terrain, origin, target, smoke=None, env=None, threshold: float = 2.0):
    """Convenience: can this be seen at all, and how clearly.

    Returns `(can_see, clarity)` with clarity in 0..1 -- one through clear air, and
    falling away exponentially with smoke, which is what an optical depth means.
    """
    blocked, depth = trace(terrain, origin, target, smoke, env)
    clarity = torch.exp(-depth) * (~blocked).float()
    return clarity > torch.exp(torch.tensor(-threshold)), clarity
