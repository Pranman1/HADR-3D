"""Grid navigation fields: cost-to-go, clearance from walls, proximity to hot cells, and
the flow (unit vectors toward the goal).  All batched over environments."""

from __future__ import annotations

import torch

NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5),
)

UNREACHABLE = 1e9


def distance_field(
    walkable: torch.Tensor,
    goal_xy: torch.Tensor,
    cell_size_m: float = 1.0,
    max_iters: int | None = None,
    cost: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, W, H) cost-to-go to the goal by the cheapest walkable path."""
    w, h = walkable.shape[-2:]
    b = len(goal_xy)
    device = walkable.device
    step = cell_size_m

    dist = torch.full((b, w, h), UNREACHABLE, device=device)
    idx = torch.arange(b, device=device)
    dist[idx, goal_xy[:, 0].long(), goal_xy[:, 1].long()] = 0.0

    blocked = ~walkable
    if blocked.dim() == 2:
        blocked = blocked[None].expand(b, w, h)
    dist = dist.masked_fill(blocked, UNREACHABLE)

    if max_iters is None:
        max_iters = w + h

    price = step if cost is None else step * cost.clamp_min(1e-3)

    for _ in range(max_iters):
        prev = dist
        best = dist
        for dx, dy, step_cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + step_cost * price)
        dist = best.masked_fill(blocked, UNREACHABLE).clamp_max(UNREACHABLE)
        if torch.equal(dist, prev):
            break

    return dist


def clearance(walkable: torch.Tensor, cell_size_m: float = 1.0) -> torch.Tensor:
    """(W, H) metres from each cell to the nearest wall."""
    w, h = walkable.shape
    dist = torch.where(
        walkable,
        torch.full((w, h), UNREACHABLE, device=walkable.device),
        torch.zeros((w, h), device=walkable.device),
    )[None]

    for _ in range(w + h):
        prev = dist
        best = dist
        for dx, dy, step_cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + step_cost * cell_size_m)
        dist = best.clamp_max(UNREACHABLE)
        if torch.equal(dist, prev):
            break
    return dist[0]


def proximity(field: torch.Tensor, radius_m: float, cell_size_m: float = 1.0,
              levels: int = 3) -> torch.Tensor:
    """(B, W, H) nearness to the nearest hot cell, falling off to 0 within `radius_m`."""
    import torch.nn.functional as Fn

    want = max(1, int(round(radius_m / cell_size_m)))
    x = field[:, None]
    out = torch.zeros_like(x)

    reach, dilation, rings = 0, 1, []
    while reach < want:
        dilation = min(dilation, want - reach)
        x = Fn.max_pool2d(Fn.pad(x, (dilation,) * 4), 3, stride=1, dilation=dilation)
        reach += dilation
        dilation *= 2
        rings.append((reach, x))

    # Weight each ring by how far it reached, so nearby fire dominates distant fire.
    for reach, ring in rings[-levels:] if levels else rings:
        out = torch.maximum(out, ring * max(0.0, 1.0 - reach / (want + 1e-6)) ** 0.5)
    out = torch.maximum(out, rings[0][1] if rings else x)
    return out[:, 0]


def flow(dist: torch.Tensor) -> torch.Tensor:
    """(B, W, H, 2) unit vectors pointing the way out."""
    offsets = torch.tensor([[dx, dy] for dx, dy, _ in NEIGHBOURS],
                           dtype=dist.dtype, device=dist.device)
    step_costs = torch.tensor([c for _, _, c in NEIGHBOURS],
                              dtype=dist.dtype, device=dist.device)

    # `_shift(d, -dx, -dy)` reads the neighbour lying at +(dx, dy) from each cell.
    neighbours = torch.stack(
        [_shift(dist, -dx, -dy) for (dx, dy, _) in NEIGHBOURS], dim=0
    )                                                          # (8, B, W, H)
    best = (neighbours + step_costs[:, None, None, None]).argmin(dim=0)
    v = offsets[best]                                          # (B, W, H, 2)

    stuck = (neighbours.amin(dim=0) >= dist) | (dist >= UNREACHABLE / 2)
    v = torch.where(stuck[..., None], torch.zeros_like(v), v)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def sample(field: torch.Tensor, xy_m: torch.Tensor, cell_size_m: float) -> torch.Tensor:
    """Read a (B, W, H[, C]) field at continuous positions (B, 2) in metres."""
    w, h = field.shape[1], field.shape[2]
    cells = (xy_m / cell_size_m).long()
    x = cells[:, 0].clamp(0, w - 1)
    y = cells[:, 1].clamp(0, h - 1)
    return field[torch.arange(len(xy_m), device=field.device), x, y]


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate a (B, W, H) field by whole cells, padding with UNREACHABLE."""
    out = torch.full_like(x, UNREACHABLE)
    w, h = x.shape[-2:]
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    out[..., sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy] = x[..., sx0:sx1, sy0:sy1]
    return out
