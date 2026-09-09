"""Which way is out.

The panic model needs a "sensible" direction to blend against -- the route a calm
person would take.  The 2D simulator recomputes A* for that, per person, per step.
That is fine for one person on a 25 x 25 grid and hopeless for thousands of parallel
environments.

Instead, solve it once per episode for the whole map: a distance field holding, for
every cell, how far it is to the exit.  Then the sensible direction at any position
is just downhill, and reading it costs one lookup no matter how many people ask.

Computed by relaxation rather than by a queue.  Repeatedly taking each cell's
distance as the best of its neighbours' plus the step between them converges on the
true shortest path, and every cell can be updated at once -- which is what makes it
run on a GPU across a whole batch.  A queue-based Dijkstra is asymptotically better
and entirely sequential, so it loses badly here.

Eight-connected, with diagonals costing sqrt(2), so paths come out at sensible angles
instead of only in the four compass directions.
"""

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
) -> torch.Tensor:
    """(B, W, H) metres to the goal, by the shortest walkable path.

    `walkable` is (W, H) and shared -- every environment walks the same city.
    `goal_xy` is (B, 2) in cells. Unreachable cells come back as UNREACHABLE, not inf,
    so that arithmetic on the field does not produce NaN.
    """
    w, h = walkable.shape
    b = len(goal_xy)
    device = walkable.device
    step = cell_size_m

    dist = torch.full((b, w, h), UNREACHABLE, device=device)
    idx = torch.arange(b, device=device)
    dist[idx, goal_xy[:, 0].long(), goal_xy[:, 1].long()] = 0.0

    blocked = ~walkable
    dist = dist.masked_fill(blocked[None], UNREACHABLE)

    # A shortest path never needs more steps than the map's diagonal, and relaxing in
    # both sweep directions each round roughly halves the rounds needed.
    if max_iters is None:
        max_iters = w + h

    for _ in range(max_iters):
        prev = dist
        best = dist
        for dx, dy, cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + cost * step)
        dist = best.masked_fill(blocked[None], UNREACHABLE).clamp_max(UNREACHABLE)
        if torch.equal(dist, prev):
            break

    return dist


def clearance(walkable: torch.Tensor, cell_size_m: float = 1.0) -> torch.Tensor:
    """(W, H) metres from each cell to the nearest wall.

    Static, so computed once per map and shared by every environment.  The social
    force model needs it to push people away from walls, and its gradient is the
    direction to push them.
    """
    w, h = walkable.shape
    dist = torch.where(
        walkable,
        torch.full((w, h), UNREACHABLE, device=walkable.device),
        torch.zeros((w, h), device=walkable.device),
    )[None]

    for _ in range(w + h):
        prev = dist
        best = dist
        for dx, dy, cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + cost * cell_size_m)
        dist = best.clamp_max(UNREACHABLE)
        if torch.equal(dist, prev):
            break
    return dist[0]


def flow(dist: torch.Tensor) -> torch.Tensor:
    """(B, W, H, 2) unit vectors pointing the way out.

    Downhill on the distance field, in cell coordinates.  Zero only where there is
    genuinely nowhere to go: at the goal, or somewhere walled off entirely.

    Two things this has to get right, and both are easy to get wrong:

      - **A wall is not "infinitely far away".**  Substituting UNREACHABLE into the
        difference makes the gradient enormous and points everyone at the nearest
        wall.  Treating a blocked neighbour as *no better than here* instead makes it
        contribute nothing, so the flow simply follows the directions that are open.
      - **Downhill, not uphill.**  `_shift(d, +1)` returns the neighbour at -x, so
        `_shift(d, 1) - _shift(d, -1)` is already `d[x-1] - d[x+1]`, which points the
        way the distance decreases.  Negating it sends everyone away from the exit.
    """
    def neighbour(dx: int, dy: int) -> torch.Tensor:
        s = _shift(dist, dx, dy)
        return torch.where(s >= UNREACHABLE / 2, dist, s)

    v = torch.stack([
        neighbour(1, 0) - neighbour(-1, 0),
        neighbour(0, 1) - neighbour(0, -1),
    ], dim=-1)
    v = torch.where(dist[..., None] >= UNREACHABLE / 2, torch.zeros_like(v), v)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def sample(field: torch.Tensor, xy_m: torch.Tensor, cell_size_m: float) -> torch.Tensor:
    """Read a (B, W, H[, C]) field at continuous positions (B, 2) given in metres.

    Nearest cell rather than bilinear: the fields this reads are already smooth at the
    scale a person moves in one step, and clamping keeps someone who wanders off the
    map from indexing out of bounds.
    """
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
