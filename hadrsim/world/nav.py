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
    cost: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, W, H) cost-to-go to the goal, by the cheapest walkable path.

    `walkable` is (W, H) and shared -- every environment walks the same city.
    `goal_xy` is (B, 2) in cells. Unreachable cells come back as UNREACHABLE, not inf,
    so that arithmetic on the field does not produce NaN.

    `cost` is an optional (B, W, H) multiplier on the price of entering each cell.
    One means ordinary ground; larger means somewhere you would rather not go, which
    is how a route bends around a fire instead of walking into it.  With no cost the
    result is plain distance in metres.
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

    price = step if cost is None else step * cost.clamp_min(1e-3)

    for _ in range(max_iters):
        prev = dist
        best = dist
        for dx, dy, step_cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + step_cost * price)
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
        for dx, dy, step_cost in NEIGHBOURS:
            best = torch.minimum(best, _shift(dist, dx, dy) + step_cost * cell_size_m)
        dist = best.clamp_max(UNREACHABLE)
        if torch.equal(dist, prev):
            break
    return dist[0]


def proximity(field: torch.Tensor, radius_m: float, cell_size_m: float = 1.0,
              levels: int = 3) -> torch.Tensor:
    """(B, W, H) how near the nearest hot thing is, 0..max(field), falling off to 0.

    "How close is the fire" is a *max* over what is nearby, not a sum of it: blurring
    measures how much fire there is, so on a small map an extended fire reads as
    maximal everywhere.  A max-pool of the right radius says the right thing.

    Doing it with one big kernel is unusably slow -- a 57x57 pool over a 300x300 map
    is three hundred million operations, and at ten times a second that is the whole
    frame budget.  Chaining small dilated pools reaches just as far for a hundredth
    of the work: dilations 1, 2, 4, 8 ... each add their own span, so a handful of
    3x3 pools cover thirty metres.
    """
    import torch.nn.functional as Fn

    want = max(1, int(round(radius_m / cell_size_m)))
    x = field[:, None]
    out = torch.zeros_like(x)

    reach, dilation, rings = 0, 1, []
    while reach < want:
        dilation = min(dilation, want - reach)
        # Padded by hand: max_pool2d caps padding at half the kernel however far the
        # dilation reaches, and zero is a safe pad for a max over a non-negative field.
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
    """(B, W, H, 2) unit vectors pointing the way out.

    **Pick the best neighbour, do not difference the field.**  A central difference
    is fine in the open and wrong wherever it matters.  Put an obstacle -- a
    building, or a fire the route is priced away from -- directly between someone and
    their goal, and the x-difference says "the way ahead is expensive, go back" while
    the y-difference is zero by symmetry, so the flow points them straight backwards
    instead of around.  The route was there; the reader of it was not.

    Taking the cheapest of the eight neighbours, one step of lookahead including what
    the step itself costs, is exact for a field built this way and degrades sensibly
    at walls, since a blocked neighbour is simply never the cheapest.

    Zero only where there is genuinely nowhere to go: at the goal, or walled off.
    """
    offsets = torch.tensor([[dx, dy] for dx, dy, _ in NEIGHBOURS],
                           dtype=dist.dtype, device=dist.device)
    step_costs = torch.tensor([c for _, _, c in NEIGHBOURS],
                              dtype=dist.dtype, device=dist.device)

    # `_shift(d, -dx, -dy)` reads the neighbour lying at +(dx, dy) from each cell.
    neighbours = torch.stack(
        [_shift(dist, -dx, -dy) for (dx, dy, _) in NEIGHBOURS], dim=0
    )                                                          # (8, B, W, H)
    # Step cost breaks ties toward the straight neighbours; it must not enter the
    # "is anywhere better" test, because on a converged field the winning neighbour
    # plus its step cost *equals* this cell's own value by construction -- so testing
    # against that marks every cell in the map as stuck.
    best = (neighbours + step_costs[:, None, None, None]).argmin(dim=0)
    v = offsets[best]                                          # (B, W, H, 2)

    stuck = (neighbours.amin(dim=0) >= dist) | (dist >= UNREACHABLE / 2)
    v = torch.where(stuck[..., None], torch.zeros_like(v), v)
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
