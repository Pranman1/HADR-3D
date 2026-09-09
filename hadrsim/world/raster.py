"""Turning shapes into grid cells.

Nothing here knows about buildings, fire, or OpenStreetMap -- it takes polygons and
polylines in metres and says which cells they cover.  Kept separate so any future
map source (a CAD floorplan, a procedural city, a hand-drawn scenario) rasterises
the same way, and so the arithmetic can be tested against exact areas.

Both functions return `(slice_pair, mask)` covering only the shape's own bounding
box rather than a full-grid mask.  On a 600 x 600 grid with a few hundred buildings
that is the difference between a fraction of a second and half a minute.
"""

from __future__ import annotations

import numpy as np

Hit = tuple[tuple[slice, slice], np.ndarray] | None


def _window(lo: np.ndarray, hi: np.ndarray, shape: tuple[int, int]):
    """Clip a shape's cell-space bounding box to the grid. None if it misses."""
    w, h = shape
    x0, y0 = np.clip(lo, 0, [w, h])
    x1, y1 = np.clip(hi, 0, [w, h])
    if x1 <= x0 or y1 <= y0:
        return None
    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    return (slice(int(x0), int(x1)), slice(int(y0), int(y1))), xs, ys


def fill_polygon(poly: np.ndarray, shape: tuple[int, int], cell: float) -> Hit:
    """Cells whose centre lies inside `poly`.

    Cell-centre sampling, which is unbiased: the area it reports converges on the
    polygon's true area as the grid gets finer, rather than systematically over- or
    under-covering the way "any overlap" or "fully contained" would.
    """
    from matplotlib.path import Path as MplPath

    win = _window(
        np.floor(poly.min(0) / cell).astype(int) - 1,
        np.ceil(poly.max(0) / cell).astype(int) + 1,
        shape,
    )
    if win is None:
        return None
    sl, xs, ys = win
    gx, gy = np.meshgrid((xs + 0.5) * cell, (ys + 0.5) * cell, indexing="ij")
    inside = MplPath(poly).contains_points(np.stack([gx.ravel(), gy.ravel()], -1))
    return sl, inside.reshape(gx.shape)


def stroke_line(line: np.ndarray, width_m: float, shape: tuple[int, int], cell: float) -> Hit:
    """Cells within width/2 of a polyline -- a capsule, with rounded ends.

    Round ends are deliberate: they let roads meeting at a junction join cleanly
    instead of leaving an unburnt notch at every corner.  It does mean the area is
    `length * width + pi * r^2`, not `length * width`.
    """
    r = width_m / 2.0
    win = _window(
        np.floor((line.min(0) - r) / cell).astype(int),
        np.ceil((line.max(0) + r) / cell).astype(int),
        shape,
    )
    if win is None:
        return None
    sl, xs, ys = win

    p = np.stack(np.meshgrid((xs + 0.5) * cell, (ys + 0.5) * cell, indexing="ij"), -1)
    p = p[..., None, :]                                   # (X, Y, 1, 2)
    a, b = line[:-1], line[1:]                            # (S, 2)
    ab = b - a
    denom = (ab * ab).sum(-1)
    denom[denom == 0] = 1e-9                              # a zero-length segment
    t = (((p - a) * ab).sum(-1) / denom).clip(0.0, 1.0)   # (X, Y, S)
    dist = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1).min(-1)
    return sl, dist <= r
