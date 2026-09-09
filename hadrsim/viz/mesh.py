"""Turning building outlines into triangles, so a real city can be drawn as one.

Nothing in the simulation uses this -- occlusion, fire and movement all read the
grid.  This exists purely so the viewer shows buildings instead of a hundred
thousand one-metre cubes, which is both unreadable and slow.

Roofs need real triangulation because OSM footprints are frequently concave: an
L-shaped block fanned from its centroid produces triangles lying outside the
building.  Ear clipping is the standard fix and is short enough to keep here rather
than take on a geometry dependency.
"""

from __future__ import annotations

import numpy as np


def _area2(poly: np.ndarray) -> float:
    """Twice the signed area. Positive means counter-clockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(p, a, b, c) -> np.ndarray:
    """Barycentric point-in-triangle, vectorised over p."""
    v0, v1, v2 = c - a, b - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.zeros(len(p), dtype=bool)
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return (u >= 0) & (v >= 0) & (u + v <= 1)


def earclip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple polygon. Returns (T, 3) indices into `poly`.

    Handles concave outlines. Assumes no self-intersections and no holes, which is
    true of essentially every OSM building outline.
    """
    n = len(poly)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)
    idx = list(range(n))
    if _area2(poly) < 0:            # work counter-clockwise
        idx.reverse()

    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * n:
        guard += 1
        for k in range(len(idx)):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i], poly[j], poly[l]
            # 2D cross product. Written out because numpy 2 deprecated np.cross here.
            turn = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if turn <= 0:                            # reflex corner, not an ear
                continue
            others = [m for m in idx if m not in (i, j, l)]
            if others and _inside(poly[others], a, b, c).any():
                continue                              # something is inside the ear
            tris.append((i, j, l))
            idx.pop(k)
            break
        else:
            break                                     # degenerate; take what we have
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def extrude(poly: np.ndarray, z0: float, z1: float):
    """A closed prism from a footprint. Returns (vertices (V,3), faces (F,3))."""
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]                              # OSM rings repeat the first point
    n = len(poly)
    if n < 3:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    verts = np.concatenate([
        np.column_stack([poly, np.full(n, z0)]),
        np.column_stack([poly, np.full(n, z1)]),
    ]).astype(np.float32)

    walls = []
    for i in range(n):
        j = (i + 1) % n
        walls += [(i, j, j + n), (i, j + n, i + n)]

    cap = earclip(poly)
    faces = np.array(walls, dtype=np.int32)
    if len(cap):
        faces = np.concatenate([faces, cap + n, cap[:, ::-1]])   # roof and floor
    return verts, faces


def merge(pieces) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate many (verts, faces) into one mesh, so the viewer gets one object."""
    vs, fs, offset = [], [], 0
    for v, f in pieces:
        if len(v) == 0:
            continue
        vs.append(v)
        fs.append(f + offset)
        offset += len(v)
    if not vs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.concatenate(vs), np.concatenate(fs)
