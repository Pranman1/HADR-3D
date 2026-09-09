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


def extrude(poly: np.ndarray, z0: float, z1: float, uv: bool = False):
    """A closed prism from a footprint. Returns (vertices (V,3), faces (F,3)[, uv]).

    With `uv=True` each wall gets its own four vertices rather than sharing the ring,
    so wall and roof texture coordinates do not fight over the same vertex.  The
    horizontal coordinate accumulates around the perimeter, which keeps window
    columns marching continuously around corners instead of restarting at each one.
    """
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]                              # OSM rings repeat the first point
    n = len(poly)
    if n < 3:
        empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        return (*empty, np.zeros((0, 2), np.float32)) if uv else empty

    if not uv:
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

    from hadrsim.viz.facade import ATLAS_CELLS, BAY_M, FLOOR_M, roof_uv

    v_top = (z1 - z0) / FLOOR_M / ATLAS_CELLS
    run = 0.0
    verts, faces, uvs = [], [], []
    for i in range(n):
        j = (i + 1) % n
        a, b = poly[i], poly[j]
        u0 = run / BAY_M / ATLAS_CELLS
        run += float(np.hypot(*(b - a)))
        u1 = run / BAY_M / ATLAS_CELLS

        k = len(verts)
        verts += [[*a, z0], [*b, z0], [*b, z1], [*a, z1]]
        uvs += [[u0, v_top], [u1, v_top], [u1, 0.0], [u0, 0.0]]
        faces += [[k, k + 1, k + 2], [k, k + 2, k + 3]]

    # Roof and floor get their own ring, aimed at a plain patch of the atlas.
    cap = earclip(poly)
    if len(cap):
        base = len(verts)
        verts += [[*p, z1] for p in poly] + [[*p, z0] for p in poly]
        uvs += [list(roof_uv())] * (2 * n)
        faces += (cap + base).tolist() + (cap[:, ::-1] + base + n).tolist()

    return (
        np.array(verts, np.float32),
        np.array(faces, np.int32),
        np.array(uvs, np.float32),
    )


def merge(pieces) -> tuple[np.ndarray, ...]:
    """Concatenate many meshes into one, so the viewer gets a single object.

    Accepts (verts, faces) or (verts, faces, uv); every piece must be the same shape.
    """
    pieces = [p for p in pieces if len(p[0])]
    width = len(pieces[0]) if pieces else 2
    if not pieces:
        blank = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
        return (*blank, np.zeros((0, 2), np.float32)) if width == 3 else blank

    vs, fs, uvs, offset = [], [], [], 0
    for piece in pieces:
        vs.append(piece[0])
        fs.append(piece[1] + offset)
        if width == 3:
            uvs.append(piece[2])
        offset += len(piece[0])
    out = (np.concatenate(vs), np.concatenate(fs))
    return (*out, np.concatenate(uvs)) if width == 3 else out
