"""Polygon triangulation and extrusion.

Drawing-only code, but worth testing: a fan from the centroid silently produces
triangles outside any concave building, and an L-shaped block is the commonest
shape in a real city.
"""

import numpy as np
import pytest

from hadrsim.viz import mesh

SQUARE = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float)          # area 16
L_SHAPE = np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], float)   # area 12
COMB = np.array([                                                    # strongly concave
    [0, 0], [6, 0], [6, 4], [5, 4], [5, 1], [4, 1],
    [4, 4], [3, 4], [3, 1], [2, 1], [2, 4], [0, 4],
], float)


def _tri_area(poly, tris):
    a, b, c = poly[tris[:, 0]], poly[tris[:, 1]], poly[tris[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return float(np.abs(cross).sum() / 2)


@pytest.mark.parametrize("poly,area,n_tris", [
    (SQUARE, 16.0, 2),
    (L_SHAPE, 12.0, 4),
    (COMB, 18.0, 10),
])
def test_triangles_tile_the_polygon_exactly(poly, area, n_tris):
    """n-gon -> n-2 triangles, and they must cover the polygon and nothing else."""
    tris = mesh.earclip(poly)
    assert len(tris) == n_tris == len(poly) - 2
    assert _tri_area(poly, tris) == pytest.approx(area)


@pytest.mark.parametrize("poly,area", [(SQUARE, 16.0), (L_SHAPE, 12.0), (COMB, 18.0)])
def test_winding_order_does_not_matter(poly, area):
    """OSM outlines come in both directions; both must triangulate the same."""
    flipped = poly[::-1].copy()
    assert _tri_area(flipped, mesh.earclip(flipped)) == pytest.approx(area)


def test_degenerate_input_is_survivable():
    assert len(mesh.earclip(np.zeros((0, 2)))) == 0
    assert len(mesh.earclip(np.array([[0.0, 0.0], [1.0, 1.0]]))) == 0
    v, f = mesh.extrude(np.array([[0.0, 0.0], [1.0, 1.0]]), 0, 5)
    assert len(v) == 0 and len(f) == 0


def test_extrusion_is_a_closed_prism():
    v, f = mesh.extrude(L_SHAPE, 2.0, 10.0)
    n = len(L_SHAPE)
    assert len(v) == 2 * n                       # one ring at the bottom, one on top
    assert len(f) == 2 * n + 2 * (n - 2)         # walls, plus a roof and a floor
    assert v[:, 2].min() == 2.0 and v[:, 2].max() == 10.0
    assert f.max() < len(v)                      # no index runs off the end

    # every edge of a closed surface is shared by exactly two triangles
    edges = {}
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[frozenset((int(a), int(b)))] = edges.get(frozenset((int(a), int(b))), 0) + 1
    assert all(c == 2 for c in edges.values()), "the prism has a hole in it"


def test_closed_rings_are_not_double_counted():
    """OSM repeats the first point at the end; extruding must not add a stray vertex."""
    closed = np.vstack([SQUARE, SQUARE[:1]])
    v_open, _ = mesh.extrude(SQUARE, 0, 5)
    v_closed, _ = mesh.extrude(closed, 0, 5)
    assert len(v_open) == len(v_closed) == 8


def test_merge_offsets_indices():
    a = mesh.extrude(SQUARE, 0, 5)
    b = mesh.extrude(L_SHAPE, 0, 9)
    v, f = mesh.merge([a, b])
    assert len(v) == len(a[0]) + len(b[0])
    assert len(f) == len(a[1]) + len(b[1])
    assert f.max() < len(v)
    assert f[len(a[1]):].min() >= len(a[0])      # the second mesh's faces were shifted
    assert mesh.merge([])[0].shape == (0, 3)
