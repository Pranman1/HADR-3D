"""The OpenStreetMap loader, checked against real downtown Berkeley data.

Runs entirely offline: the Overpass response is cached under maps/osm_cache/ and
committed, so these tests are reproducible and never touch the network.  If the
cache is missing they skip rather than silently fetching.

The load-bearing test is `test_rasterised_area_matches_polygons`.  Everything else
in the world layer -- occlusion, fire spread, where the human may walk -- reads the
grid, not the polygons, so if rasterising is wrong then every later number is wrong
in a way no downstream test would catch.
"""

import numpy as np
import pytest
import torch

from hadrsim.world import osm, raster

# Downtown Berkeley, California -- Nominatim centre 37.8701007, -122.268138, r=250 m.
# Hardcoded rather than geocoded so the test needs no network.
BERKELEY = osm.BBox(
    south=37.86785492206253,
    west=-122.27098290122997,
    north=37.87234647793748,
    east=-122.26529309877002,
)

pytestmark = pytest.mark.skipif(
    not (osm.Path("maps/osm_cache") / f"{BERKELEY.key()}.json").exists(),
    reason="OSM cache missing; run scripts/view_world.py --osm to populate it",
)


@pytest.fixture(scope="module")
def berkeley():
    return osm.load(bbox=BERKELEY, cell_size_m=0.5)


def _shoelace(p: np.ndarray) -> float:
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    x, y = p[:, 0], p[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


# --------------------------------------------------------------- the key test

def test_rasterised_area_matches_polygons(berkeley):
    """Every building must occupy exactly the cells its outline covers.

    Only buildings entirely inside the box are checked: one crossing the edge is
    *supposed* to lose the part that falls outside.
    """
    t = berkeley
    w_m, h_m = t.extent_m
    solid = [f for f in t.footprints if f.kind == "solid"]
    inside = [
        f for f in solid
        if 0 <= f.xy[:, 0].min() and f.xy[:, 0].max() <= w_m
        and 0 <= f.xy[:, 1].min() and f.xy[:, 1].max() <= h_m
    ]
    assert len(inside) > 50, "expected a decent sample of interior buildings"

    grid = np.zeros(t.shape, dtype=bool)
    for f in inside:
        hit = raster.fill_polygon(f.xy.numpy().astype(float), t.shape, t.cell_size_m)
        if hit:
            sl, mask = hit
            grid[sl][mask] = True

    polygon_area = sum(_shoelace(f.xy.numpy()) for f in inside)
    raster_area = grid.sum() * t.cell_size_m**2
    assert raster_area == pytest.approx(polygon_area, rel=0.01)


def test_rasterising_converges(berkeley):
    """Coarser and finer grids must agree: the answer is the geometry, not the mesh."""
    coarse = osm.load(bbox=BERKELEY, cell_size_m=2.0)
    fine = berkeley
    a = float((coarse.solid_top > 0).sum()) * coarse.cell_size_m**2
    b = float((fine.solid_top > 0).sum()) * fine.cell_size_m**2
    assert a == pytest.approx(b, rel=0.01)


# -------------------------------------------------------------------- heights

@pytest.mark.parametrize("text,want", [
    ("12", 12.0), ("12 m", 12.0), ("12.5", 12.5), ("4.5 m", 4.5),
    ("40 ft", 12.192), ("40'", 12.192), ("nonsense", None), ("", None),
])
def test_parse_len(text, want):
    got = osm._parse_len(text)
    assert got is None if want is None else got == pytest.approx(want, rel=1e-4)


def test_height_prefers_explicit_over_levels():
    assert osm._height_m({"height": "20", "building:levels": "2"}) == 20.0
    assert osm._height_m({"building:levels": "4"}) == 4 * osm.LEVEL_M
    assert osm._height_m({}) == osm.DEFAULT_LEVELS * osm.LEVEL_M
    # roof storeys sit on top of the building's storeys
    assert osm._height_m({"building:levels": "3", "roof:levels": "1"}) == (
        3 * osm.LEVEL_M + osm.ROOF_LEVEL_M
    )


def test_underground_buildings_are_skipped(berkeley):
    """Real case: downtown Berkeley's BART station is `building:levels=0`.

    It is underground, so nothing above ground should occlude, and extruding it to
    zero height would make a degenerate mesh.
    """
    assert osm._height_m({"building:levels": "0",
                          "building:levels:underground": "2"}) < osm.MIN_BUILDING_M
    heights = [f.top_m for f in berkeley.footprints if f.kind == "solid"]
    assert min(heights) >= osm.MIN_BUILDING_M
    assert float(berkeley.solid_top[berkeley.solid_top > 0].min()) >= osm.MIN_BUILDING_M


def test_heights_are_plausible_for_a_downtown(berkeley):
    h = [f.top_m for f in berkeley.footprints if f.kind == "solid"]
    assert 2.0 <= min(h) and max(h) <= 200.0
    assert 4.0 <= float(np.median(h)) <= 30.0


# --------------------------------------------------------------- canopies

def test_canopies_are_roofs_not_walls(berkeley):
    """A canopy blocks the view from above but leaves the ground beneath usable."""
    assert osm._is_canopy({"building": "roof"})
    assert osm._is_canopy({"man_made": "canopy"})
    assert not osm._is_canopy({"building": "house"})

    t = berkeley
    under = t.canopy_z1 > 0
    if not bool(under.any()):
        pytest.skip("no canopies in this extract")
    assert bool((t.solid_top[under] == 0).all()), "a canopy must not also be a wall"
    assert bool(t.walkable[under].all()), "you can stand under a canopy"
    assert bool((t.canopy_z0[under] > 0).all())
    assert bool((t.canopy_z1[under] > t.canopy_z0[under]).all())


def test_canopies_default_short_not_two_storeys():
    """A carport is not a two-storey building.

    Using the generic building default here would put every canopy at 6 m, which
    quietly kills the thing canopies exist for: at 6 m the low drone flies under all
    of them, so it is never forced to look sideways under a roof the high drone
    cannot see through.
    """
    assert osm._height_m({"building": "roof"}, default_m=osm.DEFAULT_CANOPY_M) == 3.0
    assert osm._height_m({"building": "house"}) == osm.DEFAULT_LEVELS * osm.LEVEL_M
    # an explicitly tagged canopy still gets its real height
    assert osm._height_m({"building": "roof", "building:levels": "4"},
                         default_m=osm.DEFAULT_CANOPY_M) == 4 * osm.LEVEL_M


def test_short_canopies_block_the_drone_but_not_the_view(berkeley):
    """The paper's asymmetry, arriving from real data rather than being stipulated."""
    t = berkeley
    under = t.canopy_z1 > 0
    if not bool(under.any()):
        pytest.skip("no canopies in this extract")

    low = under & (t.canopy_z0 <= 3.0)
    assert bool(low.any()), "expected some canopies too low to fly under"
    assert bool(t.walkable[low].all()), "a person walks under it"
    assert not bool(t.flyable[low].any()), "the low drone does not fit under it"
    assert bool((t.solid_top[low] == 0).all()), "yet nothing blocks the sideways view"


def test_flyable_is_never_more_permissive_than_walkable(berkeley):
    assert bool(berkeley.walkable[berkeley.flyable].all())


# ------------------------------------------------------------------- fuel

def test_buildings_do_not_burn(berkeley):
    built = berkeley.solid_top > 0
    assert bool((berkeley.fuel[built] == 0).all())


def test_roads_are_firebreaks(berkeley):
    """Streets must be zero-fuel, and there must be enough of them to matter."""
    t = berkeley
    zero_fuel = (t.fuel == 0).sum()
    built = (t.solid_top > 0).sum()
    road_cells = int(zero_fuel - built)
    assert road_cells > 0.05 * t.shape[0] * t.shape[1], "a city should have streets"


def test_vegetation_burns_more_than_bare_ground(berkeley):
    assert osm.FUEL_VEG > osm.FUEL_GROUND > osm.FUEL_ROAD
    assert bool((berkeley.fuel <= osm.FUEL_VEG).all())


CELL, SHAPE = 0.5, (300, 300)     # a generous 150 m x 150 m, so nothing clips


def _road_area(points, width_m=8.0, shape=SHAPE):
    _, mask = raster.stroke_line(np.array(points, float), width_m, shape, CELL)
    return mask.sum() * CELL**2


def _capsule(length_m, width_m=8.0):
    """A stroked line is a capsule: a rectangle with a half-disc at each end.

    Round ends are what we want -- they make road junctions join cleanly instead of
    leaving unburnt notches at every corner -- but it does mean the area is not
    simply length x width.
    """
    r = width_m / 2
    return length_m * width_m + np.pi * r**2


def test_stroke_paints_the_right_width():
    assert _road_area([[25, 75], [125, 75]]) == pytest.approx(_capsule(100), rel=0.01)


def test_stroke_clips_at_the_edge():
    """A road whose centreline runs along the boundary paints only its inner half."""
    assert _road_area([[25, 0], [125, 0]]) == pytest.approx(_capsule(100) / 2, rel=0.01)


def test_stroke_follows_a_corner():
    """Two 50 m legs at a right angle: 100 m of road, joined without double-counting."""
    area = _road_area([[25, 25], [75, 25], [75, 75]])
    assert 100 * 8.0 < area < 2 * _capsule(50)   # one capsule's worth of caps, not two


# ------------------------------------------------------- projection & determinism

def test_projection_puts_the_corner_at_the_origin():
    proj = osm._project(BERKELEY)
    x, y = proj(BERKELEY.south, BERKELEY.west)
    assert (abs(x), abs(y)) == pytest.approx((0.0, 0.0), abs=1e-6)

    w_m, h_m = BERKELEY.size_m()
    x, y = proj(BERKELEY.north, BERKELEY.east)
    assert (x, y) == pytest.approx((w_m, h_m), rel=1e-6)
    # a 250 m radius means a 500 m box, whichever way you measure it
    assert w_m == pytest.approx(500.0, rel=0.02)
    assert h_m == pytest.approx(500.0, rel=0.02)


def test_deterministic():
    a = osm.load(bbox=BERKELEY, cell_size_m=2.0)
    b = osm.load(bbox=BERKELEY, cell_size_m=2.0)
    assert torch.equal(a.solid_top, b.solid_top)
    assert torch.equal(a.fuel, b.fuel)
    assert torch.equal(a.flyable, b.flyable)
    assert len(a.footprints) == len(b.footprints)


def test_grid_matches_the_requested_extent(berkeley):
    w_m, h_m = BERKELEY.size_m()
    assert berkeley.extent_m[0] == pytest.approx(w_m, abs=berkeley.cell_size_m)
    assert berkeley.extent_m[1] == pytest.approx(h_m, abs=berkeley.cell_size_m)
