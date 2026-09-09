"""Building colour and the shared window atlas.

Drawing-only code, so nothing here affects a result.  It is tested because the
texture coordinates have one hard requirement -- everything must land inside
[0, 1] of a single shared atlas -- and because a colour that silently falls back to
the default for every building would turn the city back into grey blobs without
anything failing.
"""

import numpy as np
import pytest

from hadrsim.viz import facade, mesh

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


# ------------------------------------------------------------------- colour

JITTER = 0.2   # brightness scale is 0.88..1.12, times a per-channel 0.94..1.06


def test_colour_priority_is_tag_then_material_then_type():
    assert facade.colour_for({"building:colour": "#8B4513"}) == pytest.approx(
        (139, 69, 19), rel=JITTER)
    assert facade.colour_for({"building:material": "glass"}) == pytest.approx(
        facade.MATERIALS["glass"], rel=JITTER)
    assert facade.colour_for({"building": "office"}) == pytest.approx(
        facade.PALETTE["office"], rel=JITTER)
    assert facade.colour_for({}) == pytest.approx(facade.DEFAULT_COLOUR, rel=JITTER)


def test_dark_colours_keep_their_hue():
    """Jitter must scale brightness, not shift channels: a dark blue stays blue."""
    r, g, b = facade.colour_for({"building:colour": "#8B4513"}, seed=3)
    assert b < g < r, "saddle brown must stay saddle brown"
    assert b < 40, "an additive floor would have flattened this toward grey"


def test_colour_named_and_material_words_both_work():
    """`building:colour=brick` is common OSM usage and means the material."""
    assert facade.colour_for({"building:colour": "brick"}) == pytest.approx(
        facade.MATERIALS["brick"], rel=JITTER)
    assert facade.colour_for({"building:colour": "white"}) == pytest.approx(
        facade.NAMED["white"], rel=JITTER)
    # nonsense falls through to the type, rather than blowing up
    assert facade.colour_for({"building:colour": "puce!", "building": "office"}) == (
        pytest.approx(facade.PALETTE["office"], rel=JITTER))


def test_short_hex_is_expanded():
    assert facade._parse_colour("#abc") == (0xAA, 0xBB, 0xCC)


def test_jitter_is_stable_per_building_but_varies_between_them():
    assert facade.colour_for({"building": "yes"}, 5) == facade.colour_for({"building": "yes"}, 5)
    variants = {facade.colour_for({"building": "yes"}, s) for s in range(30)}
    assert len(variants) > 20, "a street of `building=yes` would look like one slab"


def test_colours_stay_in_range_and_visible():
    """Jitter must not blow a colour out, and must leave every building legible.

    Legibility is checked on the brightest channel, not on all three: a genuinely
    dark blue *should* have a near-zero blue... red channel.  Clamping every channel
    is what flattened hues before.
    """
    for tags in ({}, {"building": "garage"}, {"building:colour": "black"},
                 {"building:colour": "white"}, {"building:colour": "#8B4513"}):
        for seed in range(20):
            c = facade.colour_for(tags, seed)
            assert all(0 <= v <= 252 for v in c)
            assert max(c) > 45, "would be invisible against the ground"


# -------------------------------------------------------------------- atlas

def test_atlas_is_square_and_pale():
    img = facade.window_atlas()
    n = facade.ATLAS_CELLS
    assert img.shape[0] == img.shape[1] and img.dtype == np.uint8
    assert img.shape[0] % n == 0, "the atlas must divide evenly into window bays"
    # It multiplies each building's colour, so the wall must be near-white or every
    # building comes out darker than the palette says.
    assert img.mean() > 150


def test_atlas_has_windows_not_just_wall():
    img = facade.window_atlas()
    assert len(np.unique(img.reshape(-1, 3), axis=0)) >= 3   # wall, floor line, glass
    assert img.min() < 130, "windows should be visibly darker than the wall"


# ----------------------------------------------------------- texture coords

def test_uvs_fit_inside_the_shared_atlas():
    v, f, uv = mesh.extrude(SQUARE, 0.0, 30.0, uv=True)
    assert len(uv) == len(v)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert f.max() < len(v)


def test_uv_scale_matches_real_floors_and_bays():
    """A 30 m building of 40 m perimeter must cover the right patch of the atlas."""
    height, perimeter = 30.0, 40.0
    _, _, uv = mesh.extrude(SQUARE, 0.0, height, uv=True)
    walls = uv[: 4 * len(SQUARE)]
    assert walls[:, 0].max() == pytest.approx(
        perimeter / facade.BAY_M / facade.ATLAS_CELLS, rel=1e-5)
    assert walls[:, 1].max() == pytest.approx(
        height / facade.FLOOR_M / facade.ATLAS_CELLS, rel=1e-5)


def test_the_tallest_real_building_still_fits():
    """Manhattan tops out near 370 m; the atlas must not run out of floors."""
    _, _, uv = mesh.extrude(SQUARE, 0.0, 400.0, uv=True)
    assert uv.max() <= 1.0


def test_walls_do_not_share_vertices_with_the_roof():
    """Shared vertices would force one texture coordinate onto both."""
    n = len(SQUARE)
    v, _, _ = mesh.extrude(SQUARE, 0.0, 10.0, uv=True)
    assert len(v) == 4 * n + 2 * n     # four per wall quad, plus a roof and floor ring


def test_uv_runs_continuously_around_corners():
    """Window columns should march around the corner, not restart at each wall."""
    _, _, uv = mesh.extrude(SQUARE, 0.0, 10.0, uv=True)
    walls = uv[: 4 * len(SQUARE)]
    starts = walls[0::4, 0]
    assert np.all(np.diff(starts) > 0), "each wall must begin where the last ended"


def test_extrude_without_uv_is_unchanged():
    a = mesh.extrude(SQUARE, 0.0, 10.0)
    assert len(a) == 2 and len(a[0]) == 2 * len(SQUARE)


def test_merge_handles_uv_meshes():
    pieces = [mesh.extrude(SQUARE, 0, 10, uv=True), mesh.extrude(SQUARE + 20, 0, 20, uv=True)]
    v, f, uv = mesh.merge(pieces)
    assert len(v) == len(uv) and f.max() < len(v)
    assert mesh.merge([])[0].shape == (0, 3)
