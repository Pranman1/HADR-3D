"""The 3D terrain must reproduce the paper's 2D visibility table exactly.

pegrid/core/world_object.py states, per building class, who may pass and who may
see inside.  Our 3D version derives those from geometry instead of a lookup, so
this test checks the geometry lands on the same answers.  If it ever fails, the
3D map has stopped being a faithful reading of the paper's map.
"""

import torch

from hadrsim.world import terrain as terr

# from pegrid/core/world_object.py: walkable, LLR-flyable, HLR sees in, LLR sees in
PEGRID_TABLE = {
    "inaccessible":      (False, False, False, False),
    "evader-only":       (True,  False, False, False),
    "evader-accessible": (True,  False, False, True),
    "accessible":        (True,  True,  False, True),
}

HLR_ALT = 12.0   # looks straight down
LLR_ALT = 3.0    # looks sideways, at head height


def _sees_in(solid, canopy, looking_down):
    """Can an observer see the ground inside this cell?

    From above, anything overhead blocks -- wall or roof alike.  From the side, only
    a wall blocks: a canopy is open-sided, so a slanted line reaches under it.
    """
    if looking_down:
        return solid == 0.0 and canopy is None
    return solid == 0.0


def _flyable(solid, canopy, clearance):
    """Derived exactly as terrain.load() derives it."""
    roof = canopy[0] if canopy is not None else float("inf")
    return solid == 0.0 and roof > clearance


def test_matches_pegrid_visibility_table():
    for name, (walkable, flyable, hlr_in, llr_in) in PEGRID_TABLE.items():
        solid, canopy, w = terr.CLASSES[name]
        assert w == walkable, f"{name}: walkable"
        assert _flyable(solid, canopy, LLR_ALT) == flyable, f"{name}: LLR flyable"
        assert _sees_in(solid, canopy, True) == hlr_in, f"{name}: HLR sees in"
        assert _sees_in(solid, canopy, False) == llr_in, f"{name}: LLR sees in"


def test_rust25_loads_consistently():
    t = terr.load("maps/rust25.yaml")
    n = t.shape[0] * t.shape[1]
    assert t.shape == (25, 25)
    # fire may not cross a solid building, and may cross everything else
    assert bool(((t.fuel == 0.0) == (t.solid_top > 0)).all())
    assert int((t.fuel > 0).sum()) == n - int((t.solid_top > 0).sum())
    # a canopy is a roof with air under it, never overlapping a solid
    assert bool((t.canopy_z0[t.canopy_z1 > 0] > 0).all())
    assert bool((t.solid_top[t.canopy_z1 > 0] == 0).all())
    # anything the low drone can fly through, the human can walk through
    assert bool((t.walkable[t.flyable]).all())


def test_deterministic():
    a, b = terr.load("maps/rust25.yaml"), terr.load("maps/rust25.yaml")
    assert torch.equal(a.solid_top, b.solid_top) and torch.equal(a.fuel, b.fuel)
