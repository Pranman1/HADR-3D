"""Fire, smoke, wind and navigation.

These are the numbers every later result rests on, so the tests are about behaviour
rather than implementation: does the fire go out, does it stop at a road, does the
wind blow it the right way, does the plume lean downwind by the distance the physics
says it should.

Two of them exist because the bug they catch actually happened.  `test_wind_pushes_
the_fire_downwind` caught a sign error that sent fire directly upwind -- invisible in
still air, because the neighbour set is symmetric.  `test_flow_points_toward_the_goal`
caught the same class of error in the distance field.
"""

import math

import pytest
import torch

from hadrsim.core.clock import Clock
from hadrsim.core.rng import Rng
from hadrsim.world import fire as fire_mod
from hadrsim.world import nav
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import terrain as terr
from hadrsim.world import wind as wind_mod


class OpenField:
    """A featureless plain. Isolates the model from the map's own quirks."""

    def __init__(self, n=61, cell=1.0):
        self.cell_size_m = cell
        self.shape = (n, n)
        self.fuel = torch.ones(n, n)
        self.solid_top = torch.zeros(n, n)
        self.walkable = torch.ones(n, n, dtype=torch.bool)


def centroid(mask):
    n = mask.shape[0]
    xs = torch.arange(n).float()[:, None].expand_as(mask)
    ys = torch.arange(n).float()[None, :].expand_as(mask)
    m = mask.float()
    s = m.sum().clamp_min(1e-6)
    return float((xs * m).sum() / s), float((ys * m).sum() / s)


def burn(terrain, seconds, wind, seed=0, at=(30, 30), batch=1):
    rng = Rng(seed)
    f = fire_mod.Fire(terrain, batch)
    f.reset(torch.tensor([list(at)] * batch))
    for _ in range(seconds):
        f.step(1.0, wind, rng)
    return f


# ------------------------------------------------------------------- the clock

def test_clock_derives_everything_from_hz():
    c = Clock(physics_hz=100, control_hz=2, episode_seconds=120)
    assert c.dt == 0.01 and c.control_dt == 0.5
    assert c.substeps == 50 and c.max_steps == 240


def test_clock_rejects_a_policy_faster_than_physics():
    with pytest.raises(ValueError):
        Clock(physics_hz=1, control_hz=10)


# --------------------------------------------------------------------- the rng

def test_same_seed_same_numbers():
    assert torch.equal(Rng(7).uniform(100), Rng(7).uniform(100))
    assert not torch.equal(Rng(7).uniform(100), Rng(8).uniform(100))


def test_forked_streams_are_independent():
    """A gust of wind must not change which cells the fire ignites."""
    base = Rng(3)
    assert not torch.equal(base.fork("fire").uniform(50), base.fork("wind").uniform(50))
    assert torch.equal(Rng(3).fork("fire").uniform(50), Rng(3).fork("fire").uniform(50))


# -------------------------------------------------------------------- the fire

def test_fire_burns_out():
    """The 2D simulator's fire never goes out, so it eventually fills the map."""
    f = burn(OpenField(31), 400, wind_mod.still(1), at=(15, 15))
    assert not bool(f.burning.any()), "everything alight should eventually burn out"
    assert bool(f.burnt.any())
    assert bool((f.fuel[f.burnt] <= 0).all())


def test_fire_spreads_but_not_instantly():
    still = wind_mod.still(1)
    early = burn(OpenField(), 10, still)
    late = burn(OpenField(), 40, still)
    assert 1 < int((early.burning | early.burnt).sum()) < int((late.burning | late.burnt).sum())


def test_fire_cannot_cross_a_firebreak():
    """A road is zero fuel, and zero fuel must be impassable to fire."""
    t = OpenField(41)
    t.fuel[20, :] = 0.0                       # a road straight across the map
    f = burn(t, 300, wind_mod.still(1), at=(10, 20))
    assert not bool((f.burning | f.burnt)[0, 21:, :].any()), "fire jumped the road"
    assert bool((f.burning | f.burnt)[0, :20, :].any()), "fire did not spread at all"


def test_fire_will_not_light_where_there_is_nothing_to_burn():
    t = OpenField(21)
    t.fuel.zero_()
    f = fire_mod.Fire(t, 1)
    f.reset(torch.tensor([[10, 10]]))
    assert int(f.burning.sum()) == 0


def test_buildings_burn_hardest():
    """Not "fire respects buildings" -- buildings are what an urban fire consumes.

    The barrier in a city is the *street*.  Making buildings fireproof is a tempting
    way to stop fire walking through walls, and it means downtown Berkeley -- 44%
    building, most of the rest pavement -- cannot burn at all.
    """
    t = terr.load("maps/rust25.yaml")
    assert bool((t.fuel[t.solid_top > 0] > t.fuel[t.solid_top == 0].max() * 0.99).all())
    f = burn(t, 300, wind_mod.still(1), at=(5, 18))
    assert bool((f.burning | f.burnt)[0][t.solid_top > 0].any()), "buildings must burn"


def test_a_single_cell_ignition_is_not_a_coin_toss():
    """One square metre burns out in seconds having had few chances to catch, so
    whether a fire happens at all comes down to luck.  A real fire is not noticed
    until it is already several metres across."""
    outcomes = [int(burn(OpenField(41), 60, wind_mod.still(1), seed=s,
                         at=(20, 20)).burnt.sum()) for s in range(6)]
    assert all(o > 20 for o in outcomes), f"fire fizzled on some seeds: {outcomes}"


@pytest.mark.parametrize("bearing,axis,sign", [
    (0, 0, +1), (180, 0, -1), (90, 1, +1), (270, 1, -1),
])
def test_wind_pushes_the_fire_downwind(bearing, axis, sign):
    """The bug this exists for sent the fire directly *upwind*.

    Invisible without wind: the eight-neighbour set is symmetric, so a still-air fire
    spreads identically whichever way round the source and target are.
    """
    f = burn(OpenField(81), 60, wind_mod.make(1, speed_mps=6.0, bearing_deg=bearing),
             seed=3, at=(40, 40))
    c = centroid((f.burning | f.burnt)[0])
    drift = c[axis] - 40
    assert sign * drift > 5.0, f"fire drifted {drift:+.1f} cells; wind blows the other way"


def test_still_air_spreads_evenly():
    f = burn(OpenField(81), 60, wind_mod.still(1), seed=3, at=(40, 40))
    cx, cy = centroid((f.burning | f.burnt)[0])
    assert abs(cx - 40) < 4 and abs(cy - 40) < 4


def test_fire_is_reproducible():
    """`firegrid` draws from the unseeded global numpy generator; this does not."""
    a = burn(OpenField(41), 30, wind_mod.still(1), seed=11, at=(20, 20))
    b = burn(OpenField(41), 30, wind_mod.still(1), seed=11, at=(20, 20))
    assert torch.equal(a.state, b.state) and torch.equal(a.fuel, b.fuel)
    c = burn(OpenField(41), 30, wind_mod.still(1), seed=12, at=(20, 20))
    assert not torch.equal(a.state, c.state)


def test_fire_clock_is_independent_of_how_often_it_is_stepped():
    """Changing the control rate must not change how the fire behaves."""
    coarse = fire_mod.Fire(OpenField(41), 1)
    fine = fire_mod.Fire(OpenField(41), 1)
    coarse.reset(torch.tensor([[20, 20]]))
    fine.reset(torch.tensor([[20, 20]]))

    rng_a, rng_b = Rng(5), Rng(5)
    still = wind_mod.still(1)
    for _ in range(30):
        coarse.step(1.0, still, rng_a)
    for _ in range(300):
        fine.step(0.1, still, rng_b)
    assert torch.equal(coarse.state, fine.state)


def test_intensity_fades_as_fuel_runs_out():
    f = fire_mod.Fire(OpenField(21), 1)
    f.reset(torch.tensor([[10, 10]]))
    rng, still = Rng(0), wind_mod.still(1)
    first = float(f.intensity.max())
    for _ in range(12):
        f.step(1.0, still, rng)
    assert 0.0 < float(f.intensity[0, 10, 10]) < first


# ------------------------------------------------------------------- the smoke

def test_plume_leans_downwind_by_the_right_distance():
    """A parcel at height z has been aloft z/rise seconds, so it has drifted wind*that."""
    t = OpenField(121)
    w = wind_mod.make(1, speed_mps=6.0, bearing_deg=0.0)
    f = burn(t, 12, w, seed=1, at=(40, 60))
    s = smoke_mod.Smoke(t, 1)
    s.update(f, w)

    base = centroid(f.intensity[0] > 0)[0]
    for k in (0, 1, 2):
        layer = s.density[0, k]
        if layer.sum() < 1e-3:
            continue
        alt = float(s.layer_altitudes[k])
        expected = 6.0 * alt / s.p.rise_speed_mps
        assert centroid(layer > layer.max() * 0.1)[0] - base == pytest.approx(
            expected, abs=3.0)


def test_plume_stays_over_the_fire_in_still_air():
    t = OpenField(81)
    w = wind_mod.still(1)
    f = burn(t, 12, w, seed=1, at=(40, 40))
    s = smoke_mod.Smoke(t, 1)
    s.update(f, w)
    top = s.density[0, -1]
    if top.sum() > 1e-3:
        cx, cy = centroid(top > top.max() * 0.1)
        assert abs(cx - 40) < 5 and abs(cy - 40) < 5


def test_smoke_blocks_the_view_from_above():
    """The paper's premise, made physical: the high drone cannot see through a plume."""
    t = OpenField(81)
    w = wind_mod.still(1)
    f = burn(t, 20, w, seed=1, at=(40, 40))
    s = smoke_mod.Smoke(t, 1)
    s.update(f, w)

    tr = s.transmittance_down(40.0)[0]
    assert float(tr.min()) < 0.9, "a plume should obscure the ground beneath it"
    assert float(tr.max()) > 0.99, "clear air should stay clear"
    assert float(tr[40, 40]) < float(tr[5, 5]), "thickest right above the fire"


def test_no_fire_means_no_smoke():
    t = OpenField(41)
    s = smoke_mod.Smoke(t, 1)
    f = fire_mod.Fire(t, 1)
    s.update(f, wind_mod.still(1))
    assert float(s.density.max()) == 0.0
    assert float(s.transmittance_down(30.0).min()) == pytest.approx(1.0)


# ------------------------------------------------------------- the way out

def test_distance_field_is_exact_in_the_open():
    d = nav.distance_field(torch.ones(41, 41, dtype=torch.bool), torch.tensor([[20, 20]]), 1.0)
    for p in [(20, 25), (25, 20), (25, 25), (40, 40)]:
        assert float(d[0, p[0], p[1]]) == pytest.approx(math.dist(p, (20, 20)), rel=1e-4)


def test_flow_points_toward_the_goal():
    """The bug this exists for pointed everyone away from the exit."""
    d = nav.distance_field(torch.ones(41, 41, dtype=torch.bool), torch.tensor([[20, 20]]), 1.0)
    f = nav.flow(d)
    for p, want in [((30, 20), (-1, 0)), ((10, 20), (1, 0)),
                    ((20, 30), (0, -1)), ((20, 10), (0, 1))]:
        assert f[0, p[0], p[1]].tolist() == pytest.approx(list(want), abs=0.05)


def test_flow_still_works_beside_a_wall():
    """Treating a wall as infinitely far makes the gradient enormous and useless."""
    walk = torch.ones(41, 41, dtype=torch.bool)
    walk[:, 20] = False
    walk[20, 20] = True                       # a doorway
    d = nav.distance_field(walk, torch.tensor([[20, 0]]), 1.0)
    f = nav.flow(d)
    reachable = walk & (d[0] < nav.UNREACHABLE / 2)
    usable = f[0][reachable].norm(dim=-1) > 0.5
    assert float(usable.float().mean()) > 0.98


def test_walls_are_unreachable_not_far():
    t = terr.load("maps/rust25.yaml")
    d = nav.distance_field(t.walkable, torch.tensor([[0, 0]]), 1.0)
    assert bool((d[0][~t.walkable] >= nav.UNREACHABLE).all())
    assert float(d[0][t.walkable].max()) < nav.UNREACHABLE


def test_clearance_measures_distance_to_the_nearest_wall():
    walk = torch.ones(21, 21, dtype=torch.bool)
    walk[0, :] = False
    c = nav.clearance(walk, 1.0)
    assert float(c[0, 10]) == 0.0
    assert float(c[1, 10]) == pytest.approx(1.0)
    assert float(c[5, 10]) == pytest.approx(5.0)
