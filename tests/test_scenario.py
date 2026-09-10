"""Scenarios and the runner that steps them.

The reason this file matters more than it looks: a control panel, a video script and
the training loop all build a world from a `Scenario`, and if any of them constructs
a slightly different one then results from the panel say nothing about the policy
that was trained.  So the tests here are mostly about *sameness* -- same config,
same world, same numbers, whoever asked.
"""

import pytest
import torch

from hadrsim.runner import Runner
from hadrsim.scenario import PRESETS, Scenario, autoplace


def berkeley(**kw):
    return Scenario.preset("Downtown Berkeley", **kw)


# ------------------------------------------------------------------ scenario

def test_a_scenario_survives_a_round_trip():
    """The GUI sends a dict; it has to come back as the same world."""
    sc = berkeley(n_humans=3, wind_mps=7.5, seed=11)
    again = Scenario.from_dict(sc.to_dict())
    assert again == sc


def test_unknown_fields_are_ignored_not_fatal():
    """A control panel will send its own bookkeeping along with the config."""
    sc = Scenario.from_dict({**berkeley().to_dict(), "ui_tab": "weather"})
    assert sc.place == "Downtown Berkeley, California"


def test_presets_all_build():
    for name in PRESETS:
        sc = Scenario.preset(name, episode_seconds=10.0, cell_size_m=2.0)
        assert (sc.place is None) == bool(sc.paper_map)


def test_terrain_is_cached():
    """Rebuilding a city per parameter change makes a panel nobody waits for."""
    a, b = berkeley(cell_size_m=2.0), berkeley(cell_size_m=2.0)
    assert a.terrain() is b.terrain()
    assert berkeley(cell_size_m=4.0).terrain() is not a.terrain()


# ----------------------------------------------------------------- placement

def test_autoplace_fills_only_what_is_missing():
    """Anything a person picked -- by hand or by clicking a map -- must survive."""
    sc = autoplace(berkeley(cell_size_m=2.0, fire_at=[40.0, 40.0]))
    assert sc.fire_at == [40.0, 40.0]
    assert sc.human_start is not None and sc.human_goal is not None


def test_autoplace_is_deterministic():
    a = autoplace(berkeley(cell_size_m=2.0))
    b = autoplace(berkeley(cell_size_m=2.0))
    assert (a.fire_at, a.human_start, a.human_goal) == (b.fire_at, b.human_start, b.human_goal)


def test_autoplace_puts_everyone_somewhere_walkable():
    sc = autoplace(berkeley(cell_size_m=2.0, n_humans=4))
    t = sc.terrain()
    for x, y in sc.human_start + sc.human_goal:
        c = (int(x / t.cell_size_m), int(y / t.cell_size_m))
        assert bool(t.walkable[c]), f"spawned inside a building at {c}"


def test_autoplace_does_not_light_the_fire_on_the_exit():
    """The safe place being inside the fire is an incoherent scenario, and it is what
    a naive midpoint pick produces on a real map."""
    sc = autoplace(berkeley(cell_size_m=2.0))
    fire = torch.tensor(sc.fire_at)
    start, goal = torch.tensor(sc.human_start[0]), torch.tensor(sc.human_goal[0])
    span = float((goal - start).norm())
    assert float((fire - goal).norm()) > 0.2 * span, "fire is on top of the exit"
    assert float((fire - start).norm()) > 0.2 * span, "fire is on top of the person"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_autoplace_lights_a_fire_that_takes_hold(seed):
    """A lone verge ringed by road burns for twenty seconds and goes out.

    Tested by running it rather than by measuring fuel density around the ignition
    point: density is a proxy, the code has fallbacks that relax it on sparse maps,
    and what actually matters is only ever whether the thing catches.
    """
    r = Runner(berkeley(cell_size_m=2.0, seed=seed, episode_seconds=45.0))
    r.advance(45.0)
    assert int(r.fire.burning[0].sum()) > 20, "the fire went out"
    assert int(r.fire.burnt[0].sum()) > 0, "nothing burnt through"


def test_extra_people_do_not_stack_on_each_other():
    sc = autoplace(berkeley(cell_size_m=2.0, n_humans=3))
    starts = torch.tensor(sc.human_start)
    assert len(starts) == 3
    gaps = (starts[:, None] - starts[None]).norm(dim=-1)
    assert float(gaps[~torch.eye(3, dtype=bool)].min()) > 1.0


# ------------------------------------------------------------------- runner

def test_runner_steps_and_stays_reproducible():
    a, b = Runner(berkeley(cell_size_m=2.0, seed=3)), Runner(berkeley(cell_size_m=2.0, seed=3))
    for _ in range(3):
        a.advance(1.0)
        b.advance(1.0)
    assert torch.equal(a.human.pos, b.human.pos)
    assert torch.equal(a.fire.state, b.fire.state)


def test_a_different_seed_is_a_different_run():
    a, b = Runner(berkeley(cell_size_m=2.0, seed=3)), Runner(berkeley(cell_size_m=2.0, seed=4))
    for _ in range(6):
        a.advance(1.0)
        b.advance(1.0)
    assert not torch.equal(a.fire.state, b.fire.state)


def test_stepping_in_slices_matches_stepping_in_one_go():
    """The GUI advances in whatever slice the browser can manage; the world must not
    be able to tell."""
    a, b = Runner(berkeley(cell_size_m=2.0, seed=5)), Runner(berkeley(cell_size_m=2.0, seed=5))
    a.advance(2.0)
    for _ in range(20):
        b.advance(0.1)
    assert torch.equal(a.fire.state, b.fire.state)
    assert torch.allclose(a.human.pos, b.human.pos, atol=1e-4)


def test_reset_really_resets():
    r = Runner(berkeley(cell_size_m=2.0, seed=7))
    start = r.human.pos.clone()
    r.advance(5.0)
    assert not torch.allclose(r.human.pos, start)
    r.reset()
    assert r.time == 0.0
    assert torch.allclose(r.human.pos, start)
    assert int(r.fire.burnt.sum()) == 0


def test_an_episode_ends():
    r = Runner(berkeley(cell_size_m=2.0, episode_seconds=3.0))
    assert not r.done
    r.advance(4.0)
    assert r.done


def test_snapshot_is_json_shaped():
    import json

    r = Runner(berkeley(cell_size_m=2.0, n_humans=2))
    r.advance(1.0)
    s = r.snapshot()
    assert len(json.dumps(s.__dict__)) < 4000, "a frame must stay small enough to stream"
    assert len(s.humans) == 2
    assert set(s.humans[0]) == {"x", "y", "panic", "speed", "reached"}
    assert s.drones == [], "nothing flies yet, and the slot should say so honestly"


def test_more_people_means_more_people():
    r = Runner(berkeley(cell_size_m=2.0, n_humans=5))
    r.advance(1.0)
    assert len(r.snapshot().humans) == 5
    assert r.human.pos.shape == (5, 2)


def test_wind_changes_where_the_fire_goes():
    def scar(bearing):
        r = Runner(berkeley(cell_size_m=2.0, seed=2, wind_mps=8.0,
                            wind_bearing_deg=bearing, episode_seconds=40.0))
        r.advance(40.0)
        burnt = (r.fire.burning | r.fire.burnt)[0].float()
        xs = torch.arange(burnt.shape[0]).float()[:, None].expand_as(burnt)
        return float((xs * burnt).sum() / burnt.sum().clamp_min(1))

    assert scar(0.0) > scar(180.0) + 2.0, "the fire should run downwind"
