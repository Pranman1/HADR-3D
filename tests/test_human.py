"""The person: panic, and the body that carries it.

The load-bearing test is `test_panic_roughly_doubles_evacuation_time`.  The paper's
Figure 2 says persistent panic "nearly doubles" the time to reach safety, and that is
the one published number about the human model we can check ourselves against.  If it
drifts, either the model or the calibration has gone wrong and every improvement
figure downstream is suspect.

Several of the others exist because the bug they catch actually happened: panic
collapsing into a random walk that never arrived, a person deadlocked against a wall
forever, and panic saturating so that a fire anywhere on the map was as frightening
as a fire underfoot.
"""

import pytest
import torch

from hadrsim.agents.crowd import Crowd, CrowdParams
from hadrsim.agents.human import Human, PanicParams
from hadrsim.core.clock import Clock
from hadrsim.core.rng import Rng
from hadrsim.world import fire as fire_mod
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import terrain as terr
from hadrsim.world import wind as wind_mod

CLOCK = Clock()
DT = CLOCK.control_dt


def rust25():
    return terr.load("maps/rust25.yaml")


class OpenGround:
    """A featureless plain, for testing the body without the map getting in the way."""

    def __init__(self, n=41, cell=1.0):
        self.cell_size_m = cell
        self.shape = (n, n)
        self.fuel = torch.ones(n, n)
        self.solid_top = torch.zeros(n, n)
        self.walkable = torch.ones(n, n, dtype=torch.bool)
        self.walkable[0, :] = self.walkable[-1, :] = False
        self.walkable[:, 0] = self.walkable[:, -1] = False


def walk_out(terrain, batch=16, seed=0, gamma=None, ignite=None, params=None,
             start=(0.5, 0.5), goal=(22.5, 24.5), max_steps=None):
    """Run people from start to goal. Returns (fraction arrived, mean time)."""
    rng = Rng(seed)
    fire = fire_mod.Fire(terrain, batch)
    smoke = smoke_mod.Smoke(terrain, batch)
    wind = wind_mod.still(batch)
    human = Human(terrain, batch, params)
    crowd = Crowd(terrain)

    if ignite is not None:
        fire.reset(torch.tensor([list(ignite)] * batch))
    human.reset(torch.tensor([list(start)] * batch).float(),
                torch.tensor([list(goal)] * batch).float())

    times = []
    steps = max_steps or CLOCK.max_steps
    for i in range(1, steps + 1):
        fire.step(DT, wind, rng)
        smoke.update(fire, wind)
        want = human.desired_velocity(fire, smoke, rng, DT)
        if gamma is not None:
            human.gamma.fill_(gamma)
        before = human.reached.clone()
        human.integrate(crowd.step(human.pos, human.vel, want, DT), DT)
        times += [i * DT] * int((human.reached & ~before).sum())

    return len(times) / batch, (sum(times) / len(times) if times else float("nan"))


# ------------------------------------------------------------------- the model

def test_a_calm_person_walks_out():
    arrived, t = walk_out(rust25(), gamma=0.0)
    assert arrived == 1.0
    assert 15.0 < t < 45.0, "a 35 m walk at 1.3 m/s should take about half a minute"


def test_panic_roughly_doubles_evacuation_time():
    """The paper's Figure 2, and the one published number we can check ourselves on.

    Held at a fixed panic with no fire, so panic is the only thing varying.
    """
    _, calm = walk_out(rust25(), batch=48, gamma=0.0, max_steps=600)
    _, panicked = walk_out(rust25(), batch=48, gamma=1.0, max_steps=600)
    assert 1.6 < panicked / calm < 3.0, (
        f"panic changed evacuation time by {panicked / calm:.2f}x; "
        "the paper says it nearly doubles it"
    )


def test_mild_stress_is_not_harmful():
    """Speed rises before judgement fails -- the inverted-U in the stress literature."""
    _, calm = walk_out(rust25(), batch=48, gamma=0.0, max_steps=600)
    _, mild = walk_out(rust25(), batch=48, gamma=0.2, max_steps=600)
    assert mild <= calm * 1.1


def test_high_panic_does_not_become_a_random_walk():
    """It should cost time, not make arrival impossible.

    An earlier version drew fresh noise every step with no fire to flee from, which
    averages out to nothing; the fix was a heading that persists.  Overcorrecting the
    other way -- pure random heading -- means nobody ever arrives.
    """
    arrived, _ = walk_out(rust25(), batch=48, gamma=1.0, max_steps=600)
    assert arrived > 0.4


def test_panic_rises_near_a_fire_and_stays_low_away_from_one():
    t = rust25()
    rng = Rng(0)
    peaks = {}
    for label, ignite in (("far", None), ("near", (2, 2))):
        fire = fire_mod.Fire(t, 8)
        smoke = smoke_mod.Smoke(t, 8)
        w = wind_mod.still(8)
        h = Human(t, 8)
        if ignite:
            fire.reset(torch.tensor([list(ignite)] * 8))
        h.reset(torch.tensor([[3.5, 3.5]] * 8), torch.tensor([[22.5, 24.5]] * 8))
        peak = torch.zeros(8)
        for _ in range(40):
            fire.step(DT, w, rng)
            smoke.update(fire, w)
            h.desired_velocity(fire, smoke, rng, DT)
            peak = torch.maximum(peak, h.gamma)
        peaks[label] = float(peak.mean())
    assert peaks["near"] > peaks["far"] + 0.15, peaks


def test_panic_grades_with_distance_to_the_fire():
    """Not just "near is worse than far" -- it has to be a gradient.

    A perception radius short enough to keep panic honest also makes a burning city
    block fifty metres away completely unalarming, which is the opposite failure to
    the saturation this file's other test guards.
    """
    t = rust25()
    peaks = []
    for ignite in (None, (22, 22), (12, 12), (4, 4)):
        rng = Rng(0)
        fire, smoke = fire_mod.Fire(t, 4), smoke_mod.Smoke(t, 4)
        w = wind_mod.still(4)
        h = Human(t, 4)
        if ignite:
            fire.reset(torch.tensor([list(ignite)] * 4), radius_m=2.5)
        h.reset(torch.tensor([[2.5, 2.5]] * 4), torch.tensor([[22.5, 24.5]] * 4))
        held = h.pos.clone()
        peak = torch.zeros(4)
        for _ in range(120):
            fire.step(0.1, w, rng)
            smoke.update(fire, w)
            h.desired_velocity(fire, smoke, rng, 0.1)
            h.pos = held.clone()               # stand still; measure panic alone
            peak = torch.maximum(peak, h.gamma)
        peaks.append(float(peak.mean()))

    none, far, mid, near = peaks
    assert none < 0.15, f"nothing burning should not be frightening: {none:.2f}"
    assert none < far < mid < near, f"panic must grade with distance: {peaks}"
    assert near > 0.6, f"a fire three metres away should be terrifying: {near:.2f}"


def test_panic_does_not_saturate_on_a_distant_fire():
    """Blurring the fire measures how *much* there is, not how *close* -- and on a
    small map that pins panic at one wherever you stand."""
    t = rust25()
    rng = Rng(0)
    fire = fire_mod.Fire(t, 4)
    smoke = smoke_mod.Smoke(t, 4)
    w = wind_mod.still(4)
    h = Human(t, 4)
    fire.reset(torch.tensor([[5, 6]] * 4))
    h.reset(torch.tensor([[22.5, 24.5]] * 4), torch.tensor([[0.5, 0.5]] * 4))
    for _ in range(20):
        fire.step(DT, w, rng)
        smoke.update(fire, w)
        h.desired_velocity(fire, smoke, rng, DT)
    assert float(h.gamma.mean()) < 0.5, "a fire across the map should not be maximal"


def test_smoke_slows_people_down():
    p = PanicParams(smoke_slowdown=0.9)
    h = Human(rust25(), 2, p)
    h.reset(torch.tensor([[5.0, 5.0]] * 2), torch.tensor([[22.5, 24.5]] * 2))

    class Clear:
        def column_density(self):
            return torch.zeros(2, 25, 25)

    class Thick:
        def column_density(self):
            return torch.full((2, 25, 25), 10.0)

    assert float(h._smoke_slowdown(Clear())[0]) == pytest.approx(1.0)
    assert float(h._smoke_slowdown(Thick())[0]) < 0.2


def test_panic_relaxes_at_the_same_rate_whatever_the_step_size():
    """gamma must not depend on how often the caller happens to step."""
    def final(dt, n):
        t = rust25()
        h = Human(t, 1)
        h.reset(torch.tensor([[5.0, 5.0]]), torch.tensor([[22.5, 24.5]]))
        threat = torch.ones(1)

        class NoSmoke:
            def column_density(self):
                return torch.zeros(1, 25, 25)

        for _ in range(n):
            h._update_gamma(threat, NoSmoke(), dt)
        return float(h.gamma[0])

    assert final(1.0, 10) == pytest.approx(final(0.1, 100), abs=0.02)


# -------------------------------------------------------------------- the body

def test_a_person_never_ends_up_inside_a_building():
    t = rust25()
    rng = Rng(0)
    fire, smoke = fire_mod.Fire(t, 32), smoke_mod.Smoke(t, 32)
    w = wind_mod.still(32)
    h, crowd = Human(t, 32), Crowd(t)
    h.reset(torch.tensor([[0.5, 0.5]] * 32), torch.tensor([[22.5, 24.5]] * 32))
    for _ in range(CLOCK.max_steps):
        smoke.update(fire, w)
        h.integrate(crowd.step(h.pos, h.vel,
                               h.desired_velocity(fire, smoke, rng, DT), DT), DT)
        cells = (h.pos / t.cell_size_m).long().clamp(0, 24)
        assert bool(t.walkable[cells[:, 0], cells[:, 1]].all())


def test_a_person_slides_along_a_wall_rather_than_sticking():
    """Zeroing the velocity at a wall deadlocks: the desired direction still points
    into it next step, so they press against it forever."""
    t = OpenGround()
    crowd = Crowd(t)
    pos = torch.tensor([[1.5, 20.5]])             # hard against the western edge
    # Pushing diagonally into the wall must keep the component that is not blocked.
    v = crowd.step(pos, torch.zeros(1, 2), torch.tensor([[-2.0, 2.0]]), DT)
    assert float(v[0, 1]) > 0.0, "should still make progress along the open axis"
    assert (pos + v * DT)[0, 0] >= t.cell_size_m, "must not step into the wall"


def test_momentum_means_a_person_cannot_turn_instantly():
    """Asked to reverse, a body must still be moving forwards a moment later.

    Note the reaction time is longer than one control step here on purpose: with
    reaction 1.0 s and a 0.5 s step, one step removes exactly the whole velocity and
    lands on zero, which tests nothing.
    """
    crowd = Crowd(OpenGround(), CrowdParams(reaction_time_s=2.0))
    pos = torch.tensor([[20.5, 20.5]])
    vel = torch.tensor([[1.5, 0.0]])
    v = crowd.step(pos, vel, torch.tensor([[-1.5, 0.0]]), DT)
    assert 0.0 < float(v[0, 0]) < 1.5, "should be slowing, not stopped or unchanged"


def test_speed_is_capped():
    crowd = Crowd(OpenGround(), CrowdParams(max_speed_mps=2.0))
    v = crowd.step(torch.tensor([[20.5, 20.5]]), torch.zeros(1, 2),
                   torch.tensor([[100.0, 100.0]]), 1.0)
    assert float(v.norm()) <= 2.0 + 1e-5


def test_a_person_goes_round_a_fire_not_through_it():
    """The paper plans A* over the static map once, so the fire is something walked
    into rather than avoided.  Pricing the route and re-planning on a slow clock
    produces a detour, the way a blocked street does in life."""
    class Corridor:
        cell_size_m = 1.0
        shape = (61, 61)
        fuel = torch.ones(61, 61)
        solid_top = torch.zeros(61, 61)
        walkable = torch.ones(61, 61, dtype=torch.bool)

    def cross(ignite):
        t = Corridor()
        rng = Rng(0)
        fire, smoke = fire_mod.Fire(t, 1), smoke_mod.Smoke(t, 1)
        w = wind_mod.still(1)
        h = Human(t, 1, PanicParams(gamma_floor=0.0, time_stress_s=1e9))
        crowd = Crowd(t)
        if ignite:
            fire.reset(torch.tensor([list(ignite)]), radius_m=6.0)
        h.reset(torch.tensor([[10.0, 30.0]]), torch.tensor([[50.0, 30.0]]))

        want, detour, burned = torch.zeros(1, 2), 0.0, False
        for i in range(12000):
            if i % 10 == 0:
                smoke.update(fire, w)
                want = h.desired_velocity(fire, smoke, rng, 0.1)
            h.gamma.fill_(0.0)
            h.integrate(crowd.step(h.pos, h.vel, want, 0.01), 0.01)
            detour = max(detour, abs(float(h.pos[0, 1]) - 30.0))
            c = h.pos[0].long().clamp(0, 60)
            burned |= bool(fire.burning[0, c[0], c[1]] or fire.burnt[0, c[0], c[1]])
            if h.reached[0]:
                break
        return bool(h.reached[0]), detour, burned

    got, straight, _ = cross(None)
    assert got and straight < 3.0, "a clear route should be walked straight"

    got, detour, burned = cross((30, 30))
    assert got, "a detour exists; they should still get out"
    assert detour > 8.0, f"should go round the fire, deviated only {detour:.1f} m"
    assert not burned, "should never walk into the fire"


def test_a_person_will_not_fit_through_a_gap_narrower_than_they_are():
    """Testing the cell under someone's centre lets them clip building corners and
    thread gaps their shoulders do not fit through.

    Checked at quarter-metre cells, because at one-metre cells the narrowest gap the
    grid can even represent is a metre wide -- which a person fits through easily, so
    there is nothing to test.
    """
    class Wall:
        cell_size_m = 0.25
        shape = (81, 81)
        fuel = torch.ones(81, 81)
        solid_top = torch.zeros(81, 81)

        def __init__(self, gap_cells):
            self.walkable = torch.ones(81, 81, dtype=torch.bool)
            self.walkable[40, :] = False
            mid = 40
            self.walkable[40, mid - gap_cells // 2: mid + (gap_cells + 1) // 2] = True
            self.solid_top = (~self.walkable).float() * 6.0

    def can_pass(gap_cells):
        t = Wall(gap_cells)
        crowd = Crowd(t)
        pos = torch.tensor([[39.5 * t.cell_size_m, 40.0 * t.cell_size_m]])
        v = crowd.step(pos, torch.zeros(1, 2), torch.tensor([[1.0, 0.0]]), 0.5)
        return float(v[0, 0]) > 0.0

    assert not can_pass(2), "0.5 m gap: shoulders do not fit"
    assert can_pass(8), "2 m gap: walks through without trouble"


def test_a_person_never_clips_a_building_corner():
    """Their centre must never come closer to a wall than their own radius."""
    t = rust25()
    rng = Rng(0)
    fire, smoke = fire_mod.Fire(t, 8), smoke_mod.Smoke(t, 8)
    w = wind_mod.still(8)
    h, crowd = Human(t, 8), Crowd(t)
    h.reset(torch.tensor([[0.5, 0.5]] * 8), torch.tensor([[22.5, 24.5]] * 8))

    from hadrsim.world import nav
    clear = nav.clearance(t.walkable, t.cell_size_m)[None].expand(8, -1, -1)

    want, closest = torch.zeros(8, 2), 1e9
    for i in range(4000):
        if i % 10 == 0:
            want = h.desired_velocity(fire, smoke, rng, 0.1)
        h.integrate(crowd.step(h.pos, h.vel, want, 0.01), 0.01)
        closest = min(closest, float(nav.sample(clear, h.pos, t.cell_size_m).min()))
    assert closest >= crowd.p.body_radius_m, (
        f"a body centre came within {closest:.2f} m of a wall")


# ---------------------------------------------------------------- reproducible

def test_same_seed_same_walk():
    """Reproducibility, checked on the trajectory rather than on the summary.

    Comparing the returned means is a trap: with a fierce fire on a 25 m map nobody
    gets out, both runs summarise to (0.0, nan), and `nan != nan` fails a test that
    should pass -- while telling you nothing about whether the walk was identical.
    """
    def final_positions(seed):
        t = rust25()
        rng = Rng(seed)
        fire, smoke = fire_mod.Fire(t, 8), smoke_mod.Smoke(t, 8)
        w = wind_mod.still(8)
        h, crowd = Human(t, 8), Crowd(t)
        fire.reset(torch.tensor([[5, 18]] * 8))
        h.reset(torch.tensor([[0.5, 0.5]] * 8), torch.tensor([[22.5, 24.5]] * 8))
        for _ in range(60):
            fire.step(DT, w, rng)
            smoke.update(fire, w)
            h.integrate(crowd.step(h.pos, h.vel,
                                   h.desired_velocity(fire, smoke, rng, DT), DT), DT)
        return h.pos.clone()

    assert torch.equal(final_positions(4), final_positions(4))
    assert not torch.equal(final_positions(4), final_positions(5))
