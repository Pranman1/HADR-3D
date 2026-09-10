"""What a drone can see.

The load-bearing test is `test_the_low_drone_sees_under_what_the_high_one_cannot`.
That asymmetry is the paper's central claim, and in the 2D simulator it is a lookup
table -- `see_inside_hlp()` returns False and `see_inside_llp()` returns True.  Here
it has to *emerge*, from a roof, a ray and two viewpoints.  If it stops emerging,
the 3D geometry has stopped meaning anything and the rebuild has no reason to exist.

The rest is about the seam: every detector returns the same `Detections`, carrying
confidence and age, so swapping the oracle for something realistic is a scenario
edit rather than a rewrite.
"""

import math

import pytest
import torch

from hadrsim.agents import airframes as af
from hadrsim.agents.drone import Drone
from hadrsim.core.rng import Rng
from hadrsim.sensing import detector, fov, occlusion
from hadrsim.world import terrain as terr


class OpenGround:
    """Featureless and flat: isolates the sensor from the map's own quirks."""

    def __init__(self, n=81, cell=1.0):
        self.cell_size_m = cell
        self.shape = (n, n)
        self.fuel = torch.ones(n, n)
        self.solid_top = torch.zeros(n, n)
        self.canopy_z0 = torch.zeros(n, n)
        self.canopy_z1 = torch.zeros(n, n)
        self.walkable = torch.ones(n, n, dtype=torch.bool)


def drone_at(x, y, z, yaw=0.0, fidelity="point_mass"):
    d = Drone(1, af.X500_V2, fidelity)
    d.reset(torch.tensor([[float(x), float(y), float(z)]]), torch.tensor([float(yaw)]))
    return d


def at(x, y, z=0.9):
    return torch.tensor([[[float(x), float(y), float(z)]]])


# -------------------------------------------------------------- field of view

def test_a_downward_camera_looks_down():
    d = drone_at(40, 40, 20)
    axis = fov.look_direction(fov.DOWNWARD, d.quat)[0]
    assert float(axis[2]) == pytest.approx(-1.0, abs=1e-5)


def test_a_forward_camera_looks_ahead_and_slightly_down():
    d = drone_at(40, 40, 5)
    axis = fov.look_direction(fov.FORWARD, d.quat)[0]
    assert float(axis[0]) > 0.85, "should mostly look forward"
    assert -0.5 < float(axis[2]) < 0.0, "and slightly downward"


def test_accelerating_swings_the_downward_camera_backwards():
    """A quadrotor has to tilt to accelerate, and the camera is bolted to it.

    Note *which way*: pitching forward tilts the aircraft's up-vector toward the
    nose, so its **belly camera swings toward the tail**.  A drone hurrying somewhere
    is looking at the ground it has already passed -- which is the most interesting
    coupling in the problem, and the one a permanently level camera would delete.
    """
    level = fov.look_direction(fov.DOWNWARD, drone_at(40, 40, 20).quat)[0]
    assert float(level[0]) == pytest.approx(0.0, abs=1e-5)

    d = drone_at(40, 40, 20)
    for _ in range(40):
        d.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 0.01)
    leaning = fov.look_direction(fov.DOWNWARD, d.quat)[0]

    assert float(d.attitude[1][0]) > 0.1, "should be pitching forward"
    assert float(leaning[0]) < -0.1, "so the downward camera should swing rearward"
    assert float(leaning[2]) > -1.0, "and no longer point straight down"


def test_turning_turns_the_forward_camera():
    ahead = fov.look_direction(fov.FORWARD, drone_at(40, 40, 5, yaw=0.0).quat)[0]
    right = fov.look_direction(fov.FORWARD, drone_at(40, 40, 5, yaw=math.pi / 2).quat)[0]
    assert float(ahead[0]) > 0.85 and abs(float(ahead[1])) < 0.05
    assert float(right[1]) > 0.85 and abs(float(right[0])) < 0.05


def test_the_cone_has_edges():
    d = drone_at(40, 40, 20)
    inside, _, off_in = fov.in_view(fov.DOWNWARD, d.pos, d.quat, at(40, 40))
    outside, _, off_out = fov.in_view(fov.DOWNWARD, d.pos, d.quat, at(90, 40))
    assert bool(inside[0, 0]) and not bool(outside[0, 0])
    assert float(off_in[0, 0]) < float(off_out[0, 0])


def test_range_is_limited():
    d = drone_at(40, 40, 20)
    seen, _, _ = fov.in_view(fov.Camera(fov_deg=120, range_m=10.0, kind="downward"),
                             d.pos, d.quat, at(40, 40))
    assert not bool(seen[0, 0]), "a 10 m camera should not see 19 m"


# ---------------------------------------------------------------- occlusion

def test_clear_air_is_clear():
    t = OpenGround()
    blocked, depth = occlusion.trace(t, drone_at(40, 40, 20).pos, at(40, 40))
    assert not bool(blocked[0, 0]) and float(depth[0, 0]) == 0.0


def test_a_wall_blocks():
    t = OpenGround()
    t.solid_top[45:48, :] = 10.0
    origin = drone_at(40, 40, 2).pos
    blocked, _ = occlusion.trace(t, origin, at(55, 40, 1.0))
    assert bool(blocked[0, 0]), "looked through a ten-metre wall"
    clear, _ = occlusion.trace(t, origin, at(42, 40, 1.0))
    assert not bool(clear[0, 0]), "blocked when nothing was in the way"


def test_a_ray_samples_its_whole_length():
    """A short epsilon at each end, not a third off each end.

    At 0.35 the ray sampled only its middle third, so a drone at twelve metres saw
    straight through a three-metre building and nothing looked wrong.
    """
    assert occlusion.EPSILON < 0.1
    t = OpenGround()
    t.solid_top[40, 40] = 3.0
    for alt in (6.0, 12.0, 25.0, 50.0):
        blocked, _ = occlusion.trace(t, drone_at(40.5, 40.5, alt).pos, at(40.5, 40.5))
        assert bool(blocked[0, 0]), f"saw through a 3 m roof from {alt} m"


def test_flying_higher_clears_a_low_building():
    """Height is tested, not just footprint -- which is the point of doing it in 3D."""
    t = OpenGround()
    t.solid_top[45, :] = 5.0
    low, _ = occlusion.trace(t, drone_at(40, 40, 2).pos, at(50, 40, 1.0))
    high, _ = occlusion.trace(t, drone_at(40, 40, 60).pos, at(50, 40, 1.0))
    assert bool(low[0, 0]), "a 5 m wall should block a 2 m viewpoint"
    assert not bool(high[0, 0]), "it should not block one from 60 m"


def test_smoke_dims_rather_than_blocks():
    from hadrsim.core.rng import Rng
    from hadrsim.world import fire as fire_mod
    from hadrsim.world import smoke as smoke_mod
    from hadrsim.world import wind as wind_mod

    t = OpenGround()
    rng, wind = Rng(0), wind_mod.still(1)
    fire = fire_mod.Fire(t, 1)
    fire.reset(torch.tensor([[40, 40]]), radius_m=6.0)
    for _ in range(25):
        fire.step(1.0, wind, rng)
    smoke = smoke_mod.Smoke(t, 1)
    smoke.update(fire, wind)

    origin = drone_at(40, 40, 40).pos
    _, through = occlusion.visible(t, origin, at(40, 40), smoke)
    _, beside = occlusion.visible(t, drone_at(10, 10, 40).pos, at(10, 10), smoke)
    assert 0.0 < float(through[0, 0]) < float(beside[0, 0]), "smoke should dim, not block"


# ---------------------------------------------------------------- the asymmetry

def test_the_low_drone_sees_under_what_the_high_one_cannot():
    """The paper's central claim, from geometry rather than from a lookup table.

    A canopy is a roof with air beneath it.  Looking down, the roof is in the way at
    any altitude; looking in from the side, nothing is.
    """
    t = terr.load("maps/rust25.yaml")
    under = ((t.canopy_z1 > 0) & t.walkable).nonzero()
    assert len(under), "the map has no canopies to test with"
    x, y = float(under[0][0]) + 0.5, float(under[0][1]) + 0.5
    person = at(x, y, 0.9)

    high = detector.make("oracle", fov.DOWNWARD)(t, drone_at(x, y, 40.0), person)
    low = detector.make("oracle", fov.FORWARD)(t, drone_at(x - 12.0, y, 3.0), person)

    assert float(high.confidence[0, 0]) == 0.0, "the high drone should be roofed out"
    assert float(low.confidence[0, 0]) > 0.0, "the low drone should see underneath"


# ----------------------------------------------------------------- detectors

def test_every_detector_returns_the_same_shape():
    """The seam: swapping perception must not touch anything downstream."""
    t = OpenGround()
    d = drone_at(40, 40, 20)
    rng = Rng(0)
    for kind in ("oracle", "geometric"):
        det = detector.make(kind, fov.DOWNWARD)(t, d, at(40, 40), rng=rng)
        assert det.position.shape == (1, 4, 3)
        assert det.confidence.shape == (1, 4) and det.age_s.shape == (1, 4)
        assert det.is_true.dtype == torch.bool


def test_the_oracle_is_exact_and_certain():
    t = OpenGround()
    person = at(40, 40)
    det = detector.make("oracle", fov.DOWNWARD)(t, drone_at(40, 40, 20), person)
    assert float(det.confidence[0, 0]) == 1.0
    assert torch.allclose(det.position[0, 0], person[0, 0])


def test_the_oracle_still_obeys_geometry():
    """It cheats on the *measurement*, not on the visibility.

    Keeping the geometry means the difference between oracle and geometric is
    attributable to noise alone, which is what makes the step-3 ablation mean
    anything.
    """
    t = OpenGround()
    t.solid_top[40, 40] = 8.0
    det = detector.make("oracle", fov.DOWNWARD)(t, drone_at(40.5, 40.5, 30.0), at(40.5, 40.5))
    assert float(det.confidence[0, 0]) == 0.0


def test_detection_gets_worse_with_range():
    """A person at sixty metres is 1.7 m of low-contrast texture."""
    t = OpenGround(n=161)
    person = at(80, 80)
    rng = Rng(0)
    geo = detector.make("geometric", fov.DOWNWARD)

    rates = {}
    for alt in (12.0, 30.0, 60.0):
        hits = 0
        for _ in range(200):
            det = geo(t, drone_at(80, 80, alt), person, rng=rng)
            hits += int(det.is_true[0].any())
        rates[alt] = hits / 200
    assert rates[12.0] > rates[30.0] > rates[60.0], rates
    assert rates[12.0] > 0.5, "should be reliable close in"


def test_position_error_grows_with_range():
    """It is an angular measurement: the same pixel of error is a bigger mistake
    further away."""
    t = OpenGround(n=161)
    person = at(80, 80)
    rng = Rng(1)
    geo = detector.make("geometric", fov.DOWNWARD)

    def mean_error(alt):
        errs = []
        for _ in range(300):
            det = geo(t, drone_at(80, 80, alt), person, rng=rng)
            if bool(det.is_true[0].any()):
                i = int(det.is_true[0].float().argmax())
                errs.append(float((det.position[0, i] - person[0, 0]).norm()))
        return sum(errs) / max(len(errs), 1)

    assert mean_error(45.0) > mean_error(12.0) * 1.5


def test_the_geometric_detector_raises_false_alarms():
    """A detector that only ever misses teaches a policy that any sighting is real.

    Somewhere to fly to that turns out to be a bin bag is a different and more
    interesting problem, and real detectors create it constantly.
    """
    t = OpenGround(n=161)
    rng = Rng(2)
    geo = detector.make("geometric", fov.DOWNWARD, false_alarm_rate=0.5)
    ghosts = 0
    for _ in range(120):
        det = geo(t, drone_at(80, 80, 20), at(80, 80), rng=rng)
        ghosts += int(((det.confidence[0] > 0) & ~det.is_true[0]).sum())
    assert ghosts > 10, "no false alarms at a 50% rate"


def test_a_ghost_is_never_reported_as_certain():
    t = OpenGround(n=161)
    rng = Rng(3)
    geo = detector.make("geometric", fov.DOWNWARD, false_alarm_rate=0.6)
    for _ in range(60):
        det = geo(t, drone_at(80, 80, 20), at(80, 80), rng=rng)
        ghost = (det.confidence[0] > 0) & ~det.is_true[0]
        if bool(ghost.any()):
            assert float(det.confidence[0][ghost].max()) < 0.65


def test_a_detector_that_can_be_wrong_needs_a_generator():
    with pytest.raises(ValueError):
        detector.make("geometric", fov.DOWNWARD)(OpenGround(), drone_at(40, 40, 20), at(40, 40))


def test_the_learned_detector_is_honestly_absent():
    with pytest.raises(ValueError, match="step 5"):
        detector.make("learned", fov.DOWNWARD)
