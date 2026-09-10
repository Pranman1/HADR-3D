"""Drone dynamics, the autopilot, and the rotation maths underneath both.

Rungs 1 and 2 of the validation ladder in the plan: internal consistency, then
agreement with the airframe's published specification.  Rung 3 is agreement with
real PX4 in SITL and rung 4 is a real flight log; neither is here yet, and no claim
about real hardware should rest on this file.

The strongest single test is `test_the_same_controller_flies_both_airframes`.  The
X500 is seventy-four times the mass of a Crazyflie with a thousand times the
inertia, and if one set of PX4 gains hovers both then the physics and the control
are consistent in the way that actually matters -- rather than being a controller
tuned until one particular set of numbers stopped falling over.
"""

import math

import pytest
import torch

from hadrsim.agents import airframes as af
from hadrsim.agents.drone import GRAVITY, Drone
from hadrsim.core import rotation as rot

FIDELITIES = ("kinematic", "point_mass", "quadrotor")


def fly(command, seconds=6.0, fidelity="quadrotor", frame=af.X500_V2, n=1, alt=20.0):
    d = Drone(n, frame, fidelity)
    d.reset(torch.tensor([[0.0, 0.0, alt]] * n))
    c = torch.tensor([command] * n)
    for _ in range(int(seconds * 100)):
        d.step(c, 0.01)
    return d


# --------------------------------------------------------------- the airframes

def test_airframe_numbers_are_physically_sensible():
    for frame in (af.X500_V2, af.CRAZYFLIE):
        assert 1.3 < frame.thrust_to_weight < 4.0, f"{frame.name} could not fly"
        assert 0.4 < frame.hover_rotor_rads / frame.max_rotor_rads < 0.85, \
            f"{frame.name} hovers at an implausible throttle"
        roll, pitch, yaw = frame.max_torque
        assert roll == pitch, "roll and pitch authority are symmetric on an X frame"

    # A normal-scale quadrotor rolls far harder than it yaws, because roll comes from
    # thrust across an arm and yaw only from prop drag.  The Crazyflie is the
    # exception and not a mistake: 40 mm arms give it so little leverage that its yaw
    # authority is comparable, which is a real property of the aircraft.
    assert af.X500_V2.max_torque[0] > 4 * af.X500_V2.max_torque[2]


def test_airframes_are_looked_up_forgivingly():
    assert af.get("x500_v2") is af.X500_V2
    assert af.get("X500 V2") is af.X500_V2
    with pytest.raises(KeyError):
        af.get("spitfire")


# ---------------------------------------------------------------- the rotations

def test_attitude_round_trips():
    r, p, y = torch.tensor([0.2]), torch.tensor([-0.3]), torch.tensor([1.0])
    back = rot.to_euler(rot.from_euler(r, p, y))
    assert [round(float(v), 4) for v in back] == [0.2, -0.3, 1.0]


def test_tilt_signs_are_what_the_autopilot_assumes():
    """With z up and body x forward, positive pitch leans the up-vector forward.

    Borrowing the aerospace convention here -- z down, nose-down pitch negative --
    flies the aircraft backwards, which is exactly what it did.
    """
    up = lambda r, p: rot.rotate(
        rot.from_euler(torch.tensor([r]), torch.tensor([p]), torch.zeros(1)),
        torch.tensor([[0.0, 0.0, 1.0]]))[0]
    assert float(up(0.0, 0.35)[0]) > 0.3, "positive pitch must lean forward"
    assert float(up(0.35, 0.0)[1]) < -0.3, "positive roll must lean toward -y"


def test_integrating_a_body_rate_turns_by_the_right_amount():
    q = rot.identity(1)
    for _ in range(1000):
        q = rot.integrate(q, torch.tensor([[0.0, 0.0, 1.0]]), 0.001)
    assert float(rot.yaw_of(q)[0]) == pytest.approx(1.0, abs=1e-3)
    assert float(q.norm()) == pytest.approx(1.0, abs=1e-6), "quaternion drifted off the sphere"


# ------------------------------------------------------------------- all tiers

@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_a_drone_told_to_do_nothing_stays_put(fidelity):
    d = fly([0, 0, 0, 0], seconds=8.0, fidelity=fidelity)
    assert float(d.pos[0, 2]) == pytest.approx(20.0, abs=0.05), "sank or climbed"
    assert float(d.pos[0, :2].norm()) < 0.05, "drifted sideways"


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_forward_means_forward(fidelity):
    d = fly([1, 0, 0, 0], fidelity=fidelity)
    assert float(d.vel[0, 0]) > 3.0, "flew backwards or not at all"
    assert abs(float(d.vel[0, 1])) < 0.5, "drifted sideways"


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_left_means_left(fidelity):
    d = fly([0, 1, 0, 0], fidelity=fidelity)
    assert float(d.vel[0, 1]) > 3.0
    assert abs(float(d.vel[0, 0])) < 0.5


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_climbing_and_descending(fidelity):
    assert float(fly([0, 0, 1, 0], fidelity=fidelity).pos[0, 2]) > 24.0
    assert float(fly([0, 0, -1, 0], fidelity=fidelity).pos[0, 2]) < 16.0


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_speed_stays_inside_the_airframe_limits(fidelity):
    frame = af.X500_V2
    d = fly([1, 1, 1, 1], seconds=10.0, fidelity=fidelity)
    horizontal = float(d.vel[0, :2].norm())
    assert horizontal <= frame.max_speed_mps * 1.15, f"{horizontal:.1f} m/s"
    assert abs(float(d.vel[0, 2])) <= frame.max_climb_mps * 1.15


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_turning_turns_the_right_way(fidelity):
    d = fly([0, 0, 0, 1], seconds=1.0, fidelity=fidelity)
    assert float(d.yaw[0]) > 0.5, "yaw command turned the wrong way"
    assert float(d.pos[0, :2].norm()) < 1.0, "turning should not translate"


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_a_batch_of_drones_is_independent(fidelity):
    d = Drone(3, af.X500_V2, fidelity)
    d.reset(torch.tensor([[0.0, 0.0, 20.0]] * 3))
    cmd = torch.tensor([[1.0, 0, 0, 0], [0.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    for _ in range(300):
        d.step(cmd, 0.01)
    assert float(d.vel[0, 0]) > 1.0 and float(d.vel[2, 0]) < -1.0
    assert abs(float(d.vel[1, 0])) < 0.1


@pytest.mark.parametrize("fidelity", FIDELITIES)
def test_flight_is_deterministic(fidelity):
    a, b = fly([0.6, -0.3, 0.2, 0.4], fidelity=fidelity), fly([0.6, -0.3, 0.2, 0.4], fidelity=fidelity)
    assert torch.equal(a.pos, b.pos) and torch.equal(a.quat, b.quat)


# ---------------------------------------------------------------- the tilt

def test_leaning_to_accelerate_and_levelling_off():
    """The camera is bolted to the airframe, so this is not cosmetic.

    A permanently level camera is the single most misleading thing a point-mass drone
    can do in a project about what drones can see.
    """
    for fidelity in ("point_mass", "quadrotor"):
        d = Drone(1, af.X500_V2, fidelity)
        d.reset(torch.tensor([[0.0, 0.0, 20.0]]))
        cmd = torch.tensor([[1.0, 0, 0, 0]])
        for _ in range(40):
            d.step(cmd, 0.01)
        accelerating = float(d.attitude[1][0])
        for _ in range(1000):
            d.step(cmd, 0.01)
        cruising = float(d.attitude[1][0])
        assert accelerating > math.radians(8), f"{fidelity}: never leaned"
        assert cruising < accelerating, f"{fidelity}: never levelled off"


def test_tilt_is_capped():
    for fidelity in ("point_mass", "quadrotor"):
        d = fly([1, 1, 0, 0], seconds=2.0, fidelity=fidelity)
        tilt = math.degrees(float(rot.tilt_of(d.quat)[0]))
        assert tilt <= af.X500_V2.max_tilt_deg + 2.0, f"{fidelity}: leaned {tilt:.0f} deg"


# ------------------------------------------------------- the quadrotor proper

def test_hovering_takes_exactly_its_own_weight_in_thrust():
    frame = af.X500_V2
    d = Drone(1, frame, "quadrotor")
    d.reset(torch.tensor([[0.0, 0.0, 20.0]]))
    for _ in range(200):
        d.step(torch.zeros(1, 4), 0.01)
    thrust = float((frame.thrust_coeff * d.rotor ** 2).sum())
    assert thrust == pytest.approx(frame.mass_kg * GRAVITY, rel=0.02)
    assert all(abs(float(r) - frame.hover_rotor_rads) < 5 for r in d.rotor[0])


def test_the_same_controller_flies_both_airframes():
    """One set of PX4 gains, a 74x mass difference and a 1400x inertia difference.

    A controller tuned until one particular aircraft stopped falling over would not
    survive this; one that is actually consistent with the physics does.
    """
    for frame in (af.X500_V2, af.CRAZYFLIE):
        d = fly([0, 0, 0, 0], seconds=6.0, frame=frame)
        assert float(d.pos[0, 2]) == pytest.approx(20.0, abs=0.05), frame.name
        assert float(d.pos[0, :2].norm()) < 0.05, frame.name

        d = fly([1, 0, 0, 0], seconds=5.0, frame=frame)
        assert float(d.vel[0, 0]) > 0.4 * frame.max_speed_mps, frame.name


def test_a_quadrotor_cannot_push_sideways_without_leaning():
    """All four rotors point the same way, so horizontal acceleration *is* tilt."""
    d = Drone(1, af.X500_V2, "quadrotor")
    d.reset(torch.tensor([[0.0, 0.0, 20.0]]))
    for _ in range(30):
        d.step(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), 0.01)
    roll, _ = d.attitude
    assert abs(float(roll)) > math.radians(3), "accelerated sideways while level"


def test_motors_do_not_change_speed_instantly():
    frame = af.X500_V2
    d = Drone(1, frame, "quadrotor")
    d.reset(torch.tensor([[0.0, 0.0, 20.0]]))
    before = d.rotor.clone()
    d.step(torch.tensor([[0.0, 0.0, 1.0, 0.0]]), 0.01)
    change = float((d.rotor - before).abs().max())
    assert 0.0 < change < 200.0, "motor lag missing or absurd"


def test_kinematic_really_is_instant():
    """The paper's drones teleport; that tier exists to reproduce the paper."""
    d = Drone(1, af.X500_V2, "kinematic")
    d.reset(torch.tensor([[0.0, 0.0, 20.0]]))
    d.step(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 0.01)
    assert float(d.vel[0, 0]) == pytest.approx(af.X500_V2.max_speed_mps)
