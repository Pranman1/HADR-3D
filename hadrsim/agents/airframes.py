"""Real drones, as tables of numbers.

The equations of motion in `drone.py` are the same for every quadrotor ever built.
What differs between one aircraft and another is exactly this: mass, arm length,
inertia, how much thrust a rotor makes, how fast the motors respond.  Keeping them
here means adding an airframe is adding ten numbers, and it means the simulated
drone is a *specific* drone rather than a plausible one.

**X500 V2 is the default** because it is the standard PX4 research quadrotor -- a
real machine the group could buy, big enough to carry a camera and a companion
computer, and the one PX4's own SITL model describes.  Matching PX4's parameters is
what makes "validate against SITL" mean anything.

**Crazyflie is here as a cross-check**, not as a candidate: at 27 g it cannot carry
a camera, so it is the wrong scale for this problem entirely.  It earns its place
because it is the best-characterised quadrotor in the literature, so if the dynamics
are right for both a 2 kg machine and a 27 g one, they are probably right.

A caveat worth keeping: inertias are the least reliable numbers here.  Mass and arm
length are measurable with a scale and a ruler; a full inertia tensor is not, and
published values for a given frame vary.  Anything that turns on precise attitude
response should be checked against a real flight log first -- see the plan's rung 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Airframe:
    name: str
    mass_kg: float
    arm_m: float              # centre to rotor, along the arm
    inertia: tuple            # (Ixx, Iyy, Izz), kg m^2, body axes
    thrust_coeff: float       # thrust = k * omega^2, newtons per (rad/s)^2
    moment_coeff: float       # rotor drag torque as a fraction of its thrust
    motor_tau_s: float        # first-order lag from commanded to actual rotor speed
    max_rotor_rads: float
    drag_coeff: float         # body drag, newtons per (m/s)^2
    max_tilt_deg: float = 35.0        # how far the autopilot will lean it
    max_speed_mps: float = 12.0
    max_climb_mps: float = 3.0
    max_yaw_rate_dps: float = 90.0

    @property
    def weight_n(self) -> float:
        return self.mass_kg * 9.81

    @property
    def max_thrust_n(self) -> float:
        """All four rotors flat out."""
        return 4.0 * self.thrust_coeff * self.max_rotor_rads ** 2

    @property
    def thrust_to_weight(self) -> float:
        return self.max_thrust_n / self.weight_n

    @property
    def max_torque(self) -> tuple:
        """(roll, pitch, yaw) torque the rotors can actually produce, in N m.

        PX4's rate gains are normalised: its controller outputs a number in [-1, 1]
        that the mixer scales to whatever the airframe can do.  Converting those
        gains into physical torque needs that scale, and inventing one instead is how
        a stable controller becomes an unstable one.

        Roll and pitch come from running one side of the aircraft flat out and the
        other at nothing; yaw comes from the rotors' drag torques, which is far
        weaker -- and is why a quadrotor turns much more slowly than it rolls.
        """
        per_rotor = self.thrust_coeff * self.max_rotor_rads ** 2
        offset = self.arm_m / 2 ** 0.5
        lateral = 2.0 * per_rotor * offset
        return (lateral, lateral, 4.0 * per_rotor * self.moment_coeff)

    @property
    def hover_rotor_rads(self) -> float:
        """Rotor speed that exactly holds the aircraft up. The autopilot's anchor."""
        return (self.weight_n / (4.0 * self.thrust_coeff)) ** 0.5


# PX4's own SITL description of this airframe. 500 mm wheelbase, so the rotors sit
# 0.25 m from the centre and +-0.177 m along each body axis.
X500_V2 = Airframe(
    name="Holybro X500 V2",
    mass_kg=2.0,
    arm_m=0.25,
    inertia=(0.02, 0.02, 0.04),
    thrust_coeff=8.55e-6,
    moment_coeff=0.016,
    motor_tau_s=0.033,
    max_rotor_rads=1000.0,
    drag_coeff=0.10,
)

# Crazyflie 2.x, from the parameters gym-pybullet-drones and RotorPy both use.
CRAZYFLIE = Airframe(
    name="Crazyflie 2.1",
    mass_kg=0.027,
    arm_m=0.0397,
    inertia=(1.4e-5, 1.4e-5, 2.17e-5),
    thrust_coeff=3.16e-10 * (60 / (2 * 3.14159)) ** 2,   # published per-RPM, wanted per-rad/s
    moment_coeff=0.0251,
    motor_tau_s=0.015,
    max_rotor_rads=21714.0 * 2 * 3.14159 / 60,
    drag_coeff=0.001,
    max_tilt_deg=30.0,
    max_speed_mps=3.0,
    max_climb_mps=1.5,
)

BY_NAME = {"x500_v2": X500_V2, "crazyflie": CRAZYFLIE}


def get(name: str) -> Airframe:
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in BY_NAME:
        raise KeyError(f"no airframe {name!r}; have {sorted(BY_NAME)}")
    return BY_NAME[key]
