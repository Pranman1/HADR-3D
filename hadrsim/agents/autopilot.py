"""The autopilot: turning "go this fast" into four rotor speeds.

PX4's cascade, with PX4's default gains, because PX4 is the firmware a real X500
actually runs.  Imitating it rather than inventing something means the thing the
policy learns to command is the thing a real aircraft will do with that command.

    velocity setpoint  ->  PID          <- the policy plugs in HERE
    acceleration       ->  tilt + thrust
    attitude           ->  P
    body rates         ->  PID
                       ->  mixer  ->  four rotor speeds

Only the middle of the stack: position control sits above (the policy does that job)
and the motors below.

**This is for training.**  Evaluation runs the real firmware in SITL over PX4's own
simulator interface, which is stronger than matching it -- see the plan, section
8.2b.  What this has to be is fast enough for thousands of aircraft in parallel and
close enough that a policy trained against it is not surprised by the real thing.

The one structural thing worth knowing about a quadrotor: **it cannot push
sideways.**  All four rotors point the same way, so the only way to accelerate
horizontally is to tilt and use part of the thrust sideways -- which is why
`_attitude_from_accel` builds an attitude out of a desired acceleration rather than
tracking one it was given.
"""

from __future__ import annotations

import torch

from hadrsim.agents.airframes import Airframe
from hadrsim.core import rotation as rot

GRAVITY = 9.81

# PX4 multicopter defaults (MPC_*_VEL_*_ACC, MC_*_P, MC_*RATE_*).
VEL_P_XY, VEL_I_XY, VEL_D_XY = 1.8, 0.4, 0.2
VEL_P_Z, VEL_I_Z, VEL_D_Z = 4.0, 2.0, 0.0
ATT_P_ROLL, ATT_P_PITCH, ATT_P_YAW = 6.5, 6.5, 2.8
RATE_P = (0.15, 0.15, 0.20)
RATE_I = (0.20, 0.20, 0.10)
RATE_D = (0.003, 0.003, 0.0)


class Autopilot:
    """One PX4-style controller per drone, all stepped together."""

    def __init__(self, airframe: Airframe, batch: int, device="cpu"):
        self.af = airframe
        self.batch = batch
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        z3 = lambda: torch.zeros(self.batch, 3, device=self.device)
        self._vel_integral = z3()
        self._rate_integral = z3()
        self._last_vel_error = z3()
        self._last_rate_error = z3()   # holds the last measured rate, not the error

    # ------------------------------------------------------------------- step

    def rotor_speeds(self, drone, want_vel_world: torch.Tensor,
                     yaw_rate: torch.Tensor, dt: float) -> torch.Tensor:
        """(B, 4) rotor speeds that pursue the requested velocity."""
        accel = self._velocity_loop(drone.vel, want_vel_world, dt)
        want_q, thrust_n = self._attitude_from_accel(accel, drone.yaw + yaw_rate * dt)
        want_rate = self._attitude_loop(drone.quat, want_q, yaw_rate)
        torque = self._rate_loop(drone.omega, want_rate, dt)
        return self._mix(thrust_n, torque, drone)

    # ------------------------------------------------------------------ loops

    def _velocity_loop(self, vel, want, dt) -> torch.Tensor:
        """Velocity error to desired acceleration. PID, with the integral bounded."""
        error = want - vel
        self._vel_integral = (self._vel_integral + error * dt).clamp(-4.0, 4.0)
        derivative = (error - self._last_vel_error) / max(dt, 1e-6)
        self._last_vel_error = error

        gains_p = torch.tensor([VEL_P_XY, VEL_P_XY, VEL_P_Z], device=self.device)
        gains_i = torch.tensor([VEL_I_XY, VEL_I_XY, VEL_I_Z], device=self.device)
        gains_d = torch.tensor([VEL_D_XY, VEL_D_XY, VEL_D_Z], device=self.device)
        return gains_p * error + gains_i * self._vel_integral + gains_d * derivative

    def _attitude_from_accel(self, accel, want_yaw):
        """The attitude that would produce this acceleration, and the thrust to go with it.

        A quadrotor cannot push sideways: its thrust is always along its own up axis,
        so wanting to accelerate forward *is* wanting to lean forward.  The desired
        body-up is therefore the direction of (acceleration + gravity), and the
        aircraft is tilted to point at it -- capped, because past a certain lean there
        is not enough thrust left to hold the height.
        """
        want_up = accel.clone()
        want_up[:, 2] = want_up[:, 2] + GRAVITY

        max_tilt = torch.deg2rad(torch.tensor(self.af.max_tilt_deg, device=self.device))
        horizontal = want_up[:, :2].norm(dim=-1)
        limit = want_up[:, 2].clamp_min(1.0) * torch.tan(max_tilt)
        scale = (limit / horizontal.clamp_min(1e-6)).clamp(max=1.0)
        want_up = torch.cat([want_up[:, :2] * scale[:, None], want_up[:, 2:]], -1)

        norm = want_up.norm(dim=-1).clamp_min(1e-6)
        body_up = want_up / norm[:, None]
        thrust_n = (norm * self.af.mass_kg).clamp(0.0, self.af.max_thrust_n)

        # Attitude from a desired up-vector and a desired heading.  Rotate the wanted
        # up into the yaw frame first, so roll and pitch describe the lean relative to
        # where the aircraft is pointing rather than to the world.
        #
        # Signs verified against the rotation module rather than assumed: with z up
        # and body x forward, *positive* pitch tilts the up-vector toward +x and
        # positive roll tilts it toward -y.  Borrowing the aerospace convention here,
        # where z is down and nose-down pitch is negative, flies the aircraft
        # backwards -- which is exactly what it did.
        up_yaw = rot.rotate_inv(rot.from_yaw(want_yaw), body_up)
        pitch = torch.atan2(up_yaw[:, 0], up_yaw[:, 2].clamp_min(1e-3))
        roll = torch.atan2(-up_yaw[:, 1], up_yaw[:, 2].clamp_min(1e-3))
        return rot.from_euler(roll, pitch, want_yaw), thrust_n

    def _attitude_loop(self, quat, want_quat, yaw_rate) -> torch.Tensor:
        """Attitude error to body rates. Proportional, on the shortest rotation."""
        error = rot.multiply(_conjugate(quat), want_quat)
        # Both q and -q are the same attitude; pick the one that turns the short way.
        error = torch.where((error[:, :1] < 0).expand_as(error), -error, error)
        gains = torch.tensor([ATT_P_ROLL, ATT_P_PITCH, ATT_P_YAW], device=self.device)
        rate = 2.0 * gains * error[:, 1:]
        return torch.cat([rate[:, :2], (rate[:, 2:] + yaw_rate[:, None])], -1)

    def _rate_loop(self, omega, want_rate, dt) -> torch.Tensor:
        """Body-rate error to torque. The innermost and fastest loop.

        Two details that decide whether this is stable:

        **PX4's gains are normalised**, not physical.  Its rate controller outputs a
        number in [-1, 1] which the mixer scales to whatever torque the airframe can
        make, so turning those gains into newton-metres needs the airframe's actual
        limit.  Picking a scale factor by eye instead produced torques an order of
        magnitude too large, and a drone that could not hold a hover.

        **The derivative acts on the measurement, not on the error**, which is what
        PX4 does.  Differentiating the error means every change of setpoint arrives as
        a spike, and here the setpoint changes at every control step.
        """
        error = want_rate - omega
        self._rate_integral = (self._rate_integral + error * dt).clamp(-0.5, 0.5)
        d_measured = -(omega - self._last_rate_error) / max(dt, 1e-6)
        self._last_rate_error = omega

        gp = torch.tensor(RATE_P, device=self.device)
        gi = torch.tensor(RATE_I, device=self.device)
        gd = torch.tensor(RATE_D, device=self.device)
        limit = torch.tensor(self.af.max_torque, device=self.device)
        u = (gp * error + gi * self._rate_integral + gd * d_measured).clamp(-1.0, 1.0)
        return u * limit

    def _mix(self, thrust_n, torque, drone) -> torch.Tensor:
        """Thrust and torque to four rotor speeds.

        The inverse of the geometry in `drone.py`: each rotor contributes a quarter of
        the thrust, plus or minus its share of each torque according to where it sits
        and which way it spins.
        """
        arm = drone.rotor_xy                                   # (4, 2)
        af = self.af

        share = thrust_n[:, None] / 4.0
        share = share + torque[:, 0:1] * arm[:, 1] / (4 * af.arm_m ** 2 / 2)
        share = share - torque[:, 1:2] * arm[:, 0] / (4 * af.arm_m ** 2 / 2)
        share = share + torque[:, 2:3] * drone.spin / (4 * af.moment_coeff)

        # Thrust is proportional to the square of rotor speed, so inverting it is a
        # square root -- and a rotor cannot produce negative thrust.
        return (share.clamp_min(0.0) / af.thrust_coeff).sqrt().clamp(
            0.0, af.max_rotor_rads)


def _conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)
