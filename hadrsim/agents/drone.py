"""How a drone moves.

Three fidelity levels behind one interface.  The policy's command is identical at
every level -- a velocity setpoint in the drone's own frame, plus a turn rate -- so
the level can be changed between training and evaluation without retraining, which
is the whole reason for arranging it this way.

    kinematic    position moves at the commanded speed.  No physics at all; exists
                 to reproduce the paper, whose drones teleport between cells.
    point_mass   mass, drag, acceleration and turn-rate limits.  Captures inertia:
                 it cannot stop or turn instantly.  The training default.
    quadrotor    six degrees of freedom, four rotors, thrust and torque, motor lag,
                 real attitude.  What a real machine does.

**Written batched, not on a physics engine.**  Isaac Lab and MuJoCo exist to solve
*contact* -- things bumping, gripping, resting on each other -- and our drones fly
through open air and touch nothing.  Buying a contact solver costs a heavy
dependency, VRAM the GPU needs for training, and the batching that makes thousands
of parallel environments possible, in exchange for the one feature we never use.
The equations below are standard and published (Mellinger & Kumar); what makes the
aircraft a *particular* aircraft is the table in `airframes.py`.

**The tilt matters here more than it usually does.**  A real quadrotor has to lean
to accelerate, and the camera is bolted to the airframe, so where it looks depends
on how it is moving.  This whole project is about what the drones can see, so
`point_mass` derives a tilt from its commanded acceleration -- a real quadrotor's
lean is essentially determined by acceleration anyway -- rather than flying
permanently level, which no real drone does.
"""

from __future__ import annotations

import torch

from hadrsim.agents.airframes import Airframe, X500_V2
from hadrsim.core import rotation as rot

GRAVITY = 9.81

# X configuration, seen from above, +x forward and +y left. Rotors 0 and 2 spin one
# way and 1 and 3 the other, so their drag torques cancel in a hover and differ when
# the aircraft is asked to turn.
ROTOR_ANGLES = (0.25, 0.75, 1.25, 1.75)     # in units of pi
ROTOR_SPIN = (1.0, -1.0, 1.0, -1.0)


class Drone:
    """A batch of drones sharing one airframe and one fidelity level."""

    def __init__(self, batch: int, airframe: Airframe = X500_V2,
                 fidelity: str = "point_mass", device="cpu"):
        if fidelity not in ("kinematic", "point_mass", "quadrotor"):
            raise ValueError(f"unknown fidelity {fidelity!r}")
        self.batch = batch
        self.af = airframe
        self.fidelity = fidelity
        self.device = torch.device(device)

        self.pos = torch.zeros(batch, 3, device=self.device)
        self.vel = torch.zeros(batch, 3, device=self.device)
        self.quat = rot.identity(batch, self.device)
        self.omega = torch.zeros(batch, 3, device=self.device)
        self.rotor = torch.full((batch, 4), airframe.hover_rotor_rads, device=self.device)

        self._geometry()
        self._autopilot = None
        if fidelity == "quadrotor":
            from hadrsim.agents.autopilot import Autopilot
            self._autopilot = Autopilot(airframe, batch, self.device)

    def confine(self, x_max: float, y_max: float,
                alt_min: float = 2.0, alt_max: float = 120.0) -> None:
        """A geofence: the box the aircraft is allowed to be in.

        Real drones have all of this -- ground underneath, a regulatory ceiling
        (120 m almost everywhere), and an operator-set boundary -- so it is missing
        physics rather than a training convenience.

        It also closes a degenerate solution the environment was otherwise offering:
        with nothing stopping it, a policy learned to hold full vertical command and
        fly one drone to 286 metres *underground* and the other to 332 metres up,
        where neither can see anything and neither is scored on anything.  It is a
        perfectly good answer to the question that was actually being asked.

        Velocity into a limit is zeroed rather than kept, so an aircraft pinned to
        the fence stops pressing on it instead of accumulating speed it can never use.
        """
        low = torch.tensor([0.0, 0.0, alt_min], device=self.device)
        high = torch.tensor([x_max, y_max, alt_max], device=self.device)
        clamped = torch.maximum(torch.minimum(self.pos, high), low)
        self.vel = torch.where(clamped != self.pos, torch.zeros_like(self.vel), self.vel)
        self.pos = clamped

    def _geometry(self) -> None:
        """Where the rotors are, and how their thrust becomes force and torque."""
        import math

        af = self.af
        angles = torch.tensor([a * math.pi for a in ROTOR_ANGLES], device=self.device)
        self.rotor_xy = torch.stack([angles.cos(), angles.sin()], -1) * af.arm_m
        self.spin = torch.tensor(ROTOR_SPIN, device=self.device)
        self.inertia = torch.tensor(af.inertia, device=self.device)

    # ------------------------------------------------------------------ state

    def reset(self, pos: torch.Tensor, yaw: torch.Tensor | None = None) -> None:
        self.pos = pos.to(self.device).float().clone()
        yaw = torch.zeros(self.batch, device=self.device) if yaw is None else yaw
        self.quat = rot.from_yaw(yaw.to(self.device).float())
        self.vel.zero_()
        self.omega.zero_()
        self.rotor.fill_(self.af.hover_rotor_rads)
        if self._autopilot is not None:
            self._autopilot.reset()

    @property
    def yaw(self) -> torch.Tensor:
        return rot.yaw_of(self.quat)

    @property
    def attitude(self):
        """(roll, pitch) -- where the camera is looking, relative to level."""
        roll, pitch, _ = rot.to_euler(self.quat)
        return roll, pitch

    @property
    def speed(self) -> torch.Tensor:
        return self.vel.norm(dim=-1)

    # ------------------------------------------------------------------- step

    def step(self, command: torch.Tensor, dt: float) -> None:
        """Advance by `dt`. `command` is (B, 4) in [-1, 1].

        Forward, lateral and vertical speed in the drone's own frame, plus turn rate.
        Deliberately the native PX4 offboard setpoint, so what the policy emits in
        simulation is what you would send a real aircraft.
        """
        cmd = command.to(self.device).clamp(-1.0, 1.0)
        want_body = cmd[:, :3] * torch.tensor(
            [self.af.max_speed_mps, self.af.max_speed_mps, self.af.max_climb_mps],
            device=self.device)
        # The airframe's speed limit is on the whole velocity, not on each axis: full
        # forward *and* full left is not a licence to fly 1.4 times as fast.
        want_body = torch.cat(
            [_clamp_norm(want_body[:, :2], self.af.max_speed_mps), want_body[:, 2:]], -1)
        want_world = rot.rotate(rot.from_yaw(self.yaw), want_body)
        yaw_rate = cmd[:, 3] * torch.deg2rad(
            torch.tensor(self.af.max_yaw_rate_dps, device=self.device))

        if self.fidelity == "kinematic":
            self._step_kinematic(want_world, yaw_rate, dt)
        elif self.fidelity == "point_mass":
            self._step_point_mass(want_world, yaw_rate, dt)
        else:
            self._step_quadrotor(want_world, yaw_rate, dt)

    def _step_kinematic(self, want, yaw_rate, dt) -> None:
        """No physics. The paper's drones, for reproducing the paper's numbers."""
        self.vel = want
        self.pos = self.pos + want * dt
        self.quat = rot.from_yaw(self.yaw + yaw_rate * dt)

    def _step_point_mass(self, want, yaw_rate, dt) -> None:
        """Inertia and drag, and a tilt inferred from what it is doing.

        The lean is not decoration.  A real quadrotor accelerating forward is leaning
        forward by roughly atan(a/g), and the camera leans with it; a permanently
        level camera is the single most misleading thing a point-mass drone can do in
        a project about what drones can see.
        """
        af = self.af
        drag = af.drag_coeff / af.mass_kg * self.vel * self.vel.norm(dim=-1, keepdim=True)
        accel = (want - self.vel) / 0.35 - drag
        limit = GRAVITY * torch.tan(torch.deg2rad(torch.tensor(af.max_tilt_deg)))
        accel = torch.cat([_clamp_norm(accel[:, :2], float(limit)), accel[:, 2:]], -1)

        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt

        yaw = self.yaw + yaw_rate * dt
        want_lean = rot.rotate_inv(rot.from_yaw(yaw), accel)
        g = torch.tensor(GRAVITY, device=self.device)
        # Signs as `rotation` defines them, not as aerospace does: positive pitch
        # leans the up-vector toward +x and positive roll leans it toward -y.  The
        # autopilot had this backwards too, and it flew the aircraft in reverse.
        pitch = torch.atan2(want_lean[:, 0], g)
        roll = -torch.atan2(want_lean[:, 1], g)
        self.quat = rot.from_euler(roll, pitch, yaw)
        self.omega = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), yaw_rate], -1)

    def _step_quadrotor(self, want, yaw_rate, dt) -> None:
        """Six degrees of freedom, driven by real rotors under a real autopilot."""
        target = self._autopilot.rotor_speeds(self, want, yaw_rate, dt)

        # Motors do not change speed instantly, and the lag is a large part of why a
        # quadrotor feels the way it does.
        alpha = 1.0 - torch.exp(torch.tensor(-dt / self.af.motor_tau_s, device=self.device))
        self.rotor = self.rotor + alpha * (target - self.rotor)
        self.rotor = self.rotor.clamp(0.0, self.af.max_rotor_rads)

        thrust = self.af.thrust_coeff * self.rotor ** 2           # (B, 4)
        total = thrust.sum(-1)

        up_body = torch.zeros(self.batch, 3, device=self.device)
        up_body[:, 2] = 1.0
        force = rot.rotate(self.quat, up_body * total[:, None])
        force[:, 2] -= self.af.mass_kg * GRAVITY
        force = force - self.af.drag_coeff * self.vel * self.vel.norm(dim=-1, keepdim=True)

        # Roll and pitch come from thrust differences across the arms; yaw comes from
        # the rotors' drag torques, which is why a quadrotor turns so much more slowly
        # than it rolls.
        tau_x = (thrust * self.rotor_xy[:, 1]).sum(-1)
        tau_y = -(thrust * self.rotor_xy[:, 0]).sum(-1)
        tau_z = (thrust * self.spin).sum(-1) * self.af.moment_coeff
        torque = torch.stack([tau_x, tau_y, tau_z], -1)

        gyroscopic = torch.cross(self.omega, self.inertia * self.omega, dim=-1)
        self.omega = self.omega + (torque - gyroscopic) / self.inertia * dt
        self.quat = rot.integrate(self.quat, self.omega, dt)
        self.vel = self.vel + force / self.af.mass_kg * dt
        self.pos = self.pos + self.vel * dt


def _clamp_norm(v: torch.Tensor, limit: float) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return v * (limit / n.clamp_min(1e-9)).clamp(max=1.0)
