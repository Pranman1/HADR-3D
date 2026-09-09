"""Time, in seconds.

The paper welds every clock together: one step is 5 drone sub-moves, 3 person
sub-moves and 1 fire tick, capped at 100 steps.  That is fine on a grid and
meaningless in continuous time, so here each rate is stated in Hz and everything
else is derived.

Three rates, not one:

    physics   100 Hz   the drone's equations of motion
    control     2 Hz   how often the policy gets to decide
    fire        1 Hz   how often the fire tries to spread

They are separate because they are separate in reality: a real drone's autopilot
runs far faster than any planner above it, and a fire does not care about either.
Keeping them apart is also what lets the running environment honour a real
sense-to-act delay without the training environment having to model one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    physics_hz: float = 100.0
    control_hz: float = 2.0
    fire_hz: float = 1.0
    episode_seconds: float = 120.0

    def __post_init__(self):
        if self.physics_hz < self.control_hz:
            raise ValueError("physics must step at least as often as the policy decides")

    @property
    def dt(self) -> float:
        """One physics step, in seconds."""
        return 1.0 / self.physics_hz

    @property
    def control_dt(self) -> float:
        """How long one policy decision lasts, in seconds."""
        return 1.0 / self.control_hz

    @property
    def substeps(self) -> int:
        """Physics steps per policy decision. 50, at the default rates."""
        return max(1, round(self.physics_hz / self.control_hz))

    @property
    def max_steps(self) -> int:
        """Policy decisions in one episode. 240, at the default rates.

        The paper's cap was 100 steps of a quite different kind, so this number is
        not comparable to it and should not be quoted as if it were.
        """
        return max(1, round(self.episode_seconds * self.control_hz))

    def describe(self) -> str:
        return (
            f"{self.physics_hz:g} Hz physics, {self.control_hz:g} Hz control "
            f"({self.substeps} substeps), {self.fire_hz:g} Hz fire, "
            f"{self.episode_seconds:g} s = {self.max_steps} decisions"
        )
