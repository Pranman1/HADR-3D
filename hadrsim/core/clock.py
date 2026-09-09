"""Time, in seconds.

The paper welds every clock together: one step is 5 drone sub-moves, 3 person
sub-moves and 1 fire tick, capped at 100 steps.  That is fine on a grid and
meaningless in continuous time, so here each rate is stated in Hz and everything
else is derived.

Four rates, not one:

    physics   100 Hz   bodies move: integration of the equations of motion
    human      10 Hz   how often the person reconsiders where to go
    control     2 Hz   how often the *policy* gets to decide
    fire        1 Hz   how often the fire tries to spread

They are separate because they are separate in reality: a real drone's autopilot
runs far faster than any planner above it, a person re-aims several times a second
without replanning their route, and a fire cares about none of it.

**Deciding slowly is not the same as moving slowly.**  The policy choosing twice a
second does not mean anything jumps twice a second -- between decisions, bodies keep
integrating at 100 Hz under the last command they were given.  That is what makes
the motion continuous rather than a sequence of teleports, and it is why anything
that steps the world at the control rate alone looks, and is, wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    physics_hz: float = 100.0
    human_hz: float = 10.0
    control_hz: float = 2.0
    fire_hz: float = 1.0
    episode_seconds: float = 120.0
    viz_hz: float = 20.0        # how often a watcher gets a frame

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
    def human_substeps(self) -> int:
        """Physics steps between one decision by the person and the next. 10."""
        return max(1, round(self.physics_hz / self.human_hz))

    @property
    def viz_substeps(self) -> int:
        """Physics steps per drawn frame. 5, i.e. 20 frames a second."""
        return max(1, round(self.physics_hz / self.viz_hz))

    @property
    def max_steps(self) -> int:
        """Policy decisions in one episode. 240, at the default rates.

        The paper's cap was 100 steps of a quite different kind, so this number is
        not comparable to it and should not be quoted as if it were.
        """
        return max(1, round(self.episode_seconds * self.control_hz))

    def describe(self) -> str:
        return (
            f"{self.physics_hz:g} Hz physics, {self.human_hz:g} Hz human, "
            f"{self.control_hz:g} Hz control ({self.substeps} substeps), "
            f"{self.fire_hz:g} Hz fire, {self.episode_seconds:g} s "
            f"= {self.max_steps} decisions"
        )
