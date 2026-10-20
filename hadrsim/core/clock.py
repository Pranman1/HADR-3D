"""Rates and the episode length. Physics 100 Hz, person 10 Hz, policy 2 Hz, fire 1 Hz."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    physics_hz: float = 100.0
    human_hz: float = 10.0
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
        """Physics steps per policy decision."""
        return max(1, round(self.physics_hz / self.control_hz))

    @property
    def human_substeps(self) -> int:
        """Physics steps between one decision by the person and the next."""
        return max(1, round(self.physics_hz / self.human_hz))

    @property
    def max_steps(self) -> int:
        """Policy decisions in one episode."""
        return max(1, round(self.episode_seconds * self.control_hz))
