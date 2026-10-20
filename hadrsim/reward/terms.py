"""Reward terms: one function per thing we pay for, each reading a `View` of what the
team believes, never the world.  A term returns a magnitude >= 0; the weight carries
the sign.  Register your own with `@term` in a module named by your weights yaml."""

from __future__ import annotations

from dataclasses import dataclass

import torch

TERMS: dict[str, callable] = {}


def term(fn):
    """Register `fn(view) -> (B,) tensor` under its own name."""
    TERMS[fn.__name__] = fn
    return fn


@dataclass
class View:
    """Everything a reward may look at: the team's beliefs, not the truth."""

    drone_pos: torch.Tensor        # (B, D, 3)
    person_pos: torch.Tensor       # (B, 3) believed
    person_seen: torch.Tensor      # (B,) bool -- has anyone got them right now
    seen_by: torch.Tensor          # (B, D) bool -- which drones can see them
    confidence: torch.Tensor       # (B,) best confidence anyone has
    goal: torch.Tensor             # (B, 3) where safety is
    captured: torch.Tensor         # (B,) bool -- newly, this step
    was_captured: torch.Tensor     # (B,) bool -- at any point this episode
    fire_near: torch.Tensor        # (B, D) fire proximity at each drone, 0..1
    action: torch.Tensor           # (B, D, 4)
    harmed: torch.Tensor | None = None      # (B,) the evacuee was caught by the fire
    new_area: torch.Tensor | None = None    # (B,) fraction of map newly searched
    newly_seen: torch.Tensor | None = None  # (B,) spotted for the first time this step
    time_left: torch.Tensor | None = None   # (B,) 1 at the start, 0 at the deadline
    belief_age: torch.Tensor | None = None  # (B,) seconds since the last sighting
    newly_reached: torch.Tensor | None = None  # (B,) arrived at safety this step
    newly_harmed: torch.Tensor | None = None   # (B,) caught by the fire this step

    def _or_zero(self, field) -> torch.Tensor:
        v = getattr(self, field)
        return self.zeros() if v is None else v.float()

    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.batch, device=self.drone_pos.device)

    @property
    def batch(self) -> int:
        return len(self.drone_pos)

    @property
    def n_drones(self) -> int:
        return self.drone_pos.shape[1]
