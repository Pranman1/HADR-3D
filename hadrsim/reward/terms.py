"""Reward terms: one small function per thing we want, each reading a `View`.

Three rules, each learned the hard way (OVERNIGHT_FINDINGS.md §1, §6):

- **A term reads the team's beliefs, never the world.**  It is only ever handed a
  `View`, so degrading the drones' vision degrades the reward too.
- **A term returns a magnitude (>= 0); the weight carries the sign.**  A negative
  term with a negative weight cancels silently.
- **Weights live in `configs/reward/*.yaml`**, and every term's contribution is
  recorded separately, so "which term paid for that?" is a lookup.

`paper.py` holds the terms with a counterpart in the 2D paper (l0, l1, l2, l3, l7, the
time decay and the fire penalty); `ours.py` the ones a real city block needed.
Register a new term with `@term` in either file, or in your own module.
"""

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
    """Everything a reward may look at: what the team believes, not what is true.
    Where nobody has seen anything, `person_seen` is false and `person_pos` is noise."""

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
    dt: float = 0.5
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
