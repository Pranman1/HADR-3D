"""Adding the terms up, and keeping a record of what each one paid.

Two jobs beyond the arithmetic, and the second is the reason this is not one line.

**Weights come from the scenario**, so trying a different balance is an edit to a
yaml file rather than a change to code -- which is the whole point of reward terms
being a list of named functions.

**Every term's contribution is recorded separately.**  When a policy learns
something strange, the first question is always which term paid for it, and a scalar
sum cannot answer that.  The breakdown costs nothing to keep and turns "the reward
went up but the behaviour got worse" from a mystery into a lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hadrsim.reward.terms import ALL, View

# Weights that get a policy off the ground.  The paper's SpeedScaled values for the
# terms it has (l0, l1, l2, l3 = 1, 1, 1, 10), plus the ones it does not: a large
# payment for the capture itself, a small one for merely having eyes on -- without
# which early episodes are flat and nothing is learned about searching -- and light
# penalties for fire, effort and time.
DEFAULT = {
    "capture": 200.0,
    "reached_safety": 100.0,
    "evacuee_in_hlr_fov": 1.0,
    "evacuee_in_llr_fov": 1.0,
    "proximity": 1.0,
    "goal_progress": 10.0,
    "role_coordination": 1.0,
    "search_coverage": 0.5,
    "fire_proximity": -1.0,
    "control_effort": -0.01,
    "time_penalty": -0.02,
}


@dataclass
class Reward:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {k: round(float(v.mean()), 4) for k, v in self.parts.items()}


class Composer:
    """A weighted sum of named terms, with the working shown."""

    def __init__(self, weights: dict | None = None):
        self.weights = dict(DEFAULT if weights is None else weights)
        unknown = set(self.weights) - set(ALL)
        if unknown:
            raise KeyError(f"no such reward term: {sorted(unknown)}")

    def __call__(self, view: View) -> Reward:
        total = torch.zeros(view.batch, device=view.drone_pos.device)
        parts = {}
        for name, weight in self.weights.items():
            if weight == 0.0:
                continue
            value = ALL[name](view) * weight
            parts[name] = value
            total = total + value
        return Reward(total, parts)

    def without(self, *names: str) -> "Composer":
        """The same scoring with some terms switched off. For ablations."""
        return Composer({k: v for k, v in self.weights.items() if k not in names})
