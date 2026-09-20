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
# Balanced by what each term is *worth over an episode*, not by how the numbers read
# side by side.  Coverage pays a little on every step and proximity pays only once
# somebody is in sight, so equal-looking weights are nothing like equal: at 30
# against 1 the team was paid far more for sweeping empty ground than for closing on
# the person it had already found, and it behaved exactly that way -- the map got
# thoroughly searched and the capture rate never left zero.
DEFAULT = {
    "capture": 200.0,
    "reached_safety": 100.0,
    "evacuee_in_hlr_fov": 1.0,
    "evacuee_in_llr_fov": 2.0,
    "proximity": 20.0,
    "goal_progress": 10.0,
    "role_coordination": 1.0,
    "search_coverage": 0.5,
    "fire_proximity": -1.0,
    "control_effort": -0.01,
    "time_penalty": -0.02,

    # Not the paper's, and needed because the map is not the paper's either.  On a
    # 25 m block the evacuee is in view from the first step; on a real city block
    # finding them is the whole problem, so searching has to pay something before
    # it succeeds.
    "first_sight": 20.0,
    "coverage": 10.0,
    "evacuee_harmed": -100.0,
    # Closeness while in view -- the paper's detection window was 2.5 m, so its
    # "detected" already meant "close"; ours needs to say so separately.
    "apparent_size": 5.0,
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
