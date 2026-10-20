"""Adding the terms up, with the working shown.

    Composer()                        # configs/reward/paper.yaml, the default
    Composer("my_reward.yaml")        # your table; a `terms:` line names a .py of @term functions
    Composer({"capture": 1.0})        # any dict of term -> weight

Every term's contribution is kept separately in `Reward.parts`: when a policy learns
something strange, the first question is which term paid for it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from hadrsim.reward import paper  # noqa: F401  -- registers the paper's terms
from hadrsim.reward.terms import TERMS, View

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "reward"


def load_weights(name_or_path: str = "paper") -> dict[str, float]:
    """A weight table by name (`configs/reward/<name>.yaml`) or by path.  A `terms:`
    entry is a .py file next to the yaml whose `@term` functions get registered."""
    p = Path(name_or_path)
    p = p if p.suffix == ".yaml" else CONFIGS / f"{name_or_path}.yaml"
    table = yaml.safe_load(p.read_text())
    if module := table.pop("terms", None):
        s = importlib.util.spec_from_file_location(Path(module).stem, p.parent / module)
        s.loader.exec_module(importlib.util.module_from_spec(s))
    return {k: float(v) for k, v in table.items()}


@dataclass
class Reward:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {k: round(float(v.mean()), 4) for k, v in self.parts.items()}


class Composer:
    """A weighted sum of named terms."""

    def __init__(self, weights: dict | str | None = None):
        self.weights = dict(weights) if isinstance(weights, dict) else load_weights(weights or "paper")
        unknown = set(self.weights) - set(TERMS)
        if unknown:
            raise KeyError(f"no such reward term: {sorted(unknown)}")

    def __call__(self, view: View) -> Reward:
        parts = {name: TERMS[name](view) * w for name, w in self.weights.items() if w != 0.0}
        return Reward(sum(parts.values(), view.zeros()), parts)
