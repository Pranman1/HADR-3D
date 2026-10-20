"""Our reward for a real city block: eight terms the paper did not need.

Not part of the tool.  `city_reward.yaml` beside this file names this module, so
`hadrsim-train --reward experiments/city_reward.yaml` registers these terms and
uses that balance; the tool's own default is the paper's reward.

On the paper's 25 m grid the evacuee is in view from the first step and capture ends
the episode.  On a city block finding them is the whole problem, detection from 45 m
away is not closeness, and the rescue is only done when they reach safety.
Each one-off payment here is paid on the *edge* (the step it happens), never on the
latched state -- the latched versions dominated the episode total (findings §1).
"""

from __future__ import annotations

import torch

from hadrsim.reward.terms import View, term

PERSON_HEIGHT_M = 1.7


@term
def reached_safety(v: View) -> torch.Tensor:
    """What the capture was for: the evacuee arriving at safety *after being led*,
    paid once and larger the sooner.  Unled arrivals pay nothing."""
    sooner = 0.5 + 0.5 * v._or_zero("time_left")
    return v._or_zero("newly_reached") * v.was_captured.float() * sooner


@term
def goal_progress(v: View) -> torch.Tensor:
    """The person getting closer to safety, once they have ever been led."""
    distance = (v.person_pos - v.goal).norm(dim=-1)
    return torch.exp(-distance / 40.0) * v.was_captured.float()


@term
def search_coverage(v: View) -> torch.Tensor:
    """Anyone having eyes on at all. Small; it lifts the early episodes off flat zero."""
    return v.person_seen.float() * v.confidence


@term
def first_sight(v: View) -> torch.Tensor:
    """Paid once, the moment anyone first sees the person, larger the sooner.
    The paper reports this as "first seen"; on a city block it is the whole search."""
    return v._or_zero("newly_seen") * v._or_zero("time_left")


@term
def coverage(v: View) -> torch.Tensor:
    """Ground newly looked at this step.  The dense signal that makes searching
    learnable; *new* ground, so hovering over one spot earns nothing."""
    return v._or_zero("new_area")


@term
def apparent_size(v: View) -> torch.Tensor:
    """How large the person is in the low rescuer's camera (pinhole: height/distance,
    saturating when they fill the frame).  Closeness, but only while in view -- a real
    detector reports a box height, so on hardware nothing changes (findings §6)."""
    if v.n_drones < 2:
        return v.zeros()
    dist = (v.drone_pos[:, 1] - v.person_pos).norm(dim=-1)
    size = (PERSON_HEIGHT_M / dist.clamp_min(PERSON_HEIGHT_M)).clamp(0.0, 1.0)
    return size * v.seen_by[:, 1].float() * v.confidence


@term
def evacuee_harmed(v: View) -> torch.Tensor:
    """The failure the rescue exists to prevent. Weighted negative, paid once."""
    return v._or_zero("newly_harmed")


@term
def control_effort(v: View) -> torch.Tensor:
    """How hard the drones are flown. Weighted negative and small: prefer the calm solution."""
    return v.action.pow(2).mean(dim=(1, 2))
