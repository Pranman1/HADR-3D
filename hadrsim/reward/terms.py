"""Scoring: one small function per thing we want.

Weights live in the scenario file, so trying a different balance is an edit rather
than a code change.  That is most of why this is a list of named terms instead of
one function -- the other part being that a single reward function is impossible to
attribute when a policy learns something strange.

**Every term reads the observation, never the world.**  In the 2D simulator the
reward functions look up ground-truth visibility on their own, which means degrading
the drones' vision changes nothing: the answer still arrives through the reward
channel.  That is the single most important thing to design out here, and the rule
is enforced by these functions only being handed a `View`.

The term names follow the paper's, so `l0`, `l1`, `l2`, `l3` and `l7` are recognisable
to anyone holding it. `l7` is the explicit HLR/LLR coordination incentive the paper's
title rests on, and it is the one term worth porting rather than reinventing.

**Every term returns a positive quantity and the weight carries the sign.**  Writing
a penalty as a negative term *and* giving it a negative weight cancels, silently, and
produces a policy rewarded for flying into fires -- which is exactly what happened
here before a test caught it.  So a term answers "how much of this is there", and
whether that is wanted is the scenario's business.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class View:
    """Everything a reward may look at: what the team believes, not what is true.

    `person_*` are *beliefs* assembled from detections.  Where nobody has seen
    anything, `person_seen` is false and the position is meaningless -- a term that
    uses it without checking is reading noise and calling it a signal.
    """

    drone_pos: torch.Tensor        # (B, D, 3)
    drone_vel: torch.Tensor        # (B, D, 3)
    person_pos: torch.Tensor       # (B, 3) believed
    person_seen: torch.Tensor      # (B,) bool -- has anyone got them right now
    seen_by: torch.Tensor          # (B, D) bool -- which drones can see them
    confidence: torch.Tensor       # (B,) best confidence anyone has
    goal: torch.Tensor             # (B, 3) where safety is
    captured: torch.Tensor         # (B,) bool -- newly, this step
    was_captured: torch.Tensor     # (B,) bool -- at any point this episode
    reached_goal: torch.Tensor     # (B,) bool
    fire_near: torch.Tensor        # (B, D) fire proximity at each drone, 0..1
    action: torch.Tensor           # (B, D, 4)
    dt: float = 0.5
    harmed: torch.Tensor | None = None      # (B,) the evacuee was caught by the fire
    new_area: torch.Tensor | None = None    # (B,) fraction of map newly searched
    newly_seen: torch.Tensor | None = None  # (B,) spotted for the first time this step
    time_left: torch.Tensor | None = None   # (B,) 1 at the start, 0 at the deadline
    belief_age: torch.Tensor | None = None  # (B,) seconds since the last sighting

    def _or_zero(self, field) -> torch.Tensor:
        v = getattr(self, field)
        return torch.zeros(self.batch, device=self.drone_pos.device) if v is None else v.float()

    @property
    def batch(self) -> int:
        return len(self.drone_pos)

    @property
    def n_drones(self) -> int:
        return self.drone_pos.shape[1]


def capture(v: View) -> torch.Tensor:
    """The thing we actually want. Paid once, on the step it happens."""
    return v.captured.float()


def reached_safety(v: View) -> torch.Tensor:
    """What the capture was *for*. The paper stops at capture; this does not.

    **Only when the drones actually led them there.**  Paying for the outcome
    regardless of cause pays the team handsomely for a person who walked out
    unaided -- and early on that is the common case, so it is a large reward signal
    entirely uncorrelated with anything the policy did.  Seen directly in a training
    run: reward climbing past +60 while the capture rate sat at zero.
    """
    return v.reached_goal.float() * v.was_captured.float()


def evacuee_in_hlr_fov(v: View) -> torch.Tensor:
    """Paper's l0: the high rescuer has eyes on."""
    return v.seen_by[:, 0].float() * v.confidence


def evacuee_in_llr_fov(v: View) -> torch.Tensor:
    """Paper's l1: the low rescuer has eyes on. Worth more, since only it can capture."""
    if v.n_drones < 2:
        return torch.zeros(v.batch, device=v.drone_pos.device)
    return v.seen_by[:, 1].float() * v.confidence


PROXIMITY_SCALE_M = 25.0
BELIEF_HALFLIFE_S = 12.0


def proximity(v: View) -> torch.Tensor:
    """Paper's l2: the low rescuer closing on the person.

    Paid against the team's *last known* position, discounted by how stale that is,
    rather than only while somebody currently has eyes on.

    Paying only for live sightings looks like the careful choice and is the wrong
    one: a searcher who glimpses somebody and then loses them earns nothing at all
    for going where they just were, which is exactly the moment the incentive should
    be strongest.  Discounting by age is what separates "go and look where they were
    ten seconds ago", which is correct, from "converge forever on a two-minute-old
    guess", which is what the strict version was guarding against.
    """
    if v.n_drones < 2:
        return torch.zeros(v.batch, device=v.drone_pos.device)
    age = v._or_zero("belief_age")
    trust = torch.pow(torch.tensor(0.5, device=age.device), age / BELIEF_HALFLIFE_S)
    distance = (v.drone_pos[:, 1] - v.person_pos).norm(dim=-1)
    return torch.exp(-distance / PROXIMITY_SCALE_M) * trust


def goal_progress(v: View) -> torch.Tensor:
    """Paper's l3: the person getting closer to safety, once they are being led."""
    distance = (v.person_pos - v.goal).norm(dim=-1)
    return torch.exp(-distance / 40.0) * v.captured.float()


def role_coordination(v: View) -> torch.Tensor:
    """Paper's l7, and the term its title rests on.

    Pays the team for the *division of labour* rather than for either drone alone:
    the high rescuer holding a wide view while the low one closes in.  Reinventing
    this would make every coordination result incomparable to the published ones, so
    it is ported rather than redesigned.

    Both seeing the same person from the same place is not coordination, so the
    reward falls away as they converge.
    """
    if v.n_drones < 2:
        return torch.zeros(v.batch, device=v.drone_pos.device)

    hlr_watching = v.seen_by[:, 0].float()
    llr_closing = torch.exp(-(v.drone_pos[:, 1] - v.person_pos).norm(dim=-1) / 20.0)
    separation = (v.drone_pos[:, 0, :2] - v.drone_pos[:, 1, :2]).norm(dim=-1)
    spread = (separation / 25.0).clamp(0, 1)
    return hlr_watching * llr_closing * spread * v.person_seen.float()


def search_coverage(v: View) -> torch.Tensor:
    """A small payment for anyone having eyes on at all.

    Without it the early episodes are entirely flat: nothing is visible, so nothing
    is rewarded, so nothing is learned about looking.  This is the term that gets a
    policy from random flailing to searching.
    """
    return v.person_seen.float() * v.confidence


def fire_proximity(v: View) -> torch.Tensor:
    """How close the nearest drone is to a fire. Weighted negative.

    Note the 2D version pays this only when the evacuee is visible
    (`perl/reward/pursuer/base.py:52`), which teaches a policy that fire is safe
    whenever it has lost track of the person.  Here it is unconditional.
    """
    return v.fire_near.max(dim=-1).values


def control_effort(v: View) -> torch.Tensor:
    """How hard the drones are being flown. Weighted negative, and small: prefer the
    calm solution where two exist."""
    return v.action.pow(2).mean(dim=(1, 2))


def time_penalty(v: View) -> torch.Tensor:
    """One per step. Weighted negative: a rescue that takes all day is not a rescue."""
    return torch.ones(v.batch, device=v.drone_pos.device)


# --------------------------------------------------- for a city, not a courtyard

def first_sight(v: View) -> torch.Tensor:
    """Paid once, the moment anyone first sees the person, and larger the sooner.

    Not in the paper, and it should not be: on a 25 m block the evacuee is inside a
    rescuer's footprint from the first step, so "how long to find them" is barely a
    question.  On a real city block it is the *whole* question -- a search that takes
    ninety seconds has failed even if it succeeds -- and it is the metric the paper
    itself reports as "first seen".

    Paid once rather than continuously so that finding them is worth something even
    if they are immediately lost again, which is what makes it a search incentive
    rather than another way of paying for staring.
    """
    return v._or_zero("newly_seen") * v._or_zero("time_left")


def coverage(v: View) -> torch.Tensor:
    """Ground newly looked at this step.

    The dense signal that makes searching learnable at all.  Without it a policy
    only ever hears about the person once it has already found them, and on a map
    where that never happens by accident the reward is flat and there is nothing to
    climb -- measured directly: from the launch point the evacuee was visible in
    zero of five hundred episodes.

    Rewarding *new* ground rather than ground covered is what stops a drone hovering
    over one spot collecting the same payment forever.
    """
    return v._or_zero("new_area")


def evacuee_harmed(v: View) -> torch.Tensor:
    """The failure the rescue exists to prevent. Weighted negative."""
    return v._or_zero("harmed")


ALL = {
    "capture": capture,
    "reached_safety": reached_safety,
    "evacuee_in_hlr_fov": evacuee_in_hlr_fov,
    "evacuee_in_llr_fov": evacuee_in_llr_fov,
    "proximity": proximity,
    "goal_progress": goal_progress,
    "role_coordination": role_coordination,
    "search_coverage": search_coverage,
    "fire_proximity": fire_proximity,
    "control_effort": control_effort,
    "time_penalty": time_penalty,
    "first_sight": first_sight,
    "coverage": coverage,
    "evacuee_harmed": evacuee_harmed,
}
