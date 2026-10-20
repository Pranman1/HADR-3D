"""Terms with a counterpart in the 2D paper (Mendoza et al., GHTC 2025), restated for
continuous space and a camera.  Each reads the team's belief, not the truth."""

from __future__ import annotations

import torch

from hadrsim.reward.terms import View, term

PROXIMITY_SCALE_M = 25.0
BELIEF_HALFLIFE_S = 12.0


@term
def capture(v: View) -> torch.Tensor:
    """Paper's l3, the capture bonus: paid once, on the step it happens."""
    return v.captured.float()


@term
def evacuee_in_hlr_fov(v: View) -> torch.Tensor:
    """Paper's l0: the high rescuer has eyes on."""
    return v.seen_by[:, 0].float() * v.confidence


@term
def evacuee_in_llr_fov(v: View) -> torch.Tensor:
    """Paper's l1: the low rescuer has eyes on."""
    return v.zeros() if v.n_drones < 2 else v.seen_by[:, 1].float() * v.confidence


@term
def proximity(v: View) -> torch.Tensor:
    """Paper's l2: the low rescuer closing on the last known position, trusted less the staler it is."""
    if v.n_drones < 2:
        return v.zeros()
    age = v._or_zero("belief_age")
    trust = torch.pow(torch.tensor(0.5, device=age.device), age / BELIEF_HALFLIFE_S)
    distance = (v.drone_pos[:, 1] - v.person_pos).norm(dim=-1)
    return torch.exp(-distance / PROXIMITY_SCALE_M) * trust


@term
def role_coordination(v: View) -> torch.Tensor:
    """Paper's l7: the high rescuer watching while the low one closes, fading as they converge."""
    if v.n_drones < 2:
        return v.zeros()
    hlr_watching = v.seen_by[:, 0].float()
    llr_closing = torch.exp(-(v.drone_pos[:, 1] - v.person_pos).norm(dim=-1) / 20.0)
    separation = (v.drone_pos[:, 0, :2] - v.drone_pos[:, 1, :2]).norm(dim=-1)
    return hlr_watching * llr_closing * (separation / 25.0).clamp(0, 1) * v.person_seen.float()


@term
def time_penalty(v: View) -> torch.Tensor:
    """Paper's l6, the time decay: one per step, weighted negative."""
    return torch.ones(v.batch, device=v.drone_pos.device)


@term
def fire_proximity(v: View) -> torch.Tensor:
    """Nearest drone's fire proximity, 0..1; weighted negative."""
    return v.fire_near.max(dim=-1).values
