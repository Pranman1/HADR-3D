"""How a body actually moves.

`human.py` decides where the person wants to go.  This turns that wish into motion,
using the social force model (Helbing & Molnar 1995, whose follow-up in Nature 2000
is specifically about fire evacuation -- which makes it about as well-aimed a
citation as this project is going to find).

Three forces, and with one evacuee only the first two are non-zero:

    drive       accelerate toward the velocity you want, over a reaction time
    walls       push away from anything solid, harder the closer you are
    people      push apart from each other  (N > 1; the interface allows it)

Why this instead of just stepping in the desired direction, which is what the 2D
simulator does: a body has momentum.  It cannot reverse instantly, it swings wide
around corners, and it slows into a turn.  Over a 0.5 s control interval that is the
difference between a person who behaves like a person and one who behaves like a
cursor -- and it decides whether a drone that saw them a second ago still finds them
where it expected.

Walls are handled by a distance-to-wall field computed once per map, so the repulsion
is a lookup rather than a search over nearby geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hadrsim.world import nav


@dataclass
class CrowdParams:
    reaction_time_s: float = 0.5   # how quickly people reach their desired velocity
    wall_range_m: float = 0.8      # how close a wall has to be to push back
    wall_force: float = 4.0        # how hard it pushes, m/s^2
    person_range_m: float = 0.6    # personal space
    person_force: float = 3.0
    max_speed_mps: float = 2.5     # nobody sprints faster than this
    max_accel_mps2: float = 4.0


class Crowd:
    """The motion layer. Static per map, so the wall field is built once."""

    def __init__(self, terrain, params: CrowdParams | None = None, device=None):
        self.terrain = terrain
        self.p = params or CrowdParams()
        self.device = torch.device(device or terrain.fuel.device)
        cell = terrain.cell_size_m

        self.clearance = nav.clearance(terrain.walkable, cell).to(self.device)
        # Gradient of clearance points away from the nearest wall, which is exactly
        # the direction the repulsion should act in.
        self.away = nav.flow(-self.clearance[None])[0]

    def step(self, pos, vel, desired, dt: float, others=None):
        """One motion step. Returns the new velocity; the caller integrates position.

        Kept as "return a velocity" rather than "move them" so the human owns its own
        state and this file stays reusable by anything else that walks.
        """
        cell = self.terrain.cell_size_m
        accel = (desired - vel) / max(self.p.reaction_time_s, 1e-6)
        accel = accel + self._wall_force(pos, cell)
        if others is not None:
            accel = accel + self._person_force(pos, others)

        accel = _clamp_norm(accel, self.p.max_accel_mps2)
        new = _clamp_norm(vel + accel * dt, self.p.max_speed_mps)
        return self._stop_at_walls(pos, new, dt, cell)

    # -------------------------------------------------------------- internals

    def _wall_force(self, pos, cell):
        """Push away from walls, rising steeply as they get close."""
        d = nav.sample(self.clearance[None].expand(len(pos), -1, -1), pos, cell)
        direction = nav.sample(self.away[None].expand(len(pos), -1, -1, -1), pos, cell)
        strength = (1.0 - (d / self.p.wall_range_m).clamp(0, 1)) * self.p.wall_force
        return direction * strength[:, None]

    def _person_force(self, pos, others):
        """Keep out of each other's way. Inert while there is only one evacuee."""
        delta = pos[:, None, :] - others[None, :, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        near = (dist < self.p.person_range_m).float()
        push = (delta / dist) * near * self.p.person_force
        return push.sum(dim=1)

    def _stop_at_walls(self, pos, vel, dt, cell):
        """Last resort: never let a step end inside a building.

        The repulsion above handles this nearly always, but "nearly" is not enough --
        a person inside a wall breaks the distance field, the fire's fuel mask, and
        every visibility check downstream.

        **Slide, do not stop.**  Zeroing the whole velocity deadlocks: the desired
        direction still points at the wall next tick, so the person presses into it
        forever and never gets anywhere.  Dropping only the blocked component lets
        them run along the wall and round the corner, which is also what a person
        does.
        """
        def walkable_at(p):
            w, h = self.terrain.walkable.shape
            c = (p / cell).long()
            return self.terrain.walkable[c[:, 0].clamp(0, w - 1), c[:, 1].clamp(0, h - 1)]

        zero = torch.zeros_like(vel)
        slide_x = torch.stack([vel[:, 0], zero[:, 1]], dim=-1)
        slide_y = torch.stack([zero[:, 0], vel[:, 1]], dim=-1)

        ok_full = walkable_at(pos + vel * dt)
        ok_x = walkable_at(pos + slide_x * dt)
        ok_y = walkable_at(pos + slide_y * dt)

        out = torch.where(ok_x[:, None], slide_x, zero)
        out = torch.where(ok_y[:, None], slide_y, out)
        return torch.where(ok_full[:, None], vel, out)


def _clamp_norm(v: torch.Tensor, limit: float) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True)
    return v * (limit / n).clamp(max=1.0)
