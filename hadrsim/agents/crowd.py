"""Motion layer for people: social forces from walls and each other, and a hard stop at
walls. Returns a velocity; the caller integrates position."""

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
    body_radius_m: float = 0.28    # shoulder half-width


class Crowd:
    """The motion layer."""

    def __init__(self, terrain, params: CrowdParams | None = None, device=None):
        self.terrain = terrain
        self.p = params or CrowdParams()
        self.device = torch.device(device or terrain.fuel.device)
        cell = terrain.cell_size_m

        self.clearance = nav.clearance(terrain.walkable, cell).to(self.device)
        self.away = nav.flow(-self.clearance[None])[0]

    def step(self, pos, vel, desired, dt: float, others=None):
        """One motion step."""
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
        """Keep out of each other's way."""
        delta = pos[:, None, :] - others[None, :, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        near = (dist < self.p.person_range_m).float()
        push = (delta / dist) * near * self.p.person_force
        return push.sum(dim=1)

    def _stop_at_walls(self, pos, vel, dt, cell):
        """Never let a step end inside a building."""
        def walkable_at(p):
            w, h = self.terrain.walkable.shape
            c = (p / cell).long()
            x = c[:, 0].clamp(0, w - 1)
            y = c[:, 1].clamp(0, h - 1)
            return self.terrain.walkable[x, y] & (self.clearance[x, y] >= self.p.body_radius_m)

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
