"""The evacuee: a route to safety replanned around the fire, Trivedi-Rao panic mixing a
sensible velocity with an instinctive one, smoke slowdown, and following a rescuer once led."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hadrsim.core.rng import Rng
from hadrsim.world import nav


@dataclass
class PanicParams:
    v_calm: float = 1.3          # comfortable walking speed, m/s
    v_panic: float = 1.9         # panicked, m/s -- faster, and worse at choosing
    gamma_relax_s: float = 8.0   # how long panic takes to build or fade
    smoke_stress: float = 1.5    # smoke density that counts as fully alarming
    time_stress_s: float = 90.0  # how long before the situation alone is stressful
    gamma_floor: float = 0.05    # nobody in a burning city is perfectly calm
    perception_m: float = 28.0   # how far a fire is frightening from
    instinct_turn: float = 0.9   # how fast a panicked heading wanders, rad/sqrt(s)
    instinct_bias: float = 0.70  # how strongly it is still pulled toward the exit
    smoke_slowdown: float = 0.55 # fraction of speed lost in the thickest smoke
    follow_lag_s: float = 1.2    # how sluggishly they follow a rescuer
    follow_noise: float = 0.25   # how imprecisely
    lose_sight_smoke: float = 0.8  # smoke density at which the rescuer is lost
    body_radius_m: float = 0.28  # shoulder half-width. Sets the narrowest usable gap
    replan_every_s: float = 2.0  # nobody re-plans continuously; the fire ticks at 1 Hz
    burnt_cost: float = 20.0     # ash is crossed only if the way round is 21x longer
    fire_detour_cost: float = 60.0   # how far out of the way a fire is worth going


class Human:
    """One evacuee per environment."""

    def __init__(self, terrain, batch: int, params: PanicParams | None = None, device=None):
        self.terrain = terrain
        self.p = params or PanicParams()
        self.batch = batch
        self.device = torch.device(device or terrain.fuel.device)

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos = z(batch, 2)
        self.vel = z(batch, 2)
        self.gamma = z(batch)
        self.goal = z(batch, 2)
        self.reached = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.harmed = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.following = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.instinct_dir = z(batch, 2)
        self.elapsed = 0.0

        self._dist: torch.Tensor | None = None
        self._flow: torch.Tensor | None = None
        self._passable: torch.Tensor | None = None
        self._replanned_at = 0.0

    # ------------------------------------------------------------------ setup

    def reset(self, start_m: torch.Tensor, goal_m: torch.Tensor, gamma0: float = 0.0) -> None:
        """Place the person and work out the route to safety, once for the episode."""
        cell = self.terrain.cell_size_m
        self.pos = start_m.to(self.device).float().clone()
        self.goal = goal_m.to(self.device).float().clone()
        self.vel.zero_()
        self.gamma.fill_(gamma0)
        self.reached.zero_()
        self.harmed.zero_()
        self.following.zero_()
        self.elapsed = 0.0

        clear = nav.clearance(self.terrain.walkable, cell)
        self._passable = self.terrain.walkable & (clear >= self.p.body_radius_m)
        if not bool(self._passable[tuple((self.goal[0] / cell).long().tolist())]):
            self._passable = self.terrain.walkable      # a goal in a doorway still works

        self._replan(fire=None)
        self._replanned_at = -self.p.replan_every_s
        self.instinct_dir = nav.sample(self._flow, self.pos, cell).clone()

    def _replan(self, fire) -> None:
        """Recompute the route with burning cells as walls and ash priced high."""
        cell = self.terrain.cell_size_m
        goal = (self.goal / cell).long()
        if fire is None:
            self._dist = nav.distance_field(self._passable, goal, cell)
        else:
            near = nav.proximity(fire.intensity, 12.0, cell)
            cost = 1.0 + self.p.fire_detour_cost * near + self.p.burnt_cost * fire.burnt.float()
            passable = self._passable[None] & ~fire.burning
            self._dist = nav.distance_field(passable, goal, cell, cost=cost)

            stuck = nav.sample(self._dist, self.pos, cell) >= nav.UNREACHABLE / 2
            if bool(stuck.any()):
                worst = cost + 1e3 * fire.burning.float()
                fallback = nav.distance_field(self._passable, goal, cell, cost=worst)
                self._dist = torch.where(stuck[:, None, None], fallback, self._dist)
        self._flow = nav.flow(self._dist)
        self._replanned_at = self.elapsed

    # ------------------------------------------------------------------- step

    def desired_velocity(self, fire, smoke, rng: Rng, dt: float,
                         rescuer_xy=None) -> torch.Tensor:
        """Where they are trying to go, before the body's momentum gets a say."""
        threat, flee = self._threat(fire, smoke)
        self._update_gamma(threat, smoke, dt)

        if self.elapsed - self._replanned_at >= self.p.replan_every_s:
            self._replan(fire)

        cell = self.terrain.cell_size_m
        sensible = nav.sample(self._flow, self.pos, cell)

        turn = rng.normal(self.batch, 2) * (self.p.instinct_turn * dt ** 0.5)
        self.instinct_dir = _unit(
            self.instinct_dir + turn + self.p.instinct_bias * sensible * dt
        )
        urgency = flee.norm(dim=-1, keepdim=True)
        instinct = torch.where(urgency > 1e-3, _unit(flee), self.instinct_dir)

        g = self.gamma[:, None]
        direction = _unit((1.0 - g) * sensible + g * instinct)

        speed = self.p.v_calm + (self.p.v_panic - self.p.v_calm) * self.gamma
        speed = speed * self._smoke_slowdown(smoke)
        want = direction * speed[:, None]

        if rescuer_xy is not None:
            want = torch.where(self.following[:, None],
                               self._follow(rescuer_xy, smoke, speed, rng), want)
        return torch.where(self.reached[:, None], torch.zeros_like(want), want)

    def integrate(self, vel: torch.Tensor, dt: float) -> None:
        """Accept the velocity the motion layer settled on, and arrive if close enough."""
        self.vel = vel
        self.pos = self.pos + vel * dt
        self.elapsed += dt
        self.reached |= (self.pos - self.goal).norm(dim=-1) < 1.5

    def expose_to(self, fire) -> None:
        """Standing in a burning cell hurts; enough of it and the person is lost."""
        cell = self.terrain.cell_size_m
        c = (self.pos / cell).long()
        w, h = fire.burning.shape[-2:]
        idx = torch.arange(len(self.pos), device=self.device)
        burning = fire.burning[idx, c[:, 0].clamp(0, w - 1), c[:, 1].clamp(0, h - 1)]
        self.harmed |= burning & ~self.reached

    def led_by(self, rescuer_xy: torch.Tensor, radius_m: float) -> None:
        """A rescuer within `radius_m` picks them up; thick smoke loses them again."""
        self.following |= (self.pos - rescuer_xy).norm(dim=-1) < radius_m

    # -------------------------------------------------------------- internals

    def _threat(self, fire, smoke):
        """How near the fire is (0..1) and which way is away from it."""
        cell = self.terrain.cell_size_m
        heat = (fire.intensity + 0.4 * smoke.column_density()).clamp(0, 2)
        field = nav.proximity(heat, self.p.perception_m, cell)

        here = nav.sample(field, self.pos, cell)
        step = torch.tensor([cell, 0.0], device=self.device)
        side = torch.tensor([0.0, cell], device=self.device)
        grad = torch.stack([
            nav.sample(field, self.pos + step, cell) - nav.sample(field, self.pos - step, cell),
            nav.sample(field, self.pos + side, cell) - nav.sample(field, self.pos - side, cell),
        ], dim=-1)
        return here, -grad          # away from the heat

    def _update_gamma(self, threat: torch.Tensor, smoke, dt: float) -> None:
        """Panic relaxes toward what the situation warrants, rather than switching."""
        cell = self.terrain.cell_size_m
        fire_term = threat.clamp(0, 1)     # already a 0..1 proximity, see _threat
        smoke_term = (nav.sample(smoke.column_density(), self.pos, cell)
                      / self.p.smoke_stress).clamp(0, 1)
        time_term = min(self.elapsed / self.p.time_stress_s, 1.0)

        target = (0.55 * fire_term + 0.30 * smoke_term + 0.15 * time_term)
        target = target.clamp(self.p.gamma_floor, 1.0)

        alpha = 1.0 - float(torch.exp(torch.tensor(-dt / self.p.gamma_relax_s)))
        self.gamma = self.gamma + alpha * (target - self.gamma)

    def _smoke_slowdown(self, smoke) -> torch.Tensor:
        """Jin's result: walking speed falls sharply as smoke thickens."""
        d = nav.sample(smoke.column_density(), self.pos, self.terrain.cell_size_m)
        return 1.0 - self.p.smoke_slowdown * (d / self.p.smoke_stress).clamp(0, 1)

    def _follow(self, rescuer_xy, smoke, speed, rng: Rng) -> torch.Tensor:
        """Trail the rescuer -- late, imprecisely, and not at all in thick smoke."""
        cell = self.terrain.cell_size_m
        lost = nav.sample(smoke.column_density(), self.pos, cell) > self.p.lose_sight_smoke
        self.following &= ~lost

        to = rescuer_xy.to(self.device) - self.pos
        aim = _unit(to + rng.normal(self.batch, 2) * self.p.follow_noise)
        # A lag means they accelerate toward the rescuer rather than snapping to them.
        blend = min(1.0, 1.0 / max(self.p.follow_lag_s, 1e-6))
        return self.vel * (1 - blend) + aim * speed[:, None] * blend


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
