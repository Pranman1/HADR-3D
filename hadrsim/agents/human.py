"""The person: what they want to do.

This is the *decision* layer.  How the body then actually moves -- momentum, walls,
other people -- is `crowd.py`.  Splitting them is what lets the paper's model stay
intact while the motion underneath it gets better.

**Trivedi & Rao, kept verbatim:**

        v = (1 - gamma) * v_sensible  +  gamma * v_instinct

`gamma` is the panic level.  Reimplementing this equation differently would make
every number incomparable to the published paper, so it is ported as-is.

Four things around it are more accurate than the paper's, and none of them change
that equation:

  - **gamma responds to what the person actually experiences.**  In the paper it is
    fixed, or follows a schedule.  Trivedi & Rao themselves define panic as driven by
    stressors; the paper simplified that away.  Here it is driven by how close the
    fire is, how thick the smoke is, and how long this has been going on -- and it
    relaxes rather than jumping, because people do not switch from calm to panicked
    between one footstep and the next.
  - **Panic makes you faster and stupider.**  Speed rises with gamma while the
    sensible term loses weight.  The classic stress finding, and the parameters for
    it are already sitting unused in the paper's own `panicked.yaml`.
  - **Smoke slows people down.**  Well established in the evacuation literature
    (Jin's visibility studies).  The 2D grid could not represent this at all; we have
    a real smoke field, so it is nearly free.
  - **Following a rescuer is imperfect.**  The paper assumes that once reached, the
    person walks an ideal A* path to safety, which flatters the improvement figures.
    Here they follow with a lag, imprecisely, and lose the drone in thick smoke.

**One honest simplification.**  With a single evacuee there is no herd, so the
"instinct" term is flight from the fire plus noise rather than crowd-following.  That
is the faithful single-person reading of the model; with N > 1 the herd term becomes
the neighbours' mean velocity and this is where it goes.

**One known gap.**  Fire is sensed from a blurred threat field with a limited radius,
not through a real line-of-sight check -- so a person can still notice a fire through
a wall, just not one across the map.  The 2D simulator reads the *global* ground-truth
grid with no occlusion at all, so this is strictly better, but it is not right.  It
becomes right when `sensing/occlusion.py` exists.
"""

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
    perception_m: float = 10.0   # how far the fire can be felt
    instinct_turn: float = 1.1   # how fast a panicked heading wanders, rad/sqrt(s)
    instinct_bias: float = 0.55  # how strongly it is still pulled toward the exit
    smoke_slowdown: float = 0.55 # fraction of speed lost in the thickest smoke
    follow_lag_s: float = 1.2    # how sluggishly they follow a rescuer
    follow_noise: float = 0.25   # how imprecisely
    lose_sight_smoke: float = 0.8  # smoke density at which the rescuer is lost


class Human:
    """One evacuee per environment. Positions in metres, on the ground plane."""

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
        self.following = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self.instinct_dir = z(batch, 2)
        self.elapsed = 0.0

        self._dist: torch.Tensor | None = None
        self._flow: torch.Tensor | None = None

    # ------------------------------------------------------------------ setup

    def reset(self, start_m: torch.Tensor, goal_m: torch.Tensor, gamma0: float = 0.0) -> None:
        """Place the person and work out the route to safety, once for the episode."""
        cell = self.terrain.cell_size_m
        self.pos = start_m.to(self.device).float().clone()
        self.goal = goal_m.to(self.device).float().clone()
        self.vel.zero_()
        self.gamma.fill_(gamma0)
        self.reached.zero_()
        self.following.zero_()
        self.elapsed = 0.0

        self._dist = nav.distance_field(
            self.terrain.walkable, (self.goal / cell).long(), cell
        )
        self._flow = nav.flow(self._dist)
        # Start out heading the sensible way, so panic degrades a good plan rather
        # than starting from a random one.
        self.instinct_dir = nav.sample(self._flow, self.pos, cell).clone()

    # ------------------------------------------------------------------- step

    def desired_velocity(self, fire, smoke, rng: Rng, dt: float,
                         rescuer_xy=None) -> torch.Tensor:
        """Where they are trying to go, before the body's momentum gets a say."""
        threat, flee = self._threat(fire, smoke)
        self._update_gamma(threat, smoke, dt)

        cell = self.terrain.cell_size_m
        sensible = nav.sample(self._flow, self.pos, cell)

        # Instinct is "run from the fire" when there is a fire to run from.  With
        # none in reach it is a *persistent* poor choice of direction, not fresh noise
        # each step: independent noise averages out over a few seconds and costs
        # nothing, whereas committing to a wrong heading for several seconds is what
        # actually makes a panicked evacuation take longer.  That is the mechanism
        # behind the paper's Figure 2, and with a single evacuee it stands in for the
        # herd term, which has no meaning when there is no herd.
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

    def led_by(self, rescuer_xy: torch.Tensor, radius_m: float) -> None:
        """A rescuer within `radius_m` picks them up; thick smoke loses them again."""
        self.following |= (self.pos - rescuer_xy).norm(dim=-1) < radius_m

    # -------------------------------------------------------------- internals

    def _threat(self, fire, smoke):
        """How near the fire is, 0..1, and which way is away from it.

        **Nearest, not total.**  The obvious implementation -- blur the fire and read
        the value -- measures how much fire is nearby, not how close it is.  Blur an
        extended fire and every cell on a small map reads as maximally alarming, so
        panic pins at one and the person spends the episode fleeing from everywhere at
        once.  Proximity has to be a *max* over nearby fire, not a sum of it.

        Three dilations at increasing radius approximate that for the cost of three
        max-pools: is there fire within arm's reach, within a few metres, within
        sight.  Bounded by the intensity of a single cell however large the fire, and
        it still falls off with distance.
        """
        import torch.nn.functional as Fn

        cell = self.terrain.cell_size_m
        heat = (fire.intensity + 0.4 * smoke.column_density()).clamp(0, 2)[:, None]

        field = torch.zeros_like(heat)
        for radius_m, weight in ((1.5, 1.0), (self.p.perception_m * 0.4, 0.55),
                                 (self.p.perception_m, 0.22)):
            k = 2 * max(1, int(round(radius_m / cell))) + 1
            field = torch.maximum(field, weight * Fn.max_pool2d(heat, k, 1, k // 2))
        field = field[:, 0]

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
