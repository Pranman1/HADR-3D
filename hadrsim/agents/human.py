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
  - **They go round the fire, not through it.**  The paper's route is A* over the
    static map, computed once, so the fire is something the person walks into rather
    than something they avoid.  Here the route is priced: cells near the fire cost
    more to cross, and the route is recomputed every few seconds, so a blocked street
    produces a detour the way it would in life.  Nobody re-plans continuously, which
    is why it is on a slow clock rather than every step.
  - **A person has width.**  They cannot cut a building's corner, and they cannot
    squeeze through a gap narrower than they are.  The route is planned over cells
    with enough clearance for a body, not over cells that merely are not wall.

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
    perception_m: float = 28.0   # how far a fire is frightening from
    # Calibrated so full panic costs roughly twice the evacuation time, which is
    # what the paper's Figure 2 reports. Retuned after the flow field started
    # picking the best neighbour instead of differencing the field: better
    # routes made the calm baseline faster, so the same wander cost 3x.
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

        # Plan over cells a body actually fits in.  Planning over "not wall" sends
        # people diagonally through building corners and down gaps narrower than
        # their shoulders, both of which look exactly as wrong as they are.
        clear = nav.clearance(self.terrain.walkable, cell)
        self._passable = self.terrain.walkable & (clear >= self.p.body_radius_m)
        if not bool(self._passable[tuple((self.goal[0] / cell).long().tolist())]):
            self._passable = self.terrain.walkable      # a goal in a doorway still works

        self._replan(fire=None)
        # Marked as never planned, *after* the fire-blind plan above has set the
        # timestamp -- so the first real step replans with the fire in view.  With the
        # order reversed, the first fire-aware route came two seconds in, by which
        # time a person walking a straight line had committed to it.
        self._replanned_at = -self.p.replan_every_s
        # Start out heading the sensible way, so panic degrades a good plan rather
        # than starting from a random one.
        self.instinct_dir = nav.sample(self._flow, self.pos, cell).clone()

    def _replan(self, fire) -> None:
        """Recompute the route around the fire.

        **A burning cell cannot be entered.**  That is the paper's rule, kept exactly:
        `Fire.can_overlap` in `firegrid` is true only for the drones, so the evacuee's
        A* treats fire as a wall.  An earlier version here priced fire at 61x instead
        of blocking it, and people walked through it whenever the way round was long
        -- which it always is on a real block.

        Two things the paper has no need for, because its fire never goes out:
        burnt ground costs `burnt_cost` (nobody walks across hot ash if there is a
        street), and cells within 12 m of flame are priced up so the route keeps its
        distance rather than hugging the edge.

        If the fire has cut every route, the field is unreachable from where they
        stand; then, and only then, the block is relaxed to a very high price so
        they head for the least-bad way out instead of standing still.
        """
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

    def expose_to(self, fire) -> None:
        """Standing in a burning cell hurts.

        Without this there is no way for an evacuation to fail, so there is nothing
        for a rescuer to prevent: the person strolls through the fire and out the
        other side, and a policy that does nothing scores as well as one that works.
        A failure mode is what makes the task a task.
        """
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
