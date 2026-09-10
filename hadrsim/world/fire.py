"""Fire spread.

A cell can be unburnt, burning, or burnt out.  Burning cells try to ignite their
eight neighbours each tick, and consume fuel while they do.  That is it.

Simple, but it fixes all three things wrong with the 2D simulator's fire, and each
fix is one line of state:

  - **Fuel, so fires go out.**  `firegrid` has no fuel and no burnout: an ignited
    cell burns forever, so the fire eventually fills the map.  The authors kept
    lowering the spread probability (0.05 -> 0.01) to compensate, which slows the
    fire down without ever letting it end.
  - **Buildings and roads stop it.**  `firegrid`'s validity check tests map bounds
    only, so fire spreads straight through walls.  Here a cell with no fuel cannot
    ignite, and `terrain` already gives buildings and streets zero fuel -- so the
    street network becomes a firebreak for free, which is most of what makes a city
    fire behave like a city fire.
  - **A seed reproduces a run.**  `firegrid` draws from the unseeded global numpy
    generator. Here the generator is passed in.

Wind makes it directional: spread downwind is boosted and upwind suppressed, which
is the single biggest determinant of where a real fire goes.

**Embers cross the street.**  Cell-to-cell contact alone cannot get a fire over an
eight-metre road, so a fuel model that is otherwise right produces a fire that
politely burns one block and stops.  Real urban fires cross firebreaks by throwing
burning brands downwind, and that -- not creeping contact -- is what turns a
building fire into a conflagration.  So burning cells also seed ignitions at range,
downwind, at low probability.

**What the paper does, for comparison** (`firegrid/firegrid_env.py:125`): four
neighbours, `fire_prob=0.01`, no fuel, no burnout, and `_is_valid` checks the map
bounds and nothing else -- so its fire ignores buildings, roads and terrain
entirely, never goes out, and grows until it fills the map.  Ours has to earn its
growth against real geometry, which is the point, but it does have to actually
grow.

The fire keeps its own clock.  It ticks at `fire_hz` regardless of how often it is
stepped, so the same scenario burns the same way whether the caller runs at 100 Hz
or 2 Hz.  Without that, changing the control rate would silently change the fire.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hadrsim.core.rng import Rng
from hadrsim.world.wind import Wind

UNBURNT, BURNING, BURNT = 0, 1, 2

# The eight neighbours, as (dx, dy). Diagonals are further away, so they get a
# geometric discount rather than counting the same as an edge neighbour.
NEIGHBOURS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


@dataclass
class FireParams:
    spread_prob: float = 0.09      # per tick, per neighbour, no wind, dry fuel
    wind_boost: float = 2.5        # how much a full downwind alignment multiplies it
    wind_ref_mps: float = 5.0      # the speed at which `wind_boost` is reached
    burn_rate: float = 0.025       # fuel consumed per second while burning
    moisture_damping: float = 0.9  # how strongly moisture suppresses ignition
    fire_hz: float = 1.0           # spread attempts per second

    # Fuel decides how readily a cell takes, but not proportionally: a street with
    # trees and parked cars carries fire at a fifth the load of a building and
    # nothing like a fifth the rate.  The exponent softens that, so light fuel slows
    # a fire rather than stopping it dead.
    fuel_reference: float = 1.0
    fuel_exponent: float = 0.4

    # Brands thrown downwind. Small probability, long range: this is what gets a
    # fire across a road, and without it a correct fuel map produces a fire that
    # burns one block and stops.
    spot_prob: float = 0.020
    spot_range_m: float = 14.0
    spot_spread_m: float = 7.0

    @classmethod
    def paper(cls) -> "FireParams":
        """The 2D paper's fire, for `paper_parity`.

        `fire_prob=0.01` on four neighbours, no embers.  These defaults are tuned for
        a real city block over a two-minute episode and would consume the paper's
        entire 25 m map in seconds, which is not a bug in either -- spread rate is a
        property of the scenario, not of the model.
        """
        return cls(spread_prob=0.01, spot_prob=0.0, burn_rate=0.0,
                   wind_boost=1.0, fuel_exponent=0.0)


class Fire:
    """Fire over a batch of environments sharing one terrain.

    All state is `(B, W, H)`.  The terrain is shared, so it carries no batch: every
    environment burns the same city, from different ignition points.
    """

    def __init__(
        self,
        terrain,
        batch: int,
        params: FireParams | None = None,
        moisture: float = 0.1,
        device=None,
    ):
        self.terrain = terrain
        self.p = params or FireParams()
        self.batch = batch
        self.device = torch.device(device or terrain.fuel.device)
        w, h = terrain.shape

        self.fuel0 = terrain.fuel.to(self.device)                    # (W, H) shared
        self.fuel = self.fuel0.expand(batch, w, h).clone()
        self.state = torch.zeros(batch, w, h, dtype=torch.uint8, device=self.device)
        self.moisture = torch.full_like(self.fuel, float(moisture))
        self._elapsed = 0.0     # total seconds, so the tick count cannot drift
        self._ticks = 0

    # ------------------------------------------------------------------ state

    @property
    def burning(self) -> torch.Tensor:
        return self.state == BURNING

    @property
    def burnt(self) -> torch.Tensor:
        return self.state == BURNT

    @property
    def intensity(self) -> torch.Tensor:
        """How fiercely each cell is burning, 0..1.

        Proportional to the fuel still there, so a cell flares and then fades rather
        than switching off at full strength.  This is what the smoke plume is built
        from and what a sensor would actually see.
        """
        return torch.where(self.burning, (self.fuel / self.fuel0.clamp_min(1e-6)).clamp(0, 1),
                           torch.zeros_like(self.fuel))

    def area_m2(self) -> torch.Tensor:
        cell = self.terrain.cell_size_m ** 2
        return (self.burning | self.burnt).flatten(1).sum(-1) * cell

    # ------------------------------------------------------------------ setup

    def reset(self, ignite_xy: torch.Tensor | None = None, radius_m: float = 2.5) -> None:
        """Put the fire back to unburnt, then light it at the given cells.

        `ignite_xy` is (B, 2) integer cell coordinates, one ignition per environment.
        """
        self.fuel = self.fuel0.expand_as(self.fuel).clone()
        self.state.zero_()
        self._elapsed = 0.0
        self._ticks = 0
        if ignite_xy is not None:
            self.ignite(ignite_xy, radius_m)

    def ignite(self, xy: torch.Tensor, radius_m: float = 0.0) -> None:
        """Light a patch. Cells with nothing to burn are skipped -- a road will not take.

        A patch rather than a point, because a single square metre is a fragile thing
        to start from: it burns out in a few seconds having had only a handful of
        chances to catch, so whether the fire happens at all comes down to a coin
        toss.  A real fire is not noticed until it is already several metres across,
        which is both more realistic and far less capricious.
        """
        b = torch.arange(len(xy), device=self.device)
        x, y = xy[:, 0].long(), xy[:, 1].long()

        r = int(radius_m / self.terrain.cell_size_m)
        w, h = self.state.shape[-2:]
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                px, py = (x + dx).clamp(0, w - 1), (y + dy).clamp(0, h - 1)
                ok = self.fuel[b, px, py] > 0
                self.state[b[ok], px[ok], py[ok]] = BURNING

    # ------------------------------------------------------------------- step

    def step(self, dt: float, wind: Wind, rng: Rng) -> None:
        """Advance by `dt` seconds. Ticks at `fire_hz` however often it is called.

        The tick count is recomputed from total elapsed time rather than carried in a
        running remainder.  A remainder accumulates floating-point error -- stepping
        by 0.1 s a hundred times does not add up to exactly 10 s -- and eventually
        drops or duplicates a tick, so the same scenario would burn differently at
        100 Hz than at 2 Hz.  That is precisely the coupling this is meant to remove.
        """
        self._elapsed += dt
        period = 1.0 / self.p.fire_hz
        due = int(self._elapsed * self.p.fire_hz + 1e-9)
        while self._ticks < due:
            self._ticks += 1
            self._tick(period, wind, rng)

    def _tick(self, dt: float, wind: Wind, rng: Rng) -> None:
        self._spread(wind, rng)
        self._consume(dt)

    def _consume(self, dt: float) -> None:
        """Burn fuel, and let cells that run out go out."""
        burning = self.burning
        self.fuel = torch.where(
            burning, (self.fuel - self.p.burn_rate * dt).clamp_min(0.0), self.fuel
        )
        self.state = torch.where(
            burning & (self.fuel <= 0.0),
            torch.full_like(self.state, BURNT),
            self.state,
        )

    def _spread(self, wind: Wind, rng: Rng) -> None:
        """Each burning cell reaches for its neighbours, weighted by the wind.

        One grouped convolution rather than eight shifted multiplies.  The kernel is
        per-environment because the wind is, so the batch becomes the convolution's
        groups -- which at a thousand environments is the difference between a
        fourteen-second step and a readable one.

        The total reach is turned into a probability by `1 - exp(-rate)` rather than
        by multiplying out `(1 - p_k)` over the neighbours.  That is the standard way
        to ask "did at least one of these independent attempts succeed", it agrees
        with the product to well under a percent at these rates, and it is what lets
        the whole neighbourhood collapse into a single convolution.
        """
        burning = self.burning.float()                           # (B, W, H)
        weights = self._direction_weights(wind)                  # (B, 8)

        # `weights[k]` is how readily fire *travels* along d, so the cell it reaches
        # is at B and the source is at B - d.  In a convolution that means the kernel
        # tap for direction d sits at -d, which is the flip that decides whether a
        # windy fire runs downwind or straight into the wind.
        kernel = torch.zeros(self.batch, 1, 3, 3, device=self.device)
        for k, (dx, dy) in enumerate(NEIGHBOURS):
            kernel[:, 0, 1 - dx, 1 - dy] = weights[:, k]

        # A grouped convolution puts the batch in the *channel* dimension: one image
        # with B channels, one 3x3 filter per channel, B groups.
        w, h = burning.shape[-2:]
        stacked = F.pad(burning.reshape(1, self.batch, w, h), (1, 1, 1, 1))
        reach = F.conv2d(stacked, kernel, groups=self.batch).reshape(self.batch, w, h)
        reach = reach + self._spotting(burning, wind)

        receptive = (self.fuel > 0) & (self.state == UNBURNT)
        damp = 1.0 - self.p.moisture_damping * self.moisture.clamp(0, 1)
        p_ignite = (1.0 - torch.exp(-reach)) * damp * self._receptivity()

        lit = rng.bernoulli(p_ignite.clamp(0, 1)) & receptive
        self.state = torch.where(lit, torch.full_like(self.state, BURNING), self.state)

    def _receptivity(self) -> torch.Tensor:
        """How readily a cell takes, given what is on it. Sub-linear in fuel."""
        ratio = (self.fuel / self.p.fuel_reference).clamp(0.0, 1.0)
        return ratio ** self.p.fuel_exponent

    def _spotting(self, burning: torch.Tensor, wind: Wind) -> torch.Tensor:
        """Ignition probability from brands blown downwind.

        The plume carries burning debris ahead of the front, so the ember field is
        the burning cells shifted downwind and spread out -- the same construction as
        the smoke, for the same reason.

        **How many brands, and how far, both scale with the wind.**  Getting only the
        distance from the wind and leaving the rate fixed means a still-air fire
        showers embers evenly in every direction and steps over a firebreak it should
        never cross.  Still air makes fires creep; wind makes them run and jump.
        """
        from hadrsim.world.smoke import _blur, _translate

        cell = self.terrain.cell_size_m
        reach = (wind.speed.to(self.device) / self.p.wind_ref_mps).clamp(0.0, 1.5)
        if float(reach.max()) < 1e-3:
            return torch.zeros_like(burning)

        offset = wind.direction().to(self.device) * (self.p.spot_range_m * reach)[:, None] / cell
        sigma = max(float(self.p.spot_spread_m * reach.max()) / cell, 0.5)

        seeded = _translate(burning[:, None], offset)
        rate = (self.p.spot_prob * reach)[:, None, None, None]
        return (rate * _blur(seeded, sigma))[:, 0]

    def _direction_weights(self, wind: Wind) -> torch.Tensor:
        """Per-environment spread probability in each of the eight directions.

        Downwind is boosted and upwind suppressed, scaled by how hard it is blowing.
        Diagonals are discounted by 1/sqrt(2) because they are that much further.
        """
        offsets = torch.tensor(NEIGHBOURS, dtype=torch.float32, device=self.device)
        dist = offsets.norm(dim=-1)                                    # (8,)
        unit = offsets / dist[:, None]

        speed = wind.speed.to(self.device).clamp_min(0.0)               # (B,)
        align = wind.direction().to(self.device) @ unit.T               # (B, 8) in [-1, 1]
        strength = (speed / self.p.wind_ref_mps).clamp(0.0, 1.5)[:, None]
        gain = (1.0 + strength * (self.p.wind_boost - 1.0) * align).clamp_min(0.05)

        return (self.p.spread_prob * gain / dist).clamp(0.0, 1.0)


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate a (B, W, H) field, filling the edge with zeros.

    Zero-fill is deliberate: nothing burns in from outside the map, so the boundary
    behaves like a firebreak rather than a mirror.
    """
    pad = (max(dy, 0), max(-dy, 0), max(dx, 0), max(-dx, 0))
    out = F.pad(x, pad)
    w, h = x.shape[-2:]
    x0 = max(-dx, 0)
    y0 = max(-dy, 0)
    return out[..., x0:x0 + w, y0:y0 + h]
