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
    spread_prob: float = 0.06      # per tick, per neighbour, no wind, dry fuel
    wind_boost: float = 2.5        # how much a full downwind alignment multiplies it
    wind_ref_mps: float = 5.0      # the speed at which `wind_boost` is reached
    burn_rate: float = 0.05        # fuel consumed per second while burning
    moisture_damping: float = 0.9  # how strongly moisture suppresses ignition
    fire_hz: float = 1.0           # spread attempts per second


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

    def reset(self, ignite_xy: torch.Tensor | None = None) -> None:
        """Put the fire back to unburnt, then light it at the given cells.

        `ignite_xy` is (B, 2) integer cell coordinates, one ignition per environment.
        """
        self.fuel = self.fuel0.expand_as(self.fuel).clone()
        self.state.zero_()
        self._elapsed = 0.0
        self._ticks = 0
        if ignite_xy is not None:
            self.ignite(ignite_xy)

    def ignite(self, xy: torch.Tensor) -> None:
        """Light cells. Ignoring any that have nothing to burn -- a road will not take."""
        b = torch.arange(len(xy), device=self.device)
        x, y = xy[:, 0].long(), xy[:, 1].long()
        ok = self.fuel[b, x, y] > 0
        self.state[b[ok], x[ok], y[ok]] = BURNING

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
        """Each burning cell rolls against each neighbour, weighted by the wind."""
        burning = self.burning.float()
        weights = self._direction_weights(wind)             # (B, 8)

        # Probability that a cell survives *all* its burning neighbours this tick.
        survive = torch.ones_like(self.fuel)
        for k, (dx, dy) in enumerate(NEIGHBOURS):
            # `weights[k]` is how readily fire *travels* along d, so the cell it
            # travels to is at B and the one it comes from is at B - d.  Shifting the
            # burning mask by +d puts that source cell's state at B.
            #
            # Getting this backwards is invisible without wind -- the neighbour set is
            # symmetric, so a still-air fire spreads identically either way -- and
            # with wind it drives the fire directly upwind.
            src = _shift(burning, dx, dy)
            survive = survive * (1.0 - weights[:, k, None, None] * src)

        # Fuel and moisture decide whether an attempt takes hold.
        receptive = (self.fuel > 0) & (self.state == UNBURNT)
        damp = 1.0 - self.p.moisture_damping * self.moisture.clamp(0, 1)
        p_ignite = (1.0 - survive) * damp * self.fuel.clamp(0, 1)

        lit = rng.bernoulli(p_ignite.clamp(0, 1)) & receptive
        self.state = torch.where(lit, torch.full_like(self.state, BURNING), self.state)

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
