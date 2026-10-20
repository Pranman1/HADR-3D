"""Cellular fire: each burning cell ignites neighbours with a probability shaped by fuel,
moisture and wind, burns its fuel down, and can spot embers downwind."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hadrsim.core.rng import Rng
from hadrsim.world.wind import Wind

UNBURNT, BURNING, BURNT = 0, 1, 2

NEIGHBOURS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


@dataclass
class FireParams:
    spread_prob: float = 0.09      # per tick, per neighbour, no wind, dry fuel
    wind_boost: float = 1.8        # how much a full downwind alignment multiplies it
    wind_ref_mps: float = 5.0      # the speed at which `wind_boost` is reached
    burn_rate: float = 0.025       # fuel consumed per second while burning
    moisture_damping: float = 0.9  # how strongly moisture suppresses ignition
    fire_hz: float = 1.0           # spread attempts per second

    fuel_reference: float = 1.0
    fuel_exponent: float = 0.4

    spot_prob: float = 0.008
    spot_range_m: float = 14.0
    spot_spread_m: float = 7.0


class Fire:
    """Fire over a batch of environments sharing one terrain."""

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
        """How fiercely each cell burns, 0..1: fuel remaining as a fraction of what it started with."""
        return torch.where(self.burning, (self.fuel / self.fuel0.clamp_min(1e-6)).clamp(0, 1),
                           torch.zeros_like(self.fuel))

    # ------------------------------------------------------------------ setup

    def reset(self, ignite_xy: torch.Tensor | None = None, radius_m: float = 2.5) -> None:
        """Put the fire back to unburnt, then light it at the given cells."""
        self.fuel = self.fuel0.expand_as(self.fuel).clone()
        self.state.zero_()
        self._elapsed = 0.0
        self._ticks = 0
        if ignite_xy is not None:
            self.ignite(ignite_xy, radius_m)

    def ignite(self, xy: torch.Tensor, radius_m: float = 0.0) -> None:
        """Light a patch; cells with no fuel do not take."""
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
        """Advance by `dt`; the spread itself ticks at `fire_hz`."""
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
        """Each burning cell reaches for its eight neighbours, weighted by wind and the neighbour's fuel."""
        burning = self.burning.float()                           # (B, W, H)
        weights = self._direction_weights(wind)                  # (B, 8)

        kernel = torch.zeros(self.batch, 1, 3, 3, device=self.device)
        for k, (dx, dy) in enumerate(NEIGHBOURS):
            kernel[:, 0, 1 - dx, 1 - dy] = weights[:, k]

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
        """How readily a cell takes, given what is on it."""
        ratio = (self.fuel / self.p.fuel_reference).clamp(0.0, 1.0)
        return ratio ** self.p.fuel_exponent

    def _spotting(self, burning: torch.Tensor, wind: Wind) -> torch.Tensor:
        """Ignition from embers carried downwind of the front."""
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
        """(B, 8) spread probability per direction for the current wind."""
        offsets = torch.tensor(NEIGHBOURS, dtype=torch.float32, device=self.device)
        dist = offsets.norm(dim=-1)                                    # (8,)
        unit = offsets / dist[:, None]

        speed = wind.speed.to(self.device).clamp_min(0.0)               # (B,)
        align = wind.direction().to(self.device) @ unit.T               # (B, 8) in [-1, 1]
        strength = (speed / self.p.wind_ref_mps).clamp(0.0, 1.5)[:, None]
        gain = (1.0 + strength * (self.p.wind_boost - 1.0) * align).clamp_min(0.05)

        return (self.p.spread_prob * gain / dist).clamp(0.0, 1.0)

