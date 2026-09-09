"""Wind.

It exists for two reasons, neither of them decoration: it makes fire spread
directional, and it shears the smoke plume sideways so the column above a fire is
not a neat vertical cylinder.  Both change what the high drone can see.

Uniform over space, varying in time.  A real urban wind field channels down streets
and swirls behind buildings, which matters a great deal for firefighting and very
little for whether a drone can see through a plume 40 m up.  Left flat until
something needs otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hadrsim.core.rng import Rng


@dataclass
class Wind:
    """Wind velocity in the ground plane, one vector per environment.

    `speed_mps` and `bearing_rad` are the slowly varying mean; gusts ride on top.
    Bearing is the direction the wind blows *towards*, measured from +x.
    """

    velocity: torch.Tensor          # (B, 2) metres per second
    base_speed: torch.Tensor        # (B,)
    base_bearing: torch.Tensor      # (B,)
    gustiness: float = 0.25         # gust size as a fraction of the mean
    variability_hz: float = 0.05    # how fast the mean direction wanders
    _t: float = 0.0

    @property
    def speed(self) -> torch.Tensor:
        return self.velocity.norm(dim=-1)

    def step(self, dt: float, rng: Rng) -> None:
        """Advance the wind. Slow wander in the mean, fast noise in the gusts."""
        self._t += dt
        phase = 2 * math.pi * self.variability_hz * self._t
        # A slow sinusoid gives a mean that wanders reproducibly; the gust is where
        # the randomness lives, so the wind is not simply noise about a fixed vector.
        bearing = self.base_bearing + 0.4 * math.sin(phase) * torch.ones_like(self.base_bearing)
        speed = self.base_speed * (1.0 + 0.15 * math.sin(0.7 * phase))

        gust = rng.normal(*self.base_speed.shape, 2) * (self.gustiness * speed)[..., None]
        mean = torch.stack([speed * bearing.cos(), speed * bearing.sin()], dim=-1)
        self.velocity = mean + gust

    def direction(self) -> torch.Tensor:
        """Unit vector the wind blows towards. Zero-length where the air is still."""
        n = self.speed.clamp_min(1e-6)
        return self.velocity / n[..., None]


def make(
    batch: int,
    speed_mps: float = 3.0,
    bearing_deg: float = 45.0,
    gustiness: float = 0.25,
    spread: float = 0.0,
    rng: Rng | None = None,
    device="cpu",
) -> Wind:
    """Wind for a batch of environments.

    `spread` randomises the mean per environment -- speed by that fraction and
    bearing by that many radians -- so a batch is not a thousand copies of the same
    weather.  Zero gives every environment identical wind, which is what a
    reproducibility test wants.
    """
    speed = torch.full((batch,), float(speed_mps), device=device)
    bearing = torch.full((batch,), math.radians(bearing_deg), device=device)
    if spread > 0.0:
        if rng is None:
            raise ValueError("randomising the wind needs a generator passed in")
        speed = speed * (1.0 + rng.uniform(batch, low=-spread, high=spread))
        bearing = bearing + rng.uniform(batch, low=-spread, high=spread)

    w = Wind(
        velocity=torch.stack([speed * bearing.cos(), speed * bearing.sin()], dim=-1),
        base_speed=speed,
        base_bearing=bearing,
        gustiness=gustiness,
    )
    return w


def still(batch: int, device="cpu") -> Wind:
    """No wind. What `paper_parity` uses, because the paper has none."""
    return make(batch, speed_mps=0.0, bearing_deg=0.0, gustiness=0.0, device=device)
