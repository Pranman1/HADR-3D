"""Smoke.

Not decoration.  Smoke is what blocks the high drone's view, and that occlusion is
the paper's whole premise -- the low rescuer sees under things the high rescuer
cannot.  In the 2D simulator that asymmetry is asserted by a lookup table.  Here it
has to come from a plume that is actually in the way.

**Derived from the fire, never simulated independently.**  A parcel of smoke leaving
a burning cell rises at roughly constant speed and is carried sideways by the wind,
so by the time it reaches height `z` it has been aloft for `z / rise_speed` and has
drifted `wind * that` downwind, spreading as it goes.  Which means the smoke at
height `z` is the fire field, shifted downwind and blurred:

        z = 30 m   . . . ~~~~~~~~~        far downwind, wide and thin
        z = 20 m   . . ~~~~~~~
        z = 10 m   . ~~~~~
        z =  0 m   ###                    the burning cells themselves

That is a steady-state plume in uniform wind.  It is not a fluid solver, and it does
not need to be: what the drones care about is how much smoke sits between a camera
and the ground, which this gets right for a few convolutions per layer.

Vertical layers are coarse (5 m) because that is the scale over which occlusion
changes.  The horizontal grid is the terrain's, because the fire is already on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hadrsim.world.wind import Wind


@dataclass
class SmokeParams:
    n_layers: int = 8              # 8 x 5 m = a 40 m column, well above the high drone
    layer_height_m: float = 5.0
    rise_speed_mps: float = 3.0    # how fast the plume climbs
    spread_per_m: float = 0.12     # how fast it fattens with height
    source_sigma_m: float = 2.0    # width of the plume where it leaves the ground
    extinction: float = 0.35       # optical depth per unit density per metre
    decay_per_layer: float = 0.88  # dilution into clear air on the way up
    coarsen: int = 4               # cells per plume cell. See the note in `update`


class Smoke:
    """A plume field over a batch of environments. Rebuilt from the fire each tick."""

    def __init__(self, terrain, batch: int, params: SmokeParams | None = None, device=None):
        self.terrain = terrain
        self.p = params or SmokeParams()
        self.batch = batch
        self.device = torch.device(device or terrain.fuel.device)
        w, h = terrain.shape
        self.density = torch.zeros(batch, self.p.n_layers, w, h, device=self.device)

    # ------------------------------------------------------------------ query

    @property
    def layer_altitudes(self) -> torch.Tensor:
        """Centre height of each layer, in metres."""
        return (torch.arange(self.p.n_layers, device=self.device) + 0.5) * self.p.layer_height_m

    def column_density(self) -> torch.Tensor:
        """(B, W, H) -- total smoke overhead, which is what looking straight down hits."""
        return self.density.sum(1)

    def optical_depth_down(self, from_alt_m: float) -> torch.Tensor:
        """(B, W, H) how much smoke a camera at `from_alt_m` looks through, straight down.

        The high drone's problem in one number: above a fresh plume this is large, and
        drifting a hundred metres downwind is how it gets a clear view again.
        """
        alt = self.layer_altitudes
        below = (alt <= from_alt_m).float()[None, :, None, None]
        return (self.density * below).sum(1) * self.p.extinction * self.p.layer_height_m

    def transmittance_down(self, from_alt_m: float) -> torch.Tensor:
        """(B, W, H) in 0..1 -- the fraction of the ground still visible from above."""
        return torch.exp(-self.optical_depth_down(from_alt_m))

    # ------------------------------------------------------------------- step

    def update(self, fire, wind: Wind) -> None:
        """Rebuild the plume from the fire's current intensity.

        Stateless on purpose: smoke has no memory of its own, so a fire that goes out
        stops producing a column immediately rather than leaving a ghost behind.  Over
        the seconds that matter for seeing through it, that is the right trade.
        """
        # Computed on a coarser grid than the terrain, then resampled.  A plume is
        # diffuse -- it has no feature smaller than several metres -- so resolving it
        # at the map's own resolution buys nothing and costs everything: the shift and
        # blur run once per altitude layer, and at a thousand parallel environments
        # they were two thirds of the entire step. Coarsening by four cuts that
        # sixteenfold and changes the field by less than the model's own uncertainty.
        c = max(1, int(self.p.coarsen))
        cell = self.terrain.cell_size_m * c
        source = F.avg_pool2d(fire.intensity[:, None], c, ceil_mode=True)
        drift = wind.velocity.to(self.device)                     # (B, 2) m/s

        layers = []
        for k, z in enumerate(self.layer_altitudes.tolist()):
            age_s = z / max(self.p.rise_speed_mps, 1e-6)
            offset_cells = drift * age_s / cell                   # (B, 2)
            sigma_cells = (self.p.source_sigma_m + self.p.spread_per_m * z) / cell

            shifted = _translate(source, offset_cells)
            blurred = _blur(shifted, sigma_cells)
            layers.append(blurred[:, 0] * (self.p.decay_per_layer ** k))

        coarse = torch.stack(layers, dim=1)
        if c == 1:
            self.density = coarse
        else:
            w, h = self.terrain.shape
            self.density = F.interpolate(coarse, size=(w, h), mode="bilinear",
                                         align_corners=False)

    def reset(self) -> None:
        self.density.zero_()


# ------------------------------------------------------------------ helpers

def _translate(x: torch.Tensor, offset_cells: torch.Tensor) -> torch.Tensor:
    """Shift a (B, 1, W, H) field by a per-environment, fractional number of cells.

    Zero-filled at the edges: smoke that blows off the map is gone, not wrapped round
    to the other side.
    """
    b, _, w, h = x.shape
    # grid_sample reads output positions *from* input positions, so moving content by
    # +s means sampling at -s. Normalised coordinates run -1..1 across each axis, and
    # its last axis is x, which is our second spatial one.
    tx = -2.0 * offset_cells[:, 1] / max(h - 1, 1)
    ty = -2.0 * offset_cells[:, 0] / max(w - 1, 1)

    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _blur(x: torch.Tensor, sigma_cells: float) -> torch.Tensor:
    """Separable Gaussian blur. Conserves total mass, so spreading thins the plume."""
    if sigma_cells <= 0.05:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma_cells) ** 2)
    k = k / k.sum()

    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="constant"), k.view(1, 1, -1, 1))
    return F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="constant"), k.view(1, 1, 1, -1))
