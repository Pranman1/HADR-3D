"""Observation -> feature vector.  `flat` flattens everything; `conv` runs a small CNN
over the hazard raster and concatenates the rest."""

from __future__ import annotations

import torch
import torch.nn as nn

from hadrsim.core import spec


def flatten_obs(obs: dict, spec_: spec.ObsSpec = spec.DEFAULT_OBS) -> torch.Tensor:
    """Everything except the picture, as one vector."""
    return torch.cat([
        obs["self"],
        obs["detections"].flatten(1),
        obs["teammate"],
        obs["task"],
    ], dim=-1)


class FlatEncoder(nn.Module):
    """Flatten the picture in with everything else."""

    def __init__(self, spec_: spec.ObsSpec = spec.DEFAULT_OBS, out: int = 128):
        super().__init__()
        s = spec_
        raster = s.raster_channels * s.raster_size * s.raster_size
        self.net = nn.Sequential(
            nn.Linear(s.flat_dim + raster, 256), nn.ReLU(),
            nn.Linear(256, out), nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.net(torch.cat([flatten_obs(obs), obs["raster"].flatten(1)], -1))


class ConvEncoder(nn.Module):
    """A small CNN (4->16->32->32) over the raster, concatenated with the flat features."""

    def __init__(self, spec_: spec.ObsSpec = spec.DEFAULT_OBS, out: int = 128):
        super().__init__()
        s = spec_
        self.conv = nn.Sequential(
            nn.Conv2d(s.raster_channels, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        side = s.raster_size // 8
        self.net = nn.Sequential(
            nn.Linear(s.flat_dim + 32 * side * side, 256), nn.ReLU(),
            nn.Linear(256, out), nn.ReLU(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.net(torch.cat([flatten_obs(obs), self.conv(obs["raster"])], -1))


def make(kind: str = "flat", **kwargs) -> nn.Module:
    if kind == "flat":
        return FlatEncoder(**kwargs)
    if kind == "conv":
        return ConvEncoder(**kwargs)
    raise ValueError(f"no encoder {kind!r}")
