"""Turning an observation into a vector the network can use.

Two encoders, and the difference between them is a hyperparameter rather than a
research question -- which is worth saying plainly, because an earlier draft of the
plan called this "the CNN" and made a choice of layer sound like a contribution.
The CNN that matters is the *detector*, which reads camera images and is trained
separately; this one reads a small map the simulator hands over.

    flat    the surroundings picture flattened into the rest.  Fewer parameters,
            trains faster, and works.  The default.
    conv    a few convolutions over that picture first.  Should read spatial layout
            better -- more efficient search patterns, going round a building rather
            than into it -- and step 4 is where that gets measured rather than
            assumed.

One network flies both drones, as in the paper, so it sees both observations at
once.  What keeps a second network possible later is that each drone's observation
is assembled and encoded *separately* and only then concatenated: splitting them
means changing where the halves go, not changing the contract.
"""

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
    """Convolve the picture, then mix it with the rest.

    Sanika's `fire-encoding` branch prototyped this shape -- Conv2d 1->16->32 ahead of
    the recurrence -- though it was never validated and hardcoded a 5x5 view in three
    places.  Here the picture is drone-centred and fixed-size, so map size never
    reaches the network.
    """

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
