"""What a policy is -- anything with `act(obs, state, deterministic)` -- plus the scripted
baselines (hover, lawnmower, last_sighting) and `load()` for either kind."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import torch

from hadrsim.core import spec

_NONE = (None, None)


class Policy(Protocol):
    name: str

    def act(self, obs: dict, state=None, deterministic: bool = False): ...


class Scripted:
    """Shared scaffolding: no learning, no log-probabilities, no value."""

    name = "scripted"

    def _pack(self, hlr: torch.Tensor, llr: torch.Tensor, state):
        act = torch.stack([hlr, llr], 1).clamp(-1.0, 1.0)
        return act, None, None, state

    def act(self, obs, state=None, deterministic: bool = False):
        raise NotImplementedError


class Hover(Scripted):
    name = "hover"

    def act(self, obs, state=None, deterministic: bool = False):
        B = obs["hlr"]["self"].shape[0]
        z = torch.zeros(B, spec.ACTION_DIM, device=obs["hlr"]["self"].device)
        return self._pack(z, z, state)


class Lawnmower(Scripted):
    """The high rescuer sweeps the block in stripes; the low one follows underneath."""

    name = "lawnmower"

    def __init__(self, stripe: float = 0.15, margin: float = 0.08):
        self.stripe, self.margin = stripe, margin

    def act(self, obs, state=None, deterministic: bool = False):
        own = obs["hlr"]["self"]
        B, dev = own.shape[0], own.device
        heading = state if state is not None else torch.ones(B, device=dev)   # +1 east, -1 west
        x, y = own[:, 0], own[:, 1]
        yaw = own[:, 6] * math.pi
        want_yaw = torch.where(heading > 0, torch.zeros_like(yaw), torch.full_like(yaw, math.pi))
        # Reached an edge: flip, and let the lateral command carry us one stripe north.
        at_edge = ((heading > 0) & (x > 1 - self.margin)) | ((heading < 0) & (x < self.margin))
        heading = torch.where(at_edge, -heading, heading)
        err = torch.atan2(torch.sin(want_yaw - yaw), torch.cos(want_yaw - yaw))
        aligned = err.abs() < 0.3
        hlr = torch.stack([
            aligned.float() * 0.7,                          # forward only once pointed
            torch.where(at_edge, torch.full_like(x, 0.6), torch.zeros_like(x)),
            torch.zeros_like(x),
            (err / math.pi * 2).clamp(-1, 1),
        ], -1)
        # The low rescuer mirrors the high one, a little slower.
        llr = hlr * torch.tensor([0.8, 0.8, 0.0, 1.0], device=dev)
        return self._pack(hlr, llr, heading)


class LastSighting(Scripted):
    """Fly toward the person's last known position; hold when there is none."""

    name = "last_sighting"

    def act(self, obs, state=None, deterministic: bool = False):
        out = []
        for key in ("hlr", "llr"):
            o = obs[key]
            rel = o["detections"][:, 0, :3]
            conf, age = o["detections"][:, 0, 6], o["detections"][:, 0, 7]
            know = (conf > 0) | (age < 1.0)
            yaw = o["self"][:, 6] * math.pi
            c, s = torch.cos(yaw), torch.sin(yaw)
            fwd = c * rel[:, 0] + s * rel[:, 1]
            lat = -s * rel[:, 0] + c * rel[:, 1]
            n = torch.stack([fwd, lat], -1).norm(dim=-1).clamp_min(1e-6)
            go = torch.stack([fwd / n, lat / n], -1) * know[:, None].float()
            up = torch.zeros_like(fwd)
            turn = (torch.atan2(lat, fwd) / math.pi).clamp(-1, 1) * know.float()
            out.append(torch.cat([go, up[:, None], turn[:, None]], -1))
        return self._pack(out[0], out[1], state)


class Learned:
    """A trained network, loaded from a checkpoint that says what it is."""

    def __init__(self, net, name: str, manifest: dict):
        self.net, self.name, self.manifest = net, name, manifest

    def act(self, obs, state=None, deterministic: bool = False):
        return self.net.act(obs, state, deterministic=deterministic)

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "Learned":
        from hadrsim.policy.ppo import ActorCritic

        blob = torch.load(path, map_location=device)
        m = blob.get("manifest")
        if m is None:
            m = {"encoder": "conv" if any(".conv." in k for k in blob["net"]) else "flat",
                 "hidden": blob.get("config", {}).get("hidden", 128), "n_drones": 2,
                 "spec_version": spec.SPEC_VERSION}
        if m["spec_version"] != spec.SPEC_VERSION:
            raise ValueError(f"{path}: trained against observation contract v{m['spec_version']}, "
                             f"this code is v{spec.SPEC_VERSION}")
        net = ActorCritic(m["encoder"], m["hidden"], m["n_drones"]).to(device)
        net.load_state_dict(blob["net"])
        net.eval()
        return cls(net, Path(path).stem, m)


SCRIPTED = {"hover": Hover, "lawnmower": Lawnmower, "last_sighting": LastSighting}


def load(name_or_path: str | Path | None, device="cpu"):
    """A policy by name (`hover`, `lawnmower`, `last_sighting`) or checkpoint path."""
    if name_or_path in (None, "", "none"):
        return None
    key = str(name_or_path)
    if key in SCRIPTED:
        return SCRIPTED[key]()
    return Learned.load(key, device)
