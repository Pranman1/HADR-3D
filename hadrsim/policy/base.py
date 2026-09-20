"""What a policy is, and the ones that come for free.

A policy is anything that turns an observation (the contract in `core/spec.py`)
into a `(B, 2, 4)` action -- one velocity setpoint per drone -- and may carry state
between decisions:

    act(obs, state, deterministic) -> action, logp, value, state
    reset() -> state

Three scripted ones live here.  They exist because a table of methods needs
baselines a reader can picture, and because "does the learned policy beat a
lawnmower sweep?" is a question worth being able to answer in one line:

    hover           holds position.  The floor everything is read against.
    lawnmower       the high rescuer sweeps the block in stripes; the low one
                    follows underneath it.  The classic search pattern.
    last_sighting   fly toward wherever the team last saw the person.  The
                    obvious hand-written competitor to the learned closing
                    behaviour.

`load(name_or_path)` returns any of them or a trained network from a checkpoint,
so the panel, the evaluator and the sandbox never have to know which is which.
"""

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
    """Stripes across the block, turning at the edges.

    Reads only what the observation gives it: own position as a fraction of the
    map (`self[0:2]`) and heading (`self[6]`, yaw over pi).  Commands are in the
    body frame, so "forward" is wherever the airframe is pointing -- the sweep is
    flown by yawing 180 degrees at each edge and stepping one stripe over.
    """

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
    """Fly toward the person's last known position; hold when there is none.

    Detection slot 0 carries the freshest sighting, or the team's remembered one
    with its staleness, as relative position over camera range in the world frame;
    the body-frame command is that vector rotated by the drone's own heading.
    """

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
            # Older checkpoints carry no manifest; the weights themselves say which
            # encoder trained, and everything else is the default it had.
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
