"""Bit-for-bit regression guard for refactors.

The behavioural tests say *what* the simulator does; this says it still does exactly
the same thing after a change that was meant to be pure restructuring.  A fixed seed,
a fixed scripted action stream, a handful of decisions on the CPU, and a signature
of everything that came out -- positions, captures, every reward term -- compared to
the numbers recorded before the refactor began.

If a change is *supposed* to alter behaviour, regenerate on purpose:

    python tests/test_golden.py --record
"""

import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hadrsim.env import Env
from hadrsim.scenario import Scenario, autoplace

GOLDEN = pathlib.Path(__file__).with_name("golden.json")
CITY_REWARD = str(pathlib.Path(__file__).parents[1] / "experiments" / "city_reward.yaml")   # recorded with our reward
STEPS, BATCH = 12, 6


def _scenarios():
    yield "paper", Scenario.preset("Paper grid (25 m)", cell_size_m=1.0, episode_seconds=50.0,
                                   wind_mps=0.0, seed=3), dict(fidelity="kinematic", detector="oracle")
    yield "berkeley", Scenario.preset("Downtown Berkeley", radius_m=100.0, cell_size_m=2.0,
                                      episode_seconds=120.0, wind_mps=4.0, seed=3), dict(
        fidelity="quadrotor", detector="geometric")


def signature() -> dict:
    out = {}
    for name, sc, kw in _scenarios():
        torch.manual_seed(0)
        env = Env(autoplace(sc), batch=BATCH, device="cpu", weights=CITY_REWARD, **kw)
        env.launch_offset_m = 25.0
        obs = env.reset()
        g = torch.Generator().manual_seed(11)
        parts, reward_sum = {}, 0.0
        for _ in range(STEPS):
            act = (torch.rand(BATCH, 2, 4, generator=g) * 2 - 1)
            obs, r, done, info = env.step(act)
            reward_sum += float(r.sum())
            for k, v in info["parts"].items():
                parts[k] = parts.get(k, 0.0) + v
        out[name] = {
            "reward_sum": round(reward_sum, 4),
            "parts": {k: round(v, 5) for k, v in sorted(parts.items())},
            "captured": int(info["captured"].sum()),
            "reached": int(info["reached"].sum()),
            "harmed": int(info["harmed"].sum()),
            "drone_pos_sum": round(float(sum(d.pos.sum() for d in env.drones)), 3),
            "human_pos_sum": round(float(env.human.pos.sum()), 3),
            "fire_cells": int(env.fire.burning.sum()),
            "obs_self_sum": round(float(obs["hlr"]["self"].sum() + obs["llr"]["self"].sum()), 4),
            "obs_raster_sum": round(float(obs["hlr"]["raster"].sum() + obs["llr"]["raster"].sum()), 3),
        }
    return out


def test_behaviour_is_unchanged():
    if not GOLDEN.exists():
        pytest.skip("no golden.json recorded yet")
    want, got = json.loads(GOLDEN.read_text()), signature()
    for name in want:
        for key, w in want[name].items():
            g = got[name][key]
            if isinstance(w, dict):
                for k, wv in w.items():
                    assert abs(g[k] - wv) <= 1e-3 * max(1.0, abs(wv)), f"{name}.parts.{k}: {g[k]} != {wv}"
            else:
                tol = 1e-3 * max(1.0, abs(w)) if isinstance(w, float) else 0
                assert abs(g - w) <= tol, f"{name}.{key}: {g} != {w}"


if __name__ == "__main__":
    if "--record" in sys.argv:
        GOLDEN.write_text(json.dumps(signature(), indent=2))
        print(f"recorded {GOLDEN}")
    else:
        print(json.dumps(signature(), indent=2))
