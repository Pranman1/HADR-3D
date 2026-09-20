"""Does a policy trained on one city work in another?

    python sandbox/generalise.py checkpoints/overnight_seed0.pt

The question the 2D paper cannot ask.  Its map is a 25 m grid of abstract blocks, so
there is nothing to generalise *to*; ours are real OpenStreetMap extracts, and they
differ in the way cities actually differ -- Manhattan is 87% built and 400 m tall,
Berkeley 54% and 56 m, the Eixample a regular grid of chamfered blocks, the Berkeley
campus free-standing buildings with open ground between them.

A policy that only works where it trained has memorised a street pattern.  One that
holds up across these has learned something about searching.

Every city is scored against its own baselines, because the cities are not equally
hard: on a dense block the evacuee is occluded most of the time and even a perfect
searcher does worse.  The number that means anything is the *gap* over doing nothing,
not the capture rate.
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy.ppo import ActorCritic
from hadrsim.scenario import PRESETS, Scenario, autoplace
from hadrsim.world import osm

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--batch", type=int, default=256)
p.add_argument("--launch", type=float, default=60.0, help="metres from the evacuee")
p.add_argument("--radius", type=float, default=100.0)
p.add_argument("--seconds", type=float, default=120.0)
# Full fidelity by default -- the policies since 14 Sep trained on the quadrotor tier
# with a detector that can miss, and scoring them in an easier world would flatter them.
p.add_argument("--fidelity", default="quadrotor", choices=["kinematic", "point_mass", "quadrotor"])
p.add_argument("--detector", default="geometric", choices=["oracle", "geometric"])
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
blob = torch.load(a.checkpoint, map_location=dev)
kind = "conv" if any("conv" in k for k in blob["net"]) else "flat"
net = ActorCritic(kind, blob.get("config", {}).get("hidden", 128)).to(dev)
net.load_state_dict(blob["net"])
net.eval()
print(f"{a.checkpoint}  ({kind} encoder, {a.fidelity} + {a.detector})   launch {a.launch:.0f} m   {a.batch} episodes each\n")


def run(env, policy):
    obs, state = env.reset(), None
    z = lambda: torch.zeros(env.batch, dtype=torch.bool, device=env.device)
    cap, seen, reached, harmed = z(), z(), z(), z()
    with torch.no_grad():
        for _ in range(env.clock.max_steps):
            if policy is None:
                act = torch.zeros(env.batch, 2, 4, device=env.device)
            else:
                act, _, _, state = policy.act(obs, state, deterministic=True)
            obs, _, _, info = env.step(act)
            cap |= info["captured"]; reached |= info["reached"]
            harmed |= info["harmed"]; seen |= obs["seen_by"].any(-1)
    pct = lambda t: float(t.float().mean()) * 100
    return pct(cap), pct(seen), pct(reached), pct(harmed)


print(f"{'city':<24}{'built':>7}{'nothing':>9}{'policy':>8}{'gain':>7}"
      f"{'seen':>7}{'safe':>7}{'lost':>7}")
rows = []
for name in PRESETS:
    if name.startswith("Paper"):
        continue
    sc = Scenario.preset(name, radius_m=a.radius, cell_size_m=2.0,
                         episode_seconds=a.seconds, wind_mps=0.0, seed=0)
    # Overpass bans a host that queries it hard, so an uncached city is skipped
    # rather than allowed to stall the whole sweep.
    bbox = osm.bbox_around(sc.lat, sc.lon, sc.radius_m)
    if not (pathlib.Path("maps/osm_cache") / f"{bbox.key(osm.QUERY)}.json").exists():
        print(f"{name:<24}  not cached, skipped")
        continue

    env = TrainEnv(autoplace(sc), batch=a.batch, fidelity=a.fidelity, detector=a.detector, device=dev)
    env.launch_offset_m = a.launch
    built = float((env.terrain.solid_top > 0).float().mean()) * 100
    base = run(env, None)[0]
    cap, seen, reached, harmed = run(env, net)
    rows.append((name, cap - base))
    print(f"{name:<24}{built:6.0f}%{base:8.1f}%{cap:7.1f}%{cap-base:+6.1f}"
          f"{seen:6.1f}%{reached:6.1f}%{harmed:6.1f}%")
    del env

if rows:
    gains = [g for _, g in rows]
    held = [r for r in rows if not r[0].startswith("Downtown Berkeley")]
    print(f"\n  trained-on city gain : {rows[0][1]:+.1f} points")
    if held:
        print(f"  held-out mean gain   : {sum(g for _, g in held)/len(held):+.1f} points"
              f"  across {len(held)} cities")
    print(f"  worst city           : {min(rows, key=lambda r: r[1])[0]} "
          f"({min(gains):+.1f})")
