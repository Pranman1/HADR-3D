"""Watch a trained policy fly, and write it to a video.

    python sandbox/watch.py checkpoints/overnight_seed0.pt
    python sandbox/watch.py checkpoints/overnight_seed0.pt --city "Manhattan Midtown"
    python sandbox/watch.py checkpoints/paper_parity_40pair.pt --paper

Deterministic by default, because what you want to see is what the policy *believes*
rather than what its exploration noise happened to do -- the difference between those
two is the whole reason several earlier "capture rate" numbers were wrong.

Reads:
  blue circles      the rescuers -- dark is the high one, light is the low one
  thin rings        the ground each camera currently covers, green when it has them
  pale dot          the evacuee, reddening as they panic
  green cross       safety
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy import base
from hadrsim.scenario import Scenario, autoplace
from hadrsim.viz.render import Recorder

p = argparse.ArgumentParser()
p.add_argument("checkpoint", nargs="?", default="checkpoints/sweep_closeness.pt",
               help="a checkpoint path, or hover / lawnmower / last_sighting")
p.add_argument("--city", default="Downtown Berkeley")
p.add_argument("--paper", action="store_true", help="the 25 m grid instead of a city")
p.add_argument("--radius", type=float, default=100.0)
p.add_argument("--seconds", type=float, default=120.0)
p.add_argument("--launch", type=float, default=60.0)
p.add_argument("--episodes", type=int, default=4)
p.add_argument("-o", "--out", default="rescue.mp4")
p.add_argument("--fps", type=int, default=20)
p.add_argument("--stochastic", action="store_true")
p.add_argument("--untrained", action="store_true", help="for comparison")
# Full fidelity by default: the policies since 14 Sep trained on the quadrotor tier
# with a detector that can miss, and a video of them flying an easier world would be
# showing something they were never trained for.
p.add_argument("--fidelity", default="quadrotor", choices=["kinematic", "point_mass", "quadrotor"])
p.add_argument("--detector", default="geometric", choices=["oracle", "geometric"])
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
if a.paper:
    from paper_parity import build
    env = build(batch=1, device=dev, pairs=40)
else:
    env = TrainEnv(autoplace(Scenario.preset(
        a.city, radius_m=a.radius, cell_size_m=2.0,
        episode_seconds=a.seconds, wind_mps=0.0, seed=0)), batch=1,
        fidelity=a.fidelity, detector=a.detector, device=dev)
    env.launch_offset_m = a.launch

if a.untrained:
    from hadrsim.policy.ppo import ActorCritic
    net, kind = ActorCritic("conv").to(dev).eval(), "conv, untrained"
else:
    net = base.load(a.checkpoint, dev)
    kind = getattr(net, "manifest", {}).get("encoder", getattr(net, "name", "scripted"))

rec = Recorder(env.terrain, scale=max(2, int(420 / max(env.terrain.shape))))
print(f"{env.terrain.shape[0]}x{env.terrain.shape[1]} cells, {env.clock.max_steps} decisions, "
      f"{kind} encoder, {a.fidelity} + {a.detector}")

captured = reached = harmed = 0
for ep in range(a.episodes):
    env.scenario.seed = ep
    obs, state = env.reset(), None
    got = safe = hurt = False
    with torch.no_grad():
        for _ in range(env.clock.max_steps):
            act, _, _, state = net.act(obs, state, deterministic=not a.stochastic)
            obs, _, _, info = env.step(act)
            rec.frame(env.fire, env.smoke, env.human, env.drones, env.cameras)
            got |= bool(info["captured"][0])
            safe |= bool(info["reached"][0])
            hurt |= bool(info["harmed"][0])
    captured += got; reached += safe; harmed += hurt
    print(f"  episode {ep + 1}: {'CAPTURED' if got else 'not captured':<13}"
          f" {'reached safety' if safe else 'lost to the fire' if hurt else 'ran out of time'}")

print(f"\n  {captured}/{a.episodes} captured, {reached}/{a.episodes} reached safety, "
      f"{harmed}/{a.episodes} lost")
print(f"  wrote {rec.save(a.out, fps=a.fps)}  ({len(rec.frames)} frames)")
