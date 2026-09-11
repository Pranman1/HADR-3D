"""Watch a trained policy fly.

    python sandbox/watch.py                       # writes rescue.mp4
    python sandbox/watch.py --episodes 4 -o x.mp4

Loads a checkpoint, flies it, and writes a video.  Deterministic by default, because
what you want to see is what the policy *believes* rather than what its exploration
noise happened to do -- the difference between those two is the whole reason the
earlier "capture rate" numbers were wrong.

Reads:
  blue circles      the rescuers -- dark is the high one, light is the low one
  thin blue rings   the ground each camera currently covers
  pale dot          the evacuee, reddening as they panic
  green cross       safety
"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.policy.ppo import ActorCritic
from hadrsim.viz.render import Recorder
from paper_parity import build

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="checkpoints/paper_parity.pt")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("-o", "--out", default="rescue.mp4")
p.add_argument("--fps", type=int, default=12)
p.add_argument("--stochastic", action="store_true")
p.add_argument("--untrained", action="store_true", help="for comparison")
a = p.parse_args()

env = build(batch=64, device="cuda" if torch.cuda.is_available() else "cpu")
net = ActorCritic().to(env.device)
if not a.untrained:
    net.load_state_dict(torch.load(a.checkpoint, map_location=env.device)["net"])
net.eval()

rec = Recorder(env.terrain, scale=max(2, int(360 / max(env.terrain.shape))))
print(f"{env.terrain.shape[0]}x{env.terrain.shape[1]} map, "
      f"{env.clock.max_steps} decisions per episode")

captured = reached = 0
for ep in range(a.episodes):
    env.scenario.seed = ep
    obs, state = env.reset(), None
    got = safe = False
    with torch.no_grad():
        for _ in range(env.clock.max_steps):
            act, _, _, state = net.act(obs, state, deterministic=not a.stochastic)
            obs, _, _, info = env.step(act)
            rec.frame(env.fire, env.smoke, env.human, env.drones, env.cameras)
            got |= bool(info["captured"][0])
            safe |= bool(info["reached"][0])
    captured += got
    reached += safe
    print(f"  episode {ep + 1}: {'CAPTURED' if got else 'not captured'}"
          f"   {'reached safety' if safe else 'did not reach safety'}")

print(f"\n  {captured}/{a.episodes} captured, {reached}/{a.episodes} reached safety")
print(f"  wrote {rec.save(a.out, fps=a.fps)}  ({len(rec.frames)} frames)")
