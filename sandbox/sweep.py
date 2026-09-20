"""Which dimension of the city actually breaks learning?

Map size and episode length were changed together when the city scenario was built,
so neither has ever been tested alone.  Four short runs separate them.

The hypothesis worth testing: the paper's rescuers cover about 13% of their map with
a single look, and at a 150 m radius ours cover 5.7% -- half of it.  That would show
up as poor tracking rather than poor control, which is exactly what the parity
numbers said (capture near the paper's, time-in-view a tenth of it).
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy.ppo import PPO, PPOConfig
from hadrsim.scenario import Scenario, autoplace
from city import report

CONFIGS = [
    ("small map, short episode", 100.0, 60.0),   # the paper's ratios
    ("small map, long episode ", 100.0, 120.0),
    ("big map,   short episode", 150.0, 60.0),
    ("big map,   long episode ", 150.0, 120.0),
]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 90

print(f"{ITERS} iterations each, batch 512, launch 25 m out -- direction only, not a result\n")
for label, radius, seconds in CONFIGS:
    torch.manual_seed(0)
    sc = autoplace(Scenario.preset("Downtown Berkeley", radius_m=radius, cell_size_m=2.0,
                                   episode_seconds=seconds, wind_mps=0.0, seed=0))
    # A fixed, modest head start, so the comparison is about the world and not the
    # curriculum stage each config happens to have reached.
    start = torch.tensor(sc.human_start[0])
    goal = torch.tensor(sc.human_goal[0])
    away = (start - goal) / (start - goal).norm().clamp_min(1e-6)
    sc.drone_start = (start + away * 25.0).tolist()

    env = TrainEnv(sc, batch=512, device="cuda")
    ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
              device="cuda")
    best = 0.0
    for i in range(1, ITERS + 1):
        buf, last, info, st = ppo.collect()
        adv, ret = ppo.advantages(buf, last)
        ppo.update(buf, adv, ret, st, i / ITERS)
        best = max(best, float(info["captured"].float().mean()))
    ppo.net.eval()
    print(f"{label}  ({env.clock.max_steps:>3} decisions, {env.terrain.extent_m[0]:.0f} m)")
    print(f"    best during training {best * 100:5.1f}%")
    report(env, "    deterministic", ppo.net)
