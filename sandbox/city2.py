"""The city again, with everything the sandbox taught us.

Four changes from the run that sat at 1.7%:

  matched ratios   198 m map, 120 decisions -- the rescuers now cover 12.8% of it
                   with one look, against the paper's 13%.  At 150 m they covered
                   5.7%, half of what the published result was ever tested at.
  substep capture  checked at 100 Hz rather than 2 Hz.  The low rescuer covers six
                   metres between decisions against a three metre capture radius, so
                   it could fly straight through somebody and register nothing.
  normalised obs   and an annealed entropy bonus -- the two that took paper-parity
                   from nothing to 65%.
  a curriculum     promoted at 25%, not 45%.  The old bar was taken from the paper's
                   final number and no stage ever cleared it, so the curriculum never
                   advanced past its first rung.

Two seeds, and the deterministic policy is what counts.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig
from hadrsim.scenario import Scenario, autoplace
from city import report

PROMOTE_AT, STEP_M, WINDOW, MAX_M = 0.25, 20.0, 8, 110.0


def make(seed):
    sc = autoplace(Scenario.preset("Downtown Berkeley", radius_m=100.0, cell_size_m=2.0,
                                   episode_seconds=60.0, wind_mps=0.0, seed=seed))
    return TrainEnv(sc, batch=1024, device="cuda"), sc


def place(env, sc, metres):
    start = torch.tensor(sc.human_start[0])
    goal = torch.tensor(sc.human_goal[0])
    away = (start - goal) / (start - goal).norm().clamp_min(1e-6)
    w, h = env.terrain.extent_m
    p = start + away * metres
    sc.drone_start = [float(p[0].clamp(5, w - 5)), float(p[1].clamp(5, h - 5))]


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = {}

    for seed in (0, 1):
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        torch.manual_seed(seed)
        env, sc = make(seed)
        print(f"map {env.terrain.extent_m[0]:.0f} m, {env.clock.max_steps} decisions")
        report(env, "no drone help", None)

        dist = 25.0
        place(env, sc, dist)
        ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
                  device="cuda")
        ppo.obs, ppo.state = env.reset(), None

        recent = []
        for i in range(1, iters + 1):
            buf, last, info, st = ppo.collect()
            adv, ret = ppo.advantages(buf, last)
            ppo.update(buf, adv, ret, st, i / iters)
            recent.append(float(info["captured"].float().mean()))

            if len(recent) >= WINDOW:
                rate = sum(recent[-WINDOW:]) / WINDOW
                if rate > PROMOTE_AT and dist < MAX_M:
                    dist += STEP_M
                    place(env, sc, dist)
                    ppo.obs, ppo.state = env.reset(), None
                    recent.clear()
                    print(f"  {i:>4}  {rate * 100:5.1f}%  ->  launch now {dist:.0f} m out")
            if i % 40 == 0:
                r = recent[-WINDOW:] or [0.0]
                print(f"  {i:>4}  steps {i * 30 * env.batch:>10,}  "
                      f"capture {sum(r) / len(r) * 100:5.1f}%  at {dist:.0f} m")

        ppo.save(f"checkpoints/city2_seed{seed}.pt")
        ppo.net.eval()
        print(f"  reached {dist:.0f} m launch distance")
        out[seed] = report(env, "trained, deterministic", ppo.net)

    print(f"\n{'=' * 70}\nBOTH SEEDS -- the real city\n{'=' * 70}")
    for s, c in out.items():
        print(f"  seed {s}: capture {c:5.1f}%")
    print(f"\n  spread {abs(out[0] - out[1]):.1f} points -- "
          + ("reproducible" if abs(out[0] - out[1]) < 10 else "NOT reproducible"))
