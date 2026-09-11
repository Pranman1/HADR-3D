"""The city, learned in stages.

The two fixes that took paper-parity from nothing to 65% did not transfer here: on a
300 m block the policy is still at 1.7% after three million steps.  The difference
is not the learner -- it is the same learner -- it is that the rescuers launch a
hundred metres from somebody they have to find before they can do anything, and a
reward that only pays on success pays nothing at all for the first hundred metres.

So: start them close enough that capture happens by accident, and move the launch
point further out each time they get good at the range they are on.  The policy
never faces a version of the problem it has no gradient for.

Distance is the only thing that changes.  Same map, same fire, same person, same
reward -- so whatever it learns at 20 m is still true at 120 m, which is the property
that makes a curriculum work rather than merely postpone the problem.
"""

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig
from city import build, report

PROMOTE_AT = 0.45          # capture rate that counts as "good enough at this range"
STEP_M = 25.0              # how much further out the next stage launches
WINDOW = 6                 # iterations averaged before deciding


def place_base(env, distance_m: float) -> None:
    """Put the launch point `distance_m` from the person, along the route to safety."""
    start = torch.tensor(env.scenario.human_start[0])
    goal = torch.tensor(env.scenario.human_goal[0])
    away = (start - goal)
    away = away / away.norm().clamp_min(1e-6)
    base = start + away * distance_m
    w, h = env.terrain.extent_m
    env.scenario.drone_start = [float(base[0].clamp(5, w - 5)),
                                float(base[1].clamp(5, h - 5))]


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    results = {}

    for seed in (0, 1):
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        torch.manual_seed(seed)
        env = build(seed)
        distance = 20.0
        place_base(env, distance)
        ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
                  device=str(env.device))
        print(f"map {env.terrain.shape}; starting the curriculum at {distance:.0f} m")

        recent = []
        for i in range(1, iters + 1):
            buf, last, info, state0 = ppo.collect()
            adv, ret = ppo.advantages(buf, last)
            ppo.update(buf, adv, ret, state0, i / iters)
            recent.append(float(info["captured"].float().mean()))

            if len(recent) >= WINDOW:
                rate = sum(recent[-WINDOW:]) / WINDOW
                if rate > PROMOTE_AT and distance < 130.0:
                    distance += STEP_M
                    place_base(env, distance)
                    ppo.obs, ppo.state = env.reset(), None
                    recent.clear()
                    print(f"  {i:>4}  capture {rate * 100:5.1f}%  ->  launch moved out "
                          f"to {distance:.0f} m")
            if i % 40 == 0:
                print(f"  {i:>4}  steps {i * 30 * env.batch:>9,}  "
                      f"capture {sum(recent[-WINDOW:]) / max(len(recent[-WINDOW:]), 1) * 100:5.1f}%"
                      f"  at {distance:.0f} m")

        ppo.save(f"checkpoints/curriculum_seed{seed}.pt")
        ppo.net.eval()
        print(f"  final launch distance {distance:.0f} m")
        results[seed] = report(env, "trained, deterministic", ppo.net)

    print(f"\n{'=' * 70}\nBOTH SEEDS -- the real city, with a curriculum\n{'=' * 70}")
    for seed, cap in results.items():
        print(f"  seed {seed}: capture {cap:5.1f}%")
    spread = abs(results[0] - results[1])
    print(f"\n  spread {spread:.1f} points -- "
          + ("reproducible" if spread < 10 else "NOT reproducible"))
