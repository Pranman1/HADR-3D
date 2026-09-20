"""The real city, with the two fixes that made paper-parity work.

Same policy code, same rewards, same learner -- only the world is harder: a 300 m
block of real Berkeley, drones with mass and momentum, ray-traced occlusion, a fire
that spreads through actual fuel, and a person who panics and routes around it.

Two seeds, because one is not a result.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hadrsim.envs.train import TrainEnv
from hadrsim.eval import evaluate
from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig
from hadrsim.scenario import Scenario, autoplace


def build(seed=0, batch=1024, device="cuda"):
    sc = autoplace(Scenario.preset("Downtown Berkeley", radius_m=150.0, cell_size_m=2.0,
                                   episode_seconds=120.0, wind_mps=0.0, seed=seed))
    return TrainEnv(sc, batch=batch, device=device)


def report(env, tag, net=None, deterministic=True):
    """One printed row. Kept as a name the older scripts import; the loop itself is
    `hadrsim.eval.evaluate`, so every script counts the same things."""
    m = evaluate(env, net, deterministic=deterministic)
    print(m.line(tag))
    return m.capture


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    results = {}
    for seed in (0, 1):
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        torch.manual_seed(seed)
        env = build(seed)
        print(f"map {env.terrain.shape}, {env.clock.max_steps} decisions, "
              f"base {[round(v) for v in env.scenario.drone_start]}")
        report(env, "no drone help", None)
        report(env, "untrained", ActorCritic().to(env.device))

        ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
                  device=str(env.device))
        ppo.train(iters, log_every=25)
        ppo.save(f"checkpoints/city_seed{seed}.pt")
        ppo.net.eval()
        results[seed] = report(env, "trained, deterministic", ppo.net)

    print(f"\n{'=' * 70}\nBOTH SEEDS -- the real city\n{'=' * 70}")
    for seed, cap in results.items():
        print(f"  seed {seed}: capture {cap:5.1f}%")
    spread = abs(results[0] - results[1])
    print(f"\n  spread {spread:.1f} points -- "
          + ("reproducible" if spread < 10 else "NOT reproducible"))
