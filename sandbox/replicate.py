"""Run the paper-parity experiment twice, and report both.

Two seeds because one is not a result.  Twice in this session a single run has been
reported as progress and failed to replicate -- 22.7% and 0.9% capture from
identical code -- so nothing here prints a number without its pair.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig
from paper_parity import build, report

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
results = {}

for seed in (0, 1):
    print(f"\n{'=' * 66}\nSEED {seed}\n{'=' * 66}")
    torch.manual_seed(seed)
    env = build(batch=1024, seed=seed)
    report(env, "untrained")

    ppo = PPO(env, PPOConfig(horizon=25, epochs=3, entropy=0.01, entropy_final=0.0),
              device=str(env.device))
    ppo.train(ITERS, log_every=25)
    ppo.save(f"checkpoints/parity_seed{seed}.pt")

    ppo.net.eval()
    results[seed] = (report(env, "trained, deterministic", ppo.net),
                     report(env, "trained, stochastic", ppo.net, deterministic=False))

print(f"\n{'=' * 66}\nBOTH SEEDS\n{'=' * 66}")
print("  paper (deterministic)      71.0%")
for seed, (det, sto) in results.items():
    print(f"  seed {seed}: deterministic {det:5.1f}%   stochastic {sto:5.1f}%")
spread = abs(results[0][0] - results[1][0])
print(f"\n  deterministic spread between seeds: {spread:.1f} points")
print("  " + ("reproducible" if spread < 10 else "NOT reproducible -- one run means nothing"))
