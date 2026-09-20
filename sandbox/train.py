"""Train one policy, from the command line or a wandb sweep agent.

    python sandbox/train.py                                   # defaults below
    python sandbox/train.py --lr 1e-4 --entropy 0.02 --seed 3 --wandb hadr-sim3d

Every hyperparameter is a flag, so a sweep is a yaml file (`sandbox/sweep.yaml`)
and `wandb agent`, one agent per GPU -- what Gaby suggested.  Evaluates on full
episodes every `--eval-every` iterations through the shared evaluator, keeps the
best checkpoint, and reports the scripted baselines beside the learned number so
no run is ever read without them.
"""

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.eval import evaluate
from hadrsim.policy import base
from hadrsim.policy.ppo import PPO, PPOConfig
from hadrsim.reward.compose import DEFAULT
from hadrsim.scenario import Scenario, autoplace

p = argparse.ArgumentParser()
p.add_argument("--city", default="Downtown Berkeley")
p.add_argument("--radius", type=float, default=100.0)
p.add_argument("--seconds", type=float, default=120.0)
p.add_argument("--launch", type=float, default=25.0)
p.add_argument("--fidelity", default="quadrotor")
p.add_argument("--detector", default="geometric")
p.add_argument("--encoder", default="conv")
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--iters", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--entropy", type=float, default=0.01)
p.add_argument("--entropy-final", type=float, default=0.002)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--horizon", type=int, default=30)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--apparent-size", type=float, default=20.0)
p.add_argument("--proximity", type=float, default=5.0)
p.add_argument("--eval-every", type=int, default=25)
p.add_argument("--eval-batch", type=int, default=256)
p.add_argument("--wandb", default=None, help="project name; omit to stay offline")
p.add_argument("--tag", default=None)
a = p.parse_args()

run = None
if a.wandb:
    import wandb
    run = wandb.init(project=a.wandb, config=vars(a), name=a.tag)
    a = argparse.Namespace(**{**vars(a), **dict(wandb.config)})   # a sweep overrides flags

torch.manual_seed(a.seed)
sc = autoplace(Scenario.preset(a.city, radius_m=a.radius, cell_size_m=2.0,
                               episode_seconds=a.seconds, wind_mps=0.0, seed=a.seed))
weights = {**DEFAULT, "apparent_size": a.apparent_size, "proximity": a.proximity}
env = TrainEnv(sc, batch=a.batch, weights=weights, fidelity=a.fidelity, detector=a.detector, device="cuda")
env.launch_offset_m = a.launch
judge = TrainEnv(sc, batch=a.eval_batch, weights=weights, fidelity=a.fidelity, detector=a.detector, device="cuda")
judge.launch_offset_m = a.launch

tag = a.tag or f"train_{a.city.split()[0].lower()}_s{a.seed}"
out = pathlib.Path("checkpoints"); out.mkdir(exist_ok=True)
floors = {n: evaluate(judge, base.load(n, "cuda")).capture for n in ("hover", "last_sighting")}
print("floors:", {k: f"{v:.1f}%" for k, v in floors.items()}, flush=True)
if run: run.summary.update({f"floor/{k}": v for k, v in floors.items()})

ppo = PPO(env, PPOConfig(lr=a.lr, entropy=a.entropy, entropy_final=a.entropy_final, epochs=a.epochs,
                         horizon=a.horizon, clip=a.clip, hidden=a.hidden), encoder=a.encoder, device="cuda")
ppo.obs, ppo.state = env.reset(), None

best, t0, curve = 0.0, time.time(), []
for block in range(0, a.iters, a.eval_every):
    n = min(a.eval_every, a.iters - block)
    ppo.train(n, log_every=max(1, n), wandb_run=run, start=block, total=a.iters)
    ppo.net.eval()
    det, sto = evaluate(judge, ppo.net).capture, evaluate(judge, ppo.net, deterministic=False).capture
    ppo.net.train()
    i = block + n
    curve.append({"iteration": i, "capture": det, "capture_stochastic": sto})
    ppo.save(out / f"{tag}.pt")
    if det > best:
        best = det; ppo.save(out / f"{tag}_best.pt")
    print(f"  {i:>4}  capture {det:5.1f}% (stoch {sto:4.1f}%, best {best:.1f}%)  {(time.time()-t0)/60:.0f} min", flush=True)
    if run: run.log({"eval/capture": det, "eval/capture_stochastic": sto, "eval/best": best}, step=i)

json.dump({"args": vars(a), "floors": floors, "curve": curve, "best": best}, open(out / f"{tag}.json", "w"), indent=1)
print(f"best {best:.1f}%  vs last_sighting {floors['last_sighting']:.1f}%  vs hover {floors['hover']:.1f}%")
if run: run.summary["best"] = best; run.finish()
