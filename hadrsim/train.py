"""Train a policy with the default learner (PPO + LSTM + a CNN over the hazard map).

    hadrsim-train                                  # Berkeley, full fidelity, 300 iterations
    hadrsim-train --seeds 0 1 2 --curriculum       # three seeds, launch distance grows as it earns it
    hadrsim-train --lr 1e-4 --wandb hadr-sim3d     # every hyperparameter is a flag; wandb optional

Every `--eval-every` iterations the policy is scored deterministically on full
episodes through the shared evaluator; the best checkpoint is kept, and the scripted
baselines are printed beside the learned number so no run is read without them.
To use a different learner, ignore this file: `Env.reset()` / `Env.step()` is the gym.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

from hadrsim.env import Env
from hadrsim.eval import evaluate
from hadrsim.policy import base
from hadrsim.policy.ppo import PPO, PPOConfig
from hadrsim.reward.compose import load_weights
from hadrsim.scenario import Scenario, autoplace

CURRICULUM = dict(promote_at=40.0, step_m=15.0, max_m=110.0)   # capture % that earns the next rung


def train(city="Downtown Berkeley", radius=100.0, seconds=120.0, launch=25.0,
          fidelity="quadrotor", detector="geometric", encoder="conv", batch=1024,
          iters=300, seed=0, reward="default", weights=None, ppo: PPOConfig | None = None,
          eval_every=25, eval_batch=256, curriculum=False, tag=None,
          out="checkpoints", device="cuda", wandb_run=None) -> dict:
    """One training run. Returns {best, curve, floors, tag}; checkpoints land in `out`."""
    torch.manual_seed(seed)
    sc = autoplace(Scenario.preset(city, radius_m=radius, cell_size_m=2.0,
                                   episode_seconds=seconds, wind_mps=0.0, seed=seed))
    weights = {**load_weights(reward), **(weights or {})}   # yaml table, then overrides
    make = lambda b: Env(sc, batch=b, weights=weights, fidelity=fidelity, detector=detector, device=device)
    env, judge = make(batch), make(eval_batch)          # judge: a separate batch so scoring never disturbs training
    env.launch_offset_m = judge.launch_offset_m = launch

    tag = tag or f"{city.split()[0].lower()}_s{seed}"
    out = pathlib.Path(out); out.mkdir(exist_ok=True)
    floors = {n: evaluate(judge, base.load(n, device)).capture for n in ("hover", "last_sighting")}
    print(f"[{tag}] floors " + "  ".join(f"{k} {v:.1f}%" for k, v in floors.items()), flush=True)

    learner = PPO(env, ppo or PPOConfig(), encoder=encoder, device=device)
    learner.obs, learner.state = env.reset(), None
    best, curve, t0 = 0.0, [], time.time()
    for block in range(0, iters, eval_every):
        n = min(eval_every, iters - block)
        learner.train(n, log_every=n, wandb_run=wandb_run, start=block, total=iters)
        learner.net.eval()
        det = evaluate(judge, learner.net).capture
        sto = evaluate(judge, learner.net, deterministic=False).capture
        learner.net.train()
        i = block + n
        curve.append({"iteration": i, "launch_m": launch, "capture": det, "capture_stochastic": sto})
        learner.save(out / f"{tag}.pt")
        if det > best:
            best = det
            learner.save(out / f"{tag}_best.pt")
        print(f"[{tag}] {i:>4}  capture {det:5.1f}% (stoch {sto:4.1f}%, best {best:.1f}%)"
              f"  at {launch:.0f} m  {(time.time() - t0) / 60:.0f} min", flush=True)
        if wandb_run:
            wandb_run.log({"eval/capture": det, "eval/capture_stochastic": sto,
                           "eval/best": best, "eval/launch_m": launch}, step=i)
        # The curriculum: launch further out once the policy is good at this range,
        # judged on full episodes -- the in-run rollout flag reads far below the
        # real rate and a threshold set against it was never reached.
        if curriculum and det > CURRICULUM["promote_at"] and launch < CURRICULUM["max_m"]:
            launch += CURRICULUM["step_m"]
            env.launch_offset_m = judge.launch_offset_m = launch
            learner.obs, learner.state = env.reset(), None
            print(f"[{tag}]       launch now {launch:.0f} m out", flush=True)

    result = {"tag": tag, "best": best, "floors": floors, "curve": curve}
    json.dump(result, open(out / f"{tag}.json", "w"), indent=1)
    print(f"[{tag}] best {best:.1f}%   last_sighting {floors['last_sighting']:.1f}%   hover {floors['hover']:.1f}%")
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--city", default="Downtown Berkeley")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--launch", type=float, default=25.0)
    p.add_argument("--fidelity", default="quadrotor", choices=["kinematic", "point_mass", "quadrotor"])
    p.add_argument("--detector", default="geometric", choices=["oracle", "geometric"])
    p.add_argument("--encoder", default="conv", choices=["flat", "conv"])
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batch", type=int, default=256)
    for k, v in PPOConfig().__dict__.items():                 # lr, gamma, clip, entropy, ...
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--reward", default="default", help="a table in configs/reward/, or a yaml path")
    p.add_argument("--weight", nargs=2, action="append", metavar=("TERM", "VALUE"), default=[],
                   help="override one weight of that table, e.g. --weight proximity 0")
    p.add_argument("--wandb", default=None, help="project name; omit to stay offline")
    p.add_argument("--tag", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=a.wandb, config=vars(a), name=a.tag)
        a = argparse.Namespace(**{**vars(a), **dict(wandb.config)})   # a sweep overrides flags

    cfg = PPOConfig(**{k: getattr(a, k) for k in PPOConfig().__dict__})
    weights = {k: float(v) for k, v in a.weight}
    for seed in a.seeds:
        r = train(a.city, a.radius, a.seconds, a.launch, a.fidelity, a.detector, a.encoder, a.batch,
                  a.iters, seed, a.reward, weights, cfg, a.eval_every, a.eval_batch, a.curriculum,
                  a.tag and f"{a.tag}_s{seed}", device=a.device, wandb_run=run)
        if run:
            run.summary[f"best_s{seed}"] = r["best"]
    if run:
        run.finish()


if __name__ == "__main__":
    main()
