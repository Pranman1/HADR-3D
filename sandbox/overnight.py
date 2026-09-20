"""The long run: a CNN policy for the 3D city, two seeds, checkpointed as it goes.

Everything the session established, in one place:

  a CNN over the hazard map     36.9% against 4.3% for the flattened version in a
                                hundred iterations -- the largest single difference
                                anything made, and it had never been switched on
  altitude bands per role       the low rescuer was flying to the 120 m ceiling and
                                sitting there, because its ground footprint grows
                                with height and `coverage` pays for footprint
  normalised observations       and an annealed entropy bonus, the pair that took
                                paper-parity from nothing to 65%
  substep capture               checked at 100 Hz, since the rescuer covers six
                                metres between decisions against a three metre radius
  a curriculum                  launch distance grows as the policy earns it
  finished episodes freeze      every episode ends before the deadline, and a quarter
                                of all training steps were happening after the episode
                                they belonged to was over.  Fixing that took the
                                paper-parity control from 6.2% to 35.1% deterministic
  forty routes, four fires      one per environment, drawn at reset.  With a single
                                route the whole batch was the same episode: an
                                untrained network scored 100%, because one constant
                                action either always intercepts or never does

Checkpointed every 25 iterations because a formatting bug in a report has already
thrown away two and a half million steps of training once.
"""

import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy.ppo import PPO, ActorCritic, PPOConfig
from hadrsim.scenario import Scenario, autoplace
from city import report

PROMOTE_AT, STEP_M, WINDOW, MAX_M = 0.30, 15.0, 8, 110.0
OUT = pathlib.Path("checkpoints")


def scenario(seed):
    sc = autoplace(Scenario.preset("Downtown Berkeley", radius_m=100.0, cell_size_m=2.0,
                                   episode_seconds=120.0, wind_mps=0.0, seed=seed))
    return sc


def run(seed, iters):
    torch.manual_seed(seed)
    sc = scenario(seed)
    env = TrainEnv(sc, batch=1024, fidelity="quadrotor", detector="geometric", device="cuda")
    dist = 25.0
    env.launch_offset_m = dist

    ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
              encoder="conv", device="cuda")
    ppo.obs, ppo.state = env.reset(), None

    history, recent, best = [], [], 0.0
    t0 = time.time()
    for i in range(1, iters + 1):
        buf, last, info, st = ppo.collect()
        adv, ret = ppo.advantages(buf, last)
        stats = ppo.update(buf, adv, ret, st, i / iters)
        rate = float(info["captured"].float().mean())
        recent.append(rate)
        best = max(best, rate)
        history.append({"i": i, "capture": rate, "distance": dist, **stats})

        if len(recent) >= WINDOW:
            avg = sum(recent[-WINDOW:]) / WINDOW
            if avg > PROMOTE_AT and dist < MAX_M:
                dist += STEP_M
                env.launch_offset_m = dist
                ppo.obs, ppo.state = env.reset(), None
                recent.clear()
                print(f"  {i:>4}  {avg*100:5.1f}%  ->  launch now {dist:.0f} m out",
                      flush=True)

        if i % 25 == 0:
            ppo.save(OUT / f"overnight_seed{seed}.pt")
            json.dump(history, open(OUT / f"overnight_seed{seed}.json", "w"))
            r = recent[-WINDOW:] or [0.0]
            print(f"  {i:>4}  {i*30*env.batch:>10,} steps  "
                  f"capture {sum(r)/len(r)*100:5.1f}% (best {best*100:.1f}%)  "
                  f"at {dist:.0f} m  {(time.time()-t0)/60:.0f} min", flush=True)

    ppo.save(OUT / f"overnight_seed{seed}.pt")
    ppo.net.eval()
    env.launch_offset_m = dist
    print(f"  final launch distance {dist:.0f} m", flush=True)
    return report(env, f"seed {seed}, deterministic", ppo.net), best


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    OUT.mkdir(exist_ok=True)
    results = {}
    for seed in (0, 1):
        print(f"\n{'=' * 70}\nSEED {seed}   (CNN encoder, {iters} iterations)\n{'=' * 70}",
              flush=True)
        env_probe = TrainEnv(scenario(seed), batch=256, fidelity="quadrotor",
                             detector="geometric", device="cuda")
        env_probe.launch_offset_m = 25.0
        report(env_probe, "no drone help", None)
        report(env_probe, "untrained", ActorCritic("conv").to("cuda"))
        del env_probe
        results[seed] = run(seed, iters)

    print(f"\n{'=' * 70}\nOVERNIGHT RESULT -- 3D city, CNN policy\n{'=' * 70}")
    for s, (det, best) in results.items():
        print(f"  seed {s}: deterministic {det:5.1f}%   best in training {best*100:5.1f}%")
    spread = abs(results[0][0] - results[1][0])
    print(f"\n  spread {spread:.1f} points -- "
          + ("reproducible" if spread < 10 else "NOT reproducible"))
