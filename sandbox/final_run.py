"""The long run with the reward balance the sweep chose.

    python sandbox/final_run.py closeness 800        # variant name, iterations

Everything the sweep established, at full fidelity -- the quadrotor tier and a
detector that can miss -- for two seeds, checkpointed every 25 iterations.  The
sweep's 150-iteration ranking (per-variant gain over doing nothing, Berkeley, 25 m):

    closeness        +35.9      apparent_size 20, proximity 5
    search first      +7.8
    as shipped        +5.3
    quiet proximity   +4.9
    outcome heavy     -5.8      sparse outcomes with no dense guide got *worse*

The lesson in that table: the policy needs a dense signal to climb, and the one that
works is "how big the person is in the low rescuer's frame" -- closeness, but only
while actually in view.  The paper's 5x5 detection window (2.5 m) had been quietly
providing exactly that.
"""

import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from hadrsim.envs.train import TrainEnv
from hadrsim.policy.ppo import PPO, PPOConfig
from hadrsim.reward.compose import DEFAULT
from hadrsim.scenario import Scenario, autoplace
from city import report
from reward_sweep import FIDELITY, DETECTOR, VARIANTS

# Promotion is decided on a *full-episode, deterministic* evaluation every EVAL_EVERY
# iterations -- not on the capture flag sampled at the end of each 30-step rollout
# window.  The window metric reads far below the real capture rate (the sweep's
# 45% full-episode policy showed 15-20% on it), so a threshold set against it is
# never reached and the curriculum silently stays on its first rung.  That is what
# happened to the first 600-iteration run on 14 Sep: 25 m throughout.
PROMOTE_AT, STEP_M, MAX_M, EVAL_EVERY, EVAL_BATCH = 0.40, 15.0, 110.0, 25, 256
OUT = pathlib.Path("checkpoints")


def scenario(seed):
    return autoplace(Scenario.preset("Downtown Berkeley", radius_m=100.0, cell_size_m=2.0,
                                     episode_seconds=120.0, wind_mps=0.0, seed=seed))


@torch.no_grad()
def _capture(env, net, deterministic: bool = True) -> float:
    """Capture rate over full episodes -- the number everything is reported in,
    measured the way it is reported.  Stochastic alongside deterministic, because
    deterministic falling *below* stochastic is the signature of a policy that has
    stopped converging, and it is cheaper to see that at iteration 200 than 600."""
    obs, state = env.reset(), None
    got = torch.zeros(env.batch, dtype=torch.bool, device=env.device)
    for _ in range(env.clock.max_steps):
        act, _, _, state = net.act(obs, state, deterministic=deterministic)
        obs, _, _, info = env.step(act)
        got |= info["captured"]
    return float(got.float().mean())


def run(variant, seed, iters):
    torch.manual_seed(seed)
    weights = {**DEFAULT, **VARIANTS[variant]}
    env = TrainEnv(scenario(seed), batch=1024, weights=weights, fidelity=FIDELITY,
                   detector=DETECTOR, device="cuda")
    dist = 25.0
    env.launch_offset_m = dist
    # An entropy *floor* rather than a decay to zero -- cheap insurance, not a cure.
    # The 600-iteration run ended at 23.0% deterministic AND stochastic against 42.2%
    # at iteration 150; that looked like exploration collapsing, but log_std was
    # unchanged (0.61) and the actor means unsaturated.  What changed was behaviour:
    # the later policy flew half as far and sat lower -- it drifted toward loitering.
    # The protection against that is keeping the best evaluated checkpoint, below.
    ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.002),
              encoder="conv", device="cuda")
    ppo.obs, ppo.state = env.reset(), None
    tag = f"final_{variant.replace(' ', '_')}_seed{seed}"

    # A separate, smaller environment for the honest measurement, so evaluating never
    # disturbs the training batch's episodes or recurrent state.
    judge = TrainEnv(scenario(seed), batch=EVAL_BATCH, weights=weights, fidelity=FIDELITY,
                     detector=DETECTOR, device="cuda")

    history, best, t0 = [], 0.0, time.time()
    for i in range(1, iters + 1):
        buf, last, info, st = ppo.collect()
        adv, ret = ppo.advantages(buf, last)
        stats = ppo.update(buf, adv, ret, st, i / iters)
        history.append({"i": i, "window": float(info["captured"].float().mean()),
                        "distance": dist, **stats})
        if i % EVAL_EVERY == 0:
            judge.launch_offset_m = dist
            ppo.net.eval()
            real = _capture(judge, ppo.net)
            noisy = _capture(judge, ppo.net, deterministic=False)
            ppo.net.train()
            # Every evaluated checkpoint is kept, and the best one is copied aside.
            # The first 600-iteration run only kept the latest, and the latest was
            # half as good as iteration 150 -- the peak was overwritten and lost.
            ppo.save(OUT / f"{tag}_i{i:04d}.pt")
            ppo.save(OUT / f"{tag}.pt")
            if real > best:
                best = real
                ppo.save(OUT / f"{tag}_best.pt")
            history[-1].update({"capture": real, "capture_stochastic": noisy})
            json.dump(history, open(OUT / f"{tag}.json", "w"))
            print(f"  {i:>4}  {i*30*env.batch:>10,} steps  capture {real*100:5.1f}% "
                  f"(stoch {noisy*100:4.1f}%, best {best*100:.1f}%)  at {dist:.0f} m  "
                  f"{(time.time()-t0)/60:.0f} min", flush=True)
            # The curriculum: launch further out once the policy is good at this range.
            if real > PROMOTE_AT and dist < MAX_M:
                dist += STEP_M; env.launch_offset_m = dist
                ppo.obs, ppo.state = env.reset(), None
                print(f"  {i:>4}  launch now {dist:.0f} m out", flush=True)

    ppo.save(OUT / f"{tag}.pt")
    ppo.net.eval()
    # Scored where it trained and further out, so the number is not just the curriculum's last rung.
    out = {}
    for m in (25.0, 60.0):
        env.launch_offset_m = m
        out[m] = report(env, f"seed {seed} deterministic, {m:.0f} m", ppo.net)
    return out, best


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "closeness"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    OUT.mkdir(exist_ok=True)
    print(f"variant '{variant}': {VARIANTS[variant]}   fidelity {FIDELITY}/{DETECTOR}", flush=True)
    probe = TrainEnv(scenario(0), batch=256, fidelity=FIDELITY, detector=DETECTOR, device="cuda")
    probe.launch_offset_m = 25.0
    floor25 = report(probe, "no drone help, 25 m", None)
    probe.launch_offset_m = 60.0
    floor60 = report(probe, "no drone help, 60 m", None)
    del probe

    results = {}
    for seed in (0, 1):
        print(f"\n{'=' * 70}\nSEED {seed}   ({variant}, {iters} iterations)\n{'=' * 70}", flush=True)
        results[seed] = run(variant, seed, iters)

    print(f"\n{'=' * 70}\nFINAL RESULT -- {variant}, full fidelity\n{'=' * 70}")
    for s, (out, best) in results.items():
        print(f"  seed {s}: 25 m {out[25.0]:5.1f}% (floor {floor25:.1f})   "
              f"60 m {out[60.0]:5.1f}% (floor {floor60:.1f})   best in training {best*100:.1f}%")
    spread = abs(results[0][0][25.0] - results[1][0][25.0])
    print(f"  spread at 25 m: {spread:.1f} points -- " + ("reproducible" if spread < 10 else "NOT reproducible"))
