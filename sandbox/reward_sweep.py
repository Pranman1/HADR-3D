"""Which reward balance actually gets a rescue on a real city block?

    python sandbox/reward_sweep.py 150        # iterations per variant

The weights in `reward/compose.py` were chosen while four of the reward terms were
broken -- searching paid forever, arriving paid every step, and nothing paid for the
second half of a rescue.  They were tuned to compensate for that, so they are the
wrong weights for the fixed reward and have never been revisited.

The specific suspicion: `proximity` pays 20 a step for being near the evacuee and is
about 85% of the entire episode budget.  It is a *level*, not a progress term, so it
is maximised by parking next to somebody and staying there -- and since capture is
automatic inside 3 m that is nearly the same thing, but it leaves nothing pushing the
policy to then take them anywhere, and it swamps every search term while they are
still lost.

Each variant is scored the same way: full episodes, deterministic, against that
scenario's own do-nothing baseline.  Short runs, so this ranks variants rather than
producing a final number.
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

BASE = dict(DEFAULT)

# Full fidelity: the quadrotor tier and a detector that can miss.  Measured at 1.31x
# the cost of point-mass + oracle, which is cheap for no longer training on a cheat.
FIDELITY, DETECTOR = "quadrotor", "geometric"

VARIANTS = {
    "as shipped": {},

    # Stop the dense proximity term drowning everything else.
    "quiet proximity": {"proximity": 5.0},

    # ...and put what it loses into the outcomes it was standing in for.
    "outcome heavy": {"proximity": 5.0, "capture": 300.0, "reached_safety": 300.0},

    # Pay for the search, not for the staring: finding somebody is worth a lot once,
    # having found them is worth little per step.
    "search first": {"proximity": 8.0, "first_sight": 60.0, "coverage": 20.0,
                     "search_coverage": 0.1, "evacuee_in_hlr_fov": 0.2,
                     "evacuee_in_llr_fov": 0.4},

    # Pay for being *big in the frame* rather than for being near: closeness only
    # counts while the camera actually has them, which is what the paper's 2.5 m
    # detection window was quietly doing all along.
    "closeness": {"apparent_size": 20.0, "proximity": 5.0},

    # The cleanest test of the hypothesis: no proximity at all.  Measured on this
    # scenario, a *hovering* drone earns 1,374 points of proximity per episode and a
    # flying one 836 -- it pays more for staying put than for moving.
    "closeness only": {"apparent_size": 20.0, "proximity": 0.0},

    # The paper's own weights and nothing else, as a floor.
    "paper only": {"capture": 200.0, "evacuee_in_hlr_fov": 1.0,
                   "evacuee_in_llr_fov": 1.0, "proximity": 20.0,
                   "goal_progress": 10.0, "role_coordination": 1.0,
                   "time_penalty": -0.02, "reached_safety": 0.0, "first_sight": 0.0,
                   "coverage": 0.0, "search_coverage": 0.0, "fire_proximity": 0.0,
                   "control_effort": 0.0, "evacuee_harmed": 0.0, "apparent_size": 0.0},
}


def scenario(seed=0):
    return autoplace(Scenario.preset("Downtown Berkeley", radius_m=100.0,
                                     cell_size_m=2.0, episode_seconds=120.0,
                                     wind_mps=0.0, seed=seed))


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    out, results = pathlib.Path("checkpoints"), {}
    out.mkdir(exist_ok=True)

    probe = TrainEnv(scenario(), batch=256, fidelity=FIDELITY, detector=DETECTOR, device="cuda")
    probe.launch_offset_m = 25.0
    floor = report(probe, "no drone help (baseline)", None)
    del probe
    print()

    for name, over in VARIANTS.items():
        t0 = time.time()
        torch.manual_seed(0)
        weights = {**BASE, **over}
        env = TrainEnv(scenario(), batch=1024, weights=weights, fidelity=FIDELITY,
                       detector=DETECTOR, device="cuda")
        env.launch_offset_m = 25.0
        ppo = PPO(env, PPOConfig(horizon=30, epochs=2, entropy=0.01, entropy_final=0.0),
                  encoder="conv", device="cuda")
        ppo.obs, ppo.state = env.reset(), None
        ppo.train(iters, log_every=max(1, iters // 3))
        ppo.save(out / f"sweep_{name.replace(' ', '_')}.pt")
        ppo.net.eval()
        got = report(env, f"{name} -> deterministic", ppo.net)
        results[name] = got - floor
        print(f"    {name}: {got - floor:+.1f} points over doing nothing "
              f"({(time.time() - t0) / 60:.0f} min)\n")
        del env, ppo

    print(f"\n{'=' * 62}\nSWEEP RESULT -- reward balance, ranked by gain over doing nothing\n{'=' * 62}")
    for name, gain in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20}{gain:+6.1f}")
    json.dump(results, open(out / "reward_sweep.json", "w"), indent=2)
