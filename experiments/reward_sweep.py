"""Which reward balance gets a rescue on a real city block?  Ranks variants by gain
over doing nothing; short runs, so it ranks rather than produces a final number.

    python experiments/reward_sweep.py 150        # iterations per variant
"""

import json
import sys

from hadrsim.train import train

PAPER = {"capture": 200.0, "evacuee_in_hlr_fov": 1.0, "evacuee_in_llr_fov": 1.0, "proximity": 20.0,
         "goal_progress": 10.0, "role_coordination": 1.0, "time_penalty": -0.02}
VARIANTS = {
    "as shipped": {},
    "quiet proximity": {"proximity": 5.0},
    "outcome heavy": {"proximity": 5.0, "capture": 300.0, "reached_safety": 300.0},
    "search first": {"proximity": 8.0, "first_sight": 60.0, "coverage": 20.0, "search_coverage": 0.1,
                     "evacuee_in_hlr_fov": 0.2, "evacuee_in_llr_fov": 0.4},
    "closeness": {"apparent_size": 20.0, "proximity": 5.0},
    "closeness only": {"apparent_size": 20.0, "proximity": 0.0},
    # the paper's weights and nothing of ours
    "paper only": {**{k: 0.0 for k in ("reached_safety", "first_sight", "coverage", "search_coverage",
                                        "fire_proximity", "control_effort", "evacuee_harmed", "apparent_size")}, **PAPER},
}

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    results = {}
    for name, weights in VARIANTS.items():
        r = train(iters=iters, weights=weights, tag="sweep_" + name.replace(" ", "_"))
        results[name] = r["best"] - r["floors"]["hover"]
    print("\nreward balance, ranked by gain over doing nothing")
    for name, gain in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18}{gain:+6.1f}")
    json.dump(results, open("checkpoints/reward_sweep.json", "w"), indent=2)
