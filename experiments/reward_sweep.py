"""Which reward balance gets a rescue on a real city block?  Ranks variants by gain
over doing nothing; short runs, so it ranks rather than produces a final number.

    python experiments/reward_sweep.py 150        # iterations per variant
"""

import json
import sys

from hadrsim.train import train

VARIANTS = {
    "as shipped": {},
    "quiet proximity": {"proximity": 5.0},
    "outcome heavy": {"proximity": 5.0, "capture": 300.0, "reached_safety": 300.0},
    "search first": {"proximity": 8.0, "first_sight": 60.0, "coverage": 20.0, "search_coverage": 0.1,
                     "evacuee_in_hlr_fov": 0.2, "evacuee_in_llr_fov": 0.4},
    "closeness": {"apparent_size": 20.0, "proximity": 5.0},
    "closeness only": {"apparent_size": 20.0, "proximity": 0.0},
    "paper only": "paper",                 # configs/reward/paper.yaml, nothing of ours
}

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    results = {}
    for name, weights in VARIANTS.items():
        table = weights if isinstance(weights, str) else "default"
        r = train(iters=iters, reward=table, weights=None if isinstance(weights, str) else weights,
                  tag="sweep_" + name.replace(" ", "_"))
        results[name] = r["best"] - r["floors"]["hover"]
    print("\nreward balance, ranked by gain over doing nothing")
    for name, gain in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18}{gain:+6.1f}")
    json.dump(results, open("checkpoints/reward_sweep.json", "w"), indent=2)
