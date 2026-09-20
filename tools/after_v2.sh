#!/bin/bash
# When the keep-best run exits, re-score the sweep's closeness policy across every
# city at 256 episodes on the GPU -- the CPU table was 48 per city (+-7 points).
# Anchored pgrep with the argument, for the reason in after_final.sh.
cd "$(dirname "$0")/.."
source ~/SASTRY_GROUP/.venv/bin/activate
while pgrep -f "^python -u sandbox/final_run.py closeness 300" >/dev/null 2>&1; do sleep 60; done
echo "=== keep-best run finished $(date); cross-city at 256 episodes ===" > logs/generalise_256.log
python -u sandbox/generalise.py checkpoints/sweep_closeness.pt --batch 256 --launch 25 >> logs/generalise_256.log 2>&1
