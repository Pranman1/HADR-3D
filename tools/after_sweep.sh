#!/bin/bash
# Waits for the reward sweep to exit, then starts the long run with the winning
# balance -- detached from any session, so the GPU never idles between the two.
cd "$(dirname "$0")/.."
source ~/SASTRY_GROUP/.venv/bin/activate
while pgrep -f "^python -u sandbox/reward_sweep.py" >/dev/null 2>&1; do sleep 60; done
echo "=== sweep finished $(date) -- starting final run: closeness, 600 iterations x 2 seeds ===" > logs/final_run.log
python -u sandbox/final_run.py closeness 600 >> logs/final_run.log 2>&1
