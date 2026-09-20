#!/bin/bash
# When the 600-iteration run (seed 1) exits, start the corrected run: keep-best
# checkpoints, promotion on a full-episode evaluation, entropy floor.  300 iterations
# is enough -- the first run's peak was at or before iteration 150 and it declined
# after.  Anchored pgrep with the argument, so neither the harness's lingering shell
# wrappers nor the new run itself can match the wait.
cd "$(dirname "$0")/.."
source ~/SASTRY_GROUP/.venv/bin/activate
while pgrep -f "^python -u sandbox/final_run.py closeness 600" >/dev/null 2>&1; do sleep 60; done
echo "=== 600-iteration run finished $(date); starting keep-best run: closeness 300 x 2 seeds ===" > logs/final_run_v2.log
python -u sandbox/final_run.py closeness 300 >> logs/final_run_v2.log 2>&1
