#!/bin/bash
# After the last_sighting cross-city run releases the GPU: the paper-scale control
# again, this time with the CNN encoder and the current reward (PENDING #4).
cd "$(dirname "$0")/.."
source ~/SASTRY_GROUP/.venv/bin/activate
while pgrep -f "^python sandbox/generalise.py last_sighting" >/dev/null 2>&1; do sleep 30; done
echo "=== control rerun, CNN + current reward, 40 pairs, 300 iterations: $(date) ===" > logs/parity_cnn.log
python -u sandbox/paper_parity.py 300 40 >> logs/parity_cnn.log 2>&1
