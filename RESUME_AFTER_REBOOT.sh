#!/bin/bash
# The NVIDIA driver was upgraded by unattended-upgrades at 06:30 on 11 Sep while
# training was running (kernel module 580.173.02, new userspace 580.178.04).  New
# CUDA processes fail with error 804 until the module is reloaded.  A reboot is the
# reliable fix; `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo
# modprobe nvidia` also works if nothing is holding the GPU.
#
# Then run this.  It picks up exactly where the night stopped: the reward-weight
# sweep, which is the recommended next experiment.
set -e
cd "$(dirname "$0")"
source ~/SASTRY_GROUP/.venv/bin/activate

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("GPU still unavailable -- reboot first, then re-run this script.")
print(f"GPU back: {torch.cuda.get_device_name(0)}")
PY

mkdir -p logs checkpoints
echo "starting the reward-weight sweep (5 balances x 150 iterations, ~2 h)"
nohup python -u sandbox/reward_sweep.py 150 > logs/reward_sweep.log 2>&1 &
# (if you chain anything after this, wait on `pgrep -f "^python -u sandbox/reward_sweep.py"` --
#  the anchored form; an unanchored pgrep matches lingering shell wrappers forever)
echo "  pid $!   watch with: tail -f logs/reward_sweep.log"
