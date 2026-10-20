# hadrsim

A 3D simulator and gym for drone-assisted fire evacuation. It succeeds the 25 m grid
simulator of Mendoza et al., *Coordinated Autonomous Drones for Human-Centered Fire
Evacuation in Partially Observable Urban Environments* (IEEE GHTC 2025): one evacuee
leaving a burning city block, a high rescuer that searches and a low rescuer that
closes in and leads them out.

What the grid could not express and this can: real city blocks from OpenStreetMap
with building heights; a quadrotor with a PX4-style controller, so a camera tilts
when the drone accelerates; cameras that are occluded by roofs and smoke and a
detector that can miss; fire with fuel, wind and burnout; an evacuee whose panic
follows the fire. A thousand of these worlds step together on one GPU.

```
pip install -e .            # add [train] for wandb, [dev] for pytest
hadrsim-panel               # control panel at http://localhost:8000
hadrsim-train               # PPO + LSTM + CNN, Berkeley, 300 iterations
hadrsim-eval  checkpoints/x.pt            # every cached city against its own floor
hadrsim-watch last_sighting -o out.mp4    # any policy, any city, to video
```

## The gym

```python
from hadrsim.env import Env
from hadrsim.scenario import Scenario, autoplace

env = Env(autoplace(Scenario.preset("Downtown Berkeley")), batch=1024, device="cuda")
obs = env.reset()                       # dict of tensors, see core/spec.py
for _ in range(env.clock.max_steps):
    actions = ...                       # (batch, 2 drones, 4): body-frame velocities
    obs, reward, done, info = env.step(actions)
```

`Env` contains no learning. The shipped learner is `hadrsim.train` (PPO with an LSTM
and a CNN over a 32x32 hazard map); any loop on `reset()` / `step()` can replace it.
A policy is anything with `act(obs, state, deterministic)`; three scripted ones ship
in `policy/base.py` (`hover`, `lawnmower`, `last_sighting`) and are the floors every
learned number is read against.

## Layout

```
hadrsim/
  world/      terrain, OpenStreetMap, fire, smoke, wind, navigation
  agents/     evacuee (Trivedi-Rao panic, social force), crowd, quadrotor + autopilot
  sensing/    cameras, ray-marched occlusion, detectors (oracle, geometric)
  reward/     terms.py (View, registry) · paper.py (l0 l1 l2 l3 l7 ...) · ours.py
  configs/    reward/{default,paper}.yaml -- weights live here, not in code
  policy/     encoders, PPO, the policy seam and scripted baselines
  env.py      the gym       eval.py  the one evaluation loop    train.py  the learner
  runner.py   step a scenario for a viewer      gui/  the panel (FastAPI + three.js)
experiments/  paper_parity, reward_sweep, sweep.yaml (wandb)
tests/        183 tests; test_golden.py is the bit-for-bit regression guard
maps/         OpenStreetMap cache for the ten preset cities at 100 m radius
```

Fidelity is a flag: `--fidelity kinematic|point_mass|quadrotor`, `--detector
oracle|geometric`, `--reward default|paper|path.yaml`. Everything runs on CPU for
tests and small runs; training wants a GPU.

## Reading

`GUIDE.md` goes from the paper to every file with a reading order.
`OVERNIGHT_FINDINGS.md` is what was measured, section by section, including the
reward bugs and the result that a thirty-line rule nearly matches the learned policy.
`PENDING.md` is the ordered list of what is not done.

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" python -m pytest -q
```
