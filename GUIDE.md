# GUIDE — from the paper to every file

For someone picking the project up cold. Four parts: the paper and its 2D simulator,
what the 3D simulator adds, how one run is orchestrated end to end, and what every
file does. Then a reading order. Terms are defined the first time they appear.

---

## 1. The paper, and the 2D simulator it was built on

**The paper.** Mendoza et al., *Coordinated Autonomous Drones for Human-Centered Fire
Evacuation in Partially Observable Urban Environments*, IEEE GHTC 2025. One person
(the **evacuee**) is trying to leave a burning area. Two drones help:

- the **HLR** (high-level rescuer) flies high, sees a wide patch of ground, and
  cannot reach anyone;
- the **LLR** (low-level rescuer) flies low, sees a narrow patch, and is the only one
  that can **capture** the evacuee — get close enough to lead them out.

The research question is coordination under partial observability: neither drone
sees everything, so the high one has to find, and the low one has to close.

**Its simulator** (`hadr-games`, `pegrid`, `firegrid`): a **25 × 25 grid** of 1 m
cells. Drones teleport two cells per step. Each drone "sees" a fixed window of cells
around itself — 5×5 for the LLR, 9×9 for the HLR — perfectly, through anything.
Fire spreads to its four neighbours with probability 0.01 and never goes out. The
evacuee walks an A* path to an exit and cannot enter a fire cell. Panic is a
Trivedi–Rao mixture, `v = (1−γ)·v_sensible + γ·v_instinct`, with γ set by config.
Episodes are 100 steps. Rewards: capture, in-view bonuses for each drone, proximity,
goal progress, role coordination, a time penalty. The learner is **PPO** (a
reinforcement-learning algorithm) with an **LSTM** (a memory), one network flying
both drones. Published: **71 %** capture on the training setup (Env-I), **66 %** with
person and fire randomised (Env-IV).

Three things the grid cannot express, and which this project exists to test:
nothing is ever **occluded** (a roof never hides anyone), cameras never **tilt** (a
drone accelerating does not look elsewhere), and detection is **perfect** (no misses,
no false alarms, no staleness). Also: the fire never dies, and the random numbers are
not seeded, so no run reproduces.

---

## 2. What the 3D simulator adds, and what it keeps

**Kept on purpose, so results stay comparable:** the two roles and their asymmetry;
the Trivedi–Rao panic equation; the paper's reward terms with the paper's weights;
PPO + LSTM; the paper's four metrics; the paper's fire-is-a-wall rule for the
evacuee; a `kinematic` drone tier that *is* the paper's teleporting drone; and a
25 m paper-grid scenario (`experiments/paper_parity.py`) for a like-for-like control.

**Added:**

| aspect | 2D | 3D |
|---|---|---|
| world | abstract 25 m grid | 200 m blocks of ten real cities from OpenStreetMap, with building heights and seven fuel classes |
| drones | teleport | X500 quadrotor: 6-DOF rigid body, PX4-style cascade controller with PX4 gains; also point-mass and kinematic tiers |
| cameras | fixed cell windows, perfect | two real cones (90°/80 m down; 60°/45 m forward-and-20°-down) bolted to the airframe; ray-marched occlusion by buildings and smoke; a detector that misses |
| fire | 4-neighbour, immortal, unseeded | fuel, moisture, wind, burnout, ember spotting, seeded |
| smoke | none | buoyant plume; blocks the high camera, slows the person |
| evacuee | A*, fixed panic | panic driven by measured fire distance and smoke; social-force motion; fire a wall, ash avoided, fallback when cut off |
| policy input | hand-built vector | a frozen contract: self state, detections with confidence and age, teammate-as-detection, task, and a 4-channel 32×32 hazard raster read by a CNN |
| honesty | reward reads truth | reward may read only what the drones believe; raster channels carry provenance |
| scale | one episode at a time | ~1,000 cities in parallel on the GPU |
| tooling | scripts | control panel with 2D + 3D view, one evaluator, scripted baselines, golden regression test, wandb-ready training |

---

## 3. How one run is orchestrated

### A scenario becomes a world
`scenario.py` holds a `Scenario` (city or paper grid, radius, cell size, wind,
seed, episode length). `autoplace()` fills in what was left blank: connected start
and exit, **40 origin/destination routes** sized to the episode's walking budget
(paired by *walkable* distance), and **4 fire origins** on heavy fuel — the paper's
randomisation, generated from the map. `Scenario.terrain()` builds (and caches) the
`Terrain`: heights, canopies, walkable, flyable, fuel — from `world/osm.py` for a
city or `world/terrain.py` for the paper's yaml.

### The environment holds the batch
`env.py` → `Env(scenario, batch, fidelity, detector, weights)`. It owns
one `Fire`, one `Smoke`, one `Wind`, one `Human`, two `Drone`s, two `Camera`s, two
detectors and a reward `Composer`, each sized `(batch, …)`.

`reset()` → `_draw_episodes()` picks a route and a fire per environment →
`_launch_drones()` places both rescuers a fixed distance from the person on a random
bearing (`launch_offset_m`, the curriculum knob) → `observe()`.

### One policy decision (0.5 s of simulated time)
`step(actions)`, `actions` being `(batch, 2, 4)` in [−1, 1]:

1. finished environments have their actions zeroed (they are frozen and unpaid);
2. `_fly(actions)` runs **50 physics ticks at 100 Hz**: each drone's controller
   chases its velocity setpoint (`agents/drone.py` → `agents/autopilot.py`), the
   person integrates their intended velocity through the crowd layer, wind steps,
   fire steps on its 1 Hz schedule, and **capture is tested every tick** (the LLR
   covers 6 m per decision against a 3 m radius);
3. `_settle()`: smoke updates; the person re-plans every 2 s with the fire as a wall
   (`agents/human.py`); fire-proximity is computed once and shared; the searched map
   is swept; captures and the arrival/death edges are recorded; `observe()` builds
   the contract for each drone; `reward.Composer` scores it from the drones'
   *beliefs*; `done` is computed; the reward is masked for finished environments.

### Observation
`observe()` per drone: `self` (10: position/extent, velocity/max speed, yaw, altitude,
roll, pitch), `detections` (4 slots × 8: relative position/range, velocity, confidence,
age — slot 0 falls back to the team's last known position), `teammate` (5), `task`
(4: exit relative position, time left), `raster` (4 × 32 × 32 at 2 m: buildings,
fire, smoke, searched). Everything scaled by fixed physical constants.

### Policy and update
`policy/ppo.py`: `ActorCritic` — per drone a CNN (`policy/encoders.py`, 4→16→32→32)
plus the flat blocks → 128; both drones concatenated → LSTM(128) → actor (2 × 4
Gaussian means, learned spread, tanh) and critic. `PPO.collect()` runs 30 decisions,
`advantages()` computes GAE, `update()` replays the sequence through the LSTM
*carrying state* for a few epochs with clipping and an annealed entropy bonus.
`save()` writes weights plus a manifest (encoder, hidden, contract version).

### Scoring
`eval.py` → `evaluate(env, policy)`: runs full episodes and returns capture, seen,
reached, harmed, time-to-capture, time-to-first-sighting, time-in-view and the
reward per term. `policy/base.py` → `load(name_or_path)` gives a trained network or
a scripted baseline (`hover`, `lawnmower`, `last_sighting`).

### Watching
`runner.py` → `PolicyRunner` wraps an `Env` of batch 1 and a policy;
`gui/server.py` serves the panel and streams a frame every 50 ms of simulated time; `gui/index.html` draws the plan
view and `gui/scene3d.js` the 3D one. `viz/render.py` writes mp4s.

---

## 4. Every file

### `hadrsim/core` — the contract and the clocks
- **`spec.py`** — THE contract: `ObsSpec` shapes, `ACTION_FIELDS`, `RASTER_CHANNELS`
  with provenance, `SPEC_VERSION`, the three honesty rules. Change it deliberately.
- **`clock.py`** — one 100 Hz physics tick; derived control (2 Hz), human (10 Hz),
  fire (1 Hz) rates; `max_steps`.
- **`rotation.py`** — quaternions (w, x, y, z), body→world, Euler, yaw/tilt helpers.
- **`rng.py`** — `Rng`: seeded torch generator; `fork(tag)` gives an independent
  stream per subsystem (crc32 of the tag, so it is the same in every process).

### `hadrsim/world` — the city and the fire
- **`terrain.py`** — `Terrain` dataclass (solid_top, canopy, walkable, flyable, fuel,
  cell size); loader for the paper's yaml map; `derive_flyable()`.
- **`osm.py`** — Overpass query (mirrors, backoff, disk cache), geocoding, building
  heights from tags/levels, canopies, seven fuel classes, rasterising to a `Terrain`.
- **`raster.py`** — polygon fill and line stroke on a grid.
- **`fire.py`** — `Fire`: `state` codes (unburnt/burning/burnt), fuel and moisture,
  8-neighbour spread as grouped convolution, wind boost, burnout, ember spotting;
  `burning`/`burnt`/`intensity` are views of `state`.
- **`smoke.py`** — `Smoke`: column density derived from the fire and wind.
- **`wind.py`** — `Wind`: mean vector that wanders slowly plus gusts.
- **`nav.py`** — `distance_field` (relaxation, optional per-env blocking and cost),
  `flow` (steepest descent over 8 neighbours), `clearance`, `proximity` (dilated
  max-pools), `sample`.

### `hadrsim/agents` — who is in the world
- **`human.py`** — `Human`: `PanicParams`; `_replan()` (fire blocked, ash priced,
  12 m stand-off, fallback); `desired_velocity()` (the Trivedi–Rao mix, instinct
  wander, smoke slowdown, following a rescuer); `integrate()`; `expose_to()`;
  `_threat()`/`_update_gamma()`.
- **`crowd.py`** — social-force layer: slides along walls, separates bodies.
- **`drone.py`** — `Drone`: three tiers (`_step_kinematic`, `_step_point_mass`,
  `_step_quadrotor`), `confine()` geofence, attitude accessors.
- **`autopilot.py`** — PX4's cascade: velocity PID → acceleration → attitude +
  thrust → attitude P → body-rate P → torques → rotor mix; PX4 default gains.
- **`airframes.py`** — `Airframe` dataclass; X500 V2 and Crazyflie; `by_name()`.

### `hadrsim/sensing` — what can be seen
- **`fov.py`** — `Camera` (fov, range, pitch, kind), `look_direction()` through the
  attitude, `in_view()`, `ground_footprint()`; the two default cameras.
- **`occlusion.py`** — `trace()` ray-marching through heights and smoke;
  `visible()` with clarity.
- **`detector.py`** — `Detections` (position, velocity, confidence, age); `oracle`
  and `geometric` detectors; `make(name, camera)`.
- **`camera.py`** — `render()`: a 4×32×32 semantic image per drone (building, fire,
  smoke, objects) by marching every pixel's ray; the person and seeded street
  clutter share the objects channel, so nothing in the picture says "person".
- **`pixel.py`** — `PixelDetector`: a small CNN over that image, trained offline by
  `python -m hadrsim.sensing.pixel` on oracle labels; weights ship in
  `configs/pixel_detector.pt`. `--detector pixel` anywhere. On hardware this file is
  what a real camera and detector replace.

### `hadrsim/reward` — what the drones are paid for
- **`terms.py`** — `View` (everything a term may read: beliefs, not truth) and the
  `@term` registry.
- **`paper.py`** — the terms with a counterpart in the paper: `capture` (l3),
  `evacuee_in_hlr_fov` (l0), `evacuee_in_llr_fov` (l1), `proximity` (l2),
  `role_coordination` (l7), `time_penalty` (l6), `fire_proximity`.
- **`compose.py`** — `load_weights(name_or_path)` reads a yaml table; a `terms:`
  line in it names a .py whose `@term` functions get registered. `Composer` sums
  terms and keeps each term's contribution (`Reward.parts`).

### `hadrsim/configs` — numbers that are not code
- **`reward/paper.yaml`** — the default: the paper's terms only. Your own balance is
  a yaml path on `hadrsim-train --reward` or `Env(weights=...)`.

### `hadrsim/policy` — learning and the policy seam
- **`encoders.py`** — `FlatEncoder`, `ConvEncoder`, `make()`.
- **`ppo.py`** — `PPOConfig`, `ActorCritic` (two encoders → LSTM → actor/critic,
  manifest), `PPO` (`collect`, `advantages`, `update`, `train`, `save`).
- **`base.py`** — `Policy` protocol; `Hover`, `Lawnmower`, `LastSighting`;
  `Learned.load()` (manifest-checked); `load(name_or_path)`.

### top level
- **`env.py`** — `Env`, the gym: `ALT_BANDS` per role; `reset` → `_draw_episodes`,
  `_launch_drones`; `_detect`, `_belief` (last-known-position memory), `observe`,
  `_sweep` (searched map with decay), `_raster`; `step` → `_fly`, `_settle`; `_view`.
  No learning in it.
- **`train.py`** — `train()` and `hadrsim-train`: PPO blocks with evaluation every
  25 iterations, keep-best, scripted floors printed beside the result, optional
  launch-distance curriculum, `--wandb`, every hyperparameter a flag.
- **`scenario.py`** — `Scenario`, `PRESETS` (ten cities pinned to coordinates),
  `autoplace()`, `_variety()` (routes and fire origins), yaml/dict I/O.
- **`eval.py`** — `Metrics`, `evaluate()`; `hadrsim-eval` scores a policy on one
  city or every cached one against its own floor.
- **`runner.py`** — `Snapshot`; `Runner` (a crowd, no drones — the panel's
  people-only mode); `PolicyRunner` (one person, two drones, any policy).
- **`viz/render.py`** — `Recorder`: frames to mp4/gif; `hadrsim-watch`.
- **`gui/server.py`** — FastAPI + WebSocket: `/`, `/presets` (cached cities only),
  `/policies` (baselines + checkpoints), `/live` (build / play / step / reset /
  speed; real-time pacing). **`gui/encode.py`** — map PNG, run-length heights,
  fire/smoke overlay, flames, wind: what goes down the wire.
- **`gui/index.html`** — the form, the 2D canvases, frame tweening, 2D/3D toggle.
- **`gui/scene3d.js`** — the 3D scene: buildings, X500 airframes at true attitude,
  camera cones along the real axes, flames, wind-leaning smoke, shadows.
- **`ros.py`** — `hadrsim-ros`: the gym at batch 1 as a ROS 2 node. Publishes each
  drone's pose, its detection, the evacuee and the fire grid; takes `cmd_vel` per
  drone in the body frame, scaled by the airframe limits. Optional, never in training.

### `experiments/` — our research on top of the tool (not library code)
- **`city_reward.py` + `city_reward.yaml`** — the eight terms a city block needed
  (`reached_safety`, `goal_progress`, `search_coverage`, `first_sight`, `coverage`,
  `apparent_size`, `evacuee_harmed`, `control_effort`) and the balance every
  checkpoint so far was trained with. Bug history: `OVERNIGHT_FINDINGS.md` §1, §6.
- **`paper_parity.py`** — the control: the paper's setup at the paper's scale
  (`--pairs 1|40`).
- **`reward_sweep.py`** — the seven reward balances, through `train()`.
- **`sweep.yaml`** — a wandb sweep over `hadrsim-train`'s flags.

### `tests/` — 188 tests; `test_golden.py` is the bit-for-bit refactor guard. CI runs them on CPU.
### documents
`OVERNIGHT_FINDINGS.md` (what was found, section by section, with measurements),
`PENDING.md` (ordered next steps).

---

## 5. Reading order (two days)

1. `core/spec.py`, `core/clock.py`, `core/rng.py` — the contract.
2. `world/terrain.py`, `world/fire.py`, `world/nav.py`; skim `osm.py`.
3. `agents/human.py`, `agents/drone.py`, `agents/autopilot.py`.
4. `sensing/fov.py`, `sensing/occlusion.py`, `sensing/detector.py`.
5. `reward/terms.py`, `reward/paper.py`, `configs/reward/`, then `experiments/city_reward.*`.
6. `env.py` — one decision, start to finish.
7. `policy/encoders.py`, `policy/ppo.py`, `policy/base.py`.
8. `eval.py`, `train.py`, `scenario.py`, `runner.py`; skim the gui.
9. `OVERNIGHT_FINDINGS.md` §1–12, then `PENDING.md`.

You are done when you can draw one decision on a whiteboard and answer two
questions: *why does a thirty-line rule match the learned policy?* (at this map
scale even a hovering drone sees the person; the task is closing) and *why can't you
quote 45 %?* (one seed; peaks range 19–45 %).
