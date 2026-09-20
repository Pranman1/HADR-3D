# What we built, what we did, what came out — hadr-sim3d, 8–20 Sep 2026

Written for someone who has lost the thread. Starts from the paper. Terms are defined
where they first appear.

---

## 1. What we built

### Where we started: the paper and its 2D simulator

Mendoza et al. (GHTC 2025) trains two drones to find and lead one person out of a
burning area. The two drones are different on purpose:

- **HLR** — the *high* rescuer. Flies high, sees a wide patch of ground, cannot go
  and get anyone.
- **LLR** — the *low* rescuer. Flies low, sees a narrow patch, and is the only one
  that can "capture" the person — reach them and lead them out.

The paper's world (`hadr-games`, `pegrid`, `firegrid`) is a **25 × 25 grid** of
1-metre cells. Drones teleport two cells per step. Each drone "sees" a fixed window
of cells around it — 5×5 for the LLR, 9×9 for the HLR — perfectly, through anything.
Fire spreads to its four neighbours with probability 0.01 and never burns out. The
evacuee walks an A* path to an exit and cannot enter a fire cell. The policy is
**PPO with an LSTM** (a standard reinforcement-learning algorithm with memory), one
network flying both drones. Published capture rate: **71%** on the training setup,
**66%** with person and fire randomised.

Every result in that paper depends on assumptions the grid makes invisible: nothing
is ever occluded, cameras never tilt, detection is perfect, fire never dies. The point
of the 3D rebuild was to remove those assumptions one at a time and see what survives.

### What the 3D simulator is

`~/SASTRY_GROUP/hadr-sim3d`, ~6,300 lines, 204 tests. Written from scratch, not
ported. One rule throughout: everything is a batch of many worlds at once on the GPU
(about a thousand cities burning in parallel), because that is what makes training
affordable.

**`hadrsim/core` — the contract.** `spec.py` defines exactly what a drone observes
and what an action means; every other file agrees with it. Clocks: physics at
100 Hz, people at 10 Hz, policy decisions at 2 Hz, fire at 1 Hz. Seeded randomness
passed in explicitly, so the same seed gives the same run (the 2D sim's fire cannot
be reproduced).

**`hadrsim/world` — the city and the fire.**
- `osm.py`: pulls real OpenStreetMap data and turns it into building heights and
  seven levels of ground fuel (trees burn hotter than asphalt). Ten cities, pinned
  to coordinates rather than names after a geocoder put "Eixample, Barcelona"
  8 km outside the district. Berkeley is 54% built, 56 m tall; Manhattan 87%, 397 m.
- `fire.py`: spreads through fuel with moisture, wind, burnout and ember spotting.
- `smoke.py`: a buoyant plume derived from the fire; blocks the high drone, slows
  the person.
- `nav.py`: distance and flow fields — how the person finds the way out — and
  since Sunday they take a per-world blocked mask so the fire can be a wall.

**`hadrsim/agents` — who is in the world.**
- `human.py`: the evacuee. The paper's Trivedi–Rao panic equation is kept exactly,
  but panic is driven by how close the fire actually is rather than set by hand.
  Motion is a social-force model. Fire is a wall (the paper's rule, kept); burnt
  ground is priced 21× so it is avoided but never strands anyone.
- `drone.py`: three fidelity tiers behind one interface — `kinematic` (the paper's
  teleport), `point_mass`, and `quadrotor` (full six-degree-of-freedom dynamics).
- `autopilot.py`: PX4's cascade controller with PX4's own default gains, so the
  command the policy issues is the real aircraft's native message.
- `airframes.py`: the Holybro X500 V2 (0.25 m arms, 2 kg) and a Crazyflie.

**`hadrsim/sensing` — what can actually be seen.**
- `fov.py`: two cameras. HLR: 90° looking straight down, 80 m range. LLR: 60°
  looking forward and 20° down, 45 m range. Both are bolted to the airframe, so a
  drone that leans to accelerate points its camera somewhere else.
- `occlusion.py`: ray-marching — buildings block hard, smoke blocks by optical depth.
- `detector.py`: the seam for sim-to-real. `oracle` (perfect), `geometric`
  (misses, false alarms, confidence, staleness), and a slot for a learned CNN
  later. All return the same `Detections`, so the policy cannot tell them apart.

**`hadrsim/reward` — what the drones are paid for.** Each term is one named
function; the composer records every term's contribution separately. The paper's
terms are kept with its weights; ours are added on top. Rewards may only read what
the drones *believe* from their detections, never the true state.

**`hadrsim/envs/train.py` — training.** Thousands of episodes stepping together.
Forty origin/destination pairs and four fire origins per city, as in the paper.
Finished episodes freeze. Curriculum by launch distance.

**`hadrsim/policy` — the learner.** PPO + LSTM as in the paper, with a **CNN**
(three convolution layers, 4→16→32→32 channels) reading a 32×32 hazard map around
each drone: buildings, fire, smoke, and where it has already looked. 615 k
parameters; a quarter the size of the flat network and better.

**`hadrsim/gui` — the control panel.** `python -m hadrsim.gui` → `localhost:8000`.
Pick a city, drop a fire, choose a trained policy, press play. A 2D plan view and a
3D view (`scene3d.js`, three.js vendored so it works offline): X500 airframes at the
attitude the physics has them at, each camera's cone along its real axis, smoke
leaning with the wind, shadows. It flies policies by wrapping the *training*
environment, so what you watch is the world they trained in.

**`hadrsim/eval.py` — the one evaluation loop.** Every script that reports a
number calls `evaluate(env, policy)`; it returns capture, seen, reached, harmed,
time-to-capture, time-to-first-sighting, time-in-view and the per-term reward sums.
Four diverging copies used to live in `sandbox/`.

**`hadrsim/policy/base.py` — the policy seam.** A policy is anything with
`act(obs, state, deterministic)`. Three scripted baselines ship: `hover` (the floor),
`lawnmower` (the high rescuer sweeps in stripes), `last_sighting` (fly toward
wherever the team last saw the person). `Learned.load()` reads a checkpoint whose
manifest says what network it is; `load(name_or_path)` returns any of them, so the
panel, the evaluator and every script are indifferent to which.

**`sandbox/` — experiments.** `train.py` (one entry point, every hyperparameter a
flag, wandb-ready; `sweep.yaml` beside it), `paper_parity.py` (the control),
`reward_sweep.py`, `final_run.py`, `generalise.py` (score a policy across every
city), `watch.py` (any policy → video). `tests/test_golden.py` is the bit-for-bit
regression guard; `PENDING.md` is the ordered list of what is not done.

---

## 2. What we did

**Mon 8 – Wed 10 Sep — built it.** Layer by layer, each with tests: cities, fire,
smoke, wind, navigation, the person, the drones, cameras, occlusion, detectors,
rewards, the training environment, the panel. Twelve real bugs found and fixed along
the way (wind blowing fire *upwind*, the quadrotor flying backwards, hover sinking,
flow fields pointing away from the exit, and so on).

**Thu 11 Sep, overnight — found out why nothing learned.** Every city run had
ended at *seen 100%, captured 0%*. The method that found it: sum each reward term
over a whole episode and ask whether the numbers are proportionate. Six bugs:

1. Every world in the batch ran the *same* episode (an untrained network scored 100%).
2. Routes were longer than the person could walk in an episode (reached safety: 6%).
3. Leftover reset code wiped the "already looked" map every step, so hovering earned
   search reward forever and staring paid 20/step — watching was worth 20× a rescue.
4. Arriving and dying were paid every step after they happened (~8,600 per rescue
   against 200 for the capture).
5. The one term paying for the second half of a rescue fired once per episode.
6. A quarter of all training steps happened after the episode had ended.

Fixing them took the paper-scale control from 6.2% to 37.1%. The same night: the
old "65% parity" claim was withdrawn (measured on a setup where an *untrained*
network scores 91%), and untrained baselines were averaged over seeds because a
single draw ranges 3.7–32%. Then an automatic NVIDIA driver update killed CUDA
mid-run.

**Fri 12 Sep — the 3D view.** Real airframes, real camera cones (the LLR's had been
drawn straight down; it looks forward), attitude from the physics. The panel gained
live policies, a port fix, and a list of which cities are cached.

**Sun 14 Sep — the person, the reward, full fidelity.**
- The evacuee had been walking *through* fire (ours priced it at 61×, the paper
  makes it a wall). Kept the paper's rule exactly, added ash avoidance and a fallback;
  fixed a second bug where the first fire-aware route came two seconds late. Proof on
  a real block: 99 m around instead of 36 m through, zero steps in fire.
- Pranav's diagnosis of "detects but never approaches": the paper's LLR detection
  window is 2.5 m, so *detection was closeness there*; ours sees 45 m. New term
  `apparent_size` — how big the person is in the LLR's camera, zero unless in view
  (on hardware: the bounding-box height).
- Training switched to full fidelity — quadrotor dynamics + the detector that can
  miss — measured at only 1.31× the cost.
- Seven reward balances swept at 150 iterations each.

**Mon 15 Sep — long runs, and the cross-city test.** The winning balance trained
for 600 iterations (two seeds), then re-run with keep-best checkpoints (two seeds,
300 iterations, evaluated every 25). The sweep's best policy scored on all ten cities
at 256 episodes each. Run-to-run variance measured and written down.

---

## 3. What came out

### The control: does the learner work where a good score is known?

Paper-scale setup, 25 m grid, 40 pairs, 4 fire origins, deterministic capture:

| | capture |
|---|---|
| do nothing | 6.1% |
| untrained, mean of 8 initialisations | 7.3% ± 4.5 |
| trained, **before** the six fixes | 6.2% |
| trained, **after** the six fixes | **37.1%** |
| paper Env-IV (its own 2D sim) | 66.0% |

Deterministic beat stochastic in that run — the signature of a converged policy. A
rerun with the CNN encoder (20 Sep, one seed) did *not* converge: 7.0% deterministic,
22.9% stochastic. So the 37.1% is a flat-encoder, single-seed number that has not been
reproduced with the CNN; five seeds per encoder are needed before either is quoted.
The gap to 66% is real and not yet explained.

### The reward sweep (Berkeley, full fidelity, 150 iterations each, floor 9.4%)

| balance | capture | gain |
|---|---|---|
| **closeness** (`apparent_size` 20, `proximity` 5) | **45.3%** | +35.9 |
| closeness only (no proximity) | 21.7% | +12.3 |
| the paper's weights, none of ours | 19.2% | +9.9 |
| search first | 17.2% | +7.8 |
| as shipped (our old defaults) | 14.6% | +5.3 |
| quiet proximity | 14.3% | +4.9 |
| outcome heavy | 3.6% | −5.8 |

Three readings: the closeness term tripled everything else; removing proximity
entirely halves it (it pulls toward the last-known position once the person is
lost); our added terms as shipped were net harmful — the paper's own weights beat them.

### Longer training made it worse

| run | best | at iteration | end |
|---|---|---|---|
| sweep, seed 0 | 42.2% | 150 | — |
| 600 iterations, seed 0 / 1 | 27.5 / 32.4 (in-run) | — | 23.2% / 16.9% |
| keep-best 300, seed 0 | 30.9% | 25 | 18.5% |
| keep-best 300, seed 1 | 19.5% | 100 | 14.1% |

Not noise collapse (action spread unchanged), not saturation: the later policy
flies half as far and sits lower — it drifts toward loitering. Peaks across four
runs of one configuration ranged **19–45%**. The 45% was a fortunate run; the
honest report is the ranking, the spread and the curves.

### Cross-city — the result the 2D grid could never produce

The closeness policy, trained on Berkeley only, 256 episodes per city, each against
its own do-nothing floor:

| city | floor | policy | gain |
|---|---|---|---|
| Downtown Berkeley (trained) | 8.2% | 44.9% | +36.7 |
| SF Financial District | 6.2% | 45.3% | +39.1 |
| Tokyo Shinjuku | 10.5% | 49.6% | +39.1 |
| Venice San Marco | 10.2% | 47.7% | +37.5 |
| Paris Le Marais | 7.4% | 44.5% | +37.1 |
| UC Berkeley campus | 9.4% | 40.6% | +31.2 |
| Barcelona Eixample | 12.9% | 43.4% | +30.5 |
| Manhattan Midtown | 4.7% | 34.4% | +29.7 |
| Chicago Loop | 7.8% | 37.1% | +29.3 |
| Oakland Downtown | 8.6% | 37.9% | +29.3 |

**Held-out mean +33.6 points across nine unseen cities; worst +29.3.** But the
thirty-line `last_sighting` rule scores **+35.4** on the same nine cities. The table
shows the task rewards closing everywhere, not that the network learned to search;
the learned policy matches the rule, beating it on four cities of ten.

### Against a hand-written rule (added 20 Sep)

`last_sighting` — fly toward wherever the team last saw the person, thirty lines —
scores 45.7% at 25 m and 21.5% at 60 m on Berkeley. The best learned policy scores
48.4% and 19.5%, and closes more slowly. Finding the person is nearly free at this
map scale (a hovering drone still sees them 100% of the time); the task is closing,
and a rule that closes is about as good as what we have trained so far. The learned
policy has not yet clearly earned its place over that baseline.

### The person

Fire on the route in Berkeley: straight line 36 m, walked 99 m around it, zero steps
in fire, reached the exit. Over 96 episodes with no drone help: 77% reach the exit,
6% lost to fire, 1% genuinely stuck, 16% still walking at the 120 s deadline
(detouring around fire).

### Not done

ROS 2 runtime · PX4 SITL · a pixel-level learned detector · the comms layer ·
a five-seed variance run · the honest hazard map (fire/smoke channels are exact
within the camera footprint) · pushing the repo to a remote.
