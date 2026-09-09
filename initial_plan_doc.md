# hadr-sim3d — initial plan

Design doc. No code written yet. Last updated 2026-09-08.

Plan for a new 3D simulator replacing the 2D grid simulator behind *Coordinated
Autonomous Drones for Human-Centered Fire Evacuation in Partially Observable Urban
Environments* (Mendoza et al., IEEE GHTC 2025).

A **rebuild from scratch**, not a port — but it stays comparable to the paper, and
section 11 says exactly how we check that.

---

## 0. Vocabulary

| Term | Meaning |
|---|---|
| **policy** | the trained neural network. Looks at what a drone can see, decides where it moves next |
| **actor** | the part of the policy that actually flies. Ships with the drone |
| **critic** | a scoring helper used only during training, to teach the actor. Thrown away afterwards |
| **LOC** | lines of code. Used here as a *budget* — a cap per file, not an estimate |
| **module** | one folder or file with exactly one job |
| **batched** | running thousands of copies of the simulation at once on the GPU |
| **oracle** | the simulator handing a drone perfect information instead of making it look |
| **CNN** | the kind of neural network that reads images |
| **ROS 2** | the standard messaging system real robots use to talk to each other |
| **PPO** | the reinforcement-learning algorithm that trains the policy. Same one the paper used |
| **PX4** | the open-source autopilot software that runs on real research drones |
| **SITL** | software-in-the-loop. Running real PX4 firmware against a simulator instead of a real drone |
| **EKF2** | PX4's state estimator. Works out where the drone actually is from noisy sensors |
| **uXRCE-DDS** | the link PX4 uses to talk to ROS 2. The real one, on real hardware |
| **`.ulg`** | the log file PX4 writes on every real flight |
| **setpoint** | a command telling the autopilot what to achieve, e.g. "fly at 3 m/s forward" |
| **LLR / HLR** | low-level rescuer (flies low, can actually reach the person) / high-level rescuer (flies high, sees wide, cannot intercept) |

---

## 1. What we are building

Two drones and one panicking person in a burning 3D city.

The person runs, sometimes badly — under stress they follow instinct instead of the
sensible route. The fire spreads. The drones have to find the person and lead them out.
We train a neural network to fly both drones.

Gaby's existing version does this on a flat 25×25 checkerboard, in whole-number steps,
where a drone with the person in its viewing window is told the person's **exact**
position with zero error. Ours is continuous 3D space, continuous time, a real drone
model, and drones that have to actually see.

---

## 2. What the existing simulator really does

Worth being precise, because it is better than a summary suggests, and the real gap is
narrower and more specific than "it has no vision."

### It does have fields of view, and they are not radii

| | What it is |
|---|---|
| **LLR** | a **5×5 square placed in front of the drone**, rotated by which way it faces. Runs **real shadow-casting occlusion** — walls block line of sight (`pegrid_env.py:1086`, `grid.process_vis`) |
| **HLR** | a **9×9 square centred on the drone**. Not directional. Occlusion is computed and then **discarded** — `vis_mask = np.ones(...)` (`pegrid_env.py:1129`). It gets a random per-cell dropout (`hlp_fov_alpha`) instead |
| **Human** | a 7×7 square. Its occlusion mask is also discarded |

The asymmetry the whole paper rests on is genuinely implemented: `world_object.py:143`
gives one building class `see_inside_llp() → True` and `see_inside_hlp() → False`. That
is "the low drone sees under things the high drone cannot," written down in code.

### The actual gap: detection is binary and noiseless

```python
if OBJECT_TO_IDX["evader"] in llp_image or OBJECT_TO_IDX["evader"] in hlp_image:
    detected_evader_pd = (*self.evader_pos, self.evader_dir)   # exact truth
else:
    detected_evader_pd = (-1, -1, -1)                          # nothing at all
```

Inside the window, exact integer position and heading, zero error. Outside, nothing. No
falloff with distance, no partial confidence, no missed detections, no false alarms, no
smoke. A real camera at 60 m does not behave like this, and a policy trained against it
has never had to handle uncertainty.

### Four other things worth not reproducing

- **The person sees fire through buildings.** `perl/agent/evader/evader.py:80-92` reads
  the global ground-truth grid in a square window with no occlusion check at all.
- **The fire never goes out.** `firegrid_env.py:125-157` has no fuel and no burnout, so
  ignited cells burn forever and the fire eventually fills the map. They kept lowering
  the spread probability (0.05 → 0.01) to compensate. It also spreads *through*
  buildings — the validity check tests map bounds only.
- **The fire is not reproducible.** Spread draws from the unseeded global numpy RNG.
- **Detection and reward are separately wired to the truth.** The reward functions read
  ground-truth visibility on their own, so making the drone's *vision* worse would leave
  the policy quietly getting the answer through the reward channel. This is the single
  most important thing to design out, and section 6 does.

---

## 3. The one idea the whole design rests on

The current simulator collapses four separate things into one grid lookup: **the world**,
**the person**, **perception**, and **the runtime**.

Because "where is the person," "can the drone see the person," and "does the drone get
rewarded for seeing the person" are the same lines of code, you cannot change one without
breaking the others. That is why it cannot be extended toward realistic vision, or ROS 2,
or a real drone model, or a nicer renderer.

Split those four apart and each becomes a small local change.

```
                ┌──────────────── SCENARIO (a text file) ─────────┐
                │ which map · where the fire starts · which       │
                │ drones · how they are scored · when it ends     │
                └────────────────────────┬───────────────────────┘
                                         ▼
   WORLD  ────►  AGENTS  ────►  SENSING  ────►  COMMS  ────►  POLICY
   the city      the drones     what can        the drones     the neural
   the fire      the person     actually be     talking to     network
   the smoke                    seen from       each other
   the wind                     where

                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                  TRAINING ENV                    RUNNING ENV
                  fast, ugly, 4000 at once        one copy, real time,
                  no graphics, no ROS             on screen, over ROS 2
```

---

## 4. File structure

Each folder has one job. That is the whole trick: any one can be replaced without
touching the others.

```
hadrsim/
  core/       the shared rules — what a drone sees, what a drone can do, the clock
  world/      the city, the fire, the smoke, the wind
  agents/     the drones, their autopilot, and the panicking person
  sensing/    what a drone can actually see from where it is
  comms/      the drones talking to each other, including dropping messages
  reward/     the scoring — what counts as doing well
  scenarios/  text files that pick a map, a fire, a scoring rule
  envs/       the two environments (section 7)
  viz/        drawing it on screen
  policy/     the neural network itself
```

**Reusability constraint:** `world/`, `sensing/`, and `comms/` may not import anything
fire- or evacuation-specific. If that holds, this simulator serves an unrelated drone
project by writing a new `scenarios/` file. Cheap to hold from day one, expensive to
retrofit.

---

## 5. The LOC budget

Every file gets a cap on how many lines of code it may contain. About **3,000 lines
total** for the whole simulator.

A rule, not a prediction. A file blowing past its cap is the signal that it has quietly
absorbed a job belonging in another folder — which is how the current codebase got the
way it is. Cheapest way to enforce "minimal and readable" without re-arguing it every
commit.

For scale: the existing `perl` package is several thousand lines and does less.

```
                    budget  what it does                              [built]
core/
  spec.py           80   what a drone sees, what a drone can do. The contract   [69]
  registry.py       40   maps a name in a scenario file to the class it means
  clock.py          30   simulation step, control rate, episode length — in SECONDS [29]
  rng.py            30   randomness, seeded and passed in, never global         [25]
world/
  terrain.py       100   ground height, buildings, and what will burn           [75]
  raster.py         50   shapes -> grid cells. Knows nothing about maps         [44]
  osm.py           230   real cities, from OpenStreetMap (section 5b)          [232]
  nav.py            80   the shortest way out, solved once per episode          [72]
  fire.py          150   fire spread                                           [112]
  smoke.py         100   smoke plume, derived from the fire                     [69]
  wind.py           50   wind, varying over time                                [52]
agents/
  drone.py         220   quadrotor equations of motion, three fidelity levels
  airframes.py      40   parameter tables for real drones (X500 V2, Crazyflie)
  autopilot.py     150   PX4-style cascade controller
  human.py         120   Trivedi & Rao panic — the decision layer              [120]
  crowd.py         100   social force — the motion layer                        [58]
sensing/
  fov.py            80   is this point inside the camera's cone
  occlusion.py     120   ray tracing — do buildings or smoke block the view
  detector.py      150   oracle | geometric | learned (section 9)
comms/
  channel.py       100   message delay, loss, bandwidth. ideal | lossy | ros2
reward/
  terms.py         150   one small function per scoring rule
  compose.py        40   add them up with weights from the scenario file
scenarios/
  builder.py       100   reads a scenario file and assembles everything above
  *.yaml             -   paper_parity, urban_dense, open_field, ...
envs/
  train.py         250   the fast environment
  exec_ros.py      200   the real-time ROS 2 environment
viz/
  rerun_sink.py    170   drawing. Used by BOTH environments                    [147]
  mesh.py          110   building outlines -> triangles and texture coords     [103]
  facade.py         90   building colour and the shared window atlas            [83]
  isaac_bridge.py  150   photoreal (step 5), real meshes not procedural
policy/
  encoders.py      120   how the network reads its input (plain | CNN)
  ppo.py           150   the training loop
  ────────────────────
  total          ~3,480        built so far: 1,290 code lines, 93 tests passing
```

**Five rules that keep it there**

1. One concept per file. If describing a file needs the word "and," split it.
2. No deep inheritance. Fidelity levels are separate small classes, not one class with
   `if self.tier` branches inside.
3. A module receives its own parameters, never the whole scenario config. This is what
   makes `world/` reusable by a project unrelated to fire.
4. Everything works on thousands of copies at once. A function that only handles one
   simulation is a hidden slow loop waiting to happen.
5. Randomness is always passed in as an argument. Never module-level, never global —
   this is the bug that makes the current fire irreproducible.

---

## 5b. Real cities  *(built)*

The paper's map is a 25 x 25 checkerboard.  `world/osm.py` builds the same `Terrain`
from an actual place -- real footprints, real heights, real streets.

    python scripts/view_world.py --osm "Downtown Berkeley, California" --web

**Why this is cheap and worth doing early.** What the simulation consumes is a height
field, a fuel field, and two passability masks.  A box with a real footprint and a
real height blocks a sight line exactly as a detailed mesh would, and the policy
never sees pixels.  So *visual* fidelity buys nothing until step 5; *geometric*
fidelity buys everything, and it costs a day.

**What it produces**, measured on four real places:

| Place | Built | Buildings | Height min/median/max | Street |
|---|---|---|---|---|
| UC Berkeley campus | 37% | 145 | 3 / 6 / 30 m | 21% |
| Downtown Berkeley | 44% | 130 | 3 / 6 / 56 m | 21% |
| SF Financial District | 49% | 176 | 2 / 25 / 212 m | 28% |
| Manhattan | 63% | 181 | 4 / 20 / 366 m | 23% |

Those height distributions are the check that it works: a low-rise campus, a mid-rise
downtown, and two skylines, each coming out as itself.

**Three things fall out of the real data**

- **`building=roof` is OSM's tag for a roof with no walls** -- a carport, a bandstand,
  a fuel-station canopy.  That is exactly the canopy the paper's map implies, so the
  low-drone-sees-under-it asymmetry arrives from real data rather than being
  stipulated.
- **Streets are the firebreaks.**  Rasterising the road network to zero fuel is what
  makes a city fire behave like a city fire instead of a fire in an open field.
- **Heights are mostly missing.**  About a third of buildings carry `building:levels`
  and almost none carry `height`, so most heights are inferred at 3 m per storey.
  This is the least trustworthy thing in the world layer and it is worth remembering
  when a result depends on altitude.

### How buildings are drawn

Colour comes from `building=<type>`, not from colour tags.  Surveying the cached
extracts finds **zero** `building:colour` tags and six `building:material`, against
30-odd distinct `building=*` values on nearly every outline -- so a type palette is
the thing that actually pays.  Explicit colour and material are honoured where
present, which is rarely.

Windows come from one shared atlas: a 128 x 128 grid of window bays, with each wall
mapping into a corner of it.  A building of 8 floors and 12 bays uses u in [0, 12/128],
v in [0, 8/128].  The whole city is therefore two mesh objects and one image, and
nothing depends on the renderer repeating texture coordinates outside [0, 1].

**None of this is visible to the simulation.**  Occlusion, fire and movement all read
the grid; the policy never sees pixels.  It exists because a city of identical grey
blobs is hard to read, and being able to see what you built is how you notice it is
wrong -- which is exactly how the canopy-height bug above was caught.

**Photoreal buildings are deferred to step 5**, where they earn their place: rendering
camera images to train the frozen detector, and the demo.  That means real meshes and
textures (USD assets via Isaac Sim), not a procedural facade, and it is the only point
at which visual fidelity converts into a number.

**It does not replace `paper_parity`.**  A real city is a different scenario file, not
a different simulator.  `paper_parity.yaml` keeps the 25 x 25 grid, because comparing
against Table I is the entire point of it.

**Verification.**  For the 105 downtown-Berkeley buildings that lie entirely inside the
box, rasterised area matches polygon area to **1.0000**, and that ratio is stable from
2 m cells down to 0.25 m -- so the grid is measuring the geometry, not the mesh.
Buildings crossing the boundary are clipped, which is correct and accounts for the
whole of the apparent shortfall when they are included.

Two real bugs this shook out, both now covered by tests:

- Downtown Berkeley's BART station is tagged `building:levels=0` because it is
  underground.  Extruded naively it becomes a zero-height solid.  It is skipped.
- Canopies inherited the generic two-storey default and sat at 6 m, which quietly
  destroyed the mechanism they exist for: at 6 m the low drone flies under every
  canopy in the city and is never forced to go and look under one.  They now default
  to 3 m, and the asymmetry fires.

---

## 6. The contract

One frozen list, in `core/spec.py`, of what a drone sees and what a drone can do.
Changing it is a deliberate, versioned act.

**The most important file in the project.** It is the only thing the training environment
and the running environment share. As long as both honour it, a network trained in the
fast one runs in the pretty one unchanged — and later, on real hardware.

### What a drone can do

```
forward speed   -1 .. 1  ─┐
sideways speed  -1 .. 1   ├─ scaled to m/s, in the drone's own frame
vertical speed  -1 .. 1  ─┘
turn rate       -1 .. 1     scaled to deg/s
```

This is a **velocity setpoint**, and it is chosen because it is exactly what PX4 offboard
mode already accepts on a real drone. See section 8.2 for why this level and not another.

The paper's waypoint-style actions ship as a thin adapter on top, so the paper's setup
stays reproducible without the simulator knowing about it.

### What a drone sees

```
itself       where it is, how fast, which way, height above ground, tilt
detections   for each thing spotted: where it is, how fast,
             HOW CONFIDENT, and HOW OLD the sighting is
teammate     where the other drone is — arrives through comms, may be stale
surroundings a small picture centred on itself:
             buildings / fire / smoke / where it has already looked
task         where the exit is, how much time is left
```

Three details, each fixing a specific defect from section 2:

- **Detections carry confidence and an age.** The oracle just sets confidence to 1 and
  age to 0. So when a realistic detector is swapped in, *nothing downstream changes*.
  That turns "make vision realistic" from a rewrite into a scenario edit.
- **The teammate's position arrives through `comms`, not from the simulator.** Perfect
  communication becomes a *setting* rather than an assumption welded into the code. The
  paper names perfect communication as one of its key limitations; here it is a dial.
- **The surroundings picture is centred on the drone and always the same size.** The
  current code hardcodes a 25×25 view in at least four places, including a network layer
  written as `nn.Linear(64*13*13, ...)`. Centred and fixed-size means map size never
  touches the network.

#### Where each channel of that picture comes from

Easy to miss, and it is the same oracle hole as human detection wearing a different
hat: the fire channel must not be handed to the drones for free.  Ground truth fire
is legitimate for deciding *where the person runs* -- that is just how the simulator
works -- but the moment a drone reads it, the policy is planning around a fire map it
never had to find.

| Channel | Where it comes from | Honest? |
|---|---|---|
| **buildings** | known in advance, from the map | ✅ yes. A real deployment has building footprints -- we get ours from OSM. Nothing to earn |
| **fire** | seen within line of sight; otherwise a coarse, stale shared map | must be **observed**, not given |
| **smoke** | same as fire | must be **observed** |
| **where it has looked** | the drone's own memory | ✅ yes, it is its own |

**Fire does not need its own detector**, and this is the one place where an oracle
stays defensible after being made honest.  A person at 60 m is 1.7 m of low-contrast
texture; a fire is a large, bright, hot thing with a smoke column above it.  So
modelling fire as *reliably detected inside the sensor footprint and absent outside
it* is a fair description of reality, where the same claim about a person is not.

Fire beyond line of sight arrives the same way a real incident commander gets it: a
shared map from other aircraft, ground crews or satellite, which is **coarse and out
of date**.  That is exactly what the `comms` layer already models, so it costs nothing
new -- and it turns "where is the fire" from free knowledge into something the team
has to go and look at, which is the same problem the paper poses about the person.

### The rule that closes the oracle loophole

> **The scoring may only read what the policy reads.**

In the current code the reward functions independently look up ground truth, so degrading
vision would leave the policy still getting the answer through the back door. Making the
scoring read the same observation the policy reads closes that by construction rather
than by remembering to.

### One network now, two later

**Now:** a single network flies both drones, matching the paper.

**Structured so a second drops in later:** each drone's observation is assembled
*separately*, and teammate information arrives through the comms layer. The single
network simply happens to receive both bundles. Splitting into two actors later means
changing where the bundles go — not changing the contract, the environment, or the
observation format.

The whole cost of keeping that option open is building observations per-drone instead of
as one merged blob. Free now, expensive to retrofit.

When we do split, the standard approach applies: **two actors, one cheating critic.**
Each actor sees only its own sensors and its own arriving messages, while the critic —
which exists only during training and is discarded afterwards — sees everything. Learning
stays nearly as easy as the single-network case, but what deploys is genuinely
decentralized. Gaby's `Team` and `SPPO` pieces are the existing hooks.

---

## 7. The two environments

### Training environment — for teaching the network

- No graphics, no ROS 2, nothing pretty.
- 4,000 copies at once on the GPU.
- Only job: millions of episodes per hour. Target 50,000 simulation steps per second.

### Running environment — for watching it work and testing it

- One copy, real time, on screen, looks good.
- Each drone is a separate ROS 2 node sending real messages, like real robots.
- **Flown by real PX4 firmware in SITL** (§8.2b), not by our approximation of it.
- Loads the exact same trained network file.

### The test that keeps them honest

Start both from the same seed, feed both the same sequence of actions, check the
trajectories match. Run automatically on every change.

Without it the two environments drift apart silently and every result from the running
environment becomes unfalsifiable. Twenty lines, not optional.

---

## 8. Drones — real model, real autopilot

### 8.1 We do not invent the drone

A quadrotor's equations of motion are **standard and universal** — the same in every
paper and every simulator, with one canonical formulation (Mellinger & Kumar, ICRA 2011).
Writing them is not "our own model," it is writing down *the* model.

What differs between one real drone and another is a table of numbers: mass, arm length,
inertia, thrust coefficient, drag coefficient, motor spin-up time, maximum RPM.

So: **standard equations plus a real airframe's published parameters.**

| Airframe | Verdict |
|---|---|
| **Holybro X500 V2** | ✅ **default.** The standard PX4 research quadrotor, ~2 kg, actually flown in labs, carries a real camera and compute — which matters, since our drones need cameras. If this ever flies, it flies on hardware the group can buy |
| Crazyflie 2.1 | best-characterised in the literature, but a 27 g palm-sized toy that cannot carry a camera. Wrong scale for a city. Kept as a second parameter table for cross-checking |

`agents/airframes.py` is just those tables. Adding a drone is adding ten numbers.

### 8.2 We do not invent the autopilot either

**PX4** is the real open-source autopilot flying on actual research drones. Its control
stack and default gains are public. We implement the same cascade:

```
position setpoint  →  P controller
velocity setpoint  →  PID            ← THE POLICY PLUGS IN HERE
attitude setpoint  →  quaternion controller
body rate setpoint →  PID
                   →  mixer  →  four motor commands
```

We implement that cascade with PX4's default gains **for training**, because training
needs thousands of drones stepping in parallel and PX4 cannot do that.

**For evaluation we do not imitate PX4 — we run it.** See 8.2b. The Lee/Mellinger
geometric controller ships as a third option: cleaner, ~80 lines, more citable.

### 8.2b Running real PX4 in the running environment

**SITL** ("software in the loop") is PX4's own firmware, unmodified, running as a
program on a PC instead of on a flight controller board. It is how PX4 itself is
tested, and every serious drone simulator connects to it.

The usual setup pairs PX4 SITL with Gazebo, which supplies the physics. That is not
what we want — our world *is* the point, and Gazebo has no fire in it. PX4 exposes a
documented **simulator interface** for exactly this case: it hands out actuator
commands and expects sensor readings back, in lockstep. So:

```
   POLICY  ──velocity setpoint──►  PX4 firmware (real, unmodified)
                                          │  motor commands
                                          ▼
                                    OUR simulator
                                    quadrotor physics · city · fire · smoke
                                          │  simulated IMU, GPS, barometer
                                          └──────────────► back to PX4
```

**PX4 does the flying. We do the world.**

**Why this beats matching PX4's numbers**

- It *is* the autopilot, so there is nothing left to validate about the controller.
- We inherit PX4's **EKF2 state estimator**, which closes the one sim-to-real hole
  section 14 admits to. The drone stops being handed its own exact position and starts
  estimating it from noisy simulated sensors, drift and all, as a real drone does.
- We inherit its failsafes, arming logic and mode transitions — the unglamorous
  behaviour that decides whether a policy survives contact with hardware.
- The ROS 2 link is already the real one: PX4 talks to ROS 2 over uXRCE-DDS, which is
  what a real X500 uses.

**Why it cannot be the training environment**

SITL runs one aircraft at roughly real time. Training needs thousands stepping far
faster than real time. So the split is exactly the one we already have:

| | Drone flown by | Speed |
|---|---|---|
| **Training environment** | our own cascade, batched on GPU | thousands, far faster than real time |
| **Running environment** | **real PX4 SITL** | one, real time |

Not a compromise — the two-runtime design paying off. It also makes the parity test in
section 7 much stronger: it now compares our fast approximation against real autopilot
firmware, so a divergence is a finding rather than a broken test.

### 8.2c How we know the physics is right

Five rungs, honestly labelled. Only the last two touch reality.

| Rung | What it checks | Cost |
|---|---|---|
| 1. Internal sanity | hover thrust equals weight, no drift on zero command, energy behaves | free |
| 2. Published spec | our X500 hovers at the right throttle, hits the right top speed and climb rate | cheap |
| 3. Against PX4 SITL | our fast model's step response matches real firmware | a day, automatable |
| 4. **Against real flight logs** | replay a real drone's commands through our model, compare | cheap *if a log exists* |
| 5. Actual flight | reality | far future |

**Rung 4 is worth chasing early.** PX4 writes a `.ulg` log on every flight. If the group
has any log from an X500 or similar flying velocity setpoints, replaying it through our
model is an afternoon's work — and it is the only cheap evidence that any of this
resembles a real aircraft. **Ask Gaby whether such logs exist.**

### 8.3 What the policy outputs, and why

| Level | Policy outputs | Who is flying | Verdict |
|---|---|---|---|
| 1 | a waypoint | autopilot does everything | the paper's choice. Too coarse for continuous time |
| **2** | **velocity + turn rate** | **autopilot flies, policy decides where** | ✅ **this one** |
| 3 | acceleration / tilt angle | policy starting to fly | unnecessary here |
| 4 | body rotation rates + thrust | policy is basically flying | for aggressive-maneuvering research |
| 5 | four motor speeds | policy is the autopilot | sim-to-real disaster |

**Level 2, for a concrete reason:** PX4 offboard mode accepts a velocity setpoint as a
native message. What our policy emits in simulation is byte-for-byte what you would send
a real X500. No translation layer, ever.

Levels 4–5 are what you would pick if the research question were "can a network learn to
fly aggressively." Ours is "can two drones find a person in smoke." Making the policy
also learn to fly would burn most of the training budget on an already-solved problem.

**Known consequence:** at level 2 the autopilot keeps the drone stable whatever the policy
asks, so the policy cannot learn recklessness. That is a feature — but it means we make no
claims about agile flight, and should not.

### 8.4 Fidelity levels

The stack is identical at every level. Only the middle two boxes change.

```
  policy (2 Hz)   "forward 3 m/s, no drift, hold height, turn left 20°/s"
     │
     ▼
  AUTOPILOT       works out what the drone must physically do to obey
     │
     ▼
  DRONE MODEL     the physics, at 100 Hz
     │
     ▼
  new position, velocity, attitude in the 3D world
```

| Level | Autopilot | Physics | Captures |
|---|---|---|---|
| `kinematic` | none | position moves at the commanded speed | nothing physical. Exists only to match the paper |
| `point_mass` | one line: accelerate toward the commanded speed | mass, drag, acceleration and turn-rate limits | **inertia** — cannot stop or turn instantly |
| `quadrotor` | full PX4 cascade | 6-DOF rigid body, four rotors, thrust and torque, motor lag, real attitude | everything a real drone does |

Written to run thousands in parallel on the GPU — **not** on a physics engine like PhysX.
Our drones never touch anything, and collision solving is precisely the part of a physics
engine we do not need, and the part that would cost GPU parallelism and add a heavy
dependency. Roughly 220 lines runs 4,000 drones at once on the 5080.

### 8.5 The tilt problem, and why it matters here specifically

A real quadrotor **has to tilt to move** — lean forward to accelerate forward. A point
mass does not tilt; it accelerates while staying perfectly level.

Normally a minor detail. **Not for us**, because this entire project is about what the
drones can see, and the camera is bolted to the drone. If the drone tilts, the camera
points somewhere else. A `point_mass` drone would have a permanently level camera, which
no real drone has.

**Fix, ~5 lines:** at `point_mass`, derive a tilt angle from the commanded acceleration —
a real quadrotor's tilt is essentially determined by its acceleration anyway — and aim the
camera with it.

### 8.6 The training/evaluation split

- **Train at `point_mass`** with the derived camera tilt.
- **Evaluate at `quadrotor`** — take the trained network and run it on full physics with
  no retraining.
- **Report the gap.**

If the gap is small, `point_mass` is validated and we have saved 3× on every training run.
If it is large, retrain at `quadrotor`. Either way the gap number is itself worth
reporting.

---

## 9. The person

Keep Trivedi & Rao as the **decision** layer, so results stay comparable to the paper.
Add a proper **motion** layer underneath, and make panic actually respond to the fire.

### The paper's model, kept

```
v = (1−γ)·v_sensible  +  γ·v_instinct
```

γ is the panic level. Ported verbatim. Reimplementing it would make every result
incomparable to the published paper.

### Five accuracy upgrades, none of which break comparability

**a) γ responds to what the person actually experiences.** Currently fixed or on a
schedule. Trivedi & Rao themselves define panic as driven by stressors — the paper
simplified it. Drive γ from distance to the nearest fire, smoke density at their
position, whether an exit is visible, and time elapsed. Same equation, live inputs.

**b) Social force underneath.** The standard model in crowd dynamics (Helbing & Molnár,
1995), whose follow-up — *Simulating dynamical features of escape panic*, Nature 2000 — is
specifically about fire evacuation. Gives a pull toward the goal, push-away from walls,
and (with more than one person) person-to-person forces. Trivedi & Rao decides *where they
want to go*; social force decides *how the body actually moves*. They layer cleanly.

**c) Smoke slows them down.** Well documented in the evacuation literature (Jin's
visibility studies): walking speed falls sharply as smoke thickens. We have a real smoke
field, so this is nearly free — and it is a mechanism the paper's grid could not represent
at all.

**d) Panic raises speed but degrades judgement.** The classic stress finding.
`panicked.yaml` already carries `v_max: 1.5, v_min: 0.5` sitting unused for exactly this.

**e) Following the drone is imperfect.** The paper assumes that once reached, the person
follows an ideal path to safety. More honest: they follow with a lag, imprecisely, and can
lose the drone in smoke.

All five are small. Together they answer the paper's own limitation — *"an abstracted
representation of human behavior"* — with something defensible.

Position and speed are continuous. The person stays on the ground plane; only the drones
and the smoke are 3D.

---

## 10. The CNN

**There is one CNN worth the name: the detector.** An earlier draft of this document
counted two, which made a hyperparameter look like a research component. Correcting
that.

### What the detector does

One job: **is there a person, where, and how sure are we.** Nothing else.

Range comes from two cues, both free once you have a box around the person:

- **Box height.** Known human height ÷ box height in pixels × focal length. Always
  available. Degrades badly under partial occlusion: if the legs are hidden the box is
  short, so the person is reported further away than they are.
- **Where the feet meet the ground.** With the camera's pose known, the ray through the
  feet pixel meets the ground plane at exactly one point. **More accurate than box
  height** whenever the feet are visible.

**Their disagreement is the confidence signal, and it costs nothing.** Agreeing means
a clean sighting; disagreeing means the person is probably half-hidden, so report low
confidence and let the policy decide whether to go and look properly. That is exactly
the `confidence` field the contract already carries.

This is also why the observation carries **pitch and roll**: the ground-plane cue needs
to know where the camera is pointing, and a real quadrotor tilts to move.

*Open, one line:* should the detector also spot fire visually? Section 6 argues no —
fire inside line of sight is reliably detectable without learning anything, and adding
it is scope. Default is no.

### The other thing I previously called a CNN

The policy also reads a small map centred on the drone. Whether that map is encoded by
a few convolutions or flattened into an MLP is an **encoder choice — a hyperparameter,
not a perception system.** It is not a milestone and it does not need its own name.
(Sanika's `fire-encoding` branch prototyped exactly this, never validated, with a 5×5
view hardcoded in three places.)

### What stays ruled out

| | | Verdict |
|---|---|---|
| **Detector, trained offline and frozen** | camera in → sightings out | **this is the plan** |
| Map encoder inside the policy | convolutions vs MLP | implementation detail |
| **Raw camera frames every step of every environment** | ~100× slower | **never** |

**Standing rule: raw camera frames never enter the training loop.** The detector is
trained separately, on rendered images, then frozen — so real vision costs nothing at
training time. This single decision is what makes the project finishable.

---

## 11. Order of work

Paper comparison is **last**.

| Step | What ships | Done when |
|---|---|---|
| **1. World alone** ✅ | city, fire, smoke, person. No drones | ~~fire respects fuel and buildings and burns out; the panic model reproduces the paper's Figure 2 shape; same seed gives the same run every time~~ **all three met** — see §11b |
| **2. Drones + oracle** | X500 model, PX4 autopilot, simulator still tells them where the person is, first trained network | it trains at all, and captures the person at a sane rate |
| **3. Break the cheat** | real vision — view cone, buildings and smoke blocking sight, missed detections, false alarms, imperfect comms. Retrain | degrades gracefully, not off a cliff; an ablation shows the policy is no longer leaning on the oracle |
| **4. Add the CNN** | option A: local picture + CNN | at least as good as the plain-network baseline, and the ablation shows the picture is actually used |
| **5. Real vision** | rendering + option B frozen detector | detector accuracy reported; policy on real detections within a stated margin of step 3 |
| **6. Running environment** | ROS 2, real time, looks good, parity test running automatically | same network file runs in both environments and trajectories match |
| **7. Compare to the paper** | section 12 | — |

**Step 3 is where the actual research is.** Everything before is scaffolding; everything
after is presentation.

**Caveat on putting the comparison last:** if we run all the way to step 7 and the numbers
are bad, we will not know which of six steps broke it. So step 2 carries a cheap sanity
check — does it capture the person at roughly a sane rate at all — while the careful, full
comparison stays at step 7. Costs almost nothing, avoids debugging blind.

Each step is independently demoable. That matters more than it sounds: it is the
difference between a project with something to show at any moment and one with something
to show only at the end.

---

## 11b. Step 1, done  *(built)*

The world runs on its own: a city, a fire that spreads and goes out, a plume that
leans downwind, and a person who panics and walks out.  No drones yet.

    python scripts/view_episode.py --wind 4 --bearing 30 --web

**The three acceptance criteria, all met**

- **Fire respects fuel and buildings, and burns out.**  Fuel and burnout are one
  line of state each and they fix the three things wrong with `firegrid`: its fires
  never go out, they spread through walls, and they are irreproducible.  Streets
  become firebreaks for free, because `terrain` already gives them zero fuel.
- **The panic model reproduces Figure 2.**  Held at fixed panic with no fire, so
  panic is the only variable:

  | panic | reached | mean time | vs calm |
  |---|---|---|---|
  | 0.0 | 100% | 27.5 s | 1.00x |
  | 0.2 | 100% | 25.1 s | 0.91x |
  | 0.4 | 88% | 25.7 s | 0.93x |
  | 0.6 | 70% | 37.1 s | 1.35x |
  | 0.8 | 64% | 51.9 s | 1.89x |
  | 1.0 | 61% | 59.9 s | **2.18x** |

  The paper says persistent panic "nearly doubles" evacuation time, and that is what
  comes out.  The dip at low panic is the inverted-U from the stress literature:
  speed rises before judgement fails.  This is now a test, so it cannot drift
  unnoticed.
- **A seed reproduces a run**, and the fire ticks at its own rate, so changing the
  control rate does not change how the city burns.

**Four bugs the tests caught, each now covered**

- **The wind blew the fire upwind.**  A sign error in which neighbour ignites which:
  the weight said "how readily fire travels along d" but the shift asked "is the
  neighbour at +d alight", which propagates along -d.  Completely invisible in still
  air, because the eight-neighbour set is symmetric.
- **The flow field pointed away from the exit**, and every cell beside a wall had no
  direction at all — treating a wall as *infinitely far* rather than as *no better
  than here* makes the gradient enormous and aims everyone at the nearest wall.
- **Panic saturated.**  Blurring the fire measures how *much* of it is nearby, not
  how *close* it is; on a 25 m map that pins panic at maximum wherever you stand.
  Proximity has to be a max over nearby fire, not a sum of it.
- **A panicked person deadlocked against walls**, because stopping at an obstacle
  zeroes the whole velocity and the desired direction still points into it next step.
  Dropping only the blocked component lets them slide along and round the corner,
  which is also what a person does.

**One honest gap, unchanged.**  The person still senses fire from a proximity field
rather than a real line-of-sight check, so they can notice a fire through a wall —
just not one across town.  The 2D simulator reads the *global* ground-truth grid with
no occlusion at all, so this is strictly better and still not right.  It becomes
right when `sensing/occlusion.py` exists.

---

## 12. How we compare against the paper

Not "do we beat it." Our simulator is harder, so raw numbers will not match and should
not.

Instead, a **deliberately dumbed-down setup**, `scenarios/paper_parity.yaml`, reproducing
the paper's *assumptions* rather than our better ones:

- the same 25 × 25 m map, from `rust25.yaml`, at 1 m cells
- the same 40 person start/end pairs
- the same 4 fire starting locations
- capture distance equivalent to the paper's capture rule
- `kinematic` drones, oracle detection, perfect comms, one network, static γ — i.e.
  deliberately cheat the way the paper cheats

Then run our network on it and report the paper's own four numbers:

| Env | Capture rate ↑ | Time to capture ↓ | Time to first sighting ↓ | Time in view ↑ |
|---|---|---|---|---|
| I (0,0) | 71.0 ± 4.54 | 13.20 ± 1.17 | 8.48 ± 0.97 | 30.77 ± 1.66 |
| II (5,0) | 61.0 ± 4.88 | 12.28 ± 0.59 | 7.46 ± 0.35 | 31.60 ± 1.70 |
| III (0,5) | 68.0 ± 4.66 | 10.91 ± 0.51 | 6.75 ± 0.30 | 33.77 ± 1.83 |
| IV (5,5) | 66.0 ± 4.74 | 12.83 ± 1.17 | 8.77 ± 1.14 | 32.79 ± 1.88 |

(Env-I is the training setup unchanged; II randomises the person's start and end; III
randomises the fire; IV randomises both.)

**The bar is the same ballpark on all four numbers, and the same ordering across the four
environments** — notably Env-III best on capture time, first sighting, and time in view.
Missing that means a modelling difference we do not yet understand, worth finding.

Then turn the realism back on and report the real numbers separately. That second table is
the actual contribution.

### A problem with the reference numbers

The 2D repo currently **does not reproduce its own published table**. Running
`test/table_1_5_5.py` on `hadr-games@main-clean-merge` gives 32% where the paper says 66%,
capture time 8.0 against 12.83, time-in-view 14% against 32.8%. Five environment configs
were tried; the best was 45%.

The cause looks like config drift, not a broken result: the trained network file and all
four table scripts were committed together on 2025-06-10 (`f9ee959`), and the twenty
commits after that changed the movement rate, the speed setup, and the reward structure.
The final commit also repointed the default config to a different environment entirely.

Checking out `f9ee959` and running there should recover the published numbers. Worth doing
once, so step 7 has a reference we actually trust.

---

## 13. Changes from the original paper

Every deliberate difference, why, and what it costs. This is the table to hand Gaby.

### Kept identical on purpose

| Thing | Why |
|---|---|
| Trivedi & Rao panic equation | reimplementing it makes every result incomparable |
| The `l7` role-coordination reward term | the explicit HLR↔LLR incentive the paper's title rests on |
| Asymmetric roles — HLR sees wide but cannot intercept, LLR sees narrow but can | the paper's central claim |
| PPO with an LSTM | same algorithm, so differences are about the environment, not the learner |
| The four evaluation metrics | so section 12 is a like-for-like comparison |
| The 25 m map, 40 start/end pairs, 4 fire origins | the parity scenario needs them exactly |

### Changed, and why

| # | Paper | Ours | Why | Cost |
|---|---|---|---|---|
| 1 | 2D grid, whole-number cells | continuous 3D space | occlusion, altitude, and camera pointing are the premise; a grid cannot represent them | everything gets harder |
| 2 | discrete steps, all clocks welded (1 step = 5 drone moves = 3 person moves = 1 fire tick, cap 100) | 100 Hz physics, policy at 2 Hz, 120 s episodes | continuous time forces real units | "one step" no longer means the same thing — §12 handles it |
| 3 | drones teleport between cells | X500 V2 quadrotor + PX4 autopilot | a real drone has inertia, tilts to move, and its camera tilts with it | 3× compute at full fidelity; §8.6 measures whether it matters |
| 4 | policy outputs a waypoint | policy outputs a velocity setpoint | it is the native PX4 offboard message, so sim-to-real needs no translation | policy can no longer fly aggressively — deliberate |
| 5 | detection binary and noiseless inside a window | range falloff, misses, false alarms, confidence, staleness | this is the actual research contribution | policy must handle uncertainty; harder to train |
| 6 | reward reads ground truth independently of the observation | reward may only read what the policy reads | otherwise degrading vision changes nothing, because the answer arrives via the reward | rewards must be rewritten, not ported |
| 7 | fire never burns out, spreads through buildings, unseeded RNG | fuel, moisture, burnout, wind direction, buildings block, seeded | the current fire eventually fills the map and is irreproducible | more parameters to set |
| 8 | no smoke | 3D smoke plume, blocks the high drone, slows the person | smoke occlusion is the paper's premise, currently only asserted | a new subsystem |
| 9 | person's panic level fixed or scheduled | γ driven by fire distance, smoke, exit visibility, time | answers the paper's own "abstracted human behavior" limitation | needs calibrating |
| 10 | person moves on grid cells | social force motion layer, continuous | bodies have momentum and avoid walls | one more model to tune |
| 11 | person sees fire through buildings | occlusion-checked | it is simply a bug | — |
| 12 | person follows the drone along an ideal A* path | follows with lag, imperfectly, can lose the drone in smoke | the ideal-path assumption inflates the improvement number | improvement numbers may drop |
| 13 | perfect instant communication between drones | a comms layer with delay, loss, and bandwidth | the paper names perfect comms as a key limitation | one more thing to tune |
| 14 | one network flying both drones | one network now, structured for two later | real drones are separate machines | none now |
| 15 | observation hardcoded to 25×25 in four places | drone-centred, fixed-size, map-size independent | map size should never touch the network | — |
| 16 | one runtime | training environment and running environment, plus a parity test | you cannot demo the fast one or train in the ROS one | the parity test |
| 17 | no CNN on the live path | a frozen detector: camera in, sightings out | the paper's CNN code raises `NotImplementedError` | §10 |
| 18 | the drones are handed a perfect fire map | fire is seen within line of sight; beyond it, a coarse and stale shared map | otherwise the policy plans around a fire it never had to find — the same oracle hole as the person, in a different hat | one more thing to observe |
| 19 | evaluation uses the same hand-written dynamics as training | evaluation runs **real PX4 firmware** in SITL, flying our world | there is then nothing left to validate about the controller, and EKF2 supplies real state estimation | SITL is real-time and single-aircraft |

### Explicitly not changed

| Thing | Why not |
|---|---|
| One person, two drones | interfaces allow more; scenarios ship one. Out of scope until after step 6 |
| The person stays 2D on the ground | 3D humans buy nothing here |
| Raw camera images in the training loop | ~100× slower. The single decision that keeps this finishable |
| Rotor-level control learned by the policy | the research question is search and coordination, not learning to fly |
| A physics engine (PhysX) for the drones | our drones never touch anything; collision solving is the one part we do not need and the part that would cost GPU parallelism |

---

## 14. Sim-to-real — seams, not work

Nothing here gets built now. Named places where the gap to real hardware will eventually
be closed, each costing nothing to leave open today.

- **Commands.** A velocity setpoint is already the PX4 offboard message. No translation
  layer, ever.
- **Airframe.** X500 V2 is a real drone the group could buy, and its parameters are what
  we simulate.
- **Autopilot.** PX4's cascade with PX4's gains, checkable against real PX4 in SITL.
- **Physics.** The `quadrotor` level exists from step 2 even though training uses
  `point_mass`. Going to hardware means training at that level with randomised mass, drag,
  and motor response.
- **Randomising the physics.** Every module takes its physical parameters as arguments, so
  randomising them is a sampler in the scenario builder, not a code change.
- **Vision.** The learned detector is already trained on rendered images. The real one is
  the same class with different weights.
- **Communication.** The ROS 2 channel is already the real thing. The lossy simulated one
  gets calibrated against measured link statistics.
- **Delay.** The clock separates simulation step from control rate, so real sense-to-act
  delay is a parameter both environments already honour.
- **Knowing where you are.** ~~The one real hole.~~ **Closed by the PX4 SITL decision**
  (§8.2b). Running real firmware means running its EKF2 state estimator, so in the
  running environment the drone estimates its own position from noisy simulated
  sensors, drift and all, instead of being handed the truth. The training environment
  still hands it over — that gap is now a measurable difference between the two
  runtimes rather than an unknown.

---

## 15. Decisions

### Settled

| | Decision |
|---|---|
| Timing | 100 Hz physics, policy decides at 2 Hz, 120 s episodes |
| Networks | One now, observations built per-drone so a second drops in later |
| Airframe | Holybro X500 V2, real published parameters. Crazyflie as a cross-check |
| Autopilot (training) | PX4 cascade with PX4 default gains, ours, batched on GPU |
| Autopilot (evaluation) | **real PX4 SITL**, flying our world over PX4's simulator interface |
| Fire in the observation | observed within line of sight; a coarse stale shared map beyond it. Never handed over |
| CNN | **one** — the detector. Person present, where, how sure. The policy's map encoder is a hyperparameter, not a milestone |
| Policy output | Velocity + turn rate — the native PX4 offboard setpoint |
| Fidelity | Train at `point_mass` with derived camera tilt, evaluate at `quadrotor`, report the gap |
| Person | Trivedi & Rao decision layer, social force motion layer, γ driven by fire and smoke |
| Paper comparison | Last step, with a cheap sanity check at step 2 |

### Still open

1. **Fire level.** Simple spread with fuel is enough through step 4. Rothermel — the
   standard wildfire spread model — is a credibility upgrade for a paper and cheap to add
   later.
2. **Does step 3 land here or in Gaby's existing 2D repo?** Breaking the oracle inside the
   2D repo would produce a publishable result much sooner, at the cost of doing the work
   twice.
3. **Tell Gaby about the timing change.** 100 Hz / 2 Hz / 120 s changes what "one step"
   means relative to the paper. Not blocking, but she should know before results are
   compared.
4. **Number of people.** Interfaces allow many; scenarios ship one. Confirm more than one
   is out of scope until after step 6.

---

## 16. Immediate next steps

**Done:** `core/spec.py`, `scenarios/paper_parity.yaml`, `world/terrain.py`,
`world/raster.py`, `world/osm.py`, `viz/rerun_sink.py`, `viz/mesh.py`, `viz/facade.py`.
679 code lines, 54 tests.

**Done:** the whole world layer — `core/{spec,clock,rng}`, `world/{terrain,raster,osm,
nav,fire,smoke,wind}`, `agents/{human,crowd}`, `viz/{rerun_sink,mesh,facade}`.
1,290 code lines, 93 tests. **Step 1 of section 11 is complete.**

**Next, in order:**

1. **`agents/airframes.py`** — the X500 V2's real numbers. Ten of them.
2. **`agents/drone.py`** — quadrotor equations of motion, three fidelity levels.
3. **`agents/autopilot.py`** — PX4's cascade with PX4's gains.
4. **`sensing/{fov,occlusion,detector}.py`** — starting with the oracle detector, and
   closing the fire-provenance gap above at the same time.
5. **`reward/`, `envs/train.py`, `policy/ppo.py`** — and step 2 is training.

**To ask Gaby, in one message:**

- Does the group have any **PX4 flight logs** (`.ulg`) from an X500 or similar? Rung 4
  of §8.2c is an afternoon's work if so, and it is the only cheap evidence any of this
  resembles a real aircraft.
- Confirm the timing change: 100 Hz physics, 2 Hz decisions, 120 s episodes. It changes
  what "one step" means relative to the paper.
- Is `main-clean-merge` the branch behind Table I, and is `f9ee959` where the published
  numbers live? (§12 says HEAD does not reproduce them.)
- The 2× vs 3× panic discrepancy, still open from the website work.

---

## Appendix: environment notes for this machine

- Ubuntu 22.04, RTX 5080 (16 GB), Python 3.10.
- `~/SASTRY_GROUP/.venv` has `pegrid`, `firegrid`, and `perl` installed editable.
- **torchrl must be pinned.** Version 0.13 renamed `SyncDataCollector` to `Collector`,
  breaking `perl.rl.ppo.ppo` on import. Use `torchrl==0.9.2`, `tensordict==0.9.1`.
- `matplotlib` is missing from `hadr-games/setup.py` but imported by the table scripts.
- Isaac Sim consumes a large share of the 16 GB before any environment spawns, and turning
  cameras on cuts the parallel-environment count hard — a further argument for keeping it
  out of the training loop.
