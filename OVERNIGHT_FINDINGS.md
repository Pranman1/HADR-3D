# What happened overnight

Short version: **the policy was never broken. The reward was.** Four separate bugs
were paying the drones to do things other than rescue anybody. Fixing them took the
paper-scale control from 6% to 35% capture, and the city run from a flat 0% to
something that climbs. Details below, each with the measurement that found it.

Jargon, once: **capture** = the low rescuer gets within 3 m of the evacuee.
**episode** = one run of one scenario, 120 seconds. **deterministic** = the policy's
best guess with exploration noise switched off, which is what you would fly on real
hardware. **batch** = the thousand-odd independent copies of the city we simulate at
once on the GPU.

---

## 1. Four reward bugs

### 1a. Every environment in the batch was the same episode

One start, one destination, one fire, shared by all 1024 copies. So a policy that
did nothing but hold one constant heading either always intercepted or never did —
an **untrained network scored 100%**. There was one trajectory to memorise, not a
task to learn.

The paper draws from 40 origin/destination pairs and 4 fire origins. We now do the
same, generated from the map: reachable ground is bucketed by angle into 40 sectors,
the cell furthest from the centre in each becomes a rim point, and each is paired
with one across the map. Pairing is by **walking** distance, not straight-line —

> Pairing each rim point with the geometric opposite gave routes up to 217 m, further
> than a 1.3 m/s walker covers in a 120 s episode. The evacuee reached safety in 6% of
> episodes: the rescuers were being scored against a deadline nobody could meet.
> Routes are now sized to the episode. Reaching safety went 6% → 86%.

### 1b. Searching paid forever

Three lines inside `step()` — leftovers from `reset()` — wiped the accumulated
`searched` map and the `first_seen` flag on **every step**. Both of their reward
terms invert when you do that:

- `coverage` pays for ground *newly* looked at. With the map wiped, the drone's whole
  footprint was new again every step, so **a stationary hover collected it forever**.
- `first_sight` pays *once* for finding somebody. It became 20 points a step for
  staring at them.

Against 200 points for a capture, over a 240-step episode, **watching was worth more
than twenty rescues**. The policy had learned exactly what it was paid for. That is
the whole explanation for the result that had been constant across every city run:

```
person seen 100%   captured 0%
```

Measured after the fix, for a drone hovering and doing nothing:

```
coverage paid at step 1 / 10 / 60 :  1.06   0.005   0.005     (was ~1.06 every step)
first_sight total over 60 steps   :  0.77 per environment      (i.e. once)
```

### 1c. Outcome rewards were paid every step, not once

`reached_goal` and `harmed` are flags that latch and stay true. The terms read them
directly, so they were paid again on every remaining step of the episode:

| term | was worth | should be |
|---|---|---|
| `reached_safety` | ~8,600 per rescue | 200–300, once |
| `evacuee_harmed` | −1,340 every episode, identical whatever the drones did | −100, once |

The −1,340 was the same for a hovering drone and a flying one, so it contributed
nothing but variance. Both now pay on the edge — the step the thing actually happens.

### 1d. Nothing paid for the second half of a rescue

`goal_progress` — the one term that pays for leading someone *to safety* after
reaching them — was gated on `captured`, the single step the capture happens on,
instead of `was_captured`. It fired once per episode and contributed **0.19 points
out of 760**. The reward never cared where the person was taken.

---

## 2. The bug that mattered most: a quarter of training was on dead episodes

The batch resets together, but episodes end at different times — the evacuee reaches
safety, or the fire gets them. Measured on the paper grid:

```
episodes ending before the deadline : 100%
mean end step                       : 73 of 100
-> 27 of every 100 training steps happened AFTER the episode was over
```

Those steps were still being paid and still being trained on. Finished environments
now freeze and stop earning.

A second, quieter version of the same bug: the evacuee kept *walking* after their
episode ended, and occasionally walked into a parked rescuer — registering a capture
for an episode that was already over. **3 of every 13 recorded captures.** The reward
was already masked, so this never misled the policy, only the number everything is
reported in. Both are fixed; captures after the end are now zero.

This single change:

| paper-scale control, 40 O/D pairs | before | after |
|---|---|---|
| deterministic capture | 6.2% | **37.1%** |
| stochastic capture | 17.1% | 28.1% |

Deterministic now **beats** stochastic, which is the healthy signature — the policy's
considered answer is better than its noise. Before, noise was better, which is what a
policy that has not converged looks like.

---

## 3. A number I have to retract

The previously reported **65% paper parity was measured on a single origin/destination
pair.** On that setup:

```
do nothing   0.0%
untrained   91.4%     <-- an untrained network
```

Our drone base sits at the map centre and the one fixed route runs straight through
it, so anything that jitters near the middle intercepts. **65% was worse than an
untrained network on that task**, so it measured nothing.

To be fair to the paper: its Env-I (71%) *is* its training setup unchanged, and it is
Env-IV (66%) that randomises both the person and the fire. So a fixed pair is not by
itself wrong — the problem is that *our* fixed-pair configuration is degenerate in a
way theirs is not. The comparison to make is our 40-pair number against **Env-IV's
66%**, not against Env-I's 71%.

The honest control is the 40-pair version, where the baselines are real:

```
do nothing                              6.1%
untrained, averaged over 8 inits        7.3%  +- 4.5   (range 3.7 - 16.0)
trained, deterministic                 37.1%           <-- 5x baseline
trained, stochastic                    28.1%
paper Env-IV (both randomised)         66.0%    <-- the right comparison
paper Env-I  (training setup)          71.0%
```

**The untrained baseline has to be averaged.** A single untrained network scored
anywhere from 3.7% to 32.3% depending only on its random initialisation — a fixed
random function either happens to fly toward people or it does not. One draw is not
a baseline, and an early version of this table quoted one.

Still well short of 66%. But it is now a number that means something.

---

## 3b. The city result: it finds people, it cannot close

Seed 0, Downtown Berkeley, 200 m of real city, CNN encoder, 24.6 M steps, all six
fixes in, scored on full episodes with the corrected metric:

| | capture | evacuee seen |
|---|---|---|
| no drone help | 7.0% | 80.7% |
| untrained, 5 initialisations | 7.3% ± 1.3 | — |
| **trained, deterministic** | **18.2%** | **96.9%** |
| trained, stochastic | 26.2% | 98.2% |

Two things to read here, and the second matters more than the first.

**It learned to search.** Sightings went from 80.7% to 96.9% — the drones genuinely
find the person now, which is the half of the problem the broken `coverage` reward
had been actively preventing.

**It has not learned to close, and it has not converged.** 18.2% capture from 96.9%
sightings means it is watching people it never reaches. And stochastic (26.2%) beating
deterministic (18.2%) is the *unhealthy* signature — the policy's considered answer is
worse than its own noise, the exact opposite of what the paper-scale control now does.

The most likely cause is the one thing the night did not touch: **the reward weights
were tuned to compensate for the four broken terms, and were never revisited after
they were fixed.** `proximity` pays 20 a step for being *near* someone and is about
85% of the whole episode budget — a level, not a progress term, so it is maximised by
loitering at a comfortable distance. It also swamps every search term.

`sandbox/reward_sweep.py` tests five balances against this exact scenario and is
queued to run after seed 1. This is the recommended next step, not another seed.

---

## 4. Built tonight

- **3D view in the GUI.** Buildings extruded to their real heights, drones at
  altitude, and each camera's cone drawn as a cone — it turns green when that camera
  has the evacuee. The cone is the one thing the top-down view fundamentally cannot
  show: whether a roof is between a lens and a person. three.js is vendored into the
  repo, so the panel works with no internet.
- **The GUI now flies a real policy.** It does this by wrapping the *training*
  environment at batch 1 rather than reimplementing the physics, so what you watch is
  the world the policy trained in. Checkpoints are listed from `checkpoints/`
  automatically.
- **Ten OSM cities**, chosen for street morphology rather than fame — Manhattan is 87%
  built and 397 m tall, Berkeley 54% and 56 m, the Barcelona Eixample a regular grid,
  the Berkeley campus free-standing blocks with open ground. Six are cached and usable
  offline; four need a re-fetch (see caveat below).
- **Maps pinned to coordinates, not to names.** "Eixample, Barcelona" geocoded to a
  point 8 km outside the district and produced a map 2.7% built where the real
  Eixample is 47%. A map that changes when a gazetteer is edited is not a map a
  result can rest on.
- `sandbox/generalise.py` — scores a checkpoint across every cached city against that
  city's own baselines, since the cities are not equally hard.
- 201 tests pass, including three new ones that pin the reward contracts above.

## 5. The 3D view, made true to the simulator (afternoon, GPU still down)

Not decoration — each of these fixes something the old view got *wrong*:

- **The low rescuer's cone pointed straight down. It doesn't.** Its camera is forward,
  pitched 20° down, 60° wide, 45 m range; the high rescuer's is straight down, 90°,
  80 m. The browser now receives each camera's true world-frame axis (through the
  airframe's attitude) and draws exactly the cone the detector tests against.
- **Attitude is real.** The model is oriented from the physics' own forward/up vectors,
  so a drone accelerating visibly leans — and its "downward" cone lands off to one
  side. At one sampled instant the HLR's axis was 35° off vertical. That is the
  coupling the paper is about, now visible: *going fast costs you your view.*
- **An X500-proportioned quad**: 0.25 m arms, 10-inch counter-rotating props, nav
  lights red-left/green-right, landing gear, a gimballed camera — on the nose tilted
  20° for the LLR, under the belly for the HLR. Drawn at ×8 (slider, labelled "true
  size 0.5 m"); the cones are never scaled.
- Smoke columns lean with the live wind vector; one flickering light at the fire's
  centroid; a low sun casting real shadows; dusk sky.

Stills: `docs/shots/gui_3d_wide.png`, `gui_3d_llr_closeup.png`, `gui_3d_hlr_closeup.png`.

## 6. Sunday: the evacuee, the closeness reward, full fidelity  *(14 Sep)*

**The evacuee walked through fire.** Ours priced a burning cell at 61x and burnt
ground at nothing, so whenever the way round was long -- always, on a real block --
they went through. The paper's rule is a hard one: `Fire.can_overlap` in `firegrid`
is false for the evader, so its A* treats fire as a wall. That rule is now kept
exactly, with two additions the paper never needed because its fire never goes out:
burnt ground costs 21x (crossed only if the way round is 21x longer, so nobody is ever
stranded), and cells within 12 m of flame are priced up so routes keep their
distance. If the fire cuts every route, the block relaxes to a very high price and
they head for the least-bad way out rather than freezing.

A second bug underneath it: `reset()` marked "replan immediately" and then called the
fire-blind planner, which overwrote the mark -- so the first route that knew about the
fire came **two seconds in**, after a straight-line walker had committed. Fixed.

Measured on a real Berkeley block with a fire lit directly on the route: straight
line 36 m, walked 99 m around it, zero samples inside fire, reached safety
unharmed. Three new tests pin it (`tests/test_human_avoids_fire.py`): around a fire
wall, around an ash line, and still moving when every route is cut. 204 pass.

**Why the policy detects and never approaches.** The paper's LLR "detects" inside a
5x5 window -- 2.5 m. In the paper, detection *was* closeness. Ours sees 45 m ahead,
so a policy is paid for detection from across the street. New term `apparent_size`:
how big the person is in the low rescuer's frame (height over distance, pinhole),
zero unless actually in view. On hardware this is the detector's bounding-box height,
so nothing changes at the seam. Audit on the current weights, full fidelity:

```
hover      proximity 1374   apparent_size 1.6
untrained  proximity  836   apparent_size 1.0
```

Hovering is paid *more* proximity than flying. The sweep now has six balances,
including "closeness only" (proximity removed entirely), and runs at full fidelity:
the quadrotor tier and the geometric detector that can miss -- measured at 1.31x the
cost of point-mass + oracle, cheap for no longer training on a cheat.

## 7. The reward sweep  *(14 Sep, full fidelity, 150 iterations each)*

Six balances, Berkeley, launch 25 m, quadrotor dynamics + geometric detector.
Deterministic capture on full episodes; do-nothing floor 9.4%.

| balance | what changed | capture | gain |
|---|---|---|---|
| **closeness** | `apparent_size` 20, `proximity` 5 | **45.3%** | **+35.9** |
| closeness only | `apparent_size` 20, `proximity` 0 | 21.7% | +12.3 |
| paper only | the paper's weights, none of ours | 19.2% | +9.9 |
| search first | pay once to find, little to stare | 17.2% | +7.8 |
| as shipped | our defaults | 14.6% | +5.3 |
| quiet proximity | `proximity` 20 → 5 | 14.3% | +4.9 |
| outcome heavy | capture/rescue 300, `proximity` 5 | 3.6% | −5.8 |

Three things to read off this.

**Pranav's diagnosis was right, and by a wide margin.** Paying for how large the
person is in the low rescuer's frame — closeness, but only while in view — tripled
the result of every other balance. The paper's 2.5 m detection window had been
providing exactly that signal without anyone naming it.

**Closeness alone is not enough.** Removing `proximity` entirely halved the gain.
`apparent_size` can only pay while the person is in view; the small residual
`proximity` is what pulls the drone toward the *last known* position once they are
lost. Both are needed.

**Our extra terms, as shipped, were net harmful.** The paper's own weights with none
of our additions (19.2%) beat our defaults (14.6%). And sparse outcomes with no dense
guide ("outcome heavy") did worse than hovering — a policy needs something to climb.

The long run — `closeness`, 600 iterations, two seeds, full fidelity — started
automatically when the sweep finished. `logs/final_run.log`.

## 8. Longer was worse  *(14 Sep, evening)*

The winning balance was run for 600 iterations instead of 150. Same scenario, same
weights, same fidelity, evaluated on the same 256 episodes:

```
                        deterministic   stochastic
floor (no drones)            7.8%
150 iterations              42.2%          25.0%     healthy
600 iterations, seed 0      23.0%          23.0%     identical
600 iterations, seed 1      16.9%            --      (60 m: 8.0%, floor 3.1%)
```

Final table of the 600-iteration run: seed 0 23.2% / 12.0% at 25 / 60 m, seed 1
16.9% / 8.0%, spread 6.3 points -- reproducible, and reproducibly worse than
iteration 150.

Both seeds of the long run land well below the short one, so this is not one bad
seed -- it is what this balance does when trained longer.

Deterministic equal to stochastic *looked* like exploration collapsing to zero. It
was not -- checked, twice. The action spread is unchanged (`log_std` gives 0.61 in
both checkpoints) and the actor's means are nowhere near saturating tanh (|mu| about
0.5, none above 2). What actually changed is behaviour:

```
                         LLR flown   LLR altitude   time to capture
150 iterations              708 m       13.2 m           13 s
600 iterations              389 m        8.7 m           33 s
```

The longer-trained policy flies half as far, sits lower, and catches people late --
it drifted from active search toward loitering where routes converge, and a
loiterer's outcome barely changes when you add noise to its actions. That is why the
two numbers matched. It is the same pull toward standing still that the sweep
measured in `proximity` (hovering earned more of it than flying), still present at
weight 5, winning slowly over 600 iterations.

Fixes for the next run (the current one keeps going for its second seed): every
evaluated checkpoint is kept and the best copied aside, so a peak can never be
overwritten again -- this is the one that matters; the curriculum promotes on a
periodic full-episode evaluation rather than the rollout-window flag, because the
first run never left 25 m (the window metric reads far below the real one); and the
entropy bonus decays to a floor of 0.002 rather than zero, kept as cheap insurance,
not as the cause.

## 9. Run-to-run variance is large  *(15 Sep, early)*

The keep-best re-run of the same balance, same seed, same fidelity, evaluated on
full episodes every 25 iterations:

```
iteration      25     50     75    100    125    150
deterministic  30.9   15.2   19.5   23.0   19.9   18.8
stochastic     25.4   19.5   19.9   23.4   25.0   18.0
```

The sweep's run of this exact configuration read 42-45% at iteration 150. This one
reads 18.8%. Nothing in the setup differs except the entropy schedule's denominator
and the draw of the training stream -- which means **the 45% was a fortunate
trajectory, not the typical one.** The typical peak for this balance looks nearer
30%, reached early, followed by a noisy plateau around 20%.

The keep-best run's final table (300 iterations, two seeds, evaluated every 25):

```
                 best (kept)   at iteration   final, 25 m   final, 60 m   floor 25 / 60
seed 0              30.9%           25          18.5%          7.4%        8.6 / 2.3
seed 1              19.5%          100          14.1%          5.0%
sweep run           42.2%          150          (the same balance, one fortunate draw)
```

Every run of this balance peaks early and decays toward the floor -- seed 1 touched
8.2% at iteration 200, indistinguishable from no drones at all -- so the kept-best
checkpoint is the deliverable, not the final one. Across four runs the peak ranges
19-45%. That spread is the honest headline.

How this should be reported: as the spread across runs, with the evaluation curve,
not as one number. The claim that survives is the *ranking* -- closeness beat every
other balance by a wide margin in the sweep, and every run of it has beaten the
600-iteration plateau and the 9% floor -- not the 45% itself.

## 10. It generalises across cities  *(15 Sep, final: 256 episodes per city)*

The sweep's closeness policy -- trained on Downtown Berkeley only -- scored on every
city at full fidelity, each against that city's own do-nothing floor, 256 episodes
per city on the GPU (an earlier 48-episode CPU pass gave the same picture).

| city | built | do nothing | policy | gain |
|---|---|---|---|---|
| Downtown Berkeley *(trained here)* | 54% | 8.2% | 44.9% | +36.7 |
| UC Berkeley campus | 51% | 9.4% | 40.6% | +31.2 |
| SF Financial District | 52% | 6.2% | 45.3% | +39.1 |
| Manhattan Midtown | 87% | 4.7% | 34.4% | +29.7 |
| Chicago Loop | 51% | 7.8% | 37.1% | +29.3 |
| Barcelona Eixample | 47% | 12.9% | 43.4% | +30.5 |
| Paris Le Marais | 57% | 7.4% | 44.5% | +37.1 |
| Venice San Marco | 73% | 10.2% | 47.7% | +37.5 |
| Tokyo Shinjuku | 57% | 10.5% | 49.6% | +39.1 |
| Oakland Downtown | 38% | 8.6% | 37.9% | +29.3 |

**Held-out mean gain +33.6 points across nine cities it never saw; worst +29.3
(Chicago).** Every held-out city lands between +29 and +39, and the trained-on city
is not the best. A policy that had memorised Berkeley's street pattern would collapse
on Manhattan's 400 m towers or Venice's alleys; this one does not. What it learned is
*search* -- a question the 2D grid could never have asked.

## 11. A thirty-line rule nearly matches the learned policy  *(20 Sep)*

Building the policy seam produced three scripted baselines for free, and one of
them is the comparison the week had been missing. Berkeley, 256 episodes, full
fidelity, deterministic:

```
                            launch 25 m   launch 60 m   time to capture (25 / 60 m)
hover  (the floor)               7.8%          5.1%
lawnmower sweep                 15.6%         16.0%
last_sighting  (30 lines)       45.7%         21.5%          17 s / 28 s
sweep_closeness  (learned)      48.4%         19.5%          19 s / 46 s
keep-best seed 0 (learned)      28.5%         15.6%          19 s / 38 s
```

`last_sighting` does one thing: fly toward wherever the team last saw the person,
hold if it never has. The learned policy beats it by 2.7 points at 25 m and loses
to it at 60 m -- and is slower to close at both distances. Every other learned
checkpoint is below it everywhere.

Two things this says. First, on a 200 m block with an 80 m downward camera, *seen*
is 100% even for a hovering drone: finding the person is nearly free at this scale,
and the whole task is closing. A rule that closes does about as well as anything we
have trained. Second, the cross-city result needs this row: a rule that follows the
last sighting is map-agnostic by construction, so "it generalises" is only a claim
about learning if the learned policy generalises *better than the rule*. That run is
in progress (`logs/generalise_last_sighting.log`).

**Cross-city, the rule beside the learned policy** (256 episodes each, deterministic):

| city | last_sighting rule | learned (sweep_closeness) | learned − rule |
|---|---|---|---|
| Downtown Berkeley | 45.7% | 44.9% | -0.8 |
| UC Berkeley campus | 52.7% | 40.6% | -12.1 |
| SF Financial District | 40.2% | 45.3% | +5.1 |
| Manhattan Midtown | 46.9% | 34.4% | -12.5 |
| Chicago Loop | 43.8% | 37.1% | -6.7 |
| Barcelona Eixample | 47.7% | 43.4% | -4.3 |
| Paris Le Marais | 37.9% | 44.5% | +6.6 |
| Venice San Marco | 41.4% | 47.7% | +6.3 |
| Tokyo Shinjuku | 41.0% | 49.6% | +8.6 |
| Oakland Downtown | 40.6% | 37.9% | -2.7 |

Held-out mean gain over each city's floor: **rule +35.4, learned +33.6.** The rule
generalises at least as well -- as it must, being map-agnostic by construction. So
the cross-city table is evidence about the *task* (at this scale, closing is the whole
problem and a closing rule solves it everywhere), not evidence that the network
learned to search. The generalisation claim in section 10 is withdrawn in that form.
What remains true: the learned policy is roughly as good as the best hand rule we
wrote, on every city, and beats it on four of ten. That is a floor to build from, not
a result to lead with.

What would make the learned policy earn its place: a larger map or a narrower
camera, where search is not free; a launch further out, where the 60 m gap already
shows; or the person moving unpredictably, where anticipation beats pursuit.

---

## Caveats, honestly

- **Overpass has banned this machine** for querying it too hard (connection refused
  over IPv4, not a local network fault). This turned out not to matter: **all ten
  cities are cached at 100 m radius and work offline.** Other radii are not
  downloaded. Mirrors and backoff are now in place so it should not recur.
- **35% is not 71%.** The gap is real and unexplained. Our version is harder in ways
  that are the point of the project — occlusion, momentum, a detector seam — but I
  have not isolated how much of the gap is that versus something still wrong.
- **The paper-parity control used the flat encoder, not the CNN** — `paper_parity.py`
  defaulted to it and I did not notice until re-loading the checkpoint. The city run
  does use the CNN. Parity has been switched to the CNN for the next run of it, but
  the 37.1% above is a flat-encoder number.
- **The GPU went away at 06:30.** `unattended-upgrades` installed
  `nvidia-driver-580-open` 580.178.04 while the run was going, leaving the loaded
  kernel module at 580.173.02. New CUDA processes now fail with error 804. Seed 1 had
  already finished; **the reward sweep never got to start.** Reboot, then run
  `./RESUME_AFTER_REBOOT.sh`. Nothing was lost — all checkpoints and logs are on disk,
  and the GUI still runs the policy on CPU.
- Both city seeds finished: deterministic **16.7%** and **17.4%** on the in-run
  metric, a 0.7-point spread. The result is reproducible; it is just not good yet.
- The in-run capture percentages printed *during* the city run are inflated by a few
  points: that process started before the post-episode-capture fix, so it kept the old
  metric. The table in section 3b was re-measured afterwards with the corrected code,
  and is the one to trust.
- Still uncommitted to any remote. 20+ commits exist only on this box.
