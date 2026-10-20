# Pending work — hadr-sim3d, as of 20 Oct 2026

Ordered by what I would do next. Each line says why, and roughly how big.

## Results and method (the paper depends on these)

1. **Beat the scripted baseline, or say plainly that we don't.** `last_sighting`
   (30 lines: fly toward the last sighting) scores within 3 points of the best
   learned policy at 25 m, beats it at 60 m, and generalises across cities about as
   well. Until a learned policy clearly beats it, the learning result is "matches a
   simple rule." Candidates for where learning should win: a larger map or narrower
   camera (search is nearly free at 100 m radius with an 80 m camera — even hovering
   sees the person 100% of the time), launch further out, a person who moves less
   predictably. *Experiment design, then GPU days.*
2. **Hyperparameter sweep with wandb, on more than one GPU.** `hadrsim-train` +
   `experiments/sweep.yaml` are ready; nothing has been swept but reward weights. Five
   seeds per point — peaks ranged 19–45% across four runs of one configuration.
   *Needs Gaby's GPUs and a wandb login.*
3. **Find the decay.** Every run peaks by iteration 25–150 and slides toward the
   floor. Suspect: residual `proximity` pull toward loitering. Keep-best checkpoints
   are the workaround, not the fix. *One sweep.*
4. **Paper-scale control: five seeds per encoder.** One CNN rerun (20 Sep) gave 7.0%
   deterministic / 22.9% stochastic — not converged — against the flat run's 37.1%.
   Neither is quotable from one seed. *Part of the wandb sweep.*
5. **Honest hazard map.** The raster's fire and smoke channels are exact within the
   camera footprint — a disc, not line of sight — so a roof does not hide a fire from
   the policy. Ray-march them like the person, add detector noise. *~1 day; changes
   behaviour, so re-record `tests/golden.json` on purpose.*
6. **Report "rescued", not only "captured."** Capture-then-reached-safety is the
   outcome the paper cares about; it is computed but not headlined. *Small.*

## Simulator

7. **Pixel detector, next steps.** A minimal one exists (`sensing/camera.py`,
   `sensing/pixel.py`, `--detector pixel`): semantic 32×32 render with clutter, CNN
   trained on oracle labels. Not yet done: calibrate the geometric detector's miss and
   false-alarm rates to it; train a policy against it; richer clutter and partial
   occlusion; the camera image in the panel. *Days, as wanted.*
8. **Comms layer.** Teammate state and shared fire map arrive instantly and perfectly
   today; the contract says they arrive through comms that can be late or lost.
   *~2 days.*
9. **Person model options.** One class today. If alternatives to Trivedi–Rao are
   ever compared, `human.py` needs the same seam the detector has. *Only if asked.*
10. **Crowds with drones.** `Runner` (crowds) has no drones; `PolicyRunner` has one
    person. The observation contract is one person, so merging them means a design
    decision, not a refactor. *Design first.*
11. **Episode length vs detours.** 16% of unhelped evacuees are still walking at the
    120 s deadline because fire made them detour. Either size the episode to the
    detour budget or place fire origins off the route. *Small, but it moves numbers.*

## Code

12. **Known simulator gap.** Wind pushes fire and smoke but not the airframe.
    Small to fix; moves the golden numbers, so do it deliberately. (Buildings became
    solid for the drones on 20 Oct: on Berkeley the hover floor went 7.8 → 14.1%, the
    `last_sighting` rule 45.7 → 54.3%, the `sweep_closeness` checkpoint 48.4 → 32.0%
    -- it was trained without walls. Every checkpoint predates this change.)
13. **Research decisions, one at a time, with evidence.** Which of the 15 reward
    terms stay (`configs/reward/`); three drone tiers or two; keep the crowd
    `Runner`; which two checkpoints ship (do not delete any before deciding).
14. **Drone count.** Two, hardcoded in the network, the reward and the renderer; the
    panel's "Drones" box is inert. The paper's whole argument is the two-role
    asymmetry, so leave it unless the research question changes.
15. **Human-avoids-fire margin.** 12 m stand-off and 21× ash price are hand-set;
    the paper has no equivalent. Worth a sensitivity check.

## Infrastructure

16. **Push the repo.** ~20 commits, all local; CI (`.github/workflows/ci.yml`)
    runs on the first push. *One command, yours.*
17. **`nvidia-*` in the apt blacklist**, so unattended-upgrades cannot break CUDA
    mid-run again. *Your call.*
18. **Overpass.** Only the ten presets at 100 m radius are cached; any other radius
    or city needs a fetch, and the host was banned once for fetching too fast.

## Done this pass (for the record)

- Reproducibility across processes (`Rng.fork` used Python's salted hash).
- Golden regression test; one evaluator; policy seam with scripted baselines and
  checkpoint manifests; `Env` split into named pieces; panel exposes physics and
  detector; block-wise anneal fixed.
- Oct cleanup: `env.py` / `train.py` / `eval.py` as the three top-level modules;
  `sandbox/` → `experiments/` with only research scripts left; reward terms split
  paper/ours with weights in yaml; dead code out; packaged with four commands,
  README and CI. 183 tests, golden unchanged throughout.
