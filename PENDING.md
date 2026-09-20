# Pending work — hadr-sim3d, as of 20 Sep 2026

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
2. **Hyperparameter sweep with wandb, on more than one GPU.** `sandbox/train.py` +
   `sandbox/sweep.yaml` are ready; nothing has been swept but reward weights. Five
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

7. **Pixel-level detector at the seam.** Render each camera's view (semantic, not
   photoreal — the ray-marcher already exists), train a CNN offline to emit
   detections and bounding-box height, plug into `detector.make`. The sim-to-real
   piece. *~3 days.*
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

12. **Legacy Rerun path.** `viz/rerun_sink.py`, `viz/mesh.py`, `viz/facade.py`,
    `scripts/view_*.py` predate the panel and are used only by those scripts. Keep
    (they have tests) or delete; either is fine, they cost nothing but reading time.
13. **`sandbox/` pruning.** `city2.py`, `sweep.py`, `replicate.py`, `curriculum.py`
    are superseded by `train.py` / `final_run.py`. Delete once their numbers are in
    the findings (they are).
14. **Drone count.** Two, hardcoded in the network, the reward and the renderer; the
    panel's "Drones" box is inert. The paper's whole argument is the two-role
    asymmetry, so leave it unless the research question changes.
15. **Human-avoids-fire margin.** 12 m stand-off and 21× ash price are hand-set;
    the paper has no equivalent. Worth a sensitivity check.

## Infrastructure

16. **Push the repo.** Six commits, all local. *One command, yours.*
17. **`nvidia-*` in the apt blacklist**, so unattended-upgrades cannot break CUDA
    mid-run again. *Your call.*
18. **Overpass.** Only the ten presets at 100 m radius are cached; any other radius
    or city needs a fetch, and the host was banned once for fetching too fast.

## Done this pass (for the record)

- Reproducibility across processes (`Rng.fork` used Python's salted hash).
- Golden regression test; one evaluator; policy seam with scripted baselines and
  checkpoint manifests; `TrainEnv` split into named pieces; panel exposes physics
  and detector; wandb-ready training entry; block-wise anneal fixed. 209 tests.
