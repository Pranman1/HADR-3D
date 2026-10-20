"""Scoring a policy: the one evaluation loop, and the command that runs it everywhere.

    hadrsim-eval checkpoints/sweep_closeness.pt            # every cached city, each against its own floor
    hadrsim-eval last_sighting --city "Manhattan Midtown"  # one city, a scripted baseline

Four copies of this loop had grown in `sandbox/` -- one per experiment -- and they
had started to disagree about what they counted (whether a capture after the
episode ended counts, whether "seen" means ever or right now).  A result that
depends on which script scored it is not a result.  This is the loop; everything
that reports a number calls it.

    m = evaluate(env, policy)          # deterministic, full episodes
    print(m.line("closeness, seed 0"))

`policy` is anything with `act(obs, state, deterministic) -> (action, _, _, state)`
-- a trained network, a scripted baseline from `policy.base`, or `None`, which
means the drones hold still and gives the floor every other number is read against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Metrics:
    """What one batch of full episodes came to. Percentages, 0-100."""

    episodes: int
    capture: float            # low rescuer reached the evacuee
    seen: float               # anyone had the evacuee in view at some point
    reached: float            # evacuee reached safety (led or not)
    harmed: float             # evacuee caught by the fire
    time_to_capture_s: float  # mean, over captured episodes only; nan if none
    first_seen_s: float       # mean time to first sighting; nan if never
    in_view: float            # mean fraction of the episode someone had eyes on
    parts: dict = field(default_factory=dict)   # summed reward, per term

    def line(self, tag: str) -> str:
        """One printed row, the shape every sandbox script used to hand-roll."""
        return (f"  {tag:<28} capture {self.capture:5.1f}%   seen {self.seen:5.1f}%"
                f"   reached {self.reached:5.1f}%   harmed {self.harmed:5.1f}%"
                f"   t-capture {self.time_to_capture_s:5.1f} s")


@torch.no_grad()
def evaluate(env, policy=None, deterministic: bool = True, reset: bool = True) -> Metrics:
    """Run every environment in `env` to the end of its episode and count."""
    obs = env.reset() if reset else env.observe()
    state = None
    B, dev, dt = env.batch, env.device, env.clock.control_dt
    z = lambda: torch.zeros(B, dtype=torch.bool, device=dev)
    cap, seen, reached, harmed = z(), z(), z(), z()
    nan = torch.full((B,), float("nan"), device=dev)
    t_cap, t_seen = nan.clone(), nan.clone()
    in_view = torch.zeros(B, device=dev)
    parts: dict[str, float] = {}

    for i in range(env.clock.max_steps):
        if policy is None:
            act = torch.zeros(B, 2, 4, device=dev)
        else:
            act, _, _, state = policy.act(obs, state, deterministic=deterministic)
        obs, _, _, info = env.step(act)
        now = obs["seen_by"].any(-1)
        newly_cap = info["captured"] & ~cap
        t_cap = torch.where(newly_cap, torch.full_like(t_cap, i * dt), t_cap)
        t_seen = torch.where(now & t_seen.isnan(), torch.full_like(t_seen, i * dt), t_seen)
        in_view += now.float()
        cap |= info["captured"]; seen |= now
        reached |= info["reached"]; harmed |= info["harmed"]
        for k, v in info["parts"].items():
            parts[k] = parts.get(k, 0.0) + v

    pct = lambda t: float(t.float().mean()) * 100
    mean = lambda t: float(t[~t.isnan()].mean()) if bool((~t.isnan()).any()) else float("nan")
    return Metrics(
        episodes=B, capture=pct(cap), seen=pct(seen), reached=pct(reached), harmed=pct(harmed),
        time_to_capture_s=mean(t_cap), first_seen_s=mean(t_seen),
        in_view=float(in_view.mean()) / env.clock.max_steps * 100,
        parts={k: round(v, 4) for k, v in parts.items()},
    )


def main(argv=None) -> None:
    """Score a policy on one city or on every cached one, each against its own floor.
    Cities are not equally hard, so the number that means anything is the gain."""
    import argparse
    import pathlib

    from hadrsim.env import Env
    from hadrsim.policy import base
    from hadrsim.scenario import PRESETS, Scenario, autoplace
    from hadrsim.world import osm

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("policy", help="checkpoint path, or hover / lawnmower / last_sighting")
    p.add_argument("--city", default=None, help="one preset; default: every cached city")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--launch", type=float, default=25.0, help="metres from the evacuee")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--fidelity", default="quadrotor")
    p.add_argument("--detector", default="geometric")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)

    policy = base.load(a.policy, a.device)
    cities = [a.city] if a.city else [c for c in PRESETS if not c.startswith("Paper")]
    print(f"{a.policy}   launch {a.launch:.0f} m   {a.batch} episodes per city\n")
    print(f"{'city':<24}{'floor':>7}{'policy':>8}{'gain':>7}{'safe':>7}{'lost':>7}")
    gains = {}
    for name in cities:
        sc = Scenario.preset(name, radius_m=a.radius, cell_size_m=2.0, episode_seconds=a.seconds, wind_mps=0.0, seed=0)
        box = osm.bbox_around(sc.lat, sc.lon, sc.radius_m) if sc.lat is not None else None
        if box and not (pathlib.Path("maps/osm_cache") / f"{box.key(osm.QUERY)}.json").exists():
            print(f"{name:<24}  not cached, skipped")
            continue
        env = Env(autoplace(sc), batch=a.batch, fidelity=a.fidelity, detector=a.detector, device=a.device)
        env.launch_offset_m = a.launch
        floor, m = evaluate(env).capture, evaluate(env, policy)
        gains[name] = m.capture - floor
        print(f"{name:<24}{floor:6.1f}%{m.capture:7.1f}%{gains[name]:+6.1f}{m.reached:6.1f}%{m.harmed:6.1f}%")
    if len(gains) > 1:
        print(f"\n  mean gain {sum(gains.values()) / len(gains):+.1f} over {len(gains)} cities"
              f"   worst {min(gains, key=gains.get)} ({min(gains.values()):+.1f})")


if __name__ == "__main__":
    main()
