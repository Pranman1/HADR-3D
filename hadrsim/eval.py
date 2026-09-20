"""Scoring a policy: the one evaluation loop.

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


def baselines(env, n_untrained: int = 5, make_untrained=None) -> dict[str, float]:
    """The two floors a trained number must be read against: doing nothing, and an
    *averaged* untrained network -- a single draw ranged 3.7% to 32% by seed."""
    out = {"do nothing": evaluate(env).capture}
    if make_untrained is not None and n_untrained:
        vals = []
        for s in range(n_untrained):
            torch.manual_seed(s)
            vals.append(evaluate(env, make_untrained()).capture)
        out["untrained, mean"] = sum(vals) / len(vals)
        out["untrained, sd"] = (sum((v - out["untrained, mean"]) ** 2 for v in vals) / len(vals)) ** 0.5
    return out
