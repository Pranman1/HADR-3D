"""Running a scenario, one step at a time.

Sits between a `Scenario` and whatever is watching: a GUI stepping it in real time,
a script rendering it to a video, or eventually the training loop.  All three drive
the same object, which is the point -- a control panel that steps the world even
slightly differently from the trainer is telling you about a world nobody trained
in.

Stepping is exposed at two levels.  `step()` advances one physics tick, which is
what the clock actually means; `advance(seconds)` runs however many of those fit,
which is what a caller wanting a frame actually wants.  Keeping them apart is what
lets the GUI run at whatever rate the browser can manage without the simulation
noticing.

The drone and policy slots are wired and empty.  There is no policy yet; when there
is, it plugs in here and nothing above changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hadrsim.agents.crowd import Crowd
from hadrsim.agents.human import Human
from hadrsim.core.rng import Rng
from hadrsim.scenario import Scenario, autoplace
from hadrsim.world import fire as fire_mod
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import wind as wind_mod


@dataclass
class Snapshot:
    """What a watcher needs. Plain python, so it serialises straight to JSON."""

    t: float
    burning: int
    burnt: int
    humans: list = field(default_factory=list)   # {x, y, panic, speed, reached}
    drones: list = field(default_factory=list)
    reached: int = 0
    done: bool = False


class Runner:
    """One live scenario."""

    def __init__(self, scenario: Scenario):
        self.scenario = autoplace(scenario)
        self.terrain = self.scenario.terrain()
        self.clock = self.scenario.clock()
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        sc = self.scenario
        n = max(1, sc.n_humans)

        # A stream each. Sharing one makes the fire's spread depend on how often the
        # wind was stepped, which turns "reproducible" into a claim rather than a fact.
        base = Rng(sc.seed)
        self.rng_fire = base.fork("fire")
        self.rng_wind = base.fork("wind")
        self.rng_human = base.fork("human")

        self.fire = fire_mod.Fire(self.terrain, n)
        self.smoke = smoke_mod.Smoke(self.terrain, n)
        self.wind = (wind_mod.still(n) if sc.wind_mps <= 0 else
                     wind_mod.make(n, sc.wind_mps, sc.wind_bearing_deg))
        self.human = Human(self.terrain, n)
        self.crowd = Crowd(self.terrain)

        self.fire.reset(torch.tensor([sc.fire_at] * n).long(), radius_m=sc.fire_radius_m)
        starts = _tile(sc.human_start, n)
        goals = _tile(sc.human_goal, n)
        self.human.reset(torch.tensor(starts).float(), torch.tensor(goals).float())
        self.smoke.update(self.fire, self.wind)

        self.tick = 0
        self.want = torch.zeros(n, 2)
        self.reached_at: list[float | None] = [None] * n

    # ------------------------------------------------------------------- step

    @property
    def time(self) -> float:
        return self.tick * self.clock.dt

    @property
    def done(self) -> bool:
        return self.time >= self.scenario.episode_seconds

    def step(self) -> None:
        """One physics tick. Everything else runs on its own multiple of this."""
        c, dt = self.clock, self.clock.dt

        if self.tick % c.human_substeps == 0:
            self.want = self.human.desired_velocity(
                self.fire, self.smoke, self.rng_human, c.human_substeps * dt)
        self.human.integrate(
            self.crowd.step(self.human.pos, self.human.vel, self.want, dt), dt)

        self.wind.step(dt, self.rng_wind)
        self.fire.step(dt, self.wind, self.rng_fire)
        if self.tick % max(1, int(c.physics_hz / 2)) == 0:
            self.smoke.update(self.fire, self.wind)

        for i, done in enumerate(self.human.reached.tolist()):
            if done and self.reached_at[i] is None:
                self.reached_at[i] = self.time
        self.tick += 1

    def advance(self, seconds: float) -> None:
        """Run for a wall-clock slice. What a viewer wanting one frame asks for."""
        for _ in range(max(1, round(seconds * self.clock.physics_hz))):
            if self.done:
                break
            self.step()

    # --------------------------------------------------------------- watching

    def snapshot(self) -> Snapshot:
        cell = self.terrain.cell_size_m
        pos = self.human.pos.cpu()
        return Snapshot(
            t=round(self.time, 2),
            burning=int(self.fire.burning[0].sum()),
            burnt=int(self.fire.burnt[0].sum()),
            humans=[
                {
                    "x": round(float(pos[i, 0] / cell), 2),
                    "y": round(float(pos[i, 1] / cell), 2),
                    "panic": round(float(self.human.gamma[i]), 3),
                    "speed": round(float(self.human.vel[i].norm()), 2),
                    "reached": bool(self.human.reached[i]),
                }
                for i in range(len(pos))
            ],
            drones=[],                      # nothing flies yet
            reached=int(self.human.reached.sum()),
            done=self.done,
        )


def _tile(values: list, n: int) -> list:
    """Repeat a short list of positions up to `n`, so one entry can serve many."""
    return [values[i % len(values)] for i in range(n)]
