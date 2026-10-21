"""Two runners behind one interface (`reset`, `step`, `advance(seconds)`, `snapshot`, `time`,
`done`): `Runner` steps fire, smoke, wind and a crowd with no drones; `PolicyRunner`
wraps `Env` at batch 1 and flies a policy in it."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hadrsim.agents.crowd import Crowd
from hadrsim.agents.human import Human
from hadrsim.core import rotation as rot
from hadrsim.core import spec
from hadrsim.core.rng import Rng
from hadrsim.scenario import Scenario, autoplace
from hadrsim.world import fire as fire_mod
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import wind as wind_mod


@dataclass
class Snapshot:
    """What a watcher needs."""

    t: float
    burning: int
    burnt: int
    humans: list = field(default_factory=list)   # {x, y, panic, speed, reached, ...}
    drones: list = field(default_factory=list)
    reached: int = 0
    done: bool = False


def _person(h, i: int, cell: float, **extra) -> dict:
    """One person in cell coordinates, for either runner's snapshot."""
    return {"x": round(float(h.pos[i, 0] / cell), 2), "y": round(float(h.pos[i, 1] / cell), 2),
            "panic": round(float(h.gamma[i]), 3), "speed": round(float(h.vel[i].norm()), 2),
            "goal": [round(float(h.goal[i, 0] / cell), 2), round(float(h.goal[i, 1] / cell), 2)],
            "reached": bool(h.reached[i]), **extra}


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

    # ------------------------------------------------------------------- step

    @property
    def time(self) -> float:
        return self.tick * self.clock.dt

    @property
    def done(self) -> bool:
        return self.time >= self.scenario.episode_seconds

    def step(self) -> None:
        """One physics tick."""
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

        self.tick += 1

    def advance(self, seconds: float) -> None:
        """Run for a wall-clock slice."""
        for _ in range(max(1, round(seconds * self.clock.physics_hz))):
            if self.done:
                break
            self.step()

    # --------------------------------------------------------------- watching

    def snapshot(self) -> Snapshot:
        cell = self.terrain.cell_size_m
        return Snapshot(
            t=round(self.time, 2),
            burning=int(self.fire.burning[0].sum()), burnt=int(self.fire.burnt[0].sum()),
            humans=[_person(self.human, i, cell) for i in range(len(self.human.pos))],
            reached=int(self.human.reached.sum()), done=self.done,
        )


def _tile(values: list, n: int) -> list:
    """Repeat a short list of positions up to `n`, so one entry can serve many."""
    return [values[i % len(values)] for i in range(n)]


class PolicyRunner:
    """A live scenario with drones flying a policy: `Env` at batch 1, read out."""

    def __init__(self, scenario, checkpoint: str | None = None, device: str = "cpu",
                 fidelity: str = "quadrotor", detector: str = "geometric"):
        from hadrsim.env import Env
        from hadrsim.policy import base

        self.env = Env(scenario, batch=1, fidelity=fidelity, detector=detector, device=device)
        self.scenario = self.env.scenario
        self.terrain = self.env.terrain
        self.clock = self.env.clock
        # A checkpoint path, or one of the scripted baselines by name.
        self.net = base.load(checkpoint, device)
        self.reset()

    # ------------------------------------------------------------------ state

    @property
    def fire(self):
        return self.env.fire

    @property
    def smoke(self):
        return self.env.smoke

    @property
    def wind(self):
        return self.env.wind

    @property
    def time(self) -> float:
        return float(self.env.step_count[0]) * self.clock.control_dt

    @property
    def done(self) -> bool:
        return bool(self.env.step_count[0] >= self.clock.max_steps
                    or self.env.human.reached[0] or self.env.human.harmed[0])

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.state = None
        self.trail: list[list[float]] = [[], []]

    # ------------------------------------------------------------------- step

    def step(self) -> None:
        """One policy decision, and the half-second of flight that follows it."""
        if self.done:
            return
        with torch.no_grad():
            if self.net is None:
                action = torch.zeros(1, 2, spec.ACTION_DIM, device=self.env.device)
            else:
                action, _, _, self.state = self.net.act(
                    self.obs, self.state, deterministic=True)
            self.obs, _, _, self.info = self.env.step(action)
        cell = self.terrain.cell_size_m
        for i, d in enumerate(self.env.drones):
            self.trail[i].append([round(float(d.pos[0, 0] / cell), 2),
                                  round(float(d.pos[0, 1] / cell), 2)])
            del self.trail[i][:-120]

    def advance(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.clock.control_dt))):
            if self.done:
                break
            self.step()

    # --------------------------------------------------------------- watching

    def snapshot(self) -> Snapshot:
        from hadrsim.sensing import fov as fov_mod

        e, cell = self.env, self.terrain.cell_size_m
        seen = self.obs["seen_by"][0].tolist()
        drones = []
        for i, d in enumerate(e.drones):
            cam = e.cameras[i]
            reach = float(fov_mod.ground_footprint(cam, d.pos, d.quat)[0])
            roll, pitch = d.attitude
            axis = fov_mod.look_direction(cam, d.quat)[0].tolist()
            ex = torch.tensor([[1.0, 0.0, 0.0]], device=d.quat.device)
            ez = torch.tensor([[0.0, 0.0, 1.0]], device=d.quat.device)
            fwd = rot.rotate(d.quat[:1], ex)[0].tolist()
            up = rot.rotate(d.quat[:1], ez)[0].tolist()
            drones.append({
                "cam": {"axis": [round(v, 4) for v in axis], "fov": cam.fov_deg,
                        "range": cam.range_m, "kind": cam.kind},
                "fwd": [round(v, 4) for v in fwd],
                "up": [round(v, 4) for v in up],
                "arm_m": d.af.arm_m,
                "x": round(float(d.pos[0, 0] / cell), 2),
                "y": round(float(d.pos[0, 1] / cell), 2),
                "z": round(float(d.pos[0, 2]), 1),
                "yaw": round(float(d.yaw[0]), 3),
                "roll": round(float(roll[0]), 3),
                "pitch": round(float(pitch[0]), 3),
                "role": "hlr" if i == 0 else "llr",
                "footprint": round(reach / cell, 2),
                "sees": bool(seen[i]),
                "trail": self.trail[i],
            })
        return Snapshot(
            t=round(self.time, 2),
            burning=int(e.fire.burning[0].sum()), burnt=int(e.fire.burnt[0].sum()),
            humans=[_person(e.human, 0, cell, harmed=bool(e.human.harmed[0]), captured=bool(e.captured[0]))],
            drones=drones, reached=int(e.human.reached[0]), done=self.done,
        )
