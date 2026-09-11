"""The training environment: thousands of cities burning at once.

Everything so far has been one world at a time.  This is where the batch dimension
finally earns itself -- the whole point of writing world, agents and sensing over
`(B, ...)` tensors was so that this file could be thin.

The loop, per policy decision:

    observe   ->  what each drone can see, through comms
    act       ->  the policy commands a velocity
    fly       ->  50 physics substeps at 100 Hz, and the person walks
    score     ->  reward terms over what the team *believed*, never over truth
    reset     ->  environments that finished, in place

**No graphics, no ROS, no python loop over environments.**  A function that only
handles one simulation is a hidden slow loop waiting to happen, and the running
environment is a different file for a reason: this one exists to be fast, and that
one to be watchable and to speak to real hardware.

**Beliefs, not truth.**  Observations and rewards are both assembled from
`Detections`.  The 2D simulator's rewards read ground-truth visibility on their own,
which means degrading perception changes nothing -- the answer still arrives through
the reward channel.  Closing that is most of why this project exists, so it is
closed here by construction: the world's true state is never passed to the scorer.
"""

from __future__ import annotations

import math

import torch

from hadrsim.agents import airframes as af
from hadrsim.agents.crowd import Crowd
from hadrsim.agents.drone import Drone
from hadrsim.agents.human import Human
from hadrsim.core import spec
from hadrsim.core.clock import Clock
from hadrsim.core.rng import Rng
from hadrsim.reward.compose import Composer
from hadrsim.reward.terms import View
from hadrsim.sensing import detector as det_mod
from hadrsim.sensing import fov
from hadrsim.world import fire as fire_mod
from hadrsim.world import nav
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import wind as wind_mod

CAPTURE_RADIUS_M = 3.0
HLR, LLR = 0, 1


class TrainEnv:
    """A batch of independent episodes sharing one city."""

    ALT_MIN = 2.0        # the ground
    ALT_MAX = 120.0      # the regulatory ceiling almost everywhere

    def __init__(self, scenario, batch: int = 256, detector: str = "oracle",
                 fidelity: str = "point_mass", weights: dict | None = None,
                 cameras=None, start_alt=None, capture_radius: float | None = None,
                 airframe=None, device="cpu"):
        """`cameras`, `start_alt`, `capture_radius` and `airframe` exist so a scenario
        can be built at the paper's scale -- a 25 m block, narrow sensors, teleporting
        drones -- without any of that leaking into the defaults."""
        from hadrsim.scenario import autoplace

        self.scenario = autoplace(scenario)
        self.batch = batch
        self.device = torch.device(device)
        self.clock = self.scenario.clock()
        self.obs_spec = spec.DEFAULT_OBS

        self.terrain = self.scenario.terrain().to(self.device)
        self.cell = self.terrain.cell_size_m

        self.fire = fire_mod.Fire(self.terrain, batch, device=self.device)
        self.smoke = smoke_mod.Smoke(self.terrain, batch, device=self.device)
        self.human = Human(self.terrain, batch, device=self.device)
        self.crowd = Crowd(self.terrain, device=self.device)
        frame = airframe or af.X500_V2
        self.drones = [
            Drone(batch, frame, fidelity, self.device),           # HLR, high and wide
            Drone(batch, frame, fidelity, self.device),           # LLR, low and narrow
        ]
        self.cameras = list(cameras) if cameras else [fov.DOWNWARD, fov.FORWARD]
        self.capture_radius = (CAPTURE_RADIUS_M if capture_radius is None
                               else capture_radius)
        self.detectors = [det_mod.make(detector, c) for c in self.cameras]
        self.reward = Composer(weights)

        self._start_alt = list(start_alt) if start_alt else [40.0, 4.0]
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self, mask: torch.Tensor | None = None) -> dict:
        """Start over. `mask` restarts only the environments that finished."""
        sc = self.scenario
        if mask is None:
            self.rng = Rng(sc.seed, self.device)
            self.rng_fire = self.rng.fork("fire")
            self.rng_human = self.rng.fork("human")
            self.rng_sense = self.rng.fork("sense")
            self.rng_wind = self.rng.fork("wind")
            self.wind = (wind_mod.still(self.batch, self.device) if sc.wind_mps <= 0 else
                         wind_mod.make(self.batch, sc.wind_mps, sc.wind_bearing_deg,
                                       device=self.device))
            self.step_count = torch.zeros(self.batch, dtype=torch.long, device=self.device)
            mask = torch.ones(self.batch, dtype=torch.bool, device=self.device)

        start = torch.tensor(sc.human_start[0], device=self.device)
        goal = torch.tensor(sc.human_goal[0], device=self.device)
        self.goal = goal.expand(self.batch, 2)

        # Whole-batch resets only, for now: the fire and the person both carry state
        # that would need per-environment surgery, and the policy is trained on fixed
        # -length episodes where everything finishes together anyway.
        self.fire.reset(torch.tensor([sc.fire_at], device=self.device).long()
                        .expand(self.batch, 2), radius_m=sc.fire_radius_m)
        self.human.reset(start.expand(self.batch, 2), goal.expand(self.batch, 2))
        self.smoke.update(self.fire, self.wind)

        base = torch.tensor(sc.drone_start, device=self.device)
        for i, drone in enumerate(self.drones):
            pos = torch.cat([base.expand(self.batch, 2),
                             torch.full((self.batch, 1), self._start_alt[i],
                                        device=self.device)], -1)
            drone.reset(pos)

        self.step_count.zero_()
        self.captured = torch.zeros(self.batch, dtype=torch.bool, device=self.device)
        self._want = torch.zeros(self.batch, 2, device=self.device)
        self._fire_near = nav.proximity(self.fire.intensity, 12.0, self.cell)
        self.searched = torch.zeros(self.batch, *self.terrain.shape, device=self.device)
        self._new_area = torch.zeros(self.batch, device=self.device)
        self._first_seen = torch.full((self.batch,), -1.0, device=self.device)
        self._last_pos = torch.zeros(self.batch, 3, device=self.device)
        self._last_age = torch.full((self.batch,), 1e3, device=self.device)
        return self.observe()

    # ------------------------------------------------------------- perception

    def _detect(self):
        """What each drone can see of the person, right now."""
        target = torch.cat([self.human.pos, torch.full((self.batch, 1), 0.9,
                                                       device=self.device)], -1)[:, None]
        vel = torch.cat([self.human.vel, torch.zeros(self.batch, 1, device=self.device)],
                        -1)[:, None]
        env = torch.arange(self.batch, device=self.device)
        return [d(self.terrain, drone, target, vel, self.smoke, self.rng_sense, env)
                for d, drone in zip(self.detectors, self.drones)]

    def _belief(self, detections):
        """One team view, assembled from whoever saw anything.

        Whichever drone is most confident wins.  This is the simplest possible fusion
        and stands in for the comms layer, which is where it belongs once messages can
        be late or lost.
        """
        conf = torch.stack([d.confidence[:, 0] for d in detections], -1)   # (B, D)
        pos = torch.stack([d.position[:, 0] for d in detections], 1)        # (B, D, 3)
        best = conf.argmax(-1)
        rows = torch.arange(self.batch, device=self.device)
        fresh, strength = pos[rows, best], conf.max(-1).values

        # Remember the last sighting, and how long ago it was.
        #
        # The contract has carried an `age` on every detection since the first commit
        # and nothing ever filled it in, so a policy that lost sight of somebody was
        # told only "nothing here" -- indistinguishable from never having seen them.
        # A searcher's most valuable piece of information is where the person was ten
        # seconds ago, and it was being discarded every step.
        seen = strength > 0
        self._last_pos = torch.where(seen[:, None], fresh, self._last_pos)
        self._last_age = torch.where(seen, torch.zeros_like(self._last_age),
                                     self._last_age + self.clock.control_dt)
        return fresh, strength, conf > 0

    def observe(self) -> dict:
        """The contract, per drone, in units a network can use. See `core/spec.py`.

        **Everything is scaled into roughly [-1, 1].**  Raw values span metres of map
        (0-300), altitudes (2-120), speeds and angles all at once, fed to a freshly
        initialised MLP -- and the few dozen numbers that matter, the detection slots,
        are lost among four thousand that do not.  The policy's exploration spread
        never shrank by so much as a percent in two million steps, in every run, on
        both problems: it was never getting a signal clear enough to become confident
        about anything.

        Scaled by *known constants* rather than running statistics, deliberately.  A
        real aircraft knows its own speed limit, its ceiling and the size of its
        search area; it does not know the mean and variance of a training
        distribution it was never part of.  Fixed scales are one less thing that has
        to be reproduced on hardware.
        """
        detections = self._detect()
        believed, confidence, seen_by = self._belief(detections)
        self._last = (detections, believed, confidence, seen_by)

        w_m, h_m = self.terrain.extent_m
        extent = torch.tensor([w_m, h_m, self.ALT_MAX], device=self.device)
        speed = self.drones[0].af.max_speed_mps
        half_pi = math.pi / 2

        out = []
        for i, drone in enumerate(self.drones):
            roll, pitch = drone.attitude
            other = self.drones[1 - i]
            reach = self.cameras[i].range_m
            own = torch.cat([
                drone.pos / extent,
                drone.vel / speed,
                (drone.yaw / math.pi)[:, None],
                (drone.pos[:, 2:3] / self.ALT_MAX),
                (roll / half_pi)[:, None], (pitch / half_pi)[:, None],
            ], -1)

            d = detections[i]
            rel = d.position - drone.pos[:, None, :]
            age = d.age_s.clone()
            # When nobody has eyes on right now, slot zero carries the team's memory:
            # where they were, and how stale that is.
            stale = (d.confidence[:, 0] <= 0) & (self._last_age < 30.0)
            rel[:, 0] = torch.where(stale[:, None], self._last_pos - drone.pos, rel[:, 0])
            age[:, 0] = torch.where(stale, self._last_age, age[:, 0])
            slots = torch.cat([(rel / reach).clamp(-2, 2), d.velocity / speed,
                               d.confidence[..., None],
                               (age[..., None] / 30.0).clamp(0, 1)], -1)

            mate = torch.cat([(other.pos - drone.pos) / extent,
                              torch.ones(self.batch, 1, device=self.device),
                              torch.zeros(self.batch, 1, device=self.device)], -1)
            goal3 = torch.cat([self.goal, torch.zeros(self.batch, 1, device=self.device)], -1)
            task = torch.cat([(goal3 - drone.pos) / extent,
                              self._time_left()[:, None]], -1)

            out.append({
                "self": own,
                "detections": slots,
                "teammate": mate,
                "raster": self._raster(drone),
                "task": task,
            })
        return {"hlr": out[HLR], "llr": out[LLR],
                "believed": believed, "confidence": confidence, "seen_by": seen_by}

    def _sweep(self) -> None:
        """Mark the ground the drones can currently see as searched.

        Accumulated and slowly forgotten, rather than a snapshot of the current
        footprint.  The contract calls this channel "where it has already looked" and
        gives it provenance OWN -- the drone's own memory -- so a snapshot is not what
        it says it is.  It is also the only thing in the observation that can tell a
        policy where *not* to bother going.

        Forgetting matters as much as remembering: without decay a policy sweeps the
        map once, marks everything, and then has no reason ever to look again -- while
        the person it is hunting has walked somewhere it checked two minutes ago.
        """
        decay = 0.995
        self.searched = self.searched * decay
        before = self.searched.sum(dim=(1, 2))

        w, h = self.terrain.shape
        xs = torch.arange(w, device=self.device) * self.cell
        ys = torch.arange(h, device=self.device) * self.cell
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")

        for drone, camera in zip(self.drones, self.cameras):
            reach = fov.ground_footprint(camera, drone.pos, drone.quat)
            dist = ((gx[None] - drone.pos[:, 0, None, None]) ** 2
                    + (gy[None] - drone.pos[:, 1, None, None]) ** 2).sqrt()
            self.searched = torch.maximum(
                self.searched, (dist <= reach[:, None, None]).float())

        # Fraction of the map newly covered, which is what pays for exploring.
        self._new_area = (self.searched.sum(dim=(1, 2)) - before).clamp_min(0) / (w * h)

    def _raster(self, drone) -> torch.Tensor:
        """The small picture centred on the drone.

        Buildings are prior knowledge -- a real team has a map, and so do we, from
        OpenStreetMap.  Fire and smoke are *observed*: masked to what the drone can
        actually see from where it is, because handing a policy a perfect fire map is
        the same oracle hole as handing it the person's position.
        """
        s = self.obs_spec
        size, mpc = s.raster_size, s.raster_metres_per_cell
        half = size // 2

        cx = (drone.pos[:, 0] / mpc).long()
        cy = (drone.pos[:, 1] / mpc).long()
        offsets = torch.arange(-half, half, device=self.device)
        xs = (cx[:, None] + offsets[None]).clamp(0, int(self.terrain.shape[0] * self.cell / mpc) - 1)
        ys = (cy[:, None] + offsets[None]).clamp(0, int(self.terrain.shape[1] * self.cell / mpc) - 1)

        gx = (xs * mpc / self.cell).long().clamp(0, self.terrain.shape[0] - 1)
        gy = (ys * mpc / self.cell).long().clamp(0, self.terrain.shape[1] - 1)
        rows = torch.arange(self.batch, device=self.device)[:, None, None]

        buildings = self.terrain.solid_top[gx[:, :, None], gy[:, None, :]] / 20.0
        fire = self.fire.intensity[rows, gx[:, :, None], gy[:, None, :]]
        smoke = self.smoke.column_density()[rows, gx[:, :, None], gy[:, None, :]] / 3.0

        # Fire and smoke are masked to what has actually been looked at. Handing a
        # policy a perfect hazard map is the same oracle hole as handing it the
        # person's position.
        looked = self.searched[rows, gx[:, :, None], gy[:, None, :]]
        return torch.stack([buildings, fire * looked, smoke * looked, looked], 1)

    def _time_left(self) -> torch.Tensor:
        return 1.0 - self.step_count.float() / self.clock.max_steps

    # ------------------------------------------------------------------- step

    def step(self, actions: torch.Tensor):
        """One policy decision. `actions` is (B, 2, 4). Returns obs, reward, done, info."""
        dt = self.clock.dt
        w_m, h_m = self.terrain.extent_m
        for _ in range(self.clock.substeps):
            for i, drone in enumerate(self.drones):
                drone.step(actions[:, i], dt)
                drone.confine(w_m, h_m, self.ALT_MIN, self.ALT_MAX)
            self.human.integrate(
                self.crowd.step(self.human.pos, self.human.vel, self._want, dt), dt)
            self.wind.step(dt, self.rng_wind)
            self.fire.step(dt, self.wind, self.rng_fire)

        self.smoke.update(self.fire, self.wind)
        # Computed once and shared. "How near is the fire" was being worked out three
        # times a step -- once for the person's panic and once per drone for the
        # reward -- over the whole map each time.
        self._fire_near = nav.proximity(self.fire.intensity, 12.0, self.cell)
        self.searched = torch.zeros(self.batch, *self.terrain.shape, device=self.device)
        self._new_area = torch.zeros(self.batch, device=self.device)
        self._first_seen = torch.full((self.batch,), -1.0, device=self.device)
        self._want = self.human.desired_velocity(
            self.fire, self.smoke, self.rng_human, self.clock.control_dt)
        self.human.expose_to(self.fire)
        self._sweep()

        # Capture is the low rescuer's job alone -- the high one cannot intercept,
        # which is the asymmetry the whole paper is about.
        llr = self.drones[LLR]
        reach = (llr.pos[:, :2] - self.human.pos).norm(dim=-1) < self.capture_radius
        newly = reach & ~self.captured
        self.captured |= reach
        self.human.led_by(llr.pos[:, :2], self.capture_radius)

        obs = self.observe()
        seen = obs["seen_by"].any(-1)
        newly_seen = seen & (self._first_seen < 0)
        self._first_seen = torch.where(newly_seen, self._time_left(), self._first_seen)

        reward = self.reward(self._view(actions, newly, newly_seen))
        self.step_count += 1

        done = ((self.step_count >= self.clock.max_steps)
                | self.human.reached | self.human.harmed)
        return obs, reward.total, done, {"parts": reward.summary(),
                                         "captured": self.captured,
                                         "reached": self.human.reached,
                                         "harmed": self.human.harmed}

    def _view(self, actions, newly_captured, newly_seen=None) -> View:
        _, believed, confidence, seen_by = self._last
        goal3 = torch.cat([self.goal, torch.zeros(self.batch, 1, device=self.device)], -1)
        fire_near = torch.stack([nav.sample(self._fire_near, d.pos[:, :2], self.cell)
                                 for d in self.drones], -1)
        return View(
            drone_pos=torch.stack([d.pos for d in self.drones], 1),
            drone_vel=torch.stack([d.vel for d in self.drones], 1),
            person_pos=self._last_pos,
            person_seen=seen_by.any(-1),
            belief_age=self._last_age,
            seen_by=seen_by,
            confidence=confidence,
            goal=goal3,
            captured=newly_captured,
            was_captured=self.captured,
            reached_goal=self.human.reached,
            fire_near=fire_near,
            action=actions,
            dt=self.clock.control_dt,
            harmed=self.human.harmed,
            new_area=self._new_area,
            newly_seen=(newly_seen if newly_seen is not None
                        else torch.zeros_like(self.human.reached)),
            time_left=self._time_left(),
        )
