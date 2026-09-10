"""Spotting people.

The seam the whole project turns on.  The 2D simulator's detection is a membership
test -- `OBJECT_TO_IDX["evader"] in fov` -- and when it passes, the agent is handed
the person's exact integer pose.  Inside the window: perfect truth.  Outside:
nothing.  No falloff with range, no partial confidence, no misses, no false alarms.
A camera at sixty metres does not behave like that, and a policy trained against one
that does has never had to handle uncertainty.

Three detectors, one signature, so swapping them is a scenario edit:

    oracle       exact pose whenever geometry allows.  Parity with the paper, and the
                 source of ground-truth labels for training the learned one.
    geometric    field of view and line of sight, then: detection probability falling
                 with range and smoke, position error growing with range, false
                 alarms from clutter, and confidence reported honestly.
    learned      a frozen CNN over rendered images.  Same output type. Not yet built.

**Everything returns the same `Detections`**, carrying confidence and age.  The
oracle simply reports confidence one and age zero.  That is what turns "make
perception realistic" from a rewrite into a line of config -- and it is why the
fields exist from the very first commit rather than being added when they are
needed.

Range comes from two cues that a real detector gets for free once it has a box
around someone, and they are worth naming because their *disagreement* is the
confidence signal:

  - **box height**: known human height over apparent height.  Always available, and
    badly wrong when the legs are hidden, which reports the person as further away.
  - **where the feet meet the ground**: with the camera pose known, the ray through
    the feet pixel meets the ground plane at exactly one point.  More accurate, and
    only available when the feet are visible.

Agreeing means a clean sighting; disagreeing means the person is probably half
hidden, so confidence drops and the policy can decide to go and look properly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hadrsim.core.rng import Rng
from hadrsim.sensing import fov as fov_mod
from hadrsim.sensing import occlusion

HUMAN_HEIGHT_M = 1.7


@dataclass
class Detections:
    """What one camera reports this instant. All (B, N, ...) over N slots."""

    position: torch.Tensor      # (B, N, 3) world, as believed -- not as true
    velocity: torch.Tensor      # (B, N, 3)
    confidence: torch.Tensor    # (B, N) zero means an empty slot
    age_s: torch.Tensor         # (B, N) how stale, in seconds
    is_true: torch.Tensor       # (B, N) bool -- for scoring only, never for policies

    @property
    def any_seen(self) -> torch.Tensor:
        return (self.confidence > 0).any(-1)

    @staticmethod
    def empty(batch: int, slots: int, device="cpu") -> "Detections":
        z = lambda *s: torch.zeros(*s, device=device)
        return Detections(z(batch, slots, 3), z(batch, slots, 3), z(batch, slots),
                          z(batch, slots), torch.zeros(batch, slots, dtype=torch.bool,
                                                       device=device))


class OracleDetector:
    """Exact pose whenever the geometry allows it. The paper's detector.

    Still honours field of view and line of sight -- what it does not do is make the
    drone earn the *measurement*.  Keeping the geometry means switching to the
    geometric detector changes only the noise, not the visibility, so the difference
    between them is attributable.
    """

    name = "oracle"

    def __init__(self, camera: fov_mod.Camera, slots: int = 4):
        self.camera = camera
        self.slots = slots

    def __call__(self, terrain, drone, targets, target_vel=None,
                 smoke=None, rng: Rng | None = None, env=None) -> Detections:
        batch, n = targets.shape[:2]
        det = Detections.empty(batch, self.slots, targets.device)

        seen, distance, _ = fov_mod.in_view(self.camera, drone.pos, drone.quat, targets)
        if smoke is not None or terrain is not None:
            clear, _ = occlusion.visible(terrain, drone.pos, targets, smoke, env)
            seen = seen & clear

        keep = min(n, self.slots)
        order = torch.where(seen, distance, torch.full_like(distance, 1e9)).argsort(-1)
        idx = order[:, :keep]
        rows = torch.arange(batch, device=targets.device)[:, None]

        det.position[:, :keep] = targets[rows, idx]
        det.confidence[:, :keep] = seen[rows, idx].float()
        det.is_true[:, :keep] = seen[rows, idx]
        if target_vel is not None:
            det.velocity[:, :keep] = target_vel[rows, idx]
        det.position[:, :keep] *= det.confidence[:, :keep, None]
        return det


class GeometricDetector(OracleDetector):
    """A camera that can be wrong, in the ways a camera is wrong.

    Not a learned model -- it is a noise model driven by geometry, which is itself an
    assumption.  Real detectors fail in ways correlated with the scene: someone in
    shadow, against clutter, half behind a car.  The plan's answer is to *fit* these
    numbers to a measured detector at step 5 rather than to keep guessing them, and
    until then this is a considered guess and should be described as one.
    """

    name = "geometric"

    def __init__(self, camera, slots: int = 4, miss_rate: float = 0.08,
                 false_alarm_rate: float = 0.02, position_error_m: float = 0.02,
                 min_confidence: float = 0.15):
        super().__init__(camera, slots)
        self.miss_rate = miss_rate
        self.false_alarm_rate = false_alarm_rate
        self.position_error_m = position_error_m
        self.min_confidence = min_confidence

    def __call__(self, terrain, drone, targets, target_vel=None,
                 smoke=None, rng: Rng | None = None, env=None) -> Detections:
        if rng is None:
            raise ValueError("a detector that can be wrong needs a generator passed in")
        batch, n = targets.shape[:2]
        device = targets.device

        seen, distance, offset = fov_mod.in_view(self.camera, drone.pos, drone.quat, targets)
        clarity = torch.ones_like(distance)
        if terrain is not None:
            _, clarity = occlusion.visible(terrain, drone.pos, targets, smoke, env)
        seen = seen & (clarity > 0)

        # How many pixels tall a person is, which is what both range cues rest on.
        # Detection falls away as that shrinks, and as smoke eats the contrast.
        apparent = HUMAN_HEIGHT_M / distance.clamp_min(1.0)
        edge = 1.0 - (offset / self.camera.half_angle).clamp(0, 1) ** 4
        quality = (apparent / (HUMAN_HEIGHT_M / 8.0)).clamp(0, 1) * clarity * edge

        detected = seen & (rng.uniform(*quality.shape) > self.miss_rate + (1 - quality) * 0.9)

        # Position error grows with range because it is an angular measurement: the
        # same pixel of error is a bigger mistake further away.
        sigma = self.position_error_m * distance
        noisy = targets + rng.normal(*targets.shape) * sigma[..., None]

        confidence = (quality * detected.float()).clamp_min(0.0)
        confidence = torch.where(confidence < self.min_confidence,
                                 torch.zeros_like(confidence), confidence)

        det = Detections.empty(batch, self.slots, device)
        keep = min(n, self.slots)
        order = torch.where(confidence > 0, -confidence,
                            torch.full_like(confidence, 1e9)).argsort(-1)
        idx = order[:, :keep]
        rows = torch.arange(batch, device=device)[:, None]

        picked = confidence[rows, idx]
        det.position[:, :keep] = noisy[rows, idx] * (picked > 0)[..., None]
        det.confidence[:, :keep] = picked
        det.is_true[:, :keep] = picked > 0
        if target_vel is not None:
            det.velocity[:, :keep] = target_vel[rows, idx] * (picked > 0)[..., None]

        self._add_false_alarms(det, drone, rng)
        return det

    def _add_false_alarms(self, det: Detections, drone, rng: Rng) -> None:
        """Clutter that looks like a person, in an otherwise empty slot.

        A detector that only ever misses teaches a policy that any sighting is real.
        Somewhere to walk toward that turns out to be a bin bag is a different and
        more interesting problem, and it is one real detectors create constantly.
        """
        batch, slots = det.confidence.shape
        empty = det.confidence <= 0
        fire = rng.uniform(batch, slots) < self.false_alarm_rate
        spurious = empty & fire
        if not bool(spurious.any()):
            return

        direction = fov_mod.look_direction(self.camera, drone.quat)[:, None, :]
        reach = rng.uniform(batch, slots, low=0.2, high=1.0) * self.camera.range_m
        jitter = rng.normal(batch, slots, 3) * 3.0
        ghost = drone.pos[:, None, :] + direction * reach[..., None] + jitter

        det.position[spurious] = ghost[spurious]
        det.confidence[spurious] = rng.uniform(batch, slots)[spurious] * 0.45 + 0.15
        det.is_true[spurious] = False


def make(kind: str, camera: fov_mod.Camera, **kwargs):
    if kind == "oracle":
        return OracleDetector(camera, **kwargs)
    if kind == "geometric":
        return GeometricDetector(camera, **kwargs)
    raise ValueError(f"no detector {kind!r} (learned arrives at step 5)")
