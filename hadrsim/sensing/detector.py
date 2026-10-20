"""Detectors: same inputs, same `Detections` out, so they swap by name.
oracle = exact pose when geometry allows; geometric = oracle plus misses, range error
and false alarms; pixel = a CNN over the rendered image (`pixel.py`)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hadrsim.core.rng import Rng
from hadrsim.sensing import fov as fov_mod
from hadrsim.sensing import occlusion

HUMAN_HEIGHT_M = 1.7


@dataclass
class Detections:
    """What one camera reports this instant."""

    position: torch.Tensor      # (B, N, 3) world, as believed -- not as true
    velocity: torch.Tensor      # (B, N, 3)
    confidence: torch.Tensor    # (B, N) zero means an empty slot
    age_s: torch.Tensor         # (B, N) how stale, in seconds
    is_true: torch.Tensor       # (B, N) bool -- for scoring only, never for policies

    @staticmethod
    def empty(batch: int, slots: int, device="cpu") -> "Detections":
        z = lambda *s: torch.zeros(*s, device=device)
        return Detections(z(batch, slots, 3), z(batch, slots, 3), z(batch, slots),
                          z(batch, slots), torch.zeros(batch, slots, dtype=torch.bool,
                                                       device=device))


class OracleDetector:
    """Exact pose whenever cone and line of sight allow. The paper's detector."""

    name = "oracle"

    def __init__(self, camera: fov_mod.Camera, slots: int = 4):
        self.camera = camera
        self.slots = slots

    def __call__(self, terrain, drone, targets, target_vel=None,
                 smoke=None, rng: Rng | None = None, env=None, fire=None) -> Detections:
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
    """The oracle plus a noise model: misses rising with range and smoke, range-scaled position error, false alarms."""

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
                 smoke=None, rng: Rng | None = None, env=None, fire=None) -> Detections:
        if rng is None:
            raise ValueError("a detector that can be wrong needs a generator passed in")
        batch, n = targets.shape[:2]
        device = targets.device

        seen, distance, offset = fov_mod.in_view(self.camera, drone.pos, drone.quat, targets)
        clarity = torch.ones_like(distance)
        if terrain is not None:
            _, clarity = occlusion.visible(terrain, drone.pos, targets, smoke, env)
        seen = seen & (clarity > 0)

        apparent = HUMAN_HEIGHT_M / distance.clamp_min(1.0)
        edge = 1.0 - (offset / self.camera.half_angle).clamp(0, 1) ** 4
        quality = (apparent / (HUMAN_HEIGHT_M / 8.0)).clamp(0, 1) * clarity * edge

        detected = seen & (rng.uniform(*quality.shape) > self.miss_rate + (1 - quality) * 0.9)

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
        """A spurious detection in an empty slot, somewhere along the camera axis."""
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
    if kind == "pixel":
        from hadrsim.sensing.pixel import PixelDetector
        return PixelDetector(camera, **kwargs)
    raise ValueError(f"no detector {kind!r}: oracle, geometric or pixel")
