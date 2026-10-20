"""Cameras bolted to the airframe: where one points through the drone's attitude, and
what falls inside its cone.  No occlusion here; that is `occlusion.py`."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from hadrsim.core import rotation as rot


@dataclass(frozen=True)
class Camera:
    """A camera, in the drone's own frame."""

    fov_deg: float = 60.0
    range_m: float = 60.0
    pitch_deg: float = 0.0       # where it points relative to the airframe
    kind: str = "forward"

    @property
    def half_angle(self) -> float:
        return math.radians(self.fov_deg) * 0.5

    def axis(self, batch: int, device="cpu") -> torch.Tensor:
        """(B, 3) the direction it looks, in the drone's frame."""
        if self.kind == "downward":
            base = torch.tensor([0.0, 0.0, -1.0], device=device)
        else:
            tilt = math.radians(self.pitch_deg)
            base = torch.tensor([math.cos(tilt), 0.0, -math.sin(tilt)], device=device)
        return base.expand(batch, 3)


DOWNWARD = Camera(fov_deg=90.0, range_m=80.0, kind="downward")
FORWARD = Camera(fov_deg=60.0, range_m=45.0, pitch_deg=20.0, kind="forward")


def look_direction(camera: Camera, quat: torch.Tensor) -> torch.Tensor:
    """(B, 3) unit vector the camera points along in the world, through the drone's attitude."""
    return rot.rotate(quat, camera.axis(len(quat), quat.device))


def in_view(camera: Camera, pos: torch.Tensor, quat: torch.Tensor,
            targets: torch.Tensor) -> tuple:
    """(visible, range_m, offset_from_axis) for targets (B, N, 3)."""
    delta = targets - pos[:, None, :]
    distance = delta.norm(dim=-1)
    direction = delta / distance.clamp_min(1e-6)[..., None]

    axis = look_direction(camera, quat)                       # (B, 3)
    cos_off = (direction * axis[:, None, :]).sum(-1).clamp(-1.0, 1.0)
    offset = cos_off.acos()

    visible = (offset <= camera.half_angle) & (distance <= camera.range_m)
    return visible, distance, offset


def ground_footprint(camera: Camera, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """(B,) radius on the ground a downward camera covers, in metres."""
    height = pos[:, 2].clamp_min(0.5)
    reach = torch.minimum(height / rot.tilt_of(quat).cos().clamp_min(0.2),
                          torch.full_like(height, camera.range_m))
    return reach * math.tan(camera.half_angle)
