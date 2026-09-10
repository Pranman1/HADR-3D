"""Where a camera is pointing, and what falls inside it.

Geometry only -- whether a point lies in the cone -- with no opinion about whether
anything is in the way.  That is `occlusion`, and keeping them apart is what lets
one be swapped without the other.

Two cameras, and their difference is the paper's whole argument:

    downward   the high rescuer.  Wide, looking straight down, sees a lot of ground
               and nothing underneath anything.
    forward    the low rescuer.  Narrow, looking ahead and slightly down, sees
               under canopies and awnings that the high one cannot.

**The camera is bolted to the airframe.**  A quadrotor has to lean to accelerate, so
a drone in a hurry is looking somewhere other than where it was hovering -- which is
why `drone.attitude` exists and why the point-mass tier bothers to derive a tilt.
Modelling the camera as permanently level would quietly delete the most interesting
coupling in the problem: going fast costs you your view.
"""

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
    """(B, 3) unit vector the camera points along, in the world.

    Goes through the drone's attitude, not just its heading, so a leaning aircraft
    looks where a leaning aircraft looks.
    """
    return rot.rotate(quat, camera.axis(len(quat), quat.device))


def in_view(camera: Camera, pos: torch.Tensor, quat: torch.Tensor,
            targets: torch.Tensor) -> tuple:
    """Which targets fall inside the cone.

    `pos` and `quat` are (B, ...) for B cameras; `targets` is (B, N, 3).
    Returns (visible, range_m, offset_from_axis) -- the last two because a detector
    needs to know how far away and how far off-centre something is, not merely
    whether it is there.
    """
    delta = targets - pos[:, None, :]
    distance = delta.norm(dim=-1)
    direction = delta / distance.clamp_min(1e-6)[..., None]

    axis = look_direction(camera, quat)                       # (B, 3)
    cos_off = (direction * axis[:, None, :]).sum(-1).clamp(-1.0, 1.0)
    offset = cos_off.acos()

    visible = (offset <= camera.half_angle) & (distance <= camera.range_m)
    return visible, distance, offset


def ground_footprint(camera: Camera, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """(B,) radius on the ground the camera covers, in metres.

    Only meaningful for the downward camera, where the footprint is roughly a circle.
    Used for the hazard raster and for reporting; the actual visibility test is
    `in_view`, which does not approximate.
    """
    height = pos[:, 2].clamp_min(0.5)
    reach = torch.minimum(height / rot.tilt_of(quat).cos().clamp_min(0.2),
                          torch.full_like(height, camera.range_m))
    return reach * math.tan(camera.half_angle)
