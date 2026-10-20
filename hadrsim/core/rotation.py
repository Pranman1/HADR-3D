"""Quaternion helpers for attitude: (w, x, y, z), body-to-world."""

from __future__ import annotations

import torch


def identity(batch: int, device="cpu") -> torch.Tensor:
    q = torch.zeros(batch, 4, device=device)
    q[:, 0] = 1.0
    return q


def normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Level, pointing along `yaw`."""
    half = yaw * 0.5
    z = torch.zeros_like(yaw)
    return torch.stack([half.cos(), z, z, half.sin()], dim=-1)


def from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = (roll * 0.5).cos(), (roll * 0.5).sin()
    cp, sp = (pitch * 0.5).cos(), (pitch * 0.5).sin()
    cy, sy = (yaw * 0.5).cos(), (yaw * 0.5).sin()
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)


def to_euler(q: torch.Tensor):
    """(roll, pitch, yaw) in radians."""
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) rotating body vectors into the world."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Body vector into the world."""
    return (to_matrix(q) @ v.unsqueeze(-1)).squeeze(-1)


def rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """World vector into the body."""
    return (to_matrix(q).transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def integrate(q: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Advance attitude by body rates over `dt`."""
    wx, wy, wz = omega.unbind(-1)
    zero = torch.zeros_like(wx)
    dq = multiply(q, torch.stack([zero, wx, wy, wz], dim=-1)) * 0.5
    return normalize(q + dq * dt)


def yaw_of(q: torch.Tensor) -> torch.Tensor:
    return to_euler(q)[2]


def tilt_of(q: torch.Tensor) -> torch.Tensor:
    """Angle between the drone's own up and the world's."""
    up = rotate(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(len(q), 3))
    return up[:, 2].clamp(-1.0, 1.0).acos()
