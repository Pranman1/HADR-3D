"""A 5x32x32 semantic image per drone (building, ground, fire, smoke, objects), made by
marching each pixel's ray.  The person and seeded clutter share the objects channel, so
nothing in the picture says "person".  On hardware a real camera replaces this."""

from __future__ import annotations

import math

import torch

from hadrsim.core import rotation as rot
from hadrsim.sensing import fov, occlusion
from hadrsim.sensing.occlusion import _sample

RES = 32            # pixels per side
STEPS = 24          # samples along each pixel's ray
CHANNELS = ("building", "ground", "fire", "smoke", "objects")
PERSON_HEIGHT_M, PERSON_WIDTH_M = 1.7, 0.5
CLUTTER = 48        # person-sized things on the ground per map: bins, posts, signs, hydrants


def basis(camera: fov.Camera, quat: torch.Tensor):
    """(axis, right, up), each (B, 3) in the world, through the airframe's attitude."""
    axis = fov.look_direction(camera, quat)
    right = rot.rotate(quat, torch.tensor([0.0, -1.0, 0.0], device=quat.device).expand(len(quat), 3))   # body +y is left
    return axis, right, torch.cross(right, axis, dim=-1)


def _grid(res: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel centres as (u, v) in [-1, 1], flattened row-major; v is up."""
    lin = (torch.arange(res, device=device) + 0.5) / res * 2 - 1
    v, u = torch.meshgrid(-lin, lin, indexing="ij")
    return u.reshape(-1), v.reshape(-1)


def rays(camera: fov.Camera, quat: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unit directions through image points (u, v): (B, P, 3)."""
    axis, right, up = basis(camera, quat)
    half = math.tan(camera.half_angle)
    d = axis[:, None] + half * (u[..., None] * right[:, None] + v[..., None] * up[:, None])
    return d / d.norm(dim=-1, keepdim=True)


def project(camera: fov.Camera, pos: torch.Tensor, quat: torch.Tensor, points: torch.Tensor):
    """World points (B, N, 3) -> image (u, v) in [-1, 1] and depth along the axis."""
    axis, right, up = basis(camera, quat)
    d = points - pos[:, None]
    depth = (d * axis[:, None]).sum(-1)
    half = math.tan(camera.half_angle) * depth.clamp_min(1e-3)
    return (d * right[:, None]).sum(-1) / half, (d * up[:, None]).sum(-1) / half, depth


def clutter(terrain, k: int = CLUTTER, seed: int = 0):
    """(K, 2) positions on open ground, (K,) heights, (K,) widths; seeded, fixed per map."""
    key = ("_clutter", k, seed)
    if not hasattr(terrain, "_clutter") or terrain._clutter[0] != key:
        g = torch.Generator().manual_seed(seed)
        walk = terrain.walkable.nonzero().cpu()
        pick = walk[torch.randint(len(walk), (k,), generator=g)].float() + torch.rand(k, 2, generator=g)
        dev = terrain.walkable.device
        terrain._clutter = (key, (pick * terrain.cell_size_m).to(dev),
                            (0.5 + 2.0 * torch.rand(k, generator=g)).to(dev), (0.3 + 1.2 * torch.rand(k, generator=g)).to(dev))
    return terrain._clutter[1:]


@torch.no_grad()
def render(camera: fov.Camera, drone, terrain, fire=None, smoke=None,
           person_pos: torch.Tensor | None = None, person_visible: torch.Tensor | None = None,
           res: int = RES, steps: int = STEPS, n_clutter: int = CLUTTER) -> torch.Tensor:
    """(B, 5, res, res) images from every drone in the batch."""
    B, dev, cell = len(drone.pos), drone.pos.device, terrain.cell_size_m
    u, v = _grid(res, dev)
    dirs = rays(camera, drone.quat, u[None].expand(B, -1), v[None].expand(B, -1))     # (B, P, 3)
    t = torch.linspace(0.5, camera.range_m, steps, device=dev)
    pts = drone.pos[:, None, None] + dirs[:, :, None] * t[None, None, :, None]        # (B, P, S, 3)

    hit_building = pts[..., 2] < _sample(terrain.solid_top, pts[..., :2], cell)
    hit = hit_building | (pts[..., 2] < 0)
    any_hit = hit.any(-1)
    first = torch.where(any_hit, hit.float().argmax(-1), torch.full_like(any_hit, steps - 1, dtype=torch.long))
    take = lambda f: f.gather(-1, first[..., None]).squeeze(-1)

    building = take(hit_building.float())
    ground = any_hit.float() - building                      # the ray met the ground, not a wall
    at_hit = torch.stack([take(pts[..., 0]), take(pts[..., 1])], -1)
    fire_ch = _sample(fire.intensity, at_hit, cell) * any_hit if fire is not None else torch.zeros_like(building)

    smoke_ch = torch.zeros_like(building)
    if smoke is not None:
        layer = torch.bucketize(pts[..., 2].contiguous(), smoke.layer_altitudes).clamp(0, smoke.density.shape[1] - 1)
        w, h = smoke.density.shape[-2:]
        x = (pts[..., 0] / cell).long().clamp(0, w - 1); y = (pts[..., 1] / cell).long().clamp(0, h - 1)
        env = torch.arange(B, device=dev).view(B, 1, 1).expand_as(x)
        dens = smoke.density[env, layer, x, y] * (torch.arange(steps, device=dev) <= first[..., None])
        smoke_ch = 1 - torch.exp(-dens.sum(-1) * float(t[1] - t[0]) * smoke.p.extinction)

    # Upright silhouettes: the person and the clutter, drawn the same way.
    xy, height, width, visible = [], [], [], []
    if person_pos is not None:
        xy.append(person_pos[:, None, :2]); height.append(torch.full((B, 1), PERSON_HEIGHT_M, device=dev))
        width.append(torch.full((B, 1), PERSON_WIDTH_M, device=dev))
        visible.append(person_visible[:, None] if person_visible is not None else torch.ones(B, 1, dtype=torch.bool, device=dev))
    if n_clutter:
        cxy, ch, cw = clutter(terrain, n_clutter)
        centre = torch.cat([cxy, ch[:, None] / 2], -1)[None].expand(B, -1, -1)
        seen, _, _ = fov.in_view(camera, drone.pos, drone.quat, centre)
        clear = seen.clone()
        if bool(seen.any()):
            clear, _ = occlusion.visible(terrain, drone.pos, centre, smoke, torch.arange(B, device=dev))
        xy.append(cxy[None].expand(B, -1, -1)); height.append(ch[None].expand(B, -1))
        width.append(cw[None].expand(B, -1)); visible.append(seen & clear)
    objects = torch.zeros_like(building)
    if xy:
        xy, height, width, visible = (torch.cat(t, 1) for t in (xy, height, width, visible))
        foot = torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)
        head = torch.cat([xy, height[..., None]], -1)
        pu, pv, depth = project(camera, drone.pos, drone.quat, torch.cat([foot, head], 1))   # (B, 2N)
        n = xy.shape[1]
        dist = (foot - drone.pos[:, None]).norm(dim=-1)
        half_w = (width / 2 / (dist * math.tan(camera.half_angle)).clamp_min(1e-3)).clamp_min(0.5 / res)
        u_c = (pu[:, :n] + pu[:, n:]) / 2
        v_lo, v_hi = torch.minimum(pv[:, :n], pv[:, n:]), torch.maximum(pv[:, :n], pv[:, n:])
        ok = visible & (depth[:, :n] > 0) & (depth[:, n:] > 0)
        inside = (((u[None, :, None] - u_c[:, None]).abs() <= half_w[:, None])
                  & (v[None, :, None] >= v_lo[:, None] - 0.5 / res) & (v[None, :, None] <= v_hi[:, None] + 0.5 / res)
                  & ok[:, None])                                                              # (B, P, N)
        objects = inside.any(-1).float()

    return torch.stack([building, ground, fire_ch, smoke_ch, objects], 1).view(B, len(CHANNELS), res, res)
