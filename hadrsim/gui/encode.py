"""What the panel sends down the wire: the map once, then a small frame each tick."""

from __future__ import annotations

import base64
import io

import numpy as np

from hadrsim.viz.render import C_BURNT, C_FIRE_COOL, C_FIRE_HOT, C_SMOKE, Recorder


def _png(array: np.ndarray, mode: str | None = None) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def map_png(runner) -> str:
    """The static city."""
    return _png(Recorder(runner.terrain, scale=1)._base)


def heights(runner) -> dict:
    """Building heights, run-length encoded (150x150 cells of mostly zeros is 90 kB raw)."""
    flat = runner.terrain.solid_top.cpu().numpy().T.round(1)
    runs, prev = [], None
    for v in flat.reshape(-1).tolist():
        if v == prev:
            runs[-1][1] += 1
        else:
            runs.append([v, 1]); prev = v
    return {"w": int(flat.shape[1]), "h": int(flat.shape[0]), "runs": runs}


def wind(runner) -> list:
    """[east, north] m/s right now, for the smoke columns to lean with."""
    v = runner.wind.velocity[0]
    return [round(float(v[0]), 2), round(float(v[1]), 2)]


def flames(runner, cap: int = 600) -> list:
    """The hottest burning cells as [x, y, heat]; the 3D view stands flames on them."""
    heat = runner.fire.intensity[0]
    idx = (heat > 0.05).nonzero()
    if not len(idx):
        return []
    v = heat[idx[:, 0], idx[:, 1]]
    if len(idx) > cap:
        keep = v.topk(cap).indices
        idx, v = idx[keep], v[keep]
    return [[int(a), int(b), round(float(c), 2)] for (a, b), c in zip(idx.tolist(), v.tolist())]


def fire_png(runner) -> str:
    """Fire, burn scar and smoke as one RGBA overlay at cell resolution."""
    heat = runner.fire.intensity[0].cpu().numpy().T
    burnt = runner.fire.burnt[0].cpu().numpy().T
    col = runner.smoke.column_density()[0].cpu().numpy().T

    rgba = np.zeros(heat.shape + (4,), dtype=np.float32)
    rgba[burnt, :3], rgba[burnt, 3] = C_BURNT, 255
    alight = heat > 0
    rgba[alight, :3] = C_FIRE_COOL + heat[alight][:, None] * (C_FIRE_HOT - C_FIRE_COOL)
    rgba[alight, 3] = 255
    smoke_a = np.clip(col / 2.5, 0, 0.8) * 255
    veil = smoke_a > rgba[..., 3]
    rgba[veil, :3], rgba[veil, 3] = C_SMOKE, smoke_a[veil]
    return _png(rgba.clip(0, 255).astype(np.uint8), mode="RGBA")
