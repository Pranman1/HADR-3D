"""Drawing an episode to a video file.

The other sink streams to a live 3D viewer, which is the better tool for poking at a
scene -- and which needs a server, two ports, a browser and a URL, all of which is a
lot of apparatus between you and "did that look right".  This one is a function
call: frames in, `.mp4` out, nothing running afterwards.

Top-down, because the thing being watched is fundamentally a plan view.  Where the
fire is, where the person is, and whether the two are about to meet all read better
from above than from any perspective camera, and altitude is drawn rather than
projected: buildings are shaded by height, smoke by column density.

Deliberately *not* the training loop's business.  Rendering four thousand parallel
environments would cost more than training them; this draws one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Muted city, hot fire, cool exit -- the same split the project's website uses.
C_GROUND = np.array([34, 34, 38])
C_FUEL = np.array([58, 70, 56])
C_BUILDING = np.array([104, 104, 112])
C_ROAD = np.array([48, 48, 52])
C_BURNT = np.array([26, 22, 22])
C_FIRE_COOL = np.array([196, 58, 22])
C_FIRE_HOT = np.array([255, 216, 128])
C_SMOKE = np.array([176, 176, 182])
C_CALM = np.array([228, 230, 236])
C_PANIC = np.array([236, 92, 58])
C_GOAL = np.array([92, 208, 136])


@dataclass
class Recorder:
    """Collects frames during a run, then writes them out.

    Frames are kept as small uint8 images rather than as matplotlib figures: a
    two-minute episode is a couple of thousand frames, and holding that many figures
    open is how a rendering script runs a machine out of memory.
    """

    terrain: object
    scale: int = 2                     # screen pixels per world cell
    frames: list = field(default_factory=list)
    _base: np.ndarray | None = None

    def __post_init__(self):
        self._base = self._background()

    # ---------------------------------------------------------------- drawing

    def _background(self) -> np.ndarray:
        """The static city, drawn once and copied for every frame."""
        t = self.terrain
        fuel = t.fuel.cpu().numpy().T
        solid = t.solid_top.cpu().numpy().T

        hi = max(float(fuel.max()), 1e-6)
        img = C_GROUND + (fuel / hi)[..., None] * (C_FUEL - C_GROUND)
        img = np.where((fuel <= 0.2)[..., None], C_ROAD, img)

        built = solid > 0
        if built.any():
            # Shade by height so a plan view still reads as a city rather than a maze.
            lift = 0.75 + 0.55 * np.clip(solid / max(solid.max(), 1.0), 0, 1)[..., None]
            img = np.where(built[..., None], C_BUILDING * lift, img)
        return img.clip(0, 255).astype(np.uint8)

    def frame(self, fire=None, smoke=None, human=None, env: int = 0) -> None:
        img = self._base.astype(np.float32).copy()

        if fire is not None:
            burnt = fire.burnt[env].cpu().numpy().T
            img[burnt] = C_BURNT
            heat = fire.intensity[env].cpu().numpy().T
            alight = heat > 0
            if alight.any():
                v = heat[alight][:, None]
                img[alight] = C_FIRE_COOL + v * (C_FIRE_HOT - C_FIRE_COOL)

        if smoke is not None:
            # Smoke veils what is underneath rather than replacing it, so you can see
            # the fire glowing through the plume.
            col = smoke.column_density()[env].cpu().numpy().T
            alpha = np.clip(col / 2.5, 0, 0.82)[..., None]
            img = img * (1 - alpha) + C_SMOKE * alpha

        img = np.repeat(np.repeat(img, self.scale, 0), self.scale, 1)

        if human is not None:
            cell = self.terrain.cell_size_m
            panic = float(human.gamma[env])
            body = C_CALM * (1 - panic) + C_PANIC * panic
            g = human.goal[env].cpu().numpy() / cell
            p = human.pos[env].cpu().numpy() / cell
            _cross(img, g[1] * self.scale, g[0] * self.scale, 5 * self.scale, C_GOAL)
            _disc(img, p[1] * self.scale, p[0] * self.scale, max(2, self.scale * 2), body)

        self.frames.append(img.clip(0, 255).astype(np.uint8))

    # ---------------------------------------------------------------- writing

    def save(self, path: str, fps: int = 25) -> str:
        """Write an mp4 (or a gif, by extension). Returns the path actually written."""
        import imageio.v3 as iio

        if not self.frames:
            raise RuntimeError("nothing recorded")
        if path.endswith(".gif"):
            iio.imwrite(path, self.frames, duration=1000 / fps, loop=0)
        else:
            iio.imwrite(path, self.frames, fps=fps, codec="libx264",
                        macro_block_size=None)
        return path


def _disc(img: np.ndarray, cx: float, cy: float, r: int, colour) -> None:
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(h, int(cx + r + 1))
    y0, y1 = max(0, int(cy - r)), min(w, int(cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1))
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[x0:x1, y0:y1][inside] = colour


def _cross(img: np.ndarray, cx: float, cy: float, r: int, colour) -> None:
    h, w = img.shape[:2]
    cx, cy = int(cx), int(cy)
    t = max(1, r // 4)
    img[max(0, cx - t):min(h, cx + t), max(0, cy - r):min(w, cy + r)] = colour
    img[max(0, cx - r):min(h, cx + r), max(0, cy - t):min(w, cy + t)] = colour
