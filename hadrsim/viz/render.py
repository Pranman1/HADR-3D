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
C_HLR = np.array([90, 170, 240])     # the high rescuer: wide view, cannot intercept
C_LLR = np.array([150, 220, 255])    # the low rescuer: the only one that can
C_SIGHT = np.array([70, 130, 190])   # the ground a camera is currently covering


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

    def frame(self, fire=None, smoke=None, human=None, drones=None, cameras=None,
              env: int = 0) -> None:
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

        if drones is not None:
            cell = self.terrain.cell_size_m
            from hadrsim.sensing import fov as fov_mod
            for i, drone in enumerate(drones):
                colour = C_HLR if i == 0 else C_LLR
                p = drone.pos[env].cpu().numpy() / cell
                # The ground the camera covers, so you can see what it is missing.
                if cameras is not None:
                    r = float(fov_mod.ground_footprint(
                        cameras[i], drone.pos, drone.quat)[env]) / cell
                    _ring(img, p[1] * self.scale, p[0] * self.scale,
                          int(r * self.scale), C_SIGHT)
                _disc(img, p[1] * self.scale, p[0] * self.scale,
                      max(2, int(self.scale * 1.6)), colour)

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


def _ring(img: np.ndarray, cx: float, cy: float, r: int, colour) -> None:
    """An outline, so a camera footprint shows what it covers without hiding it."""
    h, w = img.shape[:2]
    if r < 2:
        return
    x0, x1 = max(0, int(cx - r)), min(h, int(cx + r + 1))
    y0, y1 = max(0, int(cy - r)), min(w, int(cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1))
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    edge = (d <= r) & (d >= r - max(1.0, r * 0.06))
    patch = img[x0:x1, y0:y1]
    patch[edge] = patch[edge] * 0.45 + colour * 0.55


def _cross(img: np.ndarray, cx: float, cy: float, r: int, colour) -> None:
    h, w = img.shape[:2]
    cx, cy = int(cx), int(cy)
    t = max(1, r // 4)
    img[max(0, cx - t):min(h, cx + t), max(0, cy - r):min(w, cy + r)] = colour
    img[max(0, cx - r):min(h, cx + r), max(0, cy - t):min(w, cy + t)] = colour


def main(argv=None) -> None:
    """Fly a policy and write a video.  Deterministic by default: what the policy
    believes, not what its exploration noise happened to do.

        hadrsim-watch checkpoints/sweep_closeness.pt -o rescue.mp4
        hadrsim-watch last_sighting --city "Venice San Marco" --episodes 2
    """
    import argparse

    import torch

    from hadrsim.env import Env
    from hadrsim.policy import base
    from hadrsim.scenario import Scenario, autoplace

    p = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    p.add_argument("policy", help="checkpoint path, or hover / lawnmower / last_sighting")
    p.add_argument("--city", default="Downtown Berkeley")
    p.add_argument("--radius", type=float, default=100.0)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--launch", type=float, default=25.0)
    p.add_argument("--fidelity", default="quadrotor")
    p.add_argument("--detector", default="geometric")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("-o", "--out", default="rescue.mp4")
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = Env(autoplace(Scenario.preset(a.city, radius_m=a.radius, cell_size_m=2.0,
                                        episode_seconds=a.seconds, wind_mps=0.0, seed=0)),
              batch=1, fidelity=a.fidelity, detector=a.detector, device=dev)
    env.launch_offset_m = a.launch
    policy = base.load(a.policy, dev)
    rec = Recorder(env.terrain, scale=max(2, int(420 / max(env.terrain.shape))))
    tally = {"captured": 0, "reached": 0, "lost": 0}
    with torch.no_grad():
        for ep in range(a.episodes):
            env.scenario.seed = ep
            obs, state, got = env.reset(), None, {}
            for _ in range(env.clock.max_steps):
                act, _, _, state = policy.act(obs, state, deterministic=not a.stochastic)
                obs, _, _, info = env.step(act)
                rec.frame(env.fire, env.smoke, env.human, env.drones, env.cameras)
                got = {k: got.get(k, False) or bool(info[k][0]) for k in ("captured", "reached", "harmed")}
            tally["captured"] += got["captured"]; tally["reached"] += got["reached"]; tally["lost"] += got["harmed"]
            print(f"  episode {ep + 1}: {'captured' if got['captured'] else 'not captured':<13} "
                  f"{'reached safety' if got['reached'] else 'lost to the fire' if got['harmed'] else 'ran out of time'}")
    print(f"  {tally['captured']}/{a.episodes} captured, {tally['reached']}/{a.episodes} reached safety, "
          f"{tally['lost']}/{a.episodes} lost\n  wrote {rec.save(a.out, fps=a.fps)}")


if __name__ == "__main__":
    main()
