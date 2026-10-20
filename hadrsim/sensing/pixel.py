"""The pixel detector: a small CNN over the rendered camera image, trained offline on
oracle labels.

    python -m hadrsim.sensing.pixel      # train; weights go to configs/pixel_detector.pt
    hadrsim-train --detector pixel"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hadrsim.sensing import camera as cam
from hadrsim.sensing import fov, occlusion
from hadrsim.sensing.detector import Detections

WEIGHTS = Path(__file__).resolve().parents[1] / "configs" / "pixel_detector.pt"


class Net(nn.Module):
    """5x32x32 -> (logit seen, u, v, box height, range / camera range)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(len(cam.CHANNELS), 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(),
            nn.Linear(64, 5))

    def forward(self, img):
        return self.conv(img)


def labels(camera, drone, targets):
    """What the oracle knows about the frame: seen, u, v, height, range fraction."""
    seen, dist, _ = fov.in_view(camera, drone.pos, drone.quat, targets)
    foot = torch.cat([targets[:, 0, :2], torch.zeros_like(targets[:, 0, :1])], -1)
    head = foot + torch.tensor([0.0, 0.0, cam.PERSON_HEIGHT_M], device=foot.device)
    u, v, _ = cam.project(camera, drone.pos, drone.quat, torch.stack([foot, head], 1))
    box = torch.stack([u.mean(-1), v.mean(-1), v[:, 1] - v[:, 0], dist[:, 0] / camera.range_m], -1)
    return seen[:, 0], box


class PixelDetector:
    """Same inputs and output as the other detectors; the measurement comes from the rendered image."""

    name = "pixel"

    def __init__(self, camera: fov.Camera, slots: int = 4, weights: str | Path = WEIGHTS, threshold: float = 0.5):
        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights}: train the pixel detector first (python -m hadrsim.sensing.pixel)")
        self.camera, self.slots, self.threshold = camera, slots, threshold
        self.net = Net()
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.eval()

    @torch.no_grad()
    def __call__(self, terrain, drone, targets, target_vel=None, smoke=None, rng=None, env=None, fire=None) -> Detections:
        B, dev = len(targets), targets.device
        if next(self.net.parameters()).device != dev:
            self.net.to(dev)
        seen, _, _ = fov.in_view(self.camera, drone.pos, drone.quat, targets)
        clear, _ = occlusion.visible(terrain, drone.pos, targets, smoke, env)
        truly = (seen & clear)[:, 0]
        img = cam.render(self.camera, drone, terrain, fire, smoke, targets[:, 0], truly)
        out = self.net(img)
        conf = torch.sigmoid(out[:, 0])
        found = conf > self.threshold
        direction = cam.rays(self.camera, drone.quat, out[:, 1:2], out[:, 2:3])[:, 0]
        position = drone.pos + direction * (out[:, 4:5].clamp(0.02, 1.2) * self.camera.range_m)

        det = Detections.empty(B, self.slots, dev)
        det.position[:, 0] = position * found[:, None]
        det.confidence[:, 0] = conf * found
        det.is_true[:, 0] = found & truly
        return det


def collect(env, n: int, launch=(5.0, 60.0)):
    """Rendered frames and oracle labels from random poses in `env`."""
    imgs, ys, boxes = [], [], []
    while sum(len(i) for i in imgs) < n:
        env.launch_offset_m = float(torch.empty(1).uniform_(*launch))
        env.reset()
        for _ in range(int(torch.randint(0, 6, (1,)))):
            env.step(torch.rand(env.batch, 2, 4, device=env.device) * 2 - 1)
        target = torch.cat([env.human.pos, torch.full((env.batch, 1), 0.9, device=env.device)], -1)[:, None]
        for camera, drone in zip(env.cameras, env.drones):
            seen, box = labels(camera, drone, target)
            clear, _ = occlusion.visible(env.terrain, drone.pos, target, env.smoke, torch.arange(env.batch, device=env.device))
            truly = seen & clear[:, 0]
            imgs.append(cam.render(camera, drone, env.terrain, env.fire, env.smoke, target[:, 0], truly).half())
            ys.append(truly); boxes.append(box)
    return torch.cat(imgs)[:n], torch.cat(ys)[:n], torch.cat(boxes)[:n]


def train(frames: int = 40000, epochs: int = 12, city: str = "Downtown Berkeley", batch: int = 256,
          device: str = "cuda", out: Path = WEIGHTS, seed: int = 0) -> dict:
    from hadrsim.env import Env
    from hadrsim.scenario import Scenario, autoplace

    torch.manual_seed(seed)
    env = Env(autoplace(Scenario.preset(city, radius_m=100.0, cell_size_m=2.0, seed=seed)),
              batch=batch, detector="oracle", fidelity="point_mass", device=device)
    img, y, box = collect(env, frames)
    n_val = frames // 10
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(frames - n_val, device=device) + n_val
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]
            pred = net(img[idx].float())
            seen = y[idx].float()
            loss = F.binary_cross_entropy_with_logits(pred[:, 0], seen)
            loss = loss + (F.smooth_l1_loss(pred[:, 1:], box[idx], reduction="none").sum(-1) * seen).sum() / seen.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(img[:n_val].float()); found = pred[:, 0] > 0; truth = y[:n_val]
            recall = (found & truth).sum() / truth.sum().clamp_min(1)
            false_alarm = (found & ~truth).sum() / (~truth).sum().clamp_min(1)
            range_err = ((pred[:, 4] - box[:n_val, 3]).abs() * truth).sum() / truth.sum().clamp_min(1)
        print(f"epoch {epoch + 1:>2}  recall {recall:.2f}  false alarm {false_alarm:.3f}  "
              f"range error {range_err * 100:.1f}% of camera range  (visible in {truth.float().mean():.0%} of frames)", flush=True)
    out.parent.mkdir(exist_ok=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
    print(f"wrote {out}")
    return {"recall": float(recall), "false_alarm": float(false_alarm), "range_error": float(range_err)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--city", default="Downtown Berkeley")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    train(a.frames, a.epochs, a.city, device=a.device)
