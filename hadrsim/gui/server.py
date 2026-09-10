"""A control panel for scenarios.

    python -m hadrsim.gui

Pick a city, set the wind, drop the fire and the people where you want them, press
play, and watch it run.  The point is not the picture -- the video renderer already
makes pictures -- it is being able to *change something and see what happens*
without editing a file, which is how you find out whether a policy generalises.

Two halves.  The map is static per scenario, so it is sent once as a PNG and drawn
underneath.  Everything that moves is sent as JSON over a websocket, a few hundred
bytes a frame, and drawn on top.  A live 3D streaming viewer is the wrong shape for
this: it wants to own the window, and what is wanted here is a form beside a map.

The drone and policy controls are present and inert.  There is no policy yet; when
there is, it plugs into `Runner` and the panel already knows how to ask for it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from hadrsim.runner import Runner
from hadrsim.scenario import PRESETS, Scenario
from hadrsim.viz.render import Recorder

app = FastAPI(title="hadrsim")
HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.get("/presets")
def presets() -> dict:
    return {"presets": list(PRESETS), "defaults": Scenario().to_dict()}


def _map_png(runner: Runner) -> str:
    """The static city, as a data URI. Sent once per scenario, not per frame."""
    from PIL import Image

    img = Recorder(runner.terrain, scale=1)._base
    buf = io.BytesIO()
    # Transposed because the renderer works in image order and the client draws in
    # world order; doing it here keeps the coordinate maths in the browser trivial.
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _fire_png(runner: Runner) -> str:
    """Fire, burn scar and smoke as one RGBA overlay, at cell resolution."""
    from PIL import Image
    from hadrsim.viz.render import C_BURNT, C_FIRE_COOL, C_FIRE_HOT, C_SMOKE

    heat = runner.fire.intensity[0].cpu().numpy().T
    burnt = runner.fire.burnt[0].cpu().numpy().T
    col = runner.smoke.column_density()[0].cpu().numpy().T

    h, w = heat.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[burnt, :3] = C_BURNT
    rgba[burnt, 3] = 255

    alight = heat > 0
    if alight.any():
        v = heat[alight][:, None]
        rgba[alight, :3] = C_FIRE_COOL + v * (C_FIRE_HOT - C_FIRE_COOL)
        rgba[alight, 3] = 255

    smoke_a = np.clip(col / 2.5, 0, 0.8) * 255
    veil = smoke_a > rgba[..., 3]
    rgba[veil, :3] = C_SMOKE
    rgba[veil, 3] = smoke_a[veil]

    buf = io.BytesIO()
    Image.fromarray(rgba.clip(0, 255).astype(np.uint8), mode="RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.websocket("/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    runner: Runner | None = None
    playing = False
    speed = 1.0

    async def build(cfg: dict) -> None:
        nonlocal runner, playing
        playing = False
        # Terrain fetch and rasterising blocks; off the event loop so the socket
        # stays responsive and the client can show "loading" rather than freezing.
        runner = await asyncio.to_thread(Runner, Scenario.from_dict(cfg))
        t = runner.terrain
        await ws.send_text(json.dumps({
            "type": "scenario",
            "width": t.shape[0], "height": t.shape[1],
            "cell": t.cell_size_m,
            "map": _map_png(runner),
            "config": runner.scenario.to_dict(),
        }))

    async def push() -> None:
        await ws.send_text(json.dumps({
            "type": "frame",
            "state": runner.snapshot().__dict__,
            "fire": _fire_png(runner),
        }))

    try:
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=0.05))
            except asyncio.TimeoutError:
                msg = None

            if msg:
                kind = msg.get("type")
                if kind == "build":
                    await build(msg["config"])
                    await push()
                elif kind == "play":
                    playing = True
                elif kind == "pause":
                    playing = False
                elif kind == "speed":
                    speed = float(msg["value"])
                elif kind == "reset" and runner:
                    playing = False
                    runner.reset()
                    await push()
                elif kind == "step" and runner:
                    runner.advance(0.5)
                    await push()

            if playing and runner and not runner.done:
                await asyncio.to_thread(runner.advance, 0.05 * speed)
                await push()
            elif playing and runner and runner.done:
                playing = False
                await push()
            else:
                await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass


def main() -> None:
    import socket

    import uvicorn

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except OSError:
        ip = "localhost"
    finally:
        s.close()

    print(f"\n  hadrsim control panel")
    print(f"    on this machine   http://localhost:8000")
    print(f"    from another      http://{ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
