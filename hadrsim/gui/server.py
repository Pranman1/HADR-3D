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

Two engines behind one socket.  Pick a policy and the scenario is stepped by
`PolicyRunner`, which *is* the training environment at batch 1 -- so what you are
watching is the world the policy was trained in, not a second implementation of it.
Pick none and it is `Runner`, which has no drones but will carry a crowd.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from hadrsim.policy.base import SCRIPTED
from hadrsim.runner import PolicyRunner, Runner
from hadrsim.scenario import PRESETS, Scenario
from hadrsim.viz.render import Recorder

app = FastAPI(title="hadrsim")
HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.get("/{name}.js")
def script(name: str) -> FileResponse:
    """three.js and the 3D scene, served from beside this file.

    Vendored rather than pulled from a CDN so the panel works on a machine with no
    internet, which a lab GPU box often is."""
    f = (HERE / f"{name}.js").resolve()
    if f.parent != HERE.resolve() or not f.is_file():
        raise ValueError("no such script")
    return FileResponse(f, media_type="text/javascript")


@app.get("/policies")
def policies() -> dict:
    """Checkpoints on disk, newest first, so the panel lists what actually exists."""
    root = Path.cwd() / "checkpoints"
    found = sorted(root.glob("*.pt"), key=lambda f: -f.stat().st_mtime) if root.is_dir() else []
    # The keep-best runs save a snapshot every 25 iterations (`*_i0025.pt` ...).  They
    # stay on disk for analysis but would bury the dropdown; the `_best` and final
    # checkpoints are the ones to fly.
    found = [f for f in found if not re.search(r"_i\d{4}$", f.stem)]
    scripted = [{"path": k, "name": f"baseline: {k}", "size_mb": 0.0} for k in SCRIPTED]
    return {"policies": scripted + [{"path": str(f), "name": f.stem,
                                     "size_mb": round(f.stat().st_size / 1e6, 1)} for f in found]}


@app.get("/presets")
def presets(radius_m: float = 100.0) -> dict:
    """Cities, flagged by whether their map is already on disk.

    An uncached city needs an Overpass query, and Overpass bans a host that asks too
    often -- so clicking one can hang for minutes with nothing on screen. Better to
    say so in the dropdown than to discover it in front of an audience.

    Checked at the radius actually in the form, because the cache is keyed on the
    bounding box: the same city at 100 m and at 180 m are two different downloads,
    and reporting on the preset's default radius marks cities unavailable that are
    sitting on disk at the radius about to be used.
    """
    from hadrsim.world import osm

    cache = Path.cwd() / "maps" / "osm_cache"
    out = []
    for name in PRESETS:
        sc = Scenario.preset(name, radius_m=radius_m)
        ready = True
        if sc.lat is not None:
            box = osm.bbox_around(sc.lat, sc.lon, radius_m)
            ready = (cache / f"{box.key(osm.QUERY)}.json").is_file()
        out.append({"name": name, "ready": ready})
    return {"presets": [p["name"] for p in out], "cities": out,
            "defaults": Scenario().to_dict()}


def _map_png(runner: Runner) -> str:
    """The static city, as a data URI. Sent once per scenario, not per frame."""
    from PIL import Image

    img = Recorder(runner.terrain, scale=1)._base
    buf = io.BytesIO()
    # Transposed because the renderer works in image order and the client draws in
    # world order; doing it here keeps the coordinate maths in the browser trivial.
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _heights(runner) -> dict:
    """Building heights, run-length encoded, for the 3D view to extrude.

    Sent once per scenario alongside the map.  A 300 m block at 2 m cells is 150x150
    numbers; raw JSON of that is about 90 kB and most of it is zero, so runs of equal
    height collapse it to a few kB.
    """
    h = runner.terrain.solid_top.cpu().numpy().T.round(1)
    flat, runs = h.reshape(-1), []
    prev, count = None, 0
    for v in flat.tolist():
        if v == prev:
            count += 1
        else:
            if prev is not None:
                runs.append([prev, count])
            prev, count = v, 1
    runs.append([prev, count])
    return {"w": int(h.shape[1]), "h": int(h.shape[0]), "runs": runs}


def _wind(runner) -> list:
    """The wind right now, as [east, north] m/s, for the smoke columns to lean with."""
    w = getattr(runner, "wind", None)
    if w is None:
        return [0.0, 0.0]
    v = w.velocity[0]
    return [round(float(v[0]), 2), round(float(v[1]), 2)]


def _flames(runner, cap: int = 600) -> list:
    """Where the fire actually is, as cells, for the 3D view to stand flames on.

    The ground overlay already carries this as a picture; a picture cannot be three
    dimensional.  Capped, and the hottest kept, because a late-stage fire can be
    thousands of cells and the frame budget is a few hundred.
    """
    heat = runner.fire.intensity[0]
    idx = (heat > 0.05).nonzero()
    if not len(idx):
        return []
    v = heat[idx[:, 0], idx[:, 1]]
    if len(idx) > cap:
        keep = v.topk(cap).indices
        idx, v = idx[keep], v[keep]
    return [[int(a), int(b), round(float(c), 2)]
            for (a, b), c in zip(idx.tolist(), v.tolist())]


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
        sc = Scenario.from_dict(cfg)
        pick = cfg.get("policy") or "none"
        if pick == "none":
            runner = await asyncio.to_thread(Runner, sc)
        else:
            runner = await asyncio.to_thread(
                PolicyRunner, sc, pick, "cpu",
                cfg.get("fidelity", "quadrotor"), cfg.get("detector", "geometric"))
        t = runner.terrain
        await ws.send_text(json.dumps({
            "type": "scenario",
            "width": t.shape[0], "height": t.shape[1],
            "cell": t.cell_size_m,
            "map": _map_png(runner),
            "heights": _heights(runner),
            "config": runner.scenario.to_dict(),
            "flying": bool(getattr(runner, "net", None) is not None),
        }))

    async def push() -> None:
        await ws.send_text(json.dumps({
            "type": "frame",
            "state": runner.snapshot().__dict__,
            "fire": _fire_png(runner),
            "flames": _flames(runner),
            "wind": _wind(runner),
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
                # Paced to the wall clock.  A policy runner can only advance in whole
                # decisions (0.5 s of simulated time), and pushing one every loop turn
                # ran the world ten times faster than real time in 0.5 s jumps -- which
                # is what "laggy and frame by frame" was.  Advance, then wait out the
                # simulated time at the chosen speed before the next frame.
                t0 = time.monotonic()
                before = runner.time
                await asyncio.to_thread(runner.advance, 0.05 * speed)
                await push()
                owed = (runner.time - before) / max(speed, 1e-3) - (time.monotonic() - t0)
                if owed > 0:
                    await asyncio.sleep(min(owed, 1.0))
            elif playing and runner and runner.done:
                playing = False
                await push()
            else:
                await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass


def _lan_ip() -> str:
    """This machine's address on the network, for opening the panel from a laptop."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def _free_port(start: int = 8000, tries: int = 12) -> int:
    """The first port nothing is already listening on.

    A panel left running from yesterday holds 8000, and starting a second one then
    fails to bind -- but the banner was printed *before* the bind was attempted, so
    it announced a URL that belonged to the old server and exited 0.  It looked like
    it had started.  Bind first, announce second, and move over if the port is busy.
    """
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"no free port in {start}-{start + tries - 1}")


def main() -> None:
    import uvicorn

    port = _free_port()
    if port != 8000:
        print(f"\n  note: something is already using port 8000 -- most likely a panel\n"
              f"        you left running. Using {port} instead; the old one still works.")
    print(f"\n  hadrsim control panel")
    print(f"    on this machine   http://localhost:{port}")
    print(f"    from another      http://{_lan_ip()}:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
