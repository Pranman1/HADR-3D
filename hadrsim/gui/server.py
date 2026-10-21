"""The control panel: `hadrsim-panel`, then http://localhost:8000.
FastAPI serves the page; a websocket streams frames from a `Runner` (crowd, no drones)
or a `PolicyRunner` (the gym at batch 1, flown by a policy)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from hadrsim.gui import encode
from hadrsim.policy.base import SCRIPTED
from hadrsim.runner import PolicyRunner, Runner
from hadrsim.scenario import PRESETS, Scenario

app = FastAPI(title="hadrsim")
HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.get("/{name}.js")
def script(name: str) -> FileResponse:
    """three.js and the 3D scene, vendored so the panel works with no internet."""
    f = (HERE / f"{name}.js").resolve()
    if f.parent != HERE.resolve() or not f.is_file():
        raise ValueError("no such script")
    return FileResponse(f, media_type="text/javascript")


@app.get("/policies")
def policies() -> dict:
    """Scripted baselines, then checkpoints on disk newest first; per-iteration snapshots hidden."""
    root = Path.cwd() / "checkpoints"
    found = sorted(root.glob("*.pt"), key=lambda f: -f.stat().st_mtime) if root.is_dir() else []
    found = [f for f in found if not re.search(r"_i\d{4}$", f.stem)]
    scripted = [{"path": k, "name": f"baseline: {k}", "size_mb": 0.0} for k in SCRIPTED]
    return {"policies": scripted + [{"path": str(f), "name": f.stem,
                                     "size_mb": round(f.stat().st_size / 1e6, 1)} for f in found]}


@app.get("/presets")
def presets(radius_m: float = 100.0) -> dict:
    """Cities, flagged by whether their map is cached at this radius."""
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


@app.websocket("/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    runner: Runner | None = None
    playing = False
    speed = 1.0

    async def build(cfg: dict) -> None:
        nonlocal runner, playing
        playing = False
        # Terrain fetch and rasterising block; off the event loop so the socket stays responsive.
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
            "map": encode.map_png(runner),
            "heights": encode.heights(runner),
            "config": runner.scenario.to_dict(),
            "flying": bool(getattr(runner, "net", None) is not None),
        }))

    async def push() -> None:
        await ws.send_text(json.dumps({
            "type": "frame",
            "state": runner.snapshot().__dict__,
            "fire": encode.fire_png(runner),
            "flames": encode.flames(runner),
            "wind": encode.wind(runner),
        }))

    try:
        while True:
            t0 = time.monotonic()
            try:
                msg = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=0.005 if playing else 0.05))
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
    """The first free port from 8000; bind before announcing."""
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
    print(f"    from another      http://{_lan_ip()}:{port}\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
