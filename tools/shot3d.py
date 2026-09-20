"""Screenshots of the control panel's 3D view, headless.  Kept in the repo now --
the previous copy lived in /tmp and a reboot took it.

    python tools/shot3d.py [city] [policy-substring] [steps]
"""
import asyncio, sys
from playwright.async_api import async_playwright

CITY = sys.argv[1] if len(sys.argv) > 1 else "Downtown Berkeley"
POLICY = sys.argv[2] if len(sys.argv) > 2 else "overnight_seed0"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 28
OUT = "docs/shots"

async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        pg = await b.new_page(viewport={"width": 1500, "height": 900})
        errs = []
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        await pg.goto("http://localhost:8000", wait_until="networkidle")
        await pg.wait_for_timeout(2000)
        await pg.select_option("#preset", CITY)
        await pg.fill("#radius", "100"); await pg.fill("#cell", "2")
        opts = await pg.eval_on_selector_all("#policy option", "es=>es.map(e=>e.value)")
        await pg.select_option("#policy", next(o for o in opts if POLICY in o))
        await pg.click("#build")
        await pg.wait_for_function("document.getElementById('status').textContent === 'ready'", timeout=180000)
        for _ in range(STEPS):
            await pg.click("#step"); await pg.wait_for_timeout(90)
        await pg.click("#v3d"); await pg.wait_for_timeout(3500)
        await pg.evaluate("hadr3d.orbit.pitch = 0.66; hadr3d.orbit.yaw = -0.8")
        await pg.wait_for_timeout(700)
        await pg.screenshot(path=f"{OUT}/gui_3d_wide.png")
        for role, dist, pitch, yaw, drop in (("llr", 14, 0.18, 2.3, 0), ("hlr", 40, 0.25, 0.6, 8)):
            await pg.evaluate(f"""() => {{
              const d = hadr3d._last[0].drones.find(x => x.role === '{role}'), c = hadr3d.cell;
              hadr3d.target.set(d.x * c, d.z - {drop}, d.y * c);
              hadr3d.orbit.dist = {dist}; hadr3d.orbit.pitch = {pitch}; hadr3d.orbit.yaw = {yaw};
            }}""")
            await pg.wait_for_timeout(700)
            await pg.screenshot(path=f"{OUT}/gui_3d_{role}_closeup.png")
        print("bar:", await pg.eval_on_selector("#bar", "e => e.innerText.replace(/\\n/g,' | ')"))
        print("errors:", errs[:4] or "none")
        await b.close()
asyncio.run(main())
