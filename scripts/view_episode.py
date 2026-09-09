"""Watch one episode of the world: fire, smoke, and a person trying to get out.

No drones yet -- this is step 1 of the plan, the world on its own.

    python scripts/view_episode.py --web
    python scripts/view_episode.py --osm "Downtown Berkeley, California" --web
    python scripts/view_episode.py --wind 6 --bearing 45 --web
"""

import argparse
import time

import torch

from hadrsim.agents.crowd import Crowd
from hadrsim.agents.human import Human
from hadrsim.core.clock import Clock
from hadrsim.core.rng import Rng
from hadrsim.viz import rerun_sink as viz
from hadrsim.world import fire as fire_mod
from hadrsim.world import osm
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import terrain as terr
from hadrsim.world import wind as wind_mod

p = argparse.ArgumentParser()
p.add_argument("--map", default="maps/rust25.yaml")
p.add_argument("--osm", default=None, metavar="PLACE")
p.add_argument("--radius", type=float, default=150.0)
p.add_argument("--cell", type=float, default=1.0)
p.add_argument("--wind", type=float, default=0.0, help="wind speed, m/s")
p.add_argument("--bearing", type=float, default=45.0, help="degrees, direction blown towards")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--web", action="store_true")
p.add_argument("--save", default=None)
a = p.parse_args()

t = osm.load(place=a.osm, radius_m=a.radius, cell_size_m=a.cell) if a.osm \
    else terr.load(a.map, cell_size_m=a.cell)
clock = Clock()
rng = Rng(a.seed)

fire = fire_mod.Fire(t, batch=1)
smoke = smoke_mod.Smoke(t, batch=1)
wind = wind_mod.still(1) if a.wind == 0 else wind_mod.make(1, a.wind, a.bearing)
human = Human(t, batch=1)
crowd = Crowd(t)

# Start the person and the fire at opposite ends of the walkable area, and send them
# to the far corner, so the fire has a chance to matter on the way.
walk = t.walkable.nonzero()
cell = t.cell_size_m
start = walk[walk.float().norm(dim=1).argmin()].float() * cell
goal = walk[walk.float().norm(dim=1).argmax()].float() * cell
burnable = (t.fuel > 0).nonzero()
ignite = burnable[len(burnable) // 2]

fire.reset(ignite[None])
human.reset(start[None], goal[None])

print(f"\n{a.osm or a.map}  {t.shape[0]}x{t.shape[1]} @ {cell} m")
print(clock.describe())
print(f"person {start.tolist()} -> {goal.tolist()},  fire at {ignite.tolist()}")
print(f"wind {a.wind} m/s towards {a.bearing} deg\n")

viz.start("hadrsim-episode", web=a.web, save=a.save)
viz.log_terrain(t)

dt = clock.control_dt
for step in range(clock.max_steps):
    now = step * dt
    wind.step(dt, rng)
    fire.step(dt, wind, rng)
    smoke.update(fire, wind)
    want = human.desired_velocity(fire, smoke, rng, dt)
    human.integrate(crowd.step(human.pos, human.vel, want, dt), dt)

    viz.log_time(now)
    viz.log_fire(fire)
    viz.log_smoke(smoke)
    viz.log_human(human)

    if step % 40 == 0:
        print(f"  t={now:6.1f}s  burning {int(fire.burning[0].sum()):>5}  "
              f"panic {float(human.gamma[0]):.2f}  "
              f"speed {float(human.vel[0].norm()):.2f} m/s"
              f"{'   REACHED' if human.reached[0] else ''}")
    if human.reached[0]:
        print(f"\n  reached safety at t={now:.1f}s")
        break
else:
    print(f"\n  never reached safety in {clock.episode_seconds:g}s")

if a.web:
    try:
        input("\nserving -- press enter to stop\n")
    except EOFError:
        while True:
            time.sleep(1)
