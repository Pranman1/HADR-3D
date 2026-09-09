"""Watch one episode of the world: fire, smoke, and a person trying to get out.

No drones yet -- this is step 1 of the plan, the world on its own.

Defaults to a real city, because that is the point of the thing.  `--paper` switches
to the published 25 x 25 grid, which exists only for comparing against Table I.

    python scripts/view_episode.py --web
    python scripts/view_episode.py --osm "Financial District, San Francisco" --web
    python scripts/view_episode.py --wind 8 --bearing 45 --web
    python scripts/view_episode.py --paper --web
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
from hadrsim.world import nav
from hadrsim.world import osm
from hadrsim.world import smoke as smoke_mod
from hadrsim.world import terrain as terr
from hadrsim.world import wind as wind_mod

p = argparse.ArgumentParser()
p.add_argument("--osm", default="Downtown Berkeley, California", metavar="PLACE")
p.add_argument("--paper", action="store_true",
               help="use the paper's 25x25 grid instead of a real city")
p.add_argument("--map", default="maps/rust25.yaml")
p.add_argument("--radius", type=float, default=180.0)
p.add_argument("--cell", type=float, default=1.0)
p.add_argument("--wind", type=float, default=0.0, help="wind speed, m/s")
p.add_argument("--bearing", type=float, default=45.0, help="degrees, direction blown towards")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--web", action="store_true")
p.add_argument("--save", default=None)
p.add_argument("--speed", type=float, default=2.0,
               help="playback speed once the episode is recorded")
p.add_argument("--paused", action="store_true", help="do not auto-play")
a = p.parse_args()

t = terr.load(a.map, cell_size_m=a.cell) if a.paper \
    else osm.load(place=a.osm, radius_m=a.radius, cell_size_m=a.cell)
clock = Clock()
rng = Rng(a.seed)

fire = fire_mod.Fire(t, batch=1)
smoke = smoke_mod.Smoke(t, batch=1)
wind = wind_mod.still(1) if a.wind == 0 else wind_mod.make(1, a.wind, a.bearing)
human = Human(t, batch=1)
crowd = Crowd(t)

# Choosing a start, a goal and an ignition point that actually make an episode.
# On the paper's 25 m block anything works; on a real city it does not, and both of
# the obvious choices fail:
#   - opposite corners of a 360 m map is a 500 m walk, which nobody finishes in 120 s
#   - a burnable cell picked at random is often an isolated scrap of front garden
#     surrounded by road, so the fire burns it and goes out
cell = t.cell_size_m
walk = t.walkable.nonzero()
start_cell = walk[(walk[:, 0] + walk[:, 1]).argmin()]
start = start_cell.float() * cell

# Goal: as far as the person could plausibly walk in the time available, measured
# along the streets rather than through the buildings.
reach = nav.distance_field(t.walkable, start_cell[None], cell)[0]
budget = 0.55 * Human(t, 1).p.v_calm * clock.episode_seconds
ok = t.walkable & (reach < nav.UNREACHABLE / 2)
goal_cell = ok.nonzero()[(reach[ok] - budget).abs().argmin()]
goal = goal_cell.float() * cell

# Ignition: near the midpoint of that route, and somewhere with enough around it to
# burn.  Fuel density over a 15 m window is what separates a city block from a verge.
mid = (start_cell + goal_cell).float() / 2
dense = torch.nn.functional.avg_pool2d(
    (t.fuel > 0).float()[None, None], 15, 1, 7)[0, 0]
candidates = ((t.fuel > 0) & (dense > 0.5)).nonzero().float()
if not len(candidates):
    candidates = (t.fuel > 0).nonzero().float()
ignite = candidates[(candidates - mid).norm(dim=1).argmin()].long()

fire.reset(ignite[None])
human.reset(start[None], goal[None])

print(f"\n{a.map if a.paper else a.osm}  {t.shape[0]}x{t.shape[1]} @ {cell} m")
print(clock.describe())
print(f"person {start.tolist()} -> {goal.tolist()} "
      f"({float(reach[goal_cell[0], goal_cell[1]]):.0f} m on foot), fire at {ignite.tolist()}")
print(f"wind {a.wind} m/s towards {a.bearing} deg\n")

viz.start("hadrsim-episode", web=a.web, save=a.save,
          play=not a.paused, speed=a.speed)
viz.log_terrain(t)

dt = clock.control_dt
steps_logged = 0
arrived_at = None
for step in range(clock.max_steps):
    steps_logged += 1
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
    if human.reached[0] and arrived_at is None:
        arrived_at = now
        print(f"  t={now:6.1f}s  REACHED SAFETY")

# Deliberately keep simulating after the rescue rather than cutting the recording
# short: watching the fire carry on through the block they just left is most of
# what makes the scene legible.
print(f"\n  {'reached safety at t=%.1fs' % arrived_at if arrived_at is not None else 'never reached safety'}"
      f"  ({clock.episode_seconds:g}s simulated)")

if a.web:
    print(f"\n  {steps_logged} frames recorded, playing on a loop at {a.speed:g}x.")
    print("  Scrub the timeline at the bottom of the viewer to move through it.")
    try:
        input("\nserving -- press enter to stop\n")
    except EOFError:
        while True:
            time.sleep(1)
