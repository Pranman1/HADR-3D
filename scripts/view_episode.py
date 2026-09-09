"""Watch one episode of the world: fire, smoke, and a person trying to get out.

No drones yet -- this is step 1 of the plan, the world on its own.

Defaults to a real city, because that is the point of the thing.  `--paper` switches
to the published 25 x 25 grid, which exists only for comparing against Table I.

    python scripts/view_episode.py                       # writes episode.mp4
    python scripts/view_episode.py --osm "Financial District, San Francisco"
    python scripts/view_episode.py --wind 8 --bearing 45 -o windy.mp4
    python scripts/view_episode.py --paper
    python scripts/view_episode.py --web                 # live 3D viewer instead

By default it writes a video file and exits.  `--web` gets the live 3D viewer, which
is better for poking around a scene and worse for simply looking at one, since it
needs a server, two ports, a browser and a URL.

**Where things get placed is this script's business, not the paper's.**  The paper
hand-picks 40 origin/destination pairs and 4 fire origins on one 25 m block, and
`scenarios/paper_parity.yaml` holds exactly those.  Dropping the same scenario
anywhere on Earth means choosing them automatically, and the rules below are a demo
convenience -- they are not part of the model and no result should depend on them.
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
p.add_argument("--bearing", type=float, default=45.0, help="degrees, blown towards")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--web", action="store_true",
               help="live 3D viewer instead of writing a video")
p.add_argument("-o", "--out", default="episode.mp4",
               help="video to write (.mp4 or .gif)")
p.add_argument("--fps", type=int, default=25)
p.add_argument("--save", default=None)
p.add_argument("--speed", type=float, default=2.0, help="playback speed")
p.add_argument("--paused", action="store_true", help="do not auto-play")
a = p.parse_args()

t = terr.load(a.map, cell_size_m=a.cell) if a.paper \
    else osm.load(place=a.osm, radius_m=a.radius, cell_size_m=a.cell)
clock = Clock()

# One stream each. Sharing a generator makes the fire's spread depend on how many
# times the wind happened to be stepped -- so raising the physics rate, or adding a
# gust, silently changes which cells ignite. It is exactly the class of bug that
# makes a "reproducible" simulator irreproducible in practice, and it showed up here
# as fires that mysteriously went out.
rng = Rng(a.seed)
rng_fire = rng.fork("fire")
rng_wind = rng.fork("wind")
rng_human = rng.fork("human")

fire = fire_mod.Fire(t, batch=1)
smoke = smoke_mod.Smoke(t, batch=1)
wind = wind_mod.still(1) if a.wind == 0 else wind_mod.make(1, a.wind, a.bearing)
human = Human(t, batch=1)
crowd = Crowd(t)

# --------------------------------------------------------------- placing things
cell = t.cell_size_m
walk = t.walkable.nonzero()

# The start goes a quarter of the way into the map, not in a corner.  A corner route
# on a real extract runs along the edge where the map has been clipped, so there is
# little contiguous fuel beside it and the fire fizzles -- and you never see the city.
w, h = t.shape
target = torch.tensor([w * 0.28, h * 0.28])
start_cell = walk[(walk.float() - target).norm(dim=1).argmin()]

# The exit: as far as the person could plausibly walk in the time available,
# measured along the streets rather than straight through the buildings.  Opposite
# corners of a 360 m map is a 500 m walk that nobody finishes in 120 s.
from_start = nav.distance_field(t.walkable, start_cell[None], cell)[0]
reachable = t.walkable & (from_start < nav.UNREACHABLE / 2)
budget = 0.55 * human.p.v_calm * clock.episode_seconds
goal_cell = reachable.nonzero()[(from_start[reachable] - budget).abs().argmin()]
route_m = float(from_start[goal_cell[0], goal_cell[1]])

# The fire: on the route, but well clear of both ends of it.  Lighting it beside the
# exit makes an incoherent scenario -- the safe place is inside the fire -- and
# lighting it beside the start gives the person no chance.  It also needs enough
# around it to burn: a lone scrap of front garden ringed by road burns out in
# twenty seconds.
from_goal = nav.distance_field(t.walkable, goal_cell[None], cell)[0]
# Mean fuel *load* over a 15 m window, not the fraction of cells with any fuel at
# all: once a road carries street trees and parked cars its fuel is nonzero, so
# "fraction burnable" became 1.0 everywhere and stopped selecting for anything.
density = torch.nn.functional.avg_pool2d(t.fuel[None, None], 15, 1, 7)[0, 0]

on_route = (from_start + from_goal) < route_m * 1.35
clear_of_ends = (from_start > 0.3 * route_m) & (from_goal > 0.3 * route_m)
heavy = t.fuel > 0.6                          # not a street, not bare ground
good = heavy & (density > 0.9) & on_route & clear_of_ends
for fallback in (heavy & on_route & clear_of_ends, heavy & on_route, t.fuel > 0):
    if bool(good.any()):
        break
    good = fallback
ignite = good.nonzero()[(from_start[good] - route_m / 2).abs().argmin()]

fire.reset(ignite[None])
human.reset(start_cell.float()[None] * cell, goal_cell.float()[None] * cell)

print(f"\n{a.map if a.paper else a.osm}  {t.shape[0]}x{t.shape[1]} @ {cell} m")
print(clock.describe())
print(f"person walks {route_m:.0f} m: {start_cell.tolist()} -> {goal_cell.tolist()}")
print(f"fire at {ignite.tolist()}  "
      f"({float(from_start[ignite[0], ignite[1]]):.0f} m along the route, "
      f"{float(from_goal[ignite[0], ignite[1]]):.0f} m from the exit)")
print(f"wind {a.wind} m/s towards {a.bearing} deg\n")

# ------------------------------------------------------------------- the episode
recorder = None
if a.web or a.save:
    viz.start("hadrsim-episode", web=a.web, save=a.save, play=not a.paused, speed=a.speed)
    viz.log_terrain(t)
else:
    from hadrsim.viz.render import Recorder
    recorder = Recorder(t, scale=2 if max(t.shape) < 400 else 1)

dt = clock.dt                       # 0.01 s -- bodies move at 100 Hz
want = torch.zeros(1, 2)
arrived_at = None
frames = 0
half_second = max(1, int(clock.physics_hz / 2))

for tick in range(int(clock.episode_seconds * clock.physics_hz)):
    now = tick * dt

    # Each thing on its own clock.  The person re-aims ten times a second; their body
    # integrates a hundred times a second; the fire spreads once.  Nothing teleports
    # between decisions -- that is the whole point of continuous time, and it is what
    # the drones will do too, deciding at 2 Hz while flying at 100 Hz.
    if tick % clock.human_substeps == 0:
        want = human.desired_velocity(fire, smoke, rng_human, clock.human_substeps * dt)
    human.integrate(crowd.step(human.pos, human.vel, want, dt), dt)

    wind.step(dt, rng_wind)
    fire.step(dt, wind, rng_fire)

    if tick % half_second == 0:
        smoke.update(fire, wind)

    if tick % clock.viz_substeps == 0:
        frames += 1
        if recorder is not None:
            recorder.frame(fire, smoke, human)
        else:
            viz.log_time(now)
            viz.log_human(human)
            if tick % half_second == 0:
                # Fire and smoke change slowly and the ground texture is the most
                # expensive thing logged, so they get 2 Hz while the person gets 20.
                viz.log_fire(fire)
                viz.log_smoke(smoke)

    if human.reached[0] and arrived_at is None:
        arrived_at = now
        print(f"  t={now:6.1f}s  REACHED SAFETY")
    if tick % int(clock.physics_hz * 20) == 0:
        print(f"  t={now:6.1f}s  burning {int(fire.burning[0].sum()):>5}  "
              f"panic {float(human.gamma[0]):.2f}  "
              f"speed {float(human.vel[0].norm()):.2f} m/s")

# Deliberately simulated past the rescue: watching the fire carry on through the
# block they just left is most of what makes the scene legible.
outcome = f"reached safety at t={arrived_at:.1f}s" if arrived_at is not None \
    else "never reached safety"
print(f"\n  {outcome}  ({clock.episode_seconds:g}s simulated, {frames} frames)")

if recorder is not None:
    path = recorder.save(a.out, fps=a.fps)
    seconds = frames / a.fps
    print(f"  wrote {path}  ({frames} frames, {seconds:.0f}s at {a.fps} fps)")
elif a.web:
    print(f"  playing on a loop at {a.speed:g}x. Scrub the timeline at the bottom.")
    try:
        input("\nserving -- press enter to stop\n")
    except EOFError:
        while True:
            time.sleep(1)
