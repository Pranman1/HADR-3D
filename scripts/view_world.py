"""Look at a map, real or from the paper.

    python scripts/view_world.py --web                       # the paper's grid
    python scripts/view_world.py --osm "Downtown Berkeley, California" --web
    python scripts/view_world.py --osm "..." --radius 400 --cell 0.5 --web

The first run of a new place queries OpenStreetMap and caches it under
maps/osm_cache/, so every run after that is offline and reproducible.
"""

import argparse
import time

from hadrsim.viz import rerun_sink as viz
from hadrsim.world import osm
from hadrsim.world import terrain as terr

p = argparse.ArgumentParser()
p.add_argument("--map", default="maps/rust25.yaml", help="pegrid-format map yaml")
p.add_argument("--osm", default=None, metavar="PLACE", help="load a real place instead")
p.add_argument("--radius", type=float, default=250.0, help="metres around the place")
p.add_argument("--cell", type=float, default=1.0, help="grid resolution in metres")
p.add_argument("--web", action="store_true", help="serve to a browser instead of a window")
p.add_argument("--save", default=None, help="write a .rrd file instead of showing it")
a = p.parse_args()

t0 = time.time()
if a.osm:
    t = osm.load(place=a.osm, radius_m=a.radius, cell_size_m=a.cell)
    source = f"{a.osm} (r={a.radius:.0f} m)"
else:
    t = terr.load(a.map, cell_size_m=a.cell)
    source = a.map

n_cells = t.shape[0] * t.shape[1]
built = int((t.solid_top > 0).sum())
print(f"\n{source}")
print(f"  grid            : {t.shape[0]} x {t.shape[1]} cells @ {t.cell_size_m} m "
      f"= {t.extent_m[0]:.0f} x {t.extent_m[1]:.0f} m")
if t.footprints:
    solid = sum(f.kind == "solid" for f in t.footprints)
    print(f"  buildings       : {solid} solid, {len(t.footprints) - solid} canopies")
    hs = [f.top_m for f in t.footprints if f.kind == "solid"]
    if hs:
        hs.sort()
        print(f"  heights (m)     : min {hs[0]:.0f}  median {hs[len(hs)//2]:.0f}  "
              f"max {hs[-1]:.0f}")
print(f"  built up        : {built} cells ({100 * built / n_cells:.1f}%)")
print(f"  canopy          : {int((t.canopy_z1 > 0).sum())} cells")
print(f"  burnable        : {int((t.fuel > 0).sum())} cells "
      f"({100 * int((t.fuel > 0).sum()) / n_cells:.1f}%)")
print(f"  firebreak/road  : {int((t.fuel == 0).sum()) - built} cells")
print(f"  walkable        : {int(t.walkable.sum())} cells")
print(f"  flyable (LLR)   : {int(t.flyable.sum())} cells")
print(f"  loaded in       : {time.time() - t0:.1f} s")

viz.start("hadrsim-world", web=a.web, save=a.save)
viz.log_terrain(t)
if a.web:
    try:
        input("\nserving -- press enter to stop\n")
    except EOFError:          # not a terminal (nohup, CI): just stay up
        while True:
            time.sleep(1)
