"""The city: what is solid, what you can walk through, and what will burn.

Loads the paper's 2D map and gives it a third dimension.

The 2D map has four building classes.  Read as a table of who can pass through and
who can see inside, they describe something quite specific:

    class              human walks   LLR flies   HLR sees in   LLR sees in
    inaccessible            no           no           no            no
    evader-only            yes           no           no            no
    evader-accessible      yes           no           no           YES
    accessible             yes          yes           no           YES

Every class blocks the view from above, but two of them are transparent from the
side.  That is not a building -- that is a *canopy*: a roof with open sides.  So in
3D we get the paper's asymmetry for free, out of geometry rather than a lookup:

    solid     ####   walls.  Blocks every sight line, from above and from the side.
              ####
              ####

    canopy    ‾‾‾‾   a roof slab on posts, open at the sides.  The high drone looking
              .  .   down hits the roof.  The low drone looking in from the side has
                     a clear slanted line to the ground underneath.

That is the whole premise of the paper -- "the low rescuer sees under occlusions the
high rescuer cannot" -- made physical instead of stipulated.  Sideways visibility is
therefore decided by *walls*, not by height: a canopy never blocks it.

Height decides something else: whether the low drone fits underneath.  That is the
only difference between `evader-accessible` (roof at 2.5 m -- a person fits, a drone
at 3 m does not) and `accessible` (roof at 4 m -- the drone fits too).  So `flyable`
is derived from clearance in load(), never hardcoded.

Fuel is 1.0 wherever the ground is open (including under canopies) and 0.0 under
solid buildings, which is what stops fire spreading through walls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

# height (m) of the solid part, and (bottom, top) of the canopy slab. 0 means absent.
SOLID_H = 8.0
CANOPY_LOW = (2.5, 3.0)    # a person fits under; a drone at 3 m does not
CANOPY_HIGH = (4.0, 4.5)   # the low drone fits under too
LLR_CLEARANCE_M = 3.0      # the low drone's nominal altitude

CLASSES = {
    #                      solid    canopy       walkable
    "inaccessible":      (SOLID_H, None,         False),
    "evader-only":       (3.0,     None,         True),
    "evader-accessible": (0.0,     CANOPY_LOW,   True),
    "accessible":        (0.0,     CANOPY_HIGH,  True),
}


@dataclass
class Footprint:
    """One building outline, in metres. Kept alongside the grid purely for drawing.

    The simulation itself only ever reads the grid: occlusion, fire and movement are
    all fields.  These polygons exist so a real city can be drawn as real buildings
    rather than as a pile of one-metre cubes.
    """

    xy: "torch.Tensor"     # (N, 2) outline in metres, closed implicitly
    base_m: float
    top_m: float
    kind: str              # "solid" or "canopy"
    name: str = ""
    tags: dict | None = None   # whatever the source knew; drives colour only


@dataclass
class Terrain:
    """Static map. Shared across every parallel simulation, so it carries no batch."""

    solid_top: torch.Tensor    # (W, H) height of sight-blocking solid, 0 if none
    canopy_z0: torch.Tensor    # (W, H) underside of the roof slab, 0 if none
    canopy_z1: torch.Tensor    # (W, H) top of the roof slab, 0 if none
    walkable: torch.Tensor     # (W, H) bool -- the human may stand here
    flyable: torch.Tensor      # (W, H) bool -- the low drone may fly through here
    fuel: torch.Tensor         # (W, H) how much there is to burn
    cell_size_m: float
    footprints: list[Footprint] | None = None   # for drawing only; may be absent
    origin_latlon: tuple[float, float] | None = None   # if it came from a real place

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.solid_top.shape)

    @property
    def extent_m(self) -> tuple[float, float]:
        w, h = self.shape
        return w * self.cell_size_m, h * self.cell_size_m

    def to(self, device) -> "Terrain":
        move = lambda t: t.to(device)
        return Terrain(
            move(self.solid_top), move(self.canopy_z0), move(self.canopy_z1),
            move(self.walkable), move(self.flyable), move(self.fuel), self.cell_size_m,
            self.footprints, self.origin_latlon,
        )

    def derive_flyable(self, llr_clearance_m: float = LLR_CLEARANCE_M) -> None:
        """Recompute where the low drone may fly, from the geometry alone.

        A wall stops it, and so does a roof it does not fit under.  Every loader
        calls this, so clearance is never hand-written per map.  `walkable` stays
        loader-specific: the paper's map has structures a person may enter and a
        drone may not, which no amount of geometry will tell you.
        """
        roof = torch.where(self.canopy_z1 > 0, self.canopy_z0, torch.inf)
        self.flyable = (self.solid_top == 0.0) & (roof > llr_clearance_m)


def load(
    path: str | Path,
    cell_size_m: float = 1.0,
    llr_clearance_m: float = LLR_CLEARANCE_M,
    device="cpu",
) -> Terrain:
    """Read a pegrid-format map yaml into a 3D Terrain."""
    spec = yaml.safe_load(Path(path).read_text())
    w, h = spec["dim"]["width"], spec["dim"]["height"]

    z = lambda v=0.0: torch.full((w, h), v, dtype=torch.float32, device=device)
    b = lambda v: torch.full((w, h), v, dtype=torch.bool, device=device)

    t = Terrain(z(), z(), z(), b(True), b(True), z(1.0), cell_size_m)

    for name, (solid, canopy, walkable) in CLASSES.items():
        for corners in spec["buildings"].get(name, []):
            # yaml corners are [row, col]; pegrid indexes the grid as (col, row).
            ys = [c[0] for c in corners]
            xs = [c[1] for c in corners]
            sl = (slice(min(xs), max(xs) + 1), slice(min(ys), max(ys) + 1))

            t.solid_top[sl] = solid
            if canopy is not None:
                t.canopy_z0[sl], t.canopy_z1[sl] = canopy
            t.walkable[sl] = walkable
            if solid > 0.0:
                t.fuel[sl] = 0.0   # fire cannot cross a solid building

    t.derive_flyable(llr_clearance_m)
    return t
