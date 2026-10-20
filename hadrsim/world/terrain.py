"""The static map: solid heights, canopy slabs, walkable, flyable, fuel.  One per
scenario, shared by every environment in the batch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

# height (m) of the solid part, and (bottom, top) of the canopy slab. 0 means absent.
FUEL_BUILDING = 1.4    # a building is more to burn than a patch of grass
FUEL_OPEN = 1.0
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
    """One building outline in metres; kept for the rasteriser test and for drawing."""

    xy: "torch.Tensor"     # (N, 2) outline in metres, closed implicitly
    base_m: float
    top_m: float
    kind: str              # "solid" or "canopy"
    name: str = ""
    tags: dict | None = None   # whatever the source knew; drives colour only


@dataclass
class Terrain:
    """Static map."""

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
        """Where the low drone may fly: no solid, and any canopy higher than its clearance."""
        roof = torch.where(self.canopy_z1 > 0, self.canopy_z0, torch.inf)
        self.flyable = (self.solid_top == 0.0) & (roof > llr_clearance_m)


def load(
    path: str | Path,
    cell_size_m: float = 1.0,
    llr_clearance_m: float = LLR_CLEARANCE_M,
    burnable_buildings: bool = True,
    device="cpu",
) -> Terrain:
    """Read a pegrid-format map yaml into a 3D Terrain."""
    spec = yaml.safe_load(Path(path).read_text())
    w, h = spec["dim"]["width"], spec["dim"]["height"]

    z = lambda v=0.0: torch.full((w, h), v, dtype=torch.float32, device=device)
    b = lambda v: torch.full((w, h), v, dtype=torch.bool, device=device)

    t = Terrain(z(), z(), z(), b(True), b(True), z(FUEL_OPEN), cell_size_m)

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
                t.fuel[sl] = FUEL_BUILDING if burnable_buildings else 0.0

    t.derive_flyable(llr_clearance_m)
    return t
