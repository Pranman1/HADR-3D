"""Real buildings, from OpenStreetMap.

The paper's map is a 25 x 25 checkerboard.  This loader builds the same `Terrain`
from an actual place -- real footprints, real heights, real streets -- so the
occlusion the drones have to see through is a real city block rather than a grid.

What the simulation actually consumes is unchanged: a height field, a fuel field,
and two passability masks.  A box with a real footprint and a real height blocks a
sight line exactly as a detailed mesh would, and the policy never sees pixels.  So
visual fidelity buys nothing here; *geometric* fidelity buys everything, and this is
where it comes from.

Three OSM facts drive the code below:

  - Heights are mostly absent.  Around a third of buildings carry `building:levels`
    and almost none carry `height`, so most heights are inferred.  `_height_m` is
    where that guessing lives, and it is the least trustworthy thing in this file.

  - `building=roof` is OSM's tag for a roof with no walls -- a carport, a bandstand,
    a fuel-station canopy.  That is exactly the canopy the paper's map implies, so
    the low-drone-sees-under-it asymmetry arrives from real data for free.

  - Roads are the firebreaks -- and the only real ones.  Buildings burn, and burn
    hardest; what actually stops an urban fire is the street.  Rasterising the road
    network to zero fuel is most of what makes a city fire behave like a city fire,
    and giving buildings zero fuel instead would say a city cannot have one.

Results are cached under maps/osm_cache/ by bounding box, so a scenario built from a
real place stays reproducible offline and Overpass is queried once.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from hadrsim.world.raster import fill_polygon, stroke_line
from hadrsim.world.terrain import LLR_CLEARANCE_M, Footprint, Terrain

# Several, tried in turn.  The main instance rate-limits hard -- ten district-sized
# queries in a row is enough to have it refuse connections outright for a while --
# and a map fetch failing is the difference between a night of training and none.
# All of them speak the same API and serve the same planet data.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
)
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "hadrsim/0.1 (research simulator; contact via github.com/sastry-group)"

M_PER_DEG_LAT = 111_320.0
LEVEL_M = 3.0           # a storey, where only `building:levels` is given
ROOF_LEVEL_M = 2.0      # a roof storey is shorter than a floor
DEFAULT_LEVELS = 2.0    # untagged buildings. Most of OSM, unfortunately

# Below this, a mapped building has nothing above ground worth occluding.  Real
# example from downtown Berkeley: the BART station is tagged `building:levels=0`
# with `building:levels:underground=2`, because it is underground.  Extruding that
# to zero height makes a degenerate mesh; the honest answer is to leave the ground
# clear, so it is skipped entirely rather than floored to some minimum.
MIN_BUILDING_M = 2.0

# A roof with no walls. The low drone sees underneath; the high drone does not.
CANOPY_BUILDINGS = {"roof", "carport", "canopy", "bandstand"}

# Canopies are almost never tagged with a height, and the generic building default
# (two storeys, 6 m) is far too tall for a carport or a bus shelter.  It also
# quietly destroys the mechanism they exist for: at 6 m the low drone flies under
# every canopy in the city, so the "person is under a roof the high drone cannot
# see through, and the low drone must go and look" situation never arises.
DEFAULT_CANOPY_M = 3.0
CANOPY_THICKNESS_M = 0.4
MIN_HEADROOM_M = 2.2    # a canopy you could not walk under would be a wall

# How readily each surface carries fire.  Nothing is exactly zero except open water,
# because almost nothing in a city is: a street has trees along it, cars parked on
# it, bins, market stalls and spilled fuel.  A road slows a fire and steers it; it
# does not reliably stop one, and treating it as an absolute barrier is as wrong as
# treating buildings as fireproof.
FUEL_GROUND = 0.7       # ordinary ground: some vegetation, some pavement
FUEL_VEG = 1.5          # parks, grass, woodland
FUEL_TREE = 1.8         # street trees and tree rows: the fastest thing on the block
FUEL_BUILDING = 1.4     # the biggest single load of fuel there is
FUEL_PARKING = 0.55     # a car park is a field of fuel tanks
FUEL_ROAD = 0.18        # street furniture, parked cars, debris. Slows, does not stop
FUEL_WATER = 0.0        # the one real firebreak
TREE_RADIUS_M = 3.5     # canopy of a single mapped street tree

WATER_TAGS = {"natural": {"water", "bay", "strait"},
              "landuse": {"reservoir", "basin"},
              "waterway": {"riverbank", "dock"}}

PARKING_TAGS = {"amenity": {"parking"}, "landuse": {"garages"}}

VEG_TAGS = {
    "landuse": {"grass", "forest", "meadow", "orchard", "vineyard", "cemetery",
                "village_green", "recreation_ground", "allotments"},
    "natural": {"wood", "scrub", "grassland", "heath", "wetland"},
    "leisure": {"park", "garden", "golf_course", "pitch", "nature_reserve"},
}

ROAD_WIDTH_M = {
    "motorway": 20.0, "trunk": 16.0, "primary": 14.0, "secondary": 12.0,
    "tertiary": 10.0, "residential": 8.0, "unclassified": 8.0, "service": 5.0,
    "living_street": 7.0, "pedestrian": 6.0, "footway": 2.0, "path": 2.0,
    "cycleway": 2.0, "steps": 2.0, "track": 4.0,
}
DEFAULT_ROAD_WIDTH_M = 6.0


# ------------------------------------------------------------------ fetching

@dataclass(frozen=True)
class BBox:
    south: float
    west: float
    north: float
    east: float

    @property
    def centre(self) -> tuple[float, float]:
        return (self.south + self.north) / 2, (self.west + self.east) / 2

    def size_m(self) -> tuple[float, float]:
        lat0, _ = self.centre
        return (
            (self.east - self.west) * M_PER_DEG_LAT * math.cos(math.radians(lat0)),
            (self.north - self.south) * M_PER_DEG_LAT,
        )

    def key(self, query: str = "") -> str:
        """Cache key. Includes the query, not just the box.

        Keying on the bounding box alone means that adding a feature type -- street
        trees, car parks, water -- silently serves the old response, which has none of
        them, and the new code looks broken rather than stale.
        """
        blob = f"{self.south:.6f},{self.west:.6f},{self.north:.6f},{self.east:.6f}|{query}"
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


def geocode(place: str) -> tuple[float, float]:
    """Place name -> (lat, lon). One call, then cached by whoever holds the bbox."""
    import requests

    r = requests.get(
        NOMINATIM,
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits:
        raise ValueError(f"nowhere called {place!r}")
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def bbox_around(lat: float, lon: float, radius_m: float) -> BBox:
    dlat = radius_m / M_PER_DEG_LAT
    dlon = radius_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return BBox(lat - dlat, lon - dlon, lat + dlat, lon + dlon)


QUERY = """[out:json][timeout:180];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
  way["man_made"="canopy"]({s},{w},{n},{e});
  way["highway"]({s},{w},{n},{e});
  way["landuse"]({s},{w},{n},{e});
  way["natural"]({s},{w},{n},{e});
  way["leisure"]({s},{w},{n},{e});
  way["amenity"="parking"]({s},{w},{n},{e});
  way["waterway"]({s},{w},{n},{e});
  node["natural"="tree"]({s},{w},{n},{e});
);
out body geom;
"""


def fetch(bbox: BBox, cache_dir: str | Path = "maps/osm_cache") -> dict:
    """Overpass, cached to disk by bounding box so this is queried once, ever."""
    q = QUERY.format(s=bbox.south, w=bbox.west, n=bbox.north, e=bbox.east)
    cache = Path(cache_dir) / f"{bbox.key(QUERY)}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import requests

    trouble = []
    for attempt in range(3):
        for host in OVERPASS_MIRRORS:
            try:
                r = requests.post(host, data={"data": q},
                                  headers={"User-Agent": UA}, timeout=300)
            except requests.RequestException as e:
                trouble.append(f"{host}: {type(e).__name__}")
                continue
            if r.status_code == 200:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text)
                return r.json()
            trouble.append(f"{host}: HTTP {r.status_code}")
        time.sleep(10 * (attempt + 1))   # every mirror busy; back off and go round again
    raise RuntimeError("no Overpass mirror would answer:\n  "
                       + "\n  ".join(trouble[-len(OVERPASS_MIRRORS):]))


# -------------------------------------------------------------------- tags

def _parse_len(v: str) -> float | None:
    """OSM lengths are free text: '12', '12 m', '40 ft', '12.5'."""
    v = v.strip().lower().replace("meters", "").replace("metres", "").replace("m", "")
    feet = "'" in v or "ft" in v
    v = v.replace("ft", "").replace("'", "").strip()
    try:
        x = float(v)
    except ValueError:
        return None
    return x * 0.3048 if feet else x


def _height_m(tags: dict, default_m: float | None = None) -> float:
    """Best available height. Mostly a guess, and the weakest link in this file.

    `default_m` overrides the fallback for things that are not ordinary buildings --
    a carport is not two storeys tall.
    """
    for key in ("height", "building:height"):
        if key in tags and (h := _parse_len(tags[key])) is not None:
            return h
    levels = None
    for key in ("building:levels", "levels"):
        if key in tags and (levels := _parse_len(tags[key])) is not None:
            break
    if levels is None:
        return default_m if default_m is not None else DEFAULT_LEVELS * LEVEL_M
    roof = _parse_len(tags.get("roof:levels", "0")) or 0.0
    return levels * LEVEL_M + roof * ROOF_LEVEL_M


def _is_canopy(tags: dict) -> bool:
    return (
        tags.get("building") in CANOPY_BUILDINGS
        or tags.get("man_made") == "canopy"
        or tags.get("amenity") == "shelter"
    )


def _is_vegetation(tags: dict) -> bool:
    return any(tags.get(k) in vals for k, vals in VEG_TAGS.items())


def _surface_fuel(tags: dict) -> float | None:
    """Fuel load for an area feature, or None if it is not one."""
    if any(tags.get(k) in vals for k, vals in WATER_TAGS.items()):
        return FUEL_WATER
    if any(tags.get(k) in vals for k, vals in PARKING_TAGS.items()):
        return FUEL_PARKING
    if tags.get("natural") in ("tree_row", "wood") or tags.get("landuse") == "forest":
        return FUEL_TREE
    if _is_vegetation(tags):
        return FUEL_VEG
    return None


# ------------------------------------------------------------- geometry

def _project(bbox: BBox):
    """lat/lon -> metres from the box's south-west corner. Flat earth; fine at 1 km."""
    lat0, _ = bbox.centre
    mx = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return lambda lat, lon: ((lon - bbox.west) * mx, (lat - bbox.south) * M_PER_DEG_LAT)


def _rings(el: dict, proj) -> list[np.ndarray]:
    """Outlines of one element, in metres. Handles ways and multipolygon relations."""
    out = []
    if el["type"] == "way" and el.get("geometry"):
        out.append(el["geometry"])
    elif el["type"] == "relation":
        out += [m["geometry"] for m in el.get("members", [])
                if m.get("role") == "outer" and m.get("geometry")]
    rings = []
    for g in out:
        pts = np.array([proj(p["lat"], p["lon"]) for p in g], dtype=np.float64)
        if len(pts) >= 3:
            rings.append(pts)
    return rings


# ------------------------------------------------------------------- loader

def load(
    place: str | None = None,
    bbox: BBox | None = None,
    radius_m: float = 250.0,
    cell_size_m: float = 1.0,
    llr_clearance_m: float = LLR_CLEARANCE_M,
    burnable_buildings: bool = True,
    cache_dir: str | Path = "maps/osm_cache",
    device="cpu",
) -> Terrain:
    """Build a Terrain from a real place.

    >>> load("Downtown Berkeley, California", radius_m=250)
    """
    if bbox is None:
        if place is None:
            raise ValueError("give either a place name or a bbox")
        bbox = bbox_around(*geocode(place), radius_m)

    data = fetch(bbox, cache_dir)
    proj = _project(bbox)
    ext_x, ext_y = bbox.size_m()
    shape = (max(1, int(ext_x / cell_size_m)), max(1, int(ext_y / cell_size_m)))

    solid = np.zeros(shape, dtype=np.float32)
    can_z0 = np.zeros(shape, dtype=np.float32)
    can_z1 = np.zeros(shape, dtype=np.float32)
    fuel = np.full(shape, FUEL_GROUND, dtype=np.float32)
    prints: list[Footprint] = []

    # Surfaces first: roads, then buildings, are painted over the top of them.
    for el in data["elements"]:
        tags = el.get("tags", {})
        load = _surface_fuel(tags)
        if load is None:
            continue
        for ring in _rings(el, proj):
            if (hit := fill_polygon(ring, shape, cell_size_m)) is not None:
                sl, mask = hit
                fuel[sl][mask] = load

    # Streets: the firebreaks that make a city fire behave like a city fire.
    for el in data["elements"]:
        tags = el.get("tags", {})
        if "highway" not in tags or el["type"] != "way" or not el.get("geometry"):
            continue
        line = np.array([proj(p["lat"], p["lon"]) for p in el["geometry"]])
        if len(line) < 2:
            continue
        width = _parse_len(tags["width"]) if "width" in tags else None
        width = width or ROAD_WIDTH_M.get(tags["highway"], DEFAULT_ROAD_WIDTH_M)
        if (hit := stroke_line(line, width, shape, cell_size_m)) is not None:
            sl, mask = hit
            # Take the lower of the two: a road through a park is still a road, but a
            # road already marked as water (a ford, a bridge deck over a river) stays
            # the barrier it is.
            fuel[sl][mask] = np.minimum(fuel[sl][mask], FUEL_ROAD)

    # Street trees, which OSM maps as individual points. They are what carries a fire
    # along a road that would otherwise slow it down.
    for el in data["elements"]:
        if el.get("type") != "node" or el.get("tags", {}).get("natural") != "tree":
            continue
        x, y = proj(el["lat"], el["lon"])
        r = TREE_RADIUS_M
        ring = np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]])
        if (hit := fill_polygon(ring, shape, cell_size_m)) is not None:
            sl, mask = hit
            fuel[sl][mask] = np.maximum(fuel[sl][mask], FUEL_TREE)

    # Buildings last, so they win over whatever they stand on.
    for el in data["elements"]:
        tags = el.get("tags", {})
        if "building" not in tags and not _is_canopy(tags):
            continue
        canopy = _is_canopy(tags)
        h = _height_m(tags, default_m=DEFAULT_CANOPY_M if canopy else None)
        if h < MIN_BUILDING_M:
            continue                    # underground, or mapped without a structure

        # A canopy is a slab of air-thickness at height h, with usable ground under
        # it.  A building is solid from the ground up.
        base, top = (max(h - CANOPY_THICKNESS_M, MIN_HEADROOM_M), h) if canopy else (0.0, h)

        for ring in _rings(el, proj):
            if (hit := fill_polygon(ring, shape, cell_size_m)) is None:
                continue
            sl, mask = hit
            if canopy:
                # The ground beneath stays ground: walkable, and still able to burn.
                can_z0[sl][mask] = np.maximum(can_z0[sl][mask], base)
                can_z1[sl][mask] = np.maximum(can_z1[sl][mask], top)
            else:
                solid[sl][mask] = np.maximum(solid[sl][mask], h)
                fuel[sl][mask] = FUEL_BUILDING if burnable_buildings else 0.0
            prints.append(Footprint(
                xy=torch.tensor(ring, dtype=torch.float32),
                base_m=base,
                top_m=top,
                kind="canopy" if canopy else "solid",
                name=tags.get("name", ""),
                tags=tags,
            ))

    t = torch.as_tensor
    terrain = Terrain(
        solid_top=t(solid, device=device),
        canopy_z0=t(can_z0, device=device),
        canopy_z1=t(can_z1, device=device),
        walkable=t(solid == 0.0, device=device),   # no entering buildings
        flyable=t(solid == 0.0, device=device),    # replaced immediately below
        fuel=t(fuel, device=device),
        cell_size_m=cell_size_m,
        footprints=prints,
        origin_latlon=(bbox.south, bbox.west),
    )
    terrain.derive_flyable(llr_clearance_m)
    return terrain
