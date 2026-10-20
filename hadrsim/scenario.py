"""A scenario: place, weather, episode length, where people and fire start.  Buildable
from a dict, a yaml or the panel form; `autoplace` fills in whatever was left blank."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from hadrsim.core.clock import Clock
from hadrsim.world import nav, osm
from hadrsim.world import terrain as terrain_mod

PRESETS = {
    "Downtown Berkeley":     ("Downtown Berkeley, California",   37.87010, -122.26814, 150.0),
    "UC Berkeley campus":    ("UC Berkeley campus",              37.86843, -122.26052, 180.0),
    "SF Financial District": ("Financial District, SF",          37.79365, -122.39894, 180.0),
    "Manhattan Midtown":     ("Midtown Manhattan, New York",     40.75269,  -73.97725, 180.0),
    "Chicago Loop":          ("The Loop, Chicago",               41.88161,  -87.62946, 180.0),
    "Barcelona Eixample":    ("Eixample, Barcelona",             41.39250,    2.16490, 180.0),
    "Paris Le Marais":       ("Le Marais, Paris",                48.86038,    2.36074, 160.0),
    "Venice San Marco":      ("San Marco, Venice",               45.43483,   12.33402, 160.0),
    "Tokyo Shinjuku":        ("Shinjuku, Tokyo",                 35.69376,  139.70363, 180.0),
    "Oakland Downtown":      ("Downtown Oakland, California",    37.80432, -122.27119, 160.0),
    "Paper grid (25 m)":     (None, 0.0, 0.0, 0.0),
}

_TERRAIN_CACHE: dict[tuple, object] = {}


@dataclass
class Scenario:
    """One runnable situation."""

    place: str | None = "Downtown Berkeley, California"
    radius_m: float = 150.0
    cell_size_m: float = 1.0
    paper_map: str | None = None          # set instead of `place` for the 25 m grid
    lat: float | None = None              # pinned centre; `place` is then only a label
    lon: float | None = None

    wind_mps: float = 5.0
    wind_bearing_deg: float = 40.0

    n_humans: int = 1
    n_drones: int = 2                     # the roster the policy will fly
    seed: int = 0

    # Positions in metres. Left as None for `autoplace` to fill in.
    human_start: list | None = None       # [[x, y], ...]
    human_goal: list | None = None
    fire_at: list | None = None           # [x, y]
    fire_radius_m: float = 2.5
    drone_start: list | None = None       # [x, y] launch point, shared by the team

    n_routes: int = 40
    n_fire_origins: int = 4
    routes: list | None = None            # [[[sx, sy], [gx, gy]], ...]
    fire_origins: list | None = None      # [[x, y], ...]
    randomise: bool = True

    episode_seconds: float = 120.0
    policy: str = "none"                  # checkpoint name, once there is one

    def clock(self) -> Clock:
        return Clock(episode_seconds=self.episode_seconds)

    def terrain(self):
        """The map, cached."""
        key = (self.paper_map, self.place, self.lat, self.lon,
               self.radius_m, self.cell_size_m)
        if key not in _TERRAIN_CACHE:
            if self.paper_map:
                _TERRAIN_CACHE[key] = terrain_mod.load(
                    self.paper_map, cell_size_m=self.cell_size_m)
            else:
                bbox = (osm.bbox_around(self.lat, self.lon, self.radius_m)
                        if self.lat is not None else None)
                _TERRAIN_CACHE[key] = osm.load(
                    place=None if bbox else self.place, bbox=bbox,
                    radius_m=self.radius_m, cell_size_m=self.cell_size_m)
        return _TERRAIN_CACHE[key]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def preset(cls, name: str, **overrides) -> "Scenario":
        """A named place; anything passed explicitly overrides the preset."""
        place, lat, lon, radius = PRESETS[name]
        fields = {"place": place, "lat": lat, "lon": lon, "radius_m": radius}
        if place is None:
            fields = {"place": None, "paper_map": "maps/rust25.yaml", "radius_m": 0.0}
        return cls(**{**fields, **overrides})


def autoplace(sc: Scenario) -> Scenario:
    """Fill in start, exit, fire and routes sensibly for this map, if not given."""
    t = sc.terrain()
    cell = t.cell_size_m
    walk = t.walkable.nonzero()

    if sc.human_start is None:
        w, h = t.shape
        best, best_reach = None, -1
        for fx in (0.25, 0.5, 0.75):
            for fy in (0.25, 0.5, 0.75):
                anchor = torch.tensor([w * fx, h * fy])
                seed = walk[(walk.float() - anchor).norm(dim=1).argmin()]
                d = nav.distance_field(t.walkable, seed[None], cell)[0]
                reach = int((d < nav.UNREACHABLE / 2).sum())
                if reach > best_reach:
                    best, best_reach = seed, reach
        first = best
        starts = [first]
        # Extra people are scattered near the first, not stacked on top of them.
        for i in range(1, sc.n_humans):
            offset = first.float() + torch.tensor([12.0 * (i % 3 - 1), 12.0 * (i // 3 - 1)])
            starts.append(walk[(walk.float() - offset).norm(dim=1).argmin()])
        sc.human_start = [(p.float() * cell).tolist() for p in starts]

    start_cell = (torch.tensor(sc.human_start[0]) / cell).long()
    from_start = nav.distance_field(t.walkable, start_cell[None], cell)[0]

    if sc.human_goal is None:
        budget = 0.88 * 1.3 * sc.episode_seconds
        reachable = t.walkable & (from_start < nav.UNREACHABLE / 2)
        want = min(budget, float(from_start[reachable].max()) * 0.95)
        goal_cell = reachable.nonzero()[(from_start[reachable] - want).abs().argmin()]
        sc.human_goal = [(goal_cell.float() * cell).tolist()] * len(sc.human_start)

    if sc.drone_start is None:
        w, h = t.shape
        centre = torch.tensor([w * 0.5, h * 0.5])
        sc.drone_start = (walk[(walk.float() - centre).norm(dim=1).argmin()].float()
                          * cell).tolist()

    if sc.fire_at is None:
        goal_cell = (torch.tensor(sc.human_goal[0]) / cell).long()
        from_goal = nav.distance_field(t.walkable, goal_cell[None], cell)[0]
        route = float(from_start[goal_cell[0], goal_cell[1]])

        density = torch.nn.functional.avg_pool2d(t.fuel[None, None], 15, 1, 7)[0, 0]
        heavy = t.fuel > 0.6
        on_route = (from_start + from_goal) < route * 1.35
        ends = (from_start > 0.3 * route) & (from_goal > 0.3 * route)

        good = heavy & (density > 0.9) & on_route & ends
        for fallback in (heavy & on_route & ends, heavy & on_route, t.fuel > 0):
            if bool(good.any()):
                break
            good = fallback
        chosen = good.nonzero()[(from_start[good] - route / 2).abs().argmin()]
        sc.fire_at = (chosen.float() * cell).tolist()

    if sc.randomise and sc.routes is None:
        sc.routes, sc.fire_origins = _variety(sc, t, cell)

    return sc


WALK_MPS = 1.3
ROUTE_FRACTION = 0.55
REACH_MARGIN = 0.85


def _variety(sc: Scenario, t, cell: float):
    """`n_routes` start/exit pairs and `n_fire_origins` fire cells, as the paper's 40 and 4."""
    start = (torch.tensor(sc.human_start[0]) / cell).long()
    reach = nav.distance_field(t.walkable, start[None], cell)[0]
    ok = t.walkable & (reach < nav.UNREACHABLE / 2)
    cells = ok.nonzero()
    if len(cells) < 8:
        return None, None

    centre = cells.float().mean(0)
    offset = cells.float() - centre
    angle = torch.atan2(offset[:, 1], offset[:, 0])
    radius = offset.norm(dim=-1)

    # One perimeter point per sector: the cell furthest from the middle in each.
    sectors = sc.n_routes
    bucket = ((angle + math.pi) / (2 * math.pi) * sectors).long().clamp(0, sectors - 1)
    rim = []
    for k in range(sectors):
        here = (bucket == k).nonzero().flatten()
        if len(here):
            rim.append(cells[here[radius[here].argmax()]])
    if len(rim) < 4:
        return None, None

    rim_t = torch.stack(rim)
    fields = nav.distance_field(t.walkable, rim_t, cell)          # (rim, H, W)
    between = fields[:, rim_t[:, 0], rim_t[:, 1]]                 # (rim, rim)

    budget = sc.episode_seconds * WALK_MPS
    want = budget * ROUTE_FRACTION
    far = between.clone()
    far[far > budget * REACH_MARGIN] = float("inf")               # cannot be walked
    far.fill_diagonal_(float("inf"))

    routes = []
    for i, a_ in enumerate(rim):
        j = int((far[i] - want).abs().argmin())
        if not math.isfinite(float(far[i, j])):
            continue                                             # nothing reachable
        b_ = rim[j]
        routes.append([(a_.float() * cell).tolist(), (b_.float() * cell).tolist()])
    if len(routes) < 4:
        return None, None

    # Fire origins spread around the interior, on ground that will actually carry.
    heavy = (t.fuel > 0.6) & ok
    pool = heavy.nonzero().float()
    if not len(pool):
        pool = ok.nonzero().float()
    step = max(1, len(pool) // sc.n_fire_origins)
    order = (pool - centre).norm(dim=-1).argsort()
    origins = [(pool[order[min(i * step, len(pool) - 1)]] * cell).tolist()
               for i in range(sc.n_fire_origins)]
    return routes, origins
