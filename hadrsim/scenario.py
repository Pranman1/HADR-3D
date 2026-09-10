"""A scenario: everything that makes one run different from another.

The place, the weather, where the fire starts, who is in it and where they begin.
One object, buildable from a dict, a yaml file, or a GUI form -- which matters
because all three have to produce *the same thing*.  A control panel that quietly
constructs a slightly different world from the training config is a very expensive
kind of bug.

Nothing here decides anything: it holds what was decided.  The pickers that turn
"somewhere sensible on this map" into actual coordinates live in `autoplace`, so
that a hand-specified scenario and an auto-placed one differ only in who filled the
fields in.

Terrain is cached by (place, radius, cell size).  Fetching and rasterising a city is
a second or two, and a GUI that does it on every parameter change is a GUI nobody
waits for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from hadrsim.core.clock import Clock
from hadrsim.world import nav, osm
from hadrsim.world import terrain as terrain_mod

# Places worth offering in a menu: a low-rise campus, a mid-rise downtown, and two
# skylines. Enough variety that a policy trained on one is visibly tested by another.
PRESETS = {
    "Downtown Berkeley": ("Downtown Berkeley, California", 150.0),
    "UC Berkeley campus": ("UC Berkeley campus, California", 180.0),
    "SF Financial District": ("Financial District, San Francisco", 180.0),
    "Manhattan Midtown": ("Midtown Manhattan, New York", 180.0),
    "Paper grid (25 m)": (None, 0.0),
}

_TERRAIN_CACHE: dict[tuple, object] = {}


@dataclass
class Scenario:
    """One runnable situation."""

    place: str | None = "Downtown Berkeley, California"
    radius_m: float = 150.0
    cell_size_m: float = 1.0
    paper_map: str | None = None          # set instead of `place` for the 25 m grid

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

    episode_seconds: float = 120.0
    policy: str = "none"                  # checkpoint name, once there is one

    def clock(self) -> Clock:
        return Clock(episode_seconds=self.episode_seconds)

    def terrain(self):
        """The map, cached. Rebuilding a city per parameter change is unusable."""
        key = (self.paper_map, self.place, self.radius_m, self.cell_size_m)
        if key not in _TERRAIN_CACHE:
            if self.paper_map:
                _TERRAIN_CACHE[key] = terrain_mod.load(
                    self.paper_map, cell_size_m=self.cell_size_m)
            else:
                _TERRAIN_CACHE[key] = osm.load(
                    place=self.place, radius_m=self.radius_m,
                    cell_size_m=self.cell_size_m)
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
        """A named place, with anything explicitly passed taking precedence.

        The preset's radius is a default rather than a fixture: training wants a
        smaller extract than a demo does, and having to remember which presets
        already set which fields is exactly the friction presets exist to remove.
        """
        place, radius = PRESETS[name]
        fields = {"place": place, "radius_m": radius}
        if place is None:
            fields = {"place": None, "paper_map": "maps/rust25.yaml", "radius_m": 0.0}
        return cls(**{**fields, **overrides})


def autoplace(sc: Scenario) -> Scenario:
    """Fill in whatever was left unspecified, sensibly.

    Only fields still `None` are touched, so anything a person picked -- by hand, or
    by clicking a map -- survives untouched.  This is a convenience for getting a
    runnable situation out of a half-filled form, and is emphatically not the
    paper's setup: the paper hand-picks 40 origin/destination pairs around the
    perimeter and four fire origins in the interior.  No result should depend on
    what happens here.
    """
    t = sc.terrain()
    cell = t.cell_size_m
    walk = t.walkable.nonzero()

    if sc.human_start is None:
        # A quarter of the way in, not in a corner: a corner route on a real extract
        # runs along the clipped edge, where there is little to burn and less to see.
        w, h = t.shape
        anchor = torch.tensor([w * 0.28, h * 0.28])
        first = walk[(walk.float() - anchor).norm(dim=1).argmin()]
        starts = [first]
        # Extra people are scattered near the first, not stacked on top of them.
        for i in range(1, sc.n_humans):
            offset = anchor + torch.tensor([12.0 * (i % 3 - 1), 12.0 * (i // 3 - 1)])
            starts.append(walk[(walk.float() - offset).norm(dim=1).argmin()])
        sc.human_start = [(p.float() * cell).tolist() for p in starts]

    start_cell = (torch.tensor(sc.human_start[0]) / cell).long()
    from_start = nav.distance_field(t.walkable, start_cell[None], cell)[0]

    if sc.human_goal is None:
        # As far as somebody could plausibly walk in the time available, measured
        # along the streets rather than straight through the buildings.
        budget = 0.55 * 1.3 * sc.episode_seconds
        reachable = t.walkable & (from_start < nav.UNREACHABLE / 2)
        goal_cell = reachable.nonzero()[(from_start[reachable] - budget).abs().argmin()]
        sc.human_goal = [(goal_cell.float() * cell).tolist()] * len(sc.human_start)

    if sc.drone_start is None:
        # A launch point, not the person's doorstep.  The paper puts the rescuers at a
        # fixed base near the middle of the map while the evacuee starts on the
        # perimeter; starting them on top of each other makes the task trivial and a
        # policy that captures on the first step has learned nothing about searching.
        w, h = t.shape
        centre = torch.tensor([w * 0.5, h * 0.5])
        sc.drone_start = (walk[(walk.float() - centre).norm(dim=1).argmin()].float()
                          * cell).tolist()

    if sc.fire_at is None:
        goal_cell = (torch.tensor(sc.human_goal[0]) / cell).long()
        from_goal = nav.distance_field(t.walkable, goal_cell[None], cell)[0]
        route = float(from_start[goal_cell[0], goal_cell[1]])

        # On the route, clear of both ends, and somewhere with enough around it to
        # burn. Lighting it beside the exit makes an incoherent scenario; lighting it
        # on a lone verge ringed by road gives a fire that goes out in twenty seconds.
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

    return sc
