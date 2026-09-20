"""The evacuee goes round a fire, not through it.

The paper's rule (`Fire.can_overlap` is false for the evader) makes a burning cell a
wall for the person.  These tests put a fire across the direct route with a way round
and check what actually happens to the trajectory, not what the cost field says.
"""

import math

import torch

from hadrsim.agents.human import Human
from hadrsim.core.rng import Rng
from hadrsim.world import fire as fire_mod
from hadrsim.world import nav
from hadrsim.world.terrain import Terrain


def _open_ground(n=40, cell=1.0):
    """A flat square, open ground, so the only obstacle is the fire itself."""
    solid = torch.zeros(n, n)
    return Terrain(solid_top=solid, canopy_z0=torch.zeros(n, n), canopy_z1=torch.zeros(n, n),
                   walkable=torch.ones(n, n, dtype=torch.bool),
                   flyable=torch.ones(n, n, dtype=torch.bool),
                   fuel=torch.ones(n, n), cell_size_m=cell)


def _walk(human, fire, seconds=60.0, dt=0.1):
    rng = Rng(0)
    xs = []
    for _ in range(int(seconds / dt)):
        want = human.desired_velocity(fire, _NoSmoke(), rng, dt)
        human.integrate(want, dt)
        xs.append(human.pos[0].clone())
        if bool(human.reached[0]):
            break
    return torch.stack(xs)


class _NoSmoke:
    def column_density(self):
        return torch.zeros(1, 40, 40)


def _fire_wall(t, cells, code=None):
    """A fire that is exactly these cells and nothing else, and does not spread.

    `burning`, `burnt` and `intensity` are views derived from the `state` codes, so
    the state is what gets written -- assigning into the views changes nothing.
    """
    f = fire_mod.Fire(t, 1)
    f.reset(None)
    for x, y in cells:
        f.state[0, x, y] = fire_mod.BURNING if code is None else code
    f.p = fire_mod.FireParams(spread_prob=0.0, spot_prob=0.0, burn_rate=0.0)
    return f


def test_fire_across_the_route_is_walked_around_not_through():
    t = _open_ground()
    h = Human(t, 1)
    h.reset(torch.tensor([[5.0, 20.0]]), torch.tensor([[35.0, 20.0]]))
    # A line of fire north-south at x=12, leaving only the ends open.
    wall = [(12, y) for y in range(4, 36)]
    fire = _fire_wall(t, wall)
    path = _walk(h, fire)

    burning = fire.burning[0]
    cells = path.long().clamp(0, 39)
    stepped_in = burning[cells[:, 0], cells[:, 1]]
    assert not bool(stepped_in.any()), "walked through a burning cell"
    assert bool(h.reached[0]), "never reached safety"
    straight = 30.0
    walked = float((path[1:] - path[:-1]).norm(dim=-1).sum())
    assert walked > straight * 1.25, f"no detour taken: walked {walked:.1f} m"


def test_burnt_ground_is_avoided_when_there_is_a_way_round():
    t = _open_ground()
    h = Human(t, 1)
    h.reset(torch.tensor([[5.0, 20.0]]), torch.tensor([[35.0, 20.0]]))
    fire = _fire_wall(t, [(12, y) for y in range(8, 33)], code=fire_mod.BURNT)   # ash, not flame
    path = _walk(h, fire)
    # "Around, not through": where they cross the ash line's column, they must be
    # past its end -- and brushing a corner cell for a step is not walking through it.
    crossing = path[(path[:, 0] > 11.5) & (path[:, 0] < 12.5)]
    assert len(crossing) and bool(((crossing[:, 1] < 8.5) | (crossing[:, 1] > 32.5)).all()), \
        f"went through the ash at y={crossing[:, 1].tolist()[:4]}"
    cells = path.long().clamp(0, 39)
    on_ash = fire.burnt[0][cells[:, 0], cells[:, 1]].float().mean()
    assert float(on_ash) < 0.03, f"{float(on_ash) * 100:.0f}% of the walk was on ash"
    assert bool(h.reached[0])


def test_when_the_fire_cuts_every_route_they_still_move():
    t = _open_ground()
    h = Human(t, 1)
    h.reset(torch.tensor([[5.0, 20.0]]), torch.tensor([[35.0, 20.0]]))
    fire = _fire_wall(t, [(12, y) for y in range(0, 40)])   # wall to wall
    path = _walk(h, fire, seconds=10.0)
    moved = float((path[-1] - path[0]).norm())
    assert moved > 2.0, "froze in place when every route was cut"
