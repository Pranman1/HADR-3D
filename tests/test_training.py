"""Rewards, the training environment, and the policy that learns in it.

The tests that matter most here are about *what the reward can see*.  In the 2D
simulator the reward functions look up ground-truth visibility on their own, so
degrading the drones' perception changes nothing -- the answer still arrives through
the reward channel.  Closing that is most of why this project exists, and
`test_rewards_only_read_what_the_team_believes` is what keeps it closed.

`test_the_policy_can_learn_something` is slow and deliberately weak.  It does not
check that the policy is any good; it checks that gradients flow and reward moves,
so that a broken pipeline fails here rather than after an overnight run.
"""

import pytest
import torch

from hadrsim.core import spec
from hadrsim.envs.train import TrainEnv
from hadrsim.reward.compose import DEFAULT, Composer
from hadrsim.reward.terms import ALL, View
from hadrsim.scenario import Scenario


def small(**kw):
    """A deliberately tiny world: these tests are about wiring, not about cities."""
    return Scenario.preset("Downtown Berkeley", radius_m=60.0, cell_size_m=4.0,
                           episode_seconds=20.0, **kw)


def view(**over) -> View:
    b = over.pop("batch", 1)
    base = dict(
        drone_pos=torch.tensor([[[0.0, 0.0, 40.0], [5.0, 0.0, 3.0]]] * b),
        drone_vel=torch.zeros(b, 2, 3),
        person_pos=torch.tensor([[8.0, 0.0, 0.9]] * b),
        person_seen=torch.ones(b, dtype=torch.bool),
        seen_by=torch.ones(b, 2, dtype=torch.bool),
        confidence=torch.full((b,), 0.9),
        goal=torch.tensor([[40.0, 40.0, 0.0]] * b),
        captured=torch.zeros(b, dtype=torch.bool),
        was_captured=torch.zeros(b, dtype=torch.bool),
        reached_goal=torch.zeros(b, dtype=torch.bool),
        fire_near=torch.zeros(b, 2),
        action=torch.zeros(b, 2, 4),
    )
    base.update(over)
    return View(**base)


# -------------------------------------------------------------------- rewards

def test_every_term_returns_one_number_per_environment():
    v = view(batch=5)
    for name, fn in ALL.items():
        out = fn(v)
        assert out.shape == (5,), f"{name} returned {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name} produced a non-finite value"


def test_penalties_are_penalties():
    """A negative term *and* a negative weight cancel, silently.

    That produced a policy rewarded for flying into fires, and nothing about the
    numbers looked wrong.  Terms answer "how much of this is there"; the weight
    decides whether that is wanted.
    """
    for name in ("fire_proximity", "control_effort", "time_penalty"):
        assert DEFAULT[name] < 0, f"{name} should be weighted negative"
        assert float(ALL[name](view())[0]) >= 0, f"{name} should return a magnitude"

    calm = Composer()(view(fire_near=torch.zeros(1, 2))).total
    burning = Composer()(view(fire_near=torch.ones(1, 2))).total
    assert float(burning) < float(calm), "flying into a fire must cost something"


def test_reaching_safety_pays_only_when_the_drones_helped():
    """Otherwise the team is paid handsomely for a person who walked out unaided.

    Observed in a real run: reward climbing past +60 while the capture rate sat at
    exactly zero, because the evacuee rescues themselves most of the time early on.
    """
    unaided = view(reached_goal=torch.ones(1, dtype=torch.bool))
    led = view(reached_goal=torch.ones(1, dtype=torch.bool),
               was_captured=torch.ones(1, dtype=torch.bool))

    # Checked on the term itself, not on the total: the drones in this fixture can
    # see the person, and being paid for *that* is correct.
    assert float(ALL["reached_safety"](unaided)[0]) == 0.0, "self-rescue paid out"
    assert float(ALL["reached_safety"](led)[0]) == 1.0

    gain = float(Composer()(led).total) - float(Composer()(unaided).total)
    assert gain > 50.0, "an actual rescue should be worth a great deal more"


def test_rewards_only_read_what_the_team_believes():
    """The `View` is the whole interface, and it carries beliefs, not truth.

    If a term could reach into the world it would keep working when perception got
    worse, which is precisely the hole this rebuild exists to close.
    """
    fields = set(View.__dataclass_fields__)
    assert not (fields & {"terrain", "fire", "smoke", "human", "world", "env"})
    assert "person_pos" in fields and "confidence" in fields


def test_terms_that_need_a_sighting_pay_nothing_without_one():
    blind = view(person_seen=torch.zeros(1, dtype=torch.bool),
                 seen_by=torch.zeros(1, 2, dtype=torch.bool),
                 confidence=torch.zeros(1))
    for name in ("proximity", "role_coordination", "search_coverage",
                 "evacuee_in_hlr_fov", "evacuee_in_llr_fov"):
        assert float(ALL[name](blind)[0]) == 0.0, f"{name} paid for a stale guess"


def test_coordination_rewards_a_division_of_labour():
    """The paper's l7, and the term its title rests on.

    Two drones in the same place seeing the same person is not coordination.
    """
    spread = view()
    stacked = view(drone_pos=torch.tensor([[[5.0, 0.0, 40.0], [5.0, 0.0, 3.0]]]))
    assert float(ALL["role_coordination"](spread)[0]) > \
        float(ALL["role_coordination"](stacked)[0])


def test_an_unknown_term_is_refused_loudly():
    with pytest.raises(KeyError):
        Composer({"free_lunch": 1.0})


def test_ablation_drops_terms():
    c = Composer().without("role_coordination", "capture")
    assert "role_coordination" not in c.weights and "capture" not in c.weights
    assert "proximity" in c.weights


# ---------------------------------------------------------------- environment

@pytest.fixture(scope="module")
def env():
    return TrainEnv(small(), batch=8)


def test_observations_match_the_contract(env):
    obs = env.reset()
    for who in ("hlr", "llr"):
        spec.validate(obs[who], batch=env.batch)


def test_the_raster_only_shows_what_was_looked_at(env):
    """Buildings are prior knowledge -- a real team has a map.  Fire and smoke are
    not, and handing a policy a perfect fire map is the same oracle hole as handing
    it the person's position."""
    obs = env.reset()
    raster = obs["hlr"]["raster"]
    channels = [name for name, _ in spec.RASTER_CHANNELS]
    fire = raster[:, channels.index("fire")]
    looked = raster[:, channels.index("searched")]
    assert float(fire[looked == 0].abs().max()) == 0.0, "saw fire it never looked at"


def test_stepping_returns_what_a_learner_needs(env):
    env.reset()
    obs, reward, done, info = env.step(torch.zeros(env.batch, 2, 4))
    assert reward.shape == (env.batch,) and torch.isfinite(reward).all()
    assert done.shape == (env.batch,) and done.dtype == torch.bool
    assert "parts" in info and "captured" in info


def test_an_episode_ends_on_time(env):
    env.reset()
    for _ in range(env.clock.max_steps):
        _, _, done, _ = env.step(torch.zeros(env.batch, 2, 4))
    assert bool(done.all())


def test_drones_launch_from_a_base_not_the_persons_doorstep():
    """Starting them on top of each other makes the task trivial: a policy that
    captures on the first step has learned nothing about searching."""
    e = TrainEnv(small(), batch=4)
    e.reset()
    gap = (e.drones[1].pos[:, :2] - e.human.pos).norm(dim=-1)
    assert float(gap.min()) > 10.0, "the low rescuer starts within arm's reach"


def test_the_two_drones_fly_at_different_altitudes():
    """High and wide against low and narrow is the paper's whole premise."""
    e = TrainEnv(small(), batch=4)
    e.reset()
    assert float(e.drones[0].pos[:, 2].mean()) > float(e.drones[1].pos[:, 2].mean()) + 20


def test_the_environment_is_reproducible():
    a, b = TrainEnv(small(), batch=4), TrainEnv(small(), batch=4)
    act = torch.zeros(4, 2, 4)
    for _ in range(4):
        _, ra, _, _ = a.step(act)
        _, rb, _, _ = b.step(act)
    assert torch.allclose(ra, rb)


# --------------------------------------------------------------------- policy

def test_the_policy_produces_commands_inside_the_contract():
    from hadrsim.policy.ppo import ActorCritic

    e = TrainEnv(small(), batch=4)
    net = ActorCritic()
    action, logp, value, state = net.act(e.reset())
    assert action.shape == (4, 2, spec.ACTION_DIM)
    assert float(action.detach().abs().max()) <= 1.0, "commanded outside [-1, 1]"
    assert logp.shape == (4,) and value.shape == (4,)


@pytest.mark.parametrize("encoder", ["flat", "conv"])
def test_both_encoders_work(encoder):
    from hadrsim.policy.ppo import ActorCritic

    e = TrainEnv(small(), batch=4)
    action, _, _, _ = ActorCritic(encoder).act(e.reset())
    assert torch.isfinite(action).all()


def test_the_policy_can_learn_something():
    """Slow, and deliberately weak: this is a pipeline test, not a quality one.

    It asks only that gradients flow and that reward responds, so that a broken
    pipeline fails in twenty seconds here rather than after an overnight run.
    """
    from hadrsim.policy.ppo import PPO, PPOConfig

    e = TrainEnv(small(), batch=32)
    ppo = PPO(e, PPOConfig(horizon=6, epochs=2))
    before = [p.detach().clone() for p in ppo.net.parameters()]
    history = ppo.train(3, log_every=99)

    assert len(history) == 3
    assert all(torch.isfinite(torch.tensor(h["reward"])) for h in history)
    moved = any(not torch.equal(a, b) for a, b in zip(before, ppo.net.parameters()))
    assert moved, "no parameter changed; nothing is learning"
