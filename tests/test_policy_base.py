"""The policy seam: scripted baselines, checkpoint manifests, the shared evaluator."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hadrsim.core import spec
from hadrsim.env import Env
from hadrsim.eval import evaluate
from hadrsim.policy import base
from hadrsim.policy.ppo import PPO, PPOConfig
from hadrsim.scenario import Scenario, autoplace


def _env(batch=4):
    sc = autoplace(Scenario.preset("Paper grid (25 m)", cell_size_m=1.0, episode_seconds=10.0,
                                   wind_mps=0.0, seed=1))
    return Env(sc, batch=batch, fidelity="kinematic", device="cpu")


def test_every_scripted_baseline_emits_a_contract_action():
    env = _env(); obs = env.reset()
    for name in base.SCRIPTED:
        pol = base.load(name)
        act, _, _, state = pol.act(obs, None)
        assert tuple(act.shape) == (4, 2, spec.ACTION_DIM), name
        assert float(act.abs().max()) <= 1.0, name
        act2, *_ = pol.act(obs, state)
        assert tuple(act2.shape) == (4, 2, spec.ACTION_DIM)


def test_checkpoints_carry_a_manifest_and_load_through_the_seam(tmp_path):
    env = _env()
    ppo = PPO(env, PPOConfig(horizon=2, epochs=1), encoder="conv", device="cpu")
    path = tmp_path / "tiny.pt"
    ppo.save(path)
    blob = torch.load(path, map_location="cpu")
    assert blob["manifest"]["encoder"] == "conv" and blob["manifest"]["spec_version"] == spec.SPEC_VERSION
    pol = base.load(path)
    assert isinstance(pol, base.Learned) and pol.manifest["encoder"] == "conv"
    act, *_ = pol.act(env.reset(), None, deterministic=True)
    assert tuple(act.shape) == (4, 2, spec.ACTION_DIM)


def test_a_checkpoint_from_another_contract_version_is_refused(tmp_path):
    env = _env()
    ppo = PPO(env, PPOConfig(horizon=2, epochs=1), device="cpu")
    ppo.net.manifest["spec_version"] = spec.SPEC_VERSION + 1
    path = tmp_path / "old.pt"; ppo.save(path)
    try:
        base.load(path)
    except ValueError as e:
        assert "contract" in str(e)
    else:
        raise AssertionError("loaded a checkpoint trained against a different contract")


def test_evaluate_counts_full_episodes_and_the_floor_is_zero_movement():
    env = _env(batch=6)
    m = evaluate(env, None)
    assert m.episodes == 6 and 0.0 <= m.capture <= 100.0 and 0.0 <= m.seen <= 100.0
    assert m.in_view >= 0.0 and "capture" in m.parts
    m2 = evaluate(env, base.Hover())
    assert m2.capture == m.capture, "hover must be the same floor as no policy"
