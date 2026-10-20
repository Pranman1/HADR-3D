"""Our eight city-block terms, tested beside them.  Run with `pytest experiments`."""

import pathlib

import pytest
import torch

from hadrsim.reward.compose import Composer, load_weights
from hadrsim.reward.terms import TERMS, View

CITY = load_weights(str(pathlib.Path(__file__).with_name("city_reward.yaml")))   # registers the terms
OURS = ("reached_safety", "goal_progress", "search_coverage", "first_sight",
        "coverage", "apparent_size", "evacuee_harmed", "control_effort")


def view(**over) -> View:
    b = over.pop("batch", 1)
    base = dict(
        drone_pos=torch.tensor([[[0.0, 0.0, 40.0], [5.0, 0.0, 3.0]]] * b),
        person_pos=torch.tensor([[8.0, 0.0, 0.9]] * b),
        person_seen=torch.ones(b, dtype=torch.bool),
        seen_by=torch.ones(b, 2, dtype=torch.bool),
        confidence=torch.full((b,), 0.9),
        goal=torch.tensor([[40.0, 40.0, 0.0]] * b),
        captured=torch.zeros(b, dtype=torch.bool),
        was_captured=torch.zeros(b, dtype=torch.bool),
        fire_near=torch.zeros(b, 2),
        action=torch.zeros(b, 2, 4),
    )
    base.update(over)
    return View(**base)


def test_the_yaml_registers_our_terms_and_weights_all_fifteen():
    assert set(OURS) <= set(TERMS) and len(CITY) == 15


def test_every_term_returns_one_magnitude_per_environment():
    v = view(batch=5)
    for name in OURS:
        out = TERMS[name](v)
        assert out.shape == (5,) and torch.isfinite(out).all() and bool((out >= 0).all()), name


def test_penalties_are_weighted_negative():
    for name in ("control_effort", "evacuee_harmed"):
        assert CITY[name] < 0, f"{name} should be weighted negative"


def test_reaching_safety_pays_only_when_the_drones_helped():
    """Otherwise the team is paid for a person who walked out unaided -- seen as
    reward climbing past +60 with the capture rate at zero (findings §1)."""
    one = torch.ones(1, dtype=torch.bool)
    unaided, led = view(newly_reached=one), view(newly_reached=one, was_captured=one)
    assert float(TERMS["reached_safety"](unaided)[0]) == 0.0, "self-rescue paid out"
    assert float(TERMS["reached_safety"](led)[0]) > 0.0
    gain = float(Composer(CITY)(led).total) - float(Composer(CITY)(unaided).total)
    assert gain > 50.0, "an actual rescue should be worth a great deal more"


def test_outcome_rewards_are_paid_once_not_every_step_after():
    """The latched flags stay true; paying on them was 8,600 for one rescue
    against 200 for the capture that caused it (findings §1).  The edge is paid."""
    one, zero = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    assert float(TERMS["reached_safety"](view(newly_reached=one, was_captured=one))[0]) > 0.0
    assert float(TERMS["reached_safety"](view(newly_reached=zero, was_captured=one))[0]) == 0.0
    assert float(TERMS["evacuee_harmed"](view(harmed=one, newly_harmed=one))[0]) == 1.0
    assert float(TERMS["evacuee_harmed"](view(harmed=one, newly_harmed=zero))[0]) == 0.0


def test_leading_someone_to_safety_pays_for_the_whole_walk():
    """Gated on `was_captured`, not the one-step `captured` edge, which paid 0.19
    points out of 760 for the second half of a rescue (findings §1)."""
    one = torch.ones(1, dtype=torch.bool)
    near, far, goal = torch.zeros(1, 3), torch.tensor([[200.0, 0.0, 0.0]]), torch.zeros(1, 3)
    holding = view(person_pos=far, goal=goal, was_captured=one)
    delivered = view(person_pos=near, goal=goal, was_captured=one)
    assert float(TERMS["goal_progress"](delivered)[0]) > float(TERMS["goal_progress"](holding)[0])
    assert float(TERMS["goal_progress"](view(person_pos=near, goal=goal))[0]) == 0.0


def test_terms_that_need_a_sighting_pay_nothing_without_one():
    blind = view(person_seen=torch.zeros(1, dtype=torch.bool),
                 seen_by=torch.zeros(1, 2, dtype=torch.bool), confidence=torch.zeros(1))
    for name in ("search_coverage", "apparent_size"):
        assert float(TERMS[name](blind)[0]) == 0.0, f"{name} paid without a sighting"


def test_apparent_size_grows_as_the_low_rescuer_closes():
    close = view(drone_pos=torch.tensor([[[0.0, 0.0, 40.0], [6.0, 0.0, 1.5]]]))
    assert float(TERMS["apparent_size"](close)[0]) > float(TERMS["apparent_size"](view())[0]) > 0.0
