import pytest
from model_arena.bandit import pairwise_updates, select_models, compute_stats


def test_pairwise_updates_clear_winner():
    scores = {"A": 4.0, "B": 3.0, "C": 2.0}
    wins, losses = pairwise_updates(scores, tie_epsilon=0.3)
    assert wins["A"] == 2.0  # beats B and C
    assert losses["A"] == 0.0
    assert wins["C"] == 0.0
    assert losses["C"] == 2.0
    assert wins["B"] == 1.0  # beats C
    assert losses["B"] == 1.0  # loses to A


def test_pairwise_updates_tie():
    scores = {"A": 3.5, "B": 3.3}
    wins, losses = pairwise_updates(scores, tie_epsilon=0.3)
    # Within epsilon: tie
    assert wins["A"] == 0.5
    assert wins["B"] == 0.5
    assert losses["A"] == 0.5
    assert losses["B"] == 0.5


def test_pairwise_updates_just_outside_epsilon():
    scores = {"A": 3.5, "B": 3.1}
    wins, losses = pairwise_updates(scores, tie_epsilon=0.3)
    # delta=0.4 > 0.3 → A wins
    assert wins["A"] == 1.0
    assert losses["B"] == 1.0


def test_select_models_returns_n():
    # With no history, all models are Beta(1,1) → random selection
    history = {}  # model_id -> (total_wins, total_losses)
    eligible = ["m1", "m2", "m3", "m4"]
    selected = select_models(eligible, n=3, history=history)
    assert len(selected) == 3
    assert all(m in eligible for m in selected)


def test_select_models_all_if_n_ge_eligible():
    eligible = ["m1", "m2"]
    selected = select_models(eligible, n=3, history={})
    assert set(selected) == {"m1", "m2"}


def test_select_models_favors_winner_over_time():
    # After many wins, model should be selected more often
    history = {
        "good": (50.0, 2.0),
        "bad": (2.0, 50.0),
        "neutral": (5.0, 5.0),
    }
    selections = {"good": 0, "bad": 0, "neutral": 0}
    for _ in range(200):
        picked = select_models(["good", "bad", "neutral"], n=1, history=history)
        selections[picked[0]] += 1
    # "good" should dominate
    assert selections["good"] > selections["bad"]
    assert selections["good"] > 100  # strongly favored


def test_compute_stats_empty():
    stats = compute_stats([])
    assert stats == {}


def test_compute_stats_aggregates():
    # List of (model_id, pairwise_wins, pairwise_losses) from DB
    records = [
        ("m1", 1.5, 0.5),
        ("m1", 1.0, 1.0),
        ("m2", 0.5, 1.5),
        ("m2", 1.0, 1.0),
    ]
    stats = compute_stats(records)
    assert stats["m1"] == (2.5, 1.5)
    assert stats["m2"] == (1.5, 2.5)
