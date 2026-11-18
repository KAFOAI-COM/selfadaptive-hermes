"""Offline tests; no external services required."""
import pytest
from evolution.adaptive.core import (
    ContextualBandit, ExperienceStore, Reward, choose_strategy, score_outcome
)


def test_reward_and_hard_violation():
    assert score_outcome(Reward(1.0, 0, 0)) == 1.0
    assert score_outcome(Reward(1.0, 0, 0, violation=True)) == -1.0
    with pytest.raises(ValueError):
        score_outcome(Reward(1.2, 0, 0))


def test_policy_learns_context_specific_actions():
    policy = ContextualBandit(("a", "b"), epsilon=0, seed=1)
    for _ in range(15):
        policy.update("easy", "a", 0.9)
        policy.update("easy", "b", 0.2)
        policy.update("hard", "a", 0.1)
        policy.update("hard", "b", 0.8)
    assert policy.select("easy")[0] == "a"
    assert policy.select("hard")[0] == "b"


def test_propensity_and_reproducibility():
    a = ContextualBandit(("a", "b"), epsilon=0.2, seed=3)
    b = ContextualBandit(("a", "b"), epsilon=0.2, seed=3)
    assert [a.select("x") for _ in range(30)] == [b.select("x") for _ in range(30)]
    assert all(0 < p <= 1 for _, p in [a.select("x") for _ in range(20)])


def test_mapek_roundtrip(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    policy = ContextualBandit(("safe",), epsilon=0.1)
    record = choose_strategy(
        policy, "security",
        lambda ctx, action: {"passed": True},
        lambda result: Reward(float(result["passed"]), 0.2, 0.1),
        store,
    )
    assert record["strategy"] == "safe"
    assert len(store.rows()) == 1
    assert policy.counts[("security", "safe")] == 1
    store.close()


def test_unknown_strategy_rejected():
    policy = ContextualBandit(("allowed",))
    with pytest.raises(ValueError):
        policy.update("x", "not-allowed", 1.0)
