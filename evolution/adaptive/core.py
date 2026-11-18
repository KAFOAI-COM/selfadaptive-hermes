"""Minimal, dependency-free MAPE-K control primitives.

This module does NOT train model weights. It selects strategy IDs from a
pre-approved catalogue and learns from externally verified task outcomes.
The independent verifier and Hermes executor are injected by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
import sqlite3
from pathlib import Path


@dataclass(frozen=True)
class Reward:
    quality: float       # verified, normalised [0, 1]
    cost_ratio: float    # nonnegative cost / fixed reference cost
    time_ratio: float    # nonnegative duration / fixed reference duration
    violation: bool = False


def score_outcome(outcome: Reward, cost_weight: float = 0.10,
                  time_weight: float = 0.05) -> float:
    """Fixed utility function; critical violations receive disqualifying reward.

    Do not expose this function to the optimising agent as a writable artifact.
    """
    if not 0.0 <= outcome.quality <= 1.0:
        raise ValueError("quality must be between 0 and 1")
    if outcome.cost_ratio < 0 or outcome.time_ratio < 0:
        raise ValueError("cost/time ratios must be nonnegative")
    if cost_weight < 0 or time_weight < 0:
        raise ValueError("penalty weights must be nonnegative")
    if outcome.violation:
        return -1.0
    return outcome.quality - cost_weight * outcome.cost_ratio - time_weight * outcome.time_ratio


class ContextualBandit:
    """Seedable epsilon-greedy, per-context value estimates and action counts.

    This is the FIRST baseline, not LinUCB or weight-level RL. An unseen
    context starts with balanced estimates. Use broad, predefined contexts.
    """

    def __init__(self, strategies: tuple[str, ...], epsilon: float = 0.15,
                 seed: int = 0):
        if not strategies or len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be a nonempty set of unique IDs")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.strategies = strategies
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.counts: dict[tuple[str, str], int] = {}
        self.means: dict[tuple[str, str], float] = {}

    def select(self, context: str) -> tuple[str, float]:
        if not context:
            raise ValueError("context cannot be empty")
        # Explicit uniform exploration plus uniform random tie resolution.
        values = [self.means.get((context, a), 0.0) for a in self.strategies]
        top = max(values)
        winners = [a for a, v in zip(self.strategies, values) if v == top]
        greedy_prob = (1 - self.epsilon) / len(winners)
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.strategies)
        else:
            action = self.rng.choice(winners)
        propensity = self.epsilon / len(self.strategies)
        if action in winners:
            propensity += greedy_prob
        return action, propensity

    def update(self, context: str, strategy: str, reward: float) -> None:
        if strategy not in self.strategies:
            raise ValueError("unknown strategy")
        if not context or not isinstance(reward, (int, float)):
            raise ValueError("invalid experience")
        key = (context, strategy)
        n = self.counts.get(key, 0) + 1
        mean = self.means.get(key, 0.0)
        self.means[key] = mean + (float(reward) - mean) / n
        self.counts[key] = n


class ExperienceStore:
    """Local research history only; no enterprise audit or user data required."""

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL, strategy TEXT NOT NULL,
            propensity REAL NOT NULL, reward REAL NOT NULL,
            feedback_json TEXT NOT NULL
        )""")
        self.conn.commit()

    def record(self, context: str, strategy: str, propensity: float,
               reward: float, feedback: dict) -> None:
        if not 0 < propensity <= 1:
            raise ValueError("invalid logged selection probability")
        self.conn.execute(
            "INSERT INTO experiences (context,strategy,propensity,reward,feedback_json) "
            "VALUES (?,?,?,?,?)",
            (context, strategy, propensity, reward, json.dumps(feedback)))
        self.conn.commit()

    def rows(self) -> list[tuple]:
        return list(self.conn.execute(
            "SELECT context,strategy,propensity,reward FROM experiences ORDER BY id"))

    def close(self) -> None:
        self.conn.close()


def choose_strategy(policy: ContextualBandit, context: str,
                    executor, verifier, store: ExperienceStore) -> dict:
    """One complete Monitor/Analyse/Plan/Legitimise/Execute/Knowledge iteration.

    Caller supplies a Hermes-compatible executor(context, strategy) and an
    independent verifier(result)->Reward. No production actions permitted
    by this module. The caller is responsible for sandboxing the executor.
    """
    strategy, propensity = policy.select(context)   # PLAN
    result = executor(context, strategy)            # EXECUTE
    outcome = verifier(result)                      # MONITOR / ANALYSE
    if not isinstance(outcome, Reward):
        raise TypeError("verifier must return Reward")
    reward = score_outcome(outcome)                  # LEGITIMISE
    store.record(context, strategy, propensity, reward,
                 {"quality": outcome.quality, "cost_ratio": outcome.cost_ratio,
                  "time_ratio": outcome.time_ratio, "violation": outcome.violation})
    policy.update(context, strategy, reward)        # KNOWLEDGE
    return {"context": context, "strategy": strategy,
            "propensity": propensity, "reward": reward}
