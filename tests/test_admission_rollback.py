"""Gate 6 admission composed with gate 7 checkpoint/rollback (synthetic, FakeExecutor).

``checkpoint.admit_with_rollback`` takes a caller-supplied ``admit``. Here that
step is the real gate 6 review (``admission.review_registered(apply=True)``) on
a real gate 5 ``CandidateRegistry``, so the stricter admission rule and the
rollback trigger run together. No production wiring is added.
"""
import json

import pytest

from evolution.adaptive import admission as adm
from evolution.adaptive import benchmark as bm
from evolution.adaptive import checkpoint as ck
from evolution.adaptive import evaluation as ev
from evolution.adaptive import registry as reg
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy
from evolution.adaptive.policies import Context, EpsilonGreedyPolicy

BASE_STRATEGY = Strategy("reviewer")
ARMS = ("baseline", "structured")


@pytest.fixture(scope="module")
def validation():
    return bm.load_split("validation")


@pytest.fixture
def registry(tmp_path):
    with reg.CandidateRegistry(tmp_path / "registry.db") as r:
        yield r


@pytest.fixture
def store():
    s = ck.CheckpointStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def pair(registry):
    base = registry.register_baseline("code-review", b"baseline skill\n")
    cand = registry.propose("code-review", b"evolved skill\n", parent_id=base.candidate_id,
                            provenance={"gepa_score": 0.99})
    return base, cand


def _truth(case):
    return json.dumps(ev.ground_truth_findings(case))


def _make_executor(validation, cand_id, base_id, better):
    def make(seed):
        responses = {}
        for case in validation:
            for cid, out in ((cand_id, _truth(case) if better else "[]"), (base_id, "[]")):
                responses[(case.case_id, f"eval:reviewer:{cid}")] = FakeResponse(
                    output=out, cost_usd=0.01, latency_s=2.0)
        return FakeExecutor(responses)
    return make


def _policy():
    p = EpsilonGreedyPolicy(ARMS, epsilon=0.2, seed=3)
    for _ in range(5):
        p.update(Context("simple", (1.0, 0.0, 0.0)), "baseline", 0.4)
    return p


class _Validator:
    """Fake gate 7 validation: fixed utilities; checks the phase guard on each call."""

    def __init__(self, pre, post, post_violation=False):
        self.results = [ck.Validation(pre, False, 10),
                        ck.Validation(post, post_violation, 10)]
        self.calls = 0

    def __call__(self, policy):
        assert bm.in_optimisation_phase()
        with pytest.raises(bm.SealedSplitError):   # still sealed after gate 6's phase exits
            bm.load_split("test", allow_sealed_test=True)
        result = self.results[self.calls]
        self.calls += 1
        return result


def _gate6_admit(registry, validation, pair, better, decisions):
    base, cand = pair
    make = _make_executor(validation, cand.candidate_id, base.candidate_id, better)

    def admit(policy):
        policy.update(Context("simple", (1.0, 0.0, 0.0)), "structured", 0.9)   # the change
        decision = adm.review_registered(registry, cand.candidate_id, base=BASE_STRATEGY,
                                         make_executor=make, apply=True)
        decisions.append(decision)
        if not decision.admitted:
            raise RuntimeError("gate 6 rejected: " + "; ".join(decision.reasons))
        return cand.candidate_id
    return admit


def _run(store, registry, validation, pair, *, better, pre, post, post_violation=False):
    decisions = []
    policy = _policy()
    before = ck.canonical_json(ck.policy_state(policy))
    validate = _Validator(pre, post, post_violation)
    call = lambda: ck.admit_with_rollback(   # noqa: E731
        store, policy, skill_version=pair[0].candidate_id,
        admit=_gate6_admit(registry, validation, pair, better, decisions), validate=validate)
    return call, policy, before, decisions


def test_gate6_admits_and_gate7_keeps(store, registry, validation, pair):
    call, policy, before, decisions = _run(store, registry, validation, pair,
                                           better=True, pre=0.40, post=0.45)
    result = call()
    base, cand = pair
    assert decisions[0].admitted and result.admitted
    assert result.skill_version == cand.candidate_id
    assert result.active_checkpoint.status == ck.STATUS_GOOD
    assert ck.canonical_json(ck.policy_state(policy)) != before   # change kept
    assert registry.get(cand.candidate_id).status == reg.ADMITTED
    assert adm.admission_evidence_errors(registry, cand.candidate_id) == []
    ids = {s.strategy_id for s in adm.admitted_catalogue(registry, "code-review", BASE_STRATEGY)}
    assert len(ids) == 2 and any(cand.candidate_id in i for i in ids)
    assert not bm.in_optimisation_phase()


def test_gate6_rejects_and_gate7_restores(store, registry, validation, pair):
    call, policy, before, decisions = _run(store, registry, validation, pair,
                                           better=False, pre=0.40, post=0.45)
    with pytest.raises(RuntimeError, match="gate 6 rejected"):
        call()
    assert not decisions[0].admitted
    assert registry.get(pair[1].candidate_id).status == reg.REJECTED
    assert ck.canonical_json(ck.policy_state(policy)) == before
    assert [e["event"] for e in store.log()][-1] == "restore"
    assert not bm.in_optimisation_phase()


@pytest.mark.parametrize("post, violation", [(0.30, False), (0.45, True)])
def test_gate6_admits_but_gate7_rolls_back(store, registry, validation, pair, post, violation):
    """Current limit, pinned: gate 7 rolls the policy and skill ID back to the
    parent, but the registry entry stays ``admitted`` (gate 5 allows only
    admitted -> retired, and only gate 6 may call ``transition``)."""
    call, policy, before, decisions = _run(store, registry, validation, pair, better=True,
                                           pre=0.40, post=post, post_violation=violation)
    result = call()
    base, cand = pair
    assert decisions[0].admitted and not result.admitted
    assert result.skill_version == base.candidate_id
    assert result.candidate_checkpoint.status == ck.STATUS_ROLLED_BACK
    assert ck.canonical_json(ck.policy_state(policy)) == before
    assert registry.get(cand.candidate_id).status == reg.ADMITTED   # not retired: see docstring
    assert not bm.in_optimisation_phase()
