"""Gate 6 candidate-vs-baseline admission. Offline: FakeExecutor only."""
import dataclasses
import inspect
import json
import logging
import os
import sys

import pytest

from evolution.adaptive import admission as adm
from evolution.adaptive import benchmark as bm
from evolution.adaptive import evaluation as ev
from evolution.adaptive.admission import ADMISSION_POLICY, AdmissionPolicy, ArmSpec, admit
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy

# Changing any admission threshold changes this; update it only in a reviewed change.
PINNED_POLICY_SHA256 = "1c3cd3e28dd5f9d140183bae6a747b7b0f396358adb93d14eeceff6913f0f4ed"

SEALED_FILE = bm.BENCHMARK_ROOT / "v1" / "sealed" / "test.jsonl"
CANDIDATE = ArmSpec("cand-1", Strategy("candidate"))
BASELINE = ArmSpec("base-1", Strategy("baseline"))


@pytest.fixture(scope="module")
def validation():
    return bm.load_split("validation")


def _truth(case):
    return json.dumps(ev.ground_truth_findings(case))


def factory(validation, cand=None, base=None, calls=None, during=None):
    """make_executor(seed): candidate/baseline response per case (default: silent)."""
    def response(spec, case):
        value = spec(case) if callable(spec) else spec
        if isinstance(value, FakeResponse):
            return value
        return FakeResponse(output=value if value is not None else "[]",
                            cost_usd=0.01, latency_s=2.0)

    def make_executor(seed):
        if calls is not None:
            calls.append(seed)
        responses = {}
        for case in validation:
            responses[(case.case_id, "candidate")] = response(cand, case)
            responses[(case.case_id, "baseline")] = response(base, case)
        executor = FakeExecutor(responses)
        if during is not None:
            original = executor.execute

            def execute(task, strategy):
                during(task, strategy)
                return original(task, strategy)
            executor.execute = execute
        make_executor.instances.append(executor)
        return executor
    make_executor.instances = []
    return make_executor


# ── Decisions ────────────────────────────────────────────────────────────────

def test_clearly_better_candidate_is_admitted(validation, caplog):
    with caplog.at_level(logging.INFO, logger=adm.__name__):
        d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth))
    assert d.admitted and d.reasons == ()
    assert d.mean_utility_gain >= ADMISSION_POLICY.min_margin and d.ci_low > 0
    assert d.candidate_mean_quality == 1.0
    clean_share = sum(c.is_clean for c in validation) / len(validation)
    assert d.baseline_mean_quality == pytest.approx(clean_share)
    assert d.synthetic is True and "SYNTHETIC" in d.notice
    assert d.n_cases == 10 and set(d.case_ids) == {c.case_id for c in validation}
    assert "ADMIT cand-1" in caplog.text


def test_no_improvement_is_rejected_with_both_reasons(validation, caplog):
    with caplog.at_level(logging.WARNING, logger=adm.__name__):
        d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth, base=_truth))
    assert not d.admitted and d.mean_utility_gain == 0.0
    assert [r.split(":")[0] for r in d.reasons] == ["margin_not_met", "ci_includes_zero"]
    assert "REJECT cand-1" in caplog.text and "margin_not_met" in caplog.text


def test_gain_on_one_case_meets_margin_but_ci_includes_zero(validation):
    fixed = next(c for c in validation if not c.is_clean)
    d = admit(CANDIDATE, BASELINE, factory(
        validation, cand=lambda c: _truth(c) if c.case_id == fixed.case_id else "[]"))
    assert d.mean_utility_gain >= ADMISSION_POLICY.min_margin
    assert not d.admitted and [r.split(":")[0] for r in d.reasons] == ["ci_includes_zero"]


def test_consistent_but_small_gain_fails_the_margin(validation):
    # Same reviews; candidate is slightly cheaper on every case: CI excludes 0,
    # but 0.10 * (0.006 / 0.03) = 0.02 < min_margin.
    cheap = lambda c: FakeResponse(output="[]", cost_usd=0.0, latency_s=2.0)    # noqa: E731
    dear = lambda c: FakeResponse(output="[]", cost_usd=0.006, latency_s=2.0)   # noqa: E731
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=cheap, base=dear))
    assert d.mean_utility_gain == pytest.approx(0.02) and d.ci_low > 0
    assert not d.admitted and [r.split(":")[0] for r in d.reasons] == ["margin_not_met"]


def test_any_critical_violation_rejects_even_a_better_candidate(validation):
    bad = validation[0].case_id

    def cand(case):
        tools = ("terminal",) if case.case_id == bad else ()
        return FakeResponse(output=_truth(case), cost_usd=0.01, latency_s=2.0,
                            tool_calls=tools)
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=cand))
    assert d.candidate_violations == len(ADMISSION_POLICY.seeds)
    assert not d.admitted
    assert d.reasons[0].startswith("critical_violations") and "disallowed_tool:terminal" in d.reasons[0]


def test_unknown_cost_cannot_be_admitted(validation):
    d = admit(CANDIDATE, BASELINE, factory(
        validation, cand=lambda c: FakeResponse(output=_truth(c), cost_usd=None)))
    assert not d.admitted and d.reasons[0].startswith("unscorable_outcomes")
    assert d.mean_utility_gain is None


def test_unparseable_output_scores_as_an_empty_review(validation):
    d = admit(CANDIDATE, BASELINE, factory(validation, cand="not json", base="[]"))
    assert d.unparseable_outputs == len(validation) * len(ADMISSION_POLICY.seeds)
    assert d.mean_utility_gain == 0.0 and not d.admitted


# ── Same split, executor and seeds; sealed split unreachable ─────────────────

def test_both_arms_use_same_cases_executor_factory_and_seeds(validation):
    calls = []
    make = factory(validation, cand=_truth, calls=calls)
    admit(CANDIDATE, BASELINE, make)
    seeds = list(ADMISSION_POLICY.seeds)
    assert calls == seeds + seeds                       # once per arm and seed
    assert len({id(e) for e in make.instances}) == 2 * len(seeds)   # no shared state
    cand_runs = [c for e in make.instances for c in e.calls if c[1] == "candidate"]
    base_runs = [c for e in make.instances for c in e.calls if c[1] == "baseline"]
    assert sorted(t for t, _ in cand_runs) == sorted(t for t, _ in base_runs)
    assert {t for t, _ in cand_runs} == {c.case_id for c in validation}


_OPENED = []
_RECORDING = False


def _recorder(event, args):
    if _RECORDING and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_recorder)


def test_admission_runs_in_optimisation_phase_and_never_opens_sealed(validation):
    global _RECORDING
    phases = []
    _OPENED.clear()
    _RECORDING = True
    try:
        admit(CANDIDATE, BASELINE, factory(
            validation, cand=_truth, during=lambda t, s: phases.append(bm.in_optimisation_phase())))
    finally:
        _RECORDING = False
    assert phases and all(phases)
    assert [p for p in _OPENED if bm._is_sealed_path(p)] == []


@pytest.mark.parametrize("attack", ["open_file", "unseal_loader"])
def test_admission_fails_if_executor_touches_sealed_split(validation, attack):
    def during(task, strategy):
        if attack == "open_file":
            SEALED_FILE.read_bytes()
        else:
            bm.load_split("test", allow_sealed_test=True)
    with pytest.raises(bm.SealedSplitError):
        admit(CANDIDATE, BASELINE, factory(validation, cand=_truth, during=during))


# ── Fixed configuration ──────────────────────────────────────────────────────

def test_thresholds_are_not_parameters_of_admit():
    assert list(inspect.signature(admit).parameters) == ["candidate", "baseline", "make_executor"]


def test_policy_is_frozen_and_pinned():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ADMISSION_POLICY.min_margin = 0.0
    assert ADMISSION_POLICY.fingerprint() == PINNED_POLICY_SHA256
    d_policy = ADMISSION_POLICY.to_dict()
    assert d_policy["split"] == "validation" and d_policy["max_critical_violations"] == 0


def test_decision_records_policy_and_verifier(validation):
    d = admit(CANDIDATE, BASELINE, factory(validation, cand=_truth))
    assert d.policy_sha256 == PINNED_POLICY_SHA256
    assert d.policy == ADMISSION_POLICY.to_dict()
    assert len(d.verifier_sha256) == 64
    assert json.loads(json.dumps(d.to_dict()))["admitted"] is True


@pytest.mark.parametrize("change", [dict(split="test"), dict(split="train"),
                                    dict(max_critical_violations=1), dict(min_margin=0.0),
                                    dict(seeds=()), dict(seeds=(1, 1))])
def test_policy_refuses_unsafe_settings(change):
    with pytest.raises(ValueError):
        AdmissionPolicy(**change)


# ── Input checks ─────────────────────────────────────────────────────────────

def test_candidate_shape_is_checked(validation):
    make = factory(validation)
    with pytest.raises(ValueError, match="differ"):
        admit(CANDIDATE, ArmSpec("cand-1", Strategy("baseline")), make)
    with pytest.raises(TypeError):
        admit(ArmSpec("x", "not a strategy"), BASELINE, make)
    with pytest.raises(TypeError):
        admit(object(), BASELINE, make)
    assert isinstance(CANDIDATE, adm.Arm)


def test_parse_findings():
    assert adm.parse_findings('[{"line_start": 1}]') == ([{"line_start": 1}], True)
    assert adm.parse_findings('{"findings": []}') == ([], False)
    assert adm.parse_findings("") == ([], False)


# ── Gate 5 registry review ───────────────────────────────────────────────────

from types import SimpleNamespace  # noqa: E402

BASE_STRATEGY = Strategy("reviewer")


class FakeRegistry:
    """Mirrors the gate 5 CandidateRegistry interface used by admission."""

    def __init__(self):
        self.entries, self.blobs, self.transitions = {}, {}, []

    def add(self, cid, status, parent_id=None, skill="code-review", text="skill"):
        self.entries[cid] = SimpleNamespace(candidate_id=cid, skill_name=skill,
                                            parent_id=parent_id, status=status)
        self.blobs[cid] = text.encode()

    def get(self, cid):
        return self.entries[cid]

    def content(self, cid):
        return self.blobs[cid]

    def transition(self, cid, to_status, *, reason, decided_by, evidence):
        allowed = {"proposed": {"admitted", "rejected"}}
        assert to_status in allowed.get(self.entries[cid].status, set())
        assert reason and decided_by and (evidence or to_status != "admitted")
        json.dumps(evidence, allow_nan=False)
        self.entries[cid].status = to_status
        self.transitions.append((cid, to_status, reason, decided_by, evidence))


def registry_factory(validation, cand_id, base_id, cand_spec, base_spec="[]"):
    def make_executor(seed):
        responses = {}
        for case in validation:
            for cid, spec in ((cand_id, cand_spec), (base_id, base_spec)):
                out = spec(case) if callable(spec) else spec
                responses[(case.case_id, f"reviewer:{cid}")] = FakeResponse(
                    output=out, cost_usd=0.01, latency_s=2.0)
        return FakeExecutor(responses)
    return make_executor


@pytest.fixture
def registry():
    reg = FakeRegistry()
    reg.add("code-review@base", "admitted", text="baseline skill")
    reg.add("code-review@cand", "proposed", parent_id="code-review@base", text="evolved skill")
    return reg


def test_review_is_a_dry_run_by_default(validation, registry):
    d = adm.review_registered(registry, "code-review@cand", base=BASE_STRATEGY,
                              make_executor=registry_factory(
                                  validation, "code-review@cand", "code-review@base", _truth))
    assert d.admitted and d.baseline_id == "code-review@base"
    assert registry.transitions == [] and registry.get("code-review@cand").status == "proposed"


@pytest.mark.parametrize("better", [True, False])
def test_review_applies_admission_or_rejection_with_evidence(validation, registry, better):
    make = registry_factory(validation, "code-review@cand", "code-review@base",
                            _truth if better else "[]")
    d = adm.review_registered(registry, "code-review@cand", base=BASE_STRATEGY,
                              make_executor=make, apply=True)
    ((cid, status, reason, decided_by, evidence),) = registry.transitions
    assert status == ("admitted" if better else "rejected") and d.admitted is better
    assert decided_by == f"admission-gate:{PINNED_POLICY_SHA256[:12]}"
    assert evidence["policy_sha256"] == PINNED_POLICY_SHA256 and evidence["synthetic"] is True
    assert better or "margin_not_met" in reason


def test_evaluation_arm_takes_tools_from_base_only(registry):
    arm = adm.evaluation_arm(registry, "code-review@cand", BASE_STRATEGY)
    assert arm.strategy.system_prompt == "evolved skill"
    assert arm.strategy.toolsets == BASE_STRATEGY.toolsets
    assert arm.strategy.strategy_id == "reviewer:code-review@cand"


def test_review_refuses_wrong_states(validation, registry):
    make = registry_factory(validation, "code-review@cand", "code-review@base", "[]")
    registry.add("code-review@orphan", "proposed")
    registry.add("other@cand", "proposed", parent_id="code-review@base", skill="other")
    for cid in ("code-review@base", "code-review@orphan", "other@cand"):
        with pytest.raises(ValueError):
            adm.review_registered(registry, cid, base=BASE_STRATEGY, make_executor=make)
    registry.entries["code-review@base"].status = "retired"
    with pytest.raises(ValueError, match="not admitted"):
        adm.review_registered(registry, "code-review@cand", base=BASE_STRATEGY,
                              make_executor=make)


def test_real_gate5_registry_when_available(validation, tmp_path):
    reg_mod = pytest.importorskip("evolution.adaptive.registry")
    with reg_mod.CandidateRegistry(tmp_path / "registry.db") as reg:
        base = reg.register_baseline("code-review", b"baseline skill\n")
        cand = reg.propose("code-review", b"evolved skill\n", parent_id=base.candidate_id)
        make = registry_factory(validation, cand.candidate_id, base.candidate_id, _truth)
        d = adm.review_registered(reg, cand.candidate_id, base=BASE_STRATEGY,
                                  make_executor=make, apply=True)
        assert d.admitted and reg.get(cand.candidate_id).status == reg_mod.ADMITTED
        (last,) = reg.history(cand.candidate_id)[-1:]
        assert last.evidence["policy_sha256"] == PINNED_POLICY_SHA256
        assert reg_mod.candidate_strategy(reg, cand.candidate_id, BASE_STRATEGY).system_prompt \
            == "evolved skill\n"
