"""Offline tests for the gate 8 sprint report and the one-time final evaluation."""
import ast
import copy
import json
from pathlib import Path

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive import evaluation as ev
from evolution.adaptive import final_evaluation as fe
from evolution.adaptive import report as rp
from evolution.adaptive.registry import REJECTED, CandidateRegistry
from tests.test_benchmark_integration import _scan

REPO = Path(__file__).resolve().parents[1]
SMALL = dict(seeds=3, train_episodes=40, resamples=50, verifier_bootstrap=50)


@pytest.fixture(scope="module")
def small_report():
    return rp.generate(final_ledger_path=REPO / "does-not-exist.json", **SMALL)


# ── Report content ─────────────────────────────────────────────────────────

def test_report_is_labelled_synthetic(small_report):
    md = rp.format_markdown(small_report)
    assert small_report["synthetic"] is True
    assert md.splitlines()[2].startswith("> **SYNTHETIC RESULTS.")
    assert "Nothing in this report is evidence that Hermes improves" in small_report["notice"]
    assert small_report["headline"][-1].startswith("These are results from a hypothetical")


def test_same_settings_reproduce_identical_output(small_report):
    again = rp.generate(final_ledger_path=REPO / "does-not-exist.json", **SMALL)
    assert rp.to_json(again) == rp.to_json(small_report)
    assert rp.format_markdown(again) == rp.format_markdown(small_report)
    assert "generated_at" in small_report and small_report["generated_at"] is None


def test_headline_follows_the_verdicts(small_report):
    cmp_ = copy.deepcopy(small_report["policy_comparison"])
    best = cmp_["best_fixed_baseline"]
    for c in cmp_["comparisons"]:
        c["verdict"] = "no improvement detected"
    assert rp.headline(cmp_)[0].startswith("No adaptive policy shows an improvement")
    for c in cmp_["comparisons"]:
        if c["baseline"] == best and c["phase"] == "validation" and c["policy"] == "linucb":
            c["verdict"] = "improvement"
    assert "linucb improved on the best fixed strategy" in rp.headline(cmp_)[0]


def test_full_report_states_no_improvement_over_best_fixed():
    """The committed report's headline comes from the gate 4 verdicts."""
    data = json.loads((rp.DEFAULT_OUT / "report.json").read_text(encoding="utf-8"))
    assert data["headline"][0].startswith(
        "No adaptive policy shows an improvement over the best fixed strategy")
    val = [c for c in data["policy_comparison"]["comparisons"]
           if c["baseline"] == data["policy_comparison"]["best_fixed_baseline"]
           and c["phase"] == "validation"]
    assert val and all(c["verdict"] != "improvement" for c in val)


def test_verifier_reference_scores(small_report):
    reviewers = small_report["verifier_scores"]["reviewers"]
    for split in bm.OPEN_SPLITS:
        assert reviewers["ground_truth"][split]["micro"]["f1"]["point"] == 1.0
        assert reviewers["empty"][split]["micro"]["f1"]["point"] == 0.0
        noisy = reviewers["noisy_reference"][split]["micro"]["f1"]
        assert 0.0 < noisy["point"] < 1.0 and noisy["low"] <= noisy["point"] <= noisy["high"]
    assert set(small_report["verifier_scores"]["split_sizes"]) == set(bm.OPEN_SPLITS)


def test_safety_section_and_pending_gates(small_report):
    safety = small_report["safety"]
    assert safety["sealed_access_blocked_during_generation"] == 0
    cv = safety["critical_violations"]
    assert cv["total_violations"] == sum(
        p[ph]["violations"] for p in cv["by_policy"].values() for ph in p)
    assert safety["admission"]["status"] == safety["rollbacks"]["status"] == "pending"
    assert safety["registry"]["status"] == "not supplied"
    assert small_report["live_smoke"]["status"] == "pending"
    assert small_report["final_test_evaluation"]["status"] == "not run"


def test_registry_rejections_are_reported(tmp_path):
    db = tmp_path / "registry.db"
    with CandidateRegistry(db) as reg:
        base = reg.register_baseline("code-review", b"base\n")
        cand = reg.propose("code-review", b"evolved\n", parent_id=base.candidate_id)
        reg.transition(cand.candidate_id, REJECTED, reason="worse than baseline",
                       decided_by="test", evidence={"diff": -0.1})
    before = db.read_bytes()
    summary = rp._registry_summary(str(db))
    assert db.read_bytes() == before   # opened read-only
    assert {(c["source"], c["status"], c["count"]) for c in summary["candidates"]} == {
        ("baseline", "admitted", 1), ("gepa", "rejected", 1)}
    assert summary["rejected"][0]["reason"] == "worse than baseline"


def test_live_smoke_is_labelled_as_a_handful_of_cases(tmp_path):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"source": "worker -3", "cases": [
        {"case_id": "a", "completed": True, "quality": 0.5, "cost_usd": 0.01,
         "latency_s": 9.0, "violation": False},
        {"case_id": "b", "completed": False, "quality": None, "cost_usd": None,
         "latency_s": 3.0, "violation": False}]}))
    live = rp.live_smoke_section(str(path))
    assert live["label"] == "2 live cases; a smoke test, not a benchmark result"
    assert (live["completed"], live["mean_quality"], live["mean_latency_s"]) == (1, 0.5, 6.0)
    path.write_text(json.dumps({"cases": [{"case_id": "a", "cost_usd": float("inf")}]}))
    with pytest.raises(ValueError):
        rp.live_smoke_section(str(path))


def test_report_reads_a_recorded_final_evaluation(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"status": "failed", "started_at": "t0", "finished_at": "t1",
                                  "reviewer_id": "r", "synthetic": True, "error": "boom",
                                  "unexpected": "dropped"}))
    section = rp.final_evaluation_section(ledger)
    assert section["status"] == "failed" and "unexpected" not in section
    assert "Status **failed**" in rp.format_markdown(
        {**rp.generate(final_ledger_path=ledger, **SMALL)})


def test_committed_report_matches_generator_output():
    """reports/kafo-evo-001 must be regenerated whenever merged code changes it."""
    fresh = rp.generate()
    assert (rp.DEFAULT_OUT / "report.json").read_text(encoding="utf-8") == rp.to_json(fresh)
    assert (rp.DEFAULT_OUT / "report.md").read_text(encoding="utf-8") == \
        rp.format_markdown(fresh)


def test_generator_never_references_the_final_evaluation():
    tree = ast.parse(Path(rp.__file__).read_text(encoding="utf-8"))
    names = {getattr(n, "id", getattr(n, "attr", None)) for n in ast.walk(tree)
             if isinstance(n, (ast.Name, ast.Attribute))}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert not {"final_evaluation", "run_final_evaluation", "evaluate_sealed_test"} & (
        names | imported)
    assert rp.DEFAULT_LEDGER == fe.DEFAULT_LEDGER


# ── One-time final evaluation ──────────────────────────────────────────────

def _reviewer(seen):
    def review(view):
        seen.append(view)
        return []
    return review


def test_final_evaluation_runs_once_and_records_result(tmp_path, caplog):
    ledger, seen = tmp_path / "final.json", []
    with caplog.at_level("WARNING"):
        record = fe.run_final_evaluation(_reviewer(seen), reviewer_id="empty-reference",
                                         synthetic=True, confirm=fe.CONFIRM_PHRASE,
                                         ledger_path=ledger, n_bootstrap=20)
    assert record["status"] == "completed" and record["n_cases"] == len(seen) == 10
    assert record["synthetic"] is True and record["micro"]["f1"]["point"] == 0.0
    assert record["started_at"] <= record["finished_at"]
    assert json.loads(ledger.read_text()) == record
    assert "FINAL TEST EVALUATION completed" in caplog.text
    with pytest.raises(fe.FinalEvaluationRefused, match="already completed"):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="again", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)
    assert json.loads(ledger.read_text()) == record


def test_reviewer_never_sees_labels(tmp_path):
    seen = []
    fe.run_final_evaluation(_reviewer(seen), reviewer_id="probe", synthetic=True,
                            confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "l.json",
                            n_bootstrap=5)
    for view in seen:
        assert set(view) == {"case_id", "language", "code", "context"}
        assert set(view["context"]) == {"difficulty", "domain"}
        with pytest.raises(TypeError):
            view["defects"] = []


def test_failed_run_still_counts_as_the_one_run(tmp_path):
    ledger = tmp_path / "final.json"

    def broken(view):
        raise RuntimeError("model crashed")

    with pytest.raises(RuntimeError):
        fe.run_final_evaluation(broken, reviewer_id="broken", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)
    assert json.loads(ledger.read_text())["status"] == "failed"
    with pytest.raises(fe.FinalEvaluationRefused, match="already failed"):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="retry", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=ledger)


@pytest.mark.parametrize("kwargs, error", [
    ({"confirm": "yes"}, fe.FinalEvaluationRefused),
    ({"reviewer_id": " "}, ValueError),
    ({"synthetic": "yes"}, ValueError),
])
def test_final_evaluation_needs_explicit_arguments(tmp_path, kwargs, error):
    args = dict(reviewer_id="r", synthetic=True, confirm=fe.CONFIRM_PHRASE,
                ledger_path=tmp_path / "l.json")
    args.update(kwargs)
    with pytest.raises(error):
        fe.run_final_evaluation(_reviewer([]), **args)
    assert not (tmp_path / "l.json").exists()


def test_final_evaluation_refused_inside_optimisation_phase(tmp_path):
    with bm.optimisation_phase(), pytest.raises(fe.FinalEvaluationRefused):
        fe.run_final_evaluation(_reviewer([]), reviewer_id="r", synthetic=True,
                                confirm=fe.CONFIRM_PHRASE, ledger_path=tmp_path / "l.json")
    assert not (tmp_path / "l.json").exists()


def test_evaluate_sealed_test_needs_exactly_one_review_source():
    with pytest.raises(ValueError):
        ev.evaluate_sealed_test(final_evaluation=True)
    with pytest.raises(ValueError):
        ev.evaluate_sealed_test({}, reviewer=lambda v: [], final_evaluation=True)


def test_only_the_final_step_may_call_evaluate_sealed_test(tmp_path):
    rel = Path("evolution/adaptive/final_evaluation.py")
    assert _scan(REPO / rel, rel) == []
    rogue = tmp_path / "rogue.py"
    rogue.write_text("from . import evaluation\n"
                     "def run_final_evaluation():\n"
                     "    return evaluation.evaluate_sealed_test({}, final_evaluation=True)\n"
                     "def helper():\n"
                     "    return evaluation.evaluate_sealed_test({}, final_evaluation=True)\n",
                     encoding="utf-8")
    problems = _scan(rogue, Path("evolution/adaptive/rogue.py"))
    assert len(problems) == 2 and all("evaluate_sealed_test" in p for p in problems)
    renamed = _scan(rogue, rel)   # right file, but helper() is not the allowed function
    assert len(renamed) == 1 and ":5:" in renamed[0]
