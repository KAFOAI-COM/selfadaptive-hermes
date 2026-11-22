"""The one-time, recorded final evaluation on the held-out test split.

Owner decision (KAFO-EVO-001, 2026-11-22): the final test evaluation is one
deliberate, explicitly invoked, logged, in-process step, run once. This
module is that step and the only package code allowed to call
``evaluation.evaluate_sealed_test`` (see the guard scan in
``tests/test_benchmark_integration.py``).

How it is enforced:

1. The caller must pass ``confirm=CONFIRM_PHRASE`` and a ``reviewer_id``.
2. It refuses inside ``benchmark.optimisation_phase()``.
3. Before any test case is loaded, the ledger file is created with
   exclusive-create (``open(path, "x")``) and a ``started`` record. If the
   ledger already exists, for any reason, the run is refused. A crash or a
   reviewer error therefore still counts as "the final evaluation happened";
   the ledger then records ``failed``.
4. On success the ledger records the scores, UTC timestamps, verifier and
   benchmark provenance and the reviewer identity, and the result is logged.

The reviewer is called once per test case with ``evaluation.review_input``
(code and context only, never the labelled defects). Results must not feed
back into optimisation or selection.

Owner decision (2026-11-22): the reviewer is the live ``HermesExecutor``
with the ``github-code-review`` skill version the full live run produced
(the admitted GEPA candidate if one passed gate 6, otherwise the baseline),
chosen with :func:`select_final_skill`. :class:`ExecutorReviewer` wraps any
executor, tracks cost against a hard cap, never lets one failed call abort
the run, and its :meth:`ExecutorReviewer.summary` (cost, errors,
violations) is written to the ledger together with ``skill_version``.

There is deliberately no command-line entry point; the run is an explicit
in-process call.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import admission, benchmark, evaluation
from . import registry as reg
from .executor import ExecutionOutcome, Strategy, Task

logger = logging.getLogger(__name__)

CONFIRM_PHRASE = "I am running the one-time final test evaluation"
LEDGER_SCHEMA = "kafo-final-evaluation/1"
DEFAULT_LEDGER = (Path(__file__).resolve().parents[2] / "reports" / "kafo-evo-001"
                  / "final_test_evaluation.json")


class FinalEvaluationRefused(PermissionError):
    """The final evaluation was not run (already run, not confirmed, or in a phase)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ledger(ledger_path: str | os.PathLike = DEFAULT_LEDGER) -> dict | None:
    """The recorded final evaluation, or None if it has never been started."""
    path = Path(ledger_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_replace(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class ExecutorReviewer:
    """Reviewer that runs ``executor`` with one fixed strategy per test case.

    * Prompt and parsing are gate 6's (``admission.review_prompt`` and
      ``admission.parse_findings``), so the final evaluation asks exactly what
      admission asked.
    * Spending stops (remaining cases get an empty review) once the known cost
      reaches ``max_total_cost_usd``, or after a call whose cost is unknown,
      because the cap can no longer be enforced. The reason is recorded.
    * An executor exception is recorded and scored as an empty review instead
      of aborting, so one transient failure cannot void the one-time run.
    """

    def __init__(self, executor, strategy: Strategy, *, max_total_cost_usd: float):
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must be an executor.Strategy")
        if isinstance(max_total_cost_usd, bool) or not isinstance(
                max_total_cost_usd, (int, float)) or not 0 < max_total_cost_usd < 1e6:
            raise ValueError("max_total_cost_usd must be a positive number")
        self.executor, self.strategy = executor, strategy
        self.max_total_cost_usd = float(max_total_cost_usd)
        self.cases: list[dict] = []
        self.total_cost_usd = 0.0
        self.stopped: str | None = None

    def __call__(self, view: Mapping) -> list:
        case_id = view["case_id"]
        if self.stopped is None and self.total_cost_usd >= self.max_total_cost_usd:
            self.stopped = f"cost cap USD {self.max_total_cost_usd} reached"
        if self.stopped is not None:
            self.cases.append({"case_id": case_id, "skipped": self.stopped})
            return []
        try:
            outcome = self.executor.execute(
                Task(case_id, admission.review_prompt(view["code"])), self.strategy)
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("executor must return an ExecutionOutcome")
        except Exception as exc:   # recorded, scored as an empty review
            self.cases.append({"case_id": case_id, "exception":
                               f"{type(exc).__name__}: {exc}"[:300]})
            return []
        findings, parse_ok = admission.parse_findings(outcome.output)
        if outcome.cost_usd is None:
            self.stopped = f"cost unknown after {case_id}; cap cannot be enforced"
        else:
            self.total_cost_usd += outcome.cost_usd
        self.cases.append({
            "case_id": case_id, "completed": outcome.completed, "parse_ok": parse_ok,
            "cost_usd": outcome.cost_usd, "cost_status": outcome.cost_status,
            "latency_s": outcome.latency_s, "error": outcome.error,
            "violation_reasons": list(outcome.violation_reasons), "model": outcome.model})
        return findings

    def summary(self) -> dict:
        ran = [c for c in self.cases if "completed" in c]
        return {"strategy_id": self.strategy.strategy_id,
                "toolsets": list(self.strategy.toolsets),
                "backend": getattr(self.executor, "backend", type(self.executor).__name__),
                "model": getattr(self.executor, "model", None),
                "provider": getattr(self.executor, "provider", None),
                "max_total_cost_usd": self.max_total_cost_usd,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "cost_statuses": dict(Counter(c["cost_status"] for c in ran)),
                "calls": len(ran), "completed": sum(c["completed"] for c in ran),
                "unparseable": sum(not c["parse_ok"] for c in ran),
                "violations": sum(bool(c["violation_reasons"]) for c in ran),
                "exceptions": sum("exception" in c for c in self.cases),
                "skipped": sum("skipped" in c for c in self.cases),
                "stopped": self.stopped, "cases": list(self.cases)}


def select_final_skill(registry: reg.CandidateRegistry, skill_name: str,
                       base: Strategy) -> tuple[reg.Candidate, Strategy]:
    """Owner rule: the gate-6-admitted GEPA candidate if exactly one exists,
    otherwise the admitted baseline. Admission evidence is re-checked."""
    catalogue = {s.strategy_id: s for s in admission.admitted_catalogue(
        registry, skill_name, base)}
    admitted = registry.candidates(skill_name=skill_name, status=reg.ADMITTED)
    gepa = [c for c in admitted if c.source == reg.GEPA]
    if len(gepa) > 1:
        raise ValueError(f"{len(gepa)} admitted GEPA candidates for {skill_name!r}: "
                         f"{[c.candidate_id for c in gepa]}; choose one explicitly")
    chosen = gepa or [c for c in admitted if c.source == reg.BASELINE]
    if len(chosen) != 1:
        raise ValueError(f"expected one admitted baseline for {skill_name!r}, "
                         f"found {len(chosen)}")
    return chosen[0], catalogue[f"{base.strategy_id}:{chosen[0].candidate_id}"]


def _add_summary(record: dict, reviewer) -> None:
    summary = getattr(reviewer, "summary", None)
    if callable(summary):
        try:
            record["reviewer_summary"] = summary()
        except Exception as exc:   # never lose the ledger over a summary bug
            record["reviewer_summary"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}


def run_final_evaluation(reviewer: Callable[[Mapping], Sequence], *, reviewer_id: str,
                         synthetic: bool, confirm: str,
                         skill_version: str | None = None,
                         ledger_path: str | os.PathLike = DEFAULT_LEDGER,
                         version: str = benchmark.DEFAULT_VERSION,
                         seed: int = 0, n_bootstrap: int = 1000) -> dict:
    """Run the final test evaluation once and record it. Refuses a second run."""
    if confirm != CONFIRM_PHRASE:
        raise FinalEvaluationRefused("confirm must equal final_evaluation.CONFIRM_PHRASE")
    if benchmark.in_optimisation_phase():
        raise FinalEvaluationRefused("the final evaluation cannot run inside an "
                                     "optimisation phase")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be a nonempty string")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be True or False")
    if not callable(reviewer):
        raise TypeError("reviewer must be callable")

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if skill_version is not None and (not isinstance(skill_version, str)
                                      or not skill_version.strip()):
        raise ValueError("skill_version must be a nonempty string or None")
    record = {"schema": LEDGER_SCHEMA, "status": "started", "started_at": _now(),
              "reviewer_id": reviewer_id, "synthetic": synthetic,
              "skill_version": skill_version,
              "benchmark": {"name": benchmark.BENCHMARK_NAME, "version": version,
                            "manifest_sha256": benchmark.PINNED_MANIFEST_SHA256.get(version)},
              "verifier_bootstrap": {"seed": seed, "resamples": n_bootstrap}}
    try:
        with open(path, "x", encoding="utf-8") as handle:   # atomic: one run only
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        previous = read_ledger(path) or {}
        raise FinalEvaluationRefused(
            f"the final evaluation was already {previous.get('status', 'recorded')} at "
            f"{previous.get('started_at', '?')} ({path}); it runs once") from None

    logger.warning("FINAL TEST EVALUATION started by reviewer %r (ledger %s); results "
                   "must not feed back into optimisation or selection", reviewer_id, path)
    try:
        report = evaluation.evaluate_sealed_test(
            reviewer=reviewer, final_evaluation=True, version=version,
            seed=seed, n_bootstrap=n_bootstrap)
    except BaseException as exc:
        record.update(status="failed", finished_at=_now(),
                      error=f"{type(exc).__name__}: {exc}"[:500])
        _add_summary(record, reviewer)
        _write_replace(path, record)
        logger.error("FINAL TEST EVALUATION failed: %s", record["error"])
        raise
    full = report.to_dict()
    record.update(
        status="completed", finished_at=_now(), n_cases=len(report.cases),
        micro=full["micro"], macro=full["macro"],
        totals={k: sum(c[k] for c in full["cases"])
                for k in ("true_positives", "false_positives", "false_negatives")},
        verifier={k: full[k] for k in ("verifier_version", "match_rule",
                                       "verifier_sha256", "config")})
    _add_summary(record, reviewer)
    _write_replace(path, record)
    logger.warning("FINAL TEST EVALUATION completed: %d cases, micro F1 %.3f",
                   record["n_cases"], record["micro"]["f1"]["point"])
    return record
