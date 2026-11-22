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

There is deliberately no command-line entry point. Usage, once the owner
decides which reviewer to evaluate::

    from evolution.adaptive.final_evaluation import CONFIRM_PHRASE, run_final_evaluation
    run_final_evaluation(my_reviewer, reviewer_id="hermes:structured@<skill>",
                         synthetic=False, confirm=CONFIRM_PHRASE)
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import benchmark, evaluation

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


def run_final_evaluation(reviewer: Callable[[Mapping], Sequence], *, reviewer_id: str,
                         synthetic: bool, confirm: str,
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
    record = {"schema": LEDGER_SCHEMA, "status": "started", "started_at": _now(),
              "reviewer_id": reviewer_id, "synthetic": synthetic,
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
    _write_replace(path, record)
    logger.warning("FINAL TEST EVALUATION completed: %d cases, micro F1 %.3f",
                   record["n_cases"], record["micro"]["f1"]["point"])
    return record
