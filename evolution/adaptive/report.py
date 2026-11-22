"""Reproducible KAFO-EVO-001 sprint report: quality, utility, cost, latency,
uncertainty and safety.

Usage (offline, seeded; writes only to ``--out``)::

    python -m evolution.adaptive.report            # -> reports/kafo-evo-001/report.{md,json}

What it reports, all from merged sprint code:

* Gate 4 policy comparison (``compare.compare``) on the synthetic simulator
  (``synthetic.py``), with seed-level bootstrap CIs. The headline sentences
  are derived from the comparison verdicts, never hard-coded.
* Gate 3 verifier scores on the gate 2 benchmark's open splits (train,
  validation) for reference reviews: the labelled defects (ceiling), an
  empty review (floor) and an invented, seeded "noisy reference" built from
  the labels. None of these is a model.
* Cost and latency (simulated), and safety: critical violations, sealed
  split access attempts during generation, registry rejections and
  retirements, admission and rollback (pluggable, pending until gates 6 and
  7 merge), an optional live Hermes smoke run, and the final test
  evaluation ledger (read only; this module never runs it).

Everything runs inside ``benchmark.optimisation_phase()``, so the sealed test
split cannot be loaded or opened while the report is generated. The output
contains no wall-clock time unless ``--generated-at`` is given, so the same
code and seed reproduce byte-identical files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
from typing import Mapping, Optional, Sequence

from . import benchmark, compare, evaluation, synthetic, verifier
from .benchmark import optimisation_phase, sealed_access_attempts

SCHEMA = "kafo-sprint-report/1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "kafo-evo-001"
DEFAULT_LEDGER = DEFAULT_OUT / "final_test_evaluation.json"
NOTICE = ("SYNTHETIC RESULTS. The policy comparison uses a hypothetical simulator with "
          "invented success rates, costs and latencies; the verifier scores use reference "
          "reviews derived from the benchmark labels, not a model. Nothing in this report "
          "is evidence that Hermes improves on any real task.")
PENDING_ADMISSION = ("pending: gate 6 (candidate-vs-baseline admission) is not merged; "
                     "no candidate has been admitted or rejected by an admission check")
NO_CHECKPOINT_STORE = ("no checkpoint store supplied (--checkpoints). The gate 7 rollback "
                       "mechanism (checkpoint.admit_with_rollback) is merged, but no admission "
                       "has run, so there is no rollback history: 0 rollbacks is not a safety "
                       "result")
DEFAULT_LIVE_SMOKE = DEFAULT_OUT / "live_smoke_2026-11-22.json"
NOISY_KEEP, NOISY_FALSE_ALARM = 0.7, 0.25


# ── Verifier section ─────────────────────────────────────────────────────

def noisy_reference_review(case: verifier.BenchmarkCase, seed: int) -> list[dict]:
    """Invented reviewer for showing non-trivial intervals: keeps each labelled
    defect with probability 0.7 (line shifted by -1/0/+1) and adds one false
    alarm with probability 0.25. Derived from labels; NOT a model."""
    rng = random.Random(f"noisy-reference:{seed}:{case.case_id}")
    n_lines = max(1, len(case.code.splitlines()))
    findings = []
    for d in case.defects:
        if rng.random() < NOISY_KEEP:
            shift = rng.choice((-1, 0, 1))
            start = min(max(1, d.line_start + shift), n_lines)
            findings.append({"line_start": start,
                             "line_end": max(start, min(d.line_end + shift, n_lines)),
                             "category": d.category})
    if rng.random() < NOISY_FALSE_ALARM:
        findings.append({"line_start": rng.randint(1, n_lines), "category": "style"})
    return findings


def _interval(iv: Mapping) -> dict:
    return {"point": iv["point"], "low": iv["low"], "high": iv["high"]}


def _summarise(rep: verifier.EvaluationReport) -> dict:
    d = rep.to_dict()
    return {"n_cases": d["n_cases"],
            "totals": {k: sum(c[k] for c in d["cases"])
                       for k in ("true_positives", "false_positives", "false_negatives")},
            "micro": {m: _interval(d["micro"][m]) for m in ("precision", "recall", "f1")},
            "macro": {m: _interval(d["macro"][m]) for m in ("precision", "recall", "f1")},
            "missing_reviews": len(d["missing_reviews"])}


def verifier_section(*, seed: int, n_bootstrap: int) -> dict:
    cases = evaluation.open_cases()
    all_cases = [c for split in cases.values() for c in split]
    reviewers = {
        "ground_truth": {c.case_id: evaluation.ground_truth_findings(c) for c in all_cases},
        "empty": {c.case_id: [] for c in all_cases},
        "noisy_reference": {c.case_id: noisy_reference_review(c, seed) for c in all_cases},
    }
    out = {}
    for name, reviews in reviewers.items():
        reports = evaluation.evaluate_open(reviews, seed=seed, n_bootstrap=n_bootstrap)
        out[name] = {split: _summarise(r) for split, r in reports.items()}
    any_report = verifier.evaluate(all_cases[:1], {}, n_bootstrap=1).to_dict()
    provenance = {k: any_report[k] for k in ("verifier_version", "match_rule",
                                             "verifier_sha256", "config")}
    return {"reviewers": out, "provenance": provenance,
            "split_sizes": {s: len(c) for s, c in cases.items()}}


# ── Policy comparison and safety ─────────────────────────────────────────

def comparison_section(*, seeds: int, train_episodes: int, benchmark_seed: int,
                       resamples: int) -> tuple[dict, dict]:
    bench = synthetic.SyntheticBenchmark(seed=benchmark_seed)
    encoder = synthetic.synthetic_encoder(bench)
    result = compare.compare(
        bench, synthetic.ScriptedExecutor(bench.response_table()), synthetic.synthetic_quality,
        synthetic.synthetic_catalogue(), encoder,
        compare.default_policy_specs(synthetic.STRATEGIES, encoder.dim),
        seeds=list(range(seeds)), train_episodes=train_episodes,
        scoring=synthetic.REFERENCE_SCORING, n_resamples=resamples,
        skill_version="synthetic-catalogue/1")
    counts: dict = {}
    for r in result.records:
        entry = counts.setdefault(r.policy, {}).setdefault(r.phase, Counter())
        entry["steps"] += 1
        entry["violations"] += int(r.violation)
    violations = {p: {ph: dict(c) for ph, c in phases.items()} for p, phases in counts.items()}
    return result.report, violations


def headline(comparison: Mapping) -> list[str]:
    """Plain-language findings derived from the comparison verdicts."""
    best = comparison["best_fixed_baseline"]
    rows = comparison["comparisons"]
    lines = []

    def pick(baseline, phase):
        return [c for c in rows if c["baseline"] == baseline and c["phase"] == phase]

    def fmt(c):
        return f"{c['policy']} {c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"

    val = pick(best, "validation")
    better = [c for c in val if c["verdict"] == "improvement"]
    if val and not better:
        lines.append(f"No adaptive policy shows an improvement over the best fixed strategy "
                     f"(`{best}`) on held-out validation: "
                     + "; ".join(f"{fmt(c)} ({c['verdict']})" for c in val) + ".")
    elif better:
        lines.append(f"On validation, {', '.join(c['policy'] for c in better)} improved on "
                     f"the best fixed strategy (`{best}`): " + "; ".join(map(fmt, better)) + ".")
    train = [c for c in pick(best, "train") if c["verdict"] == "improvement"]
    if train:
        lines.append("On train, " + "; ".join(map(fmt, train)) + f" against `{best}`. That "
                     "baseline was chosen in hindsight on the same train data, so this does "
                     "not show generalisation.")
    rand = [c for c in rows if c["baseline"] == "random"]
    if rand and all(c["verdict"] == "improvement" for c in rand):
        lines.append("Every adaptive policy beats uniform random selection in both phases.")
    lines.append("These are results from a hypothetical simulator, not from Hermes.")
    return lines


def _with_live(lines: list[str], live: Mapping) -> list[str]:
    if live["status"] != "supplied":
        return lines
    note = (f"A live Hermes smoke run scored {live['n_cases']} {live['split']} cases "
            f"({live['completed']} completed, {live['violations']} violations). With so few "
            "cases this is a plumbing check, not a quality estimate.")
    return lines[:-1] + [note, "Apart from that smoke run, these are results from a "
                         "hypothetical simulator, not from Hermes."]


def _registry_summary(path: Optional[str]) -> dict:
    if path is None:
        return {"status": "not supplied", "note": "no candidate registry was given "
                "(--registry); counts below are unknown, not zero"}
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        by = conn.execute("SELECT source, status, COUNT(*) FROM candidates "
                          "GROUP BY source, status ORDER BY source, status").fetchall()
        moves = conn.execute(
            "SELECT candidate_id, from_status, to_status, reason, decided_by FROM transitions "
            "WHERE to_status IN ('rejected', 'retired') ORDER BY id").fetchall()
    finally:
        conn.close()
    return {"status": "supplied",
            "candidates": [{"source": s, "status": st, "count": n} for s, st, n in by],
            "rejected": [dict(zip(("candidate_id", "from", "to", "reason", "decided_by"), m))
                         for m in moves if m[2] == "rejected"],
            "retired": [dict(zip(("candidate_id", "from", "to", "reason", "decided_by"), m))
                        for m in moves if m[2] == "retired"]}


def checkpoint_section(path: Optional[str]) -> dict:
    """Read-only summary of a gate 7 checkpoint store (statuses and log events)."""
    if path is None:
        return {"status": "not supplied", "note": NO_CHECKPOINT_STORE}
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        statuses = conn.execute("SELECT status, COUNT(*) FROM policy_checkpoints "
                                "GROUP BY status ORDER BY status").fetchall()
        events = conn.execute("SELECT event, COUNT(*) FROM checkpoint_log "
                              "GROUP BY event ORDER BY event").fetchall()
        rollbacks = conn.execute(
            "SELECT checkpoint_id, from_checkpoint_id, reason FROM checkpoint_log "
            "WHERE event='rollback' ORDER BY id").fetchall()
    finally:
        conn.close()
    return {"status": "supplied",
            "checkpoints": {k: n for k, n in statuses},
            "events": {k: n for k, n in events},
            "rollbacks": [{"to": t, "from": f, "reason": r} for t, f, r in rollbacks]}


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite number or null, got {value!r}")
    return float(value)


def live_smoke_section(path: Optional[str]) -> dict:
    """Optional live Hermes smoke results: JSON {source, cases: [{case_id, completed,
    quality?, cost_usd?, latency_s?, violation}]}. A handful of cases only."""
    if path is None:
        return {"status": "pending", "note": "no live Hermes smoke results supplied "
                "(--live-smoke); nothing in this report comes from a live Hermes run"}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("live smoke file needs a nonempty 'cases' list")
    rows = []
    for c in cases:
        if not isinstance(c, dict) or not isinstance(c.get("case_id"), str):
            raise ValueError("each live smoke case needs a case_id")
        rows.append({"case_id": c["case_id"], "completed": bool(c.get("completed")),
                     "quality": _finite_or_none(c.get("quality")),
                     "cost_usd": _finite_or_none(c.get("cost_usd")),
                     "latency_s": _finite_or_none(c.get("latency_s")),
                     "violation": bool(c.get("violation"))})

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {"status": "supplied", "source": str(data.get("source", "unspecified")),
            "model": str(data.get("model", "unspecified")),
            "split": str(data.get("split", "unspecified")),
            "note": str(data.get("note", "")),
            "label": f"{len(rows)} live cases; a smoke test, not a benchmark result",
            "n_cases": len(rows), "completed": sum(r["completed"] for r in rows),
            "violations": sum(r["violation"] for r in rows),
            "mean_quality": mean("quality"), "mean_cost_usd": mean("cost_usd"),
            "mean_latency_s": mean("latency_s"), "cases": rows}


def final_evaluation_section(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {"status": "not run", "note": "the one-time final test evaluation has not "
                "been run; when and on which reviewer is an owner decision"}
    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    keep = ("status", "started_at", "finished_at", "reviewer_id", "synthetic", "n_cases",
            "totals", "micro", "macro", "benchmark", "verifier", "error")
    return {k: record[k] for k in keep if k in record}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


# ── Generation ───────────────────────────────────────────────────────────

def generate(*, seeds: int = 20, train_episodes: int = 300, benchmark_seed: int = 0,
             resamples: int = 2000, verifier_seed: int = 0, verifier_bootstrap: int = 1000,
             registry_path: Optional[str] = None, checkpoint_path: Optional[str] = None,
             live_smoke_path: Optional[str] = str(DEFAULT_LIVE_SMOKE),
             final_ledger_path: Path = DEFAULT_LEDGER,
             generated_at: Optional[str] = None) -> dict:
    """Build the report dict. Runs entirely inside optimisation_phase()."""
    with optimisation_phase():
        attempts_before = sealed_access_attempts()
        comparison, violations = comparison_section(
            seeds=seeds, train_episodes=train_episodes, benchmark_seed=benchmark_seed,
            resamples=resamples)
        verifier_scores = verifier_section(seed=verifier_seed, n_bootstrap=verifier_bootstrap)
        registry = _registry_summary(registry_path)
        checkpoints = checkpoint_section(checkpoint_path)
        live = live_smoke_section(live_smoke_path)
        final = final_evaluation_section(Path(final_ledger_path))
        blocked = sealed_access_attempts() - attempts_before
    total_steps = sum(c["steps"] for p in violations.values() for c in p.values())
    total_viol = sum(c["violations"] for p in violations.values() for c in p.values())
    pending = [PENDING_ADMISSION]
    if checkpoints["status"] == "not supplied":
        pending.append("rollback history: none yet, because no admission has run")
    if live["status"] == "pending":
        pending.append("live Hermes smoke run: no results supplied")
    if final["status"] == "not run":
        pending.append("final test evaluation: not run (owner decision)")
    return {
        "schema": SCHEMA, "title": "KAFO-EVO-001 sprint report", "synthetic": True,
        "notice": NOTICE if live["status"] != "supplied" else NOTICE + (
            f" The only live data is a {live['n_cases']}-case Hermes smoke test, reported "
            "separately as a plumbing check."),
        "generated_at": generated_at,
        "settings": {"seeds": seeds, "train_episodes": train_episodes,
                     "benchmark_seed": benchmark_seed, "comparison_resamples": resamples,
                     "verifier_seed": verifier_seed, "verifier_resamples": verifier_bootstrap,
                     "registry": None if registry_path is None else Path(registry_path).name,
                     "checkpoints": None if checkpoint_path is None
                     else Path(checkpoint_path).name,
                     "live_smoke": None if live_smoke_path is None else Path(live_smoke_path).name,
                     "final_ledger": _rel(Path(final_ledger_path))},
        "provenance": {"benchmark": {"name": benchmark.BENCHMARK_NAME,
                                     "version": benchmark.DEFAULT_VERSION,
                                     "manifest_sha256": benchmark.PINNED_MANIFEST_SHA256.get(
                                         benchmark.DEFAULT_VERSION)},
                       "verifier": verifier_scores.pop("provenance")},
        "headline": _with_live(headline(comparison), live),
        "policy_comparison": comparison,
        "verifier_scores": verifier_scores,
        "safety": {"critical_violations": {"total_steps": total_steps,
                                           "total_violations": total_viol,
                                           "by_policy": violations},
                   "sealed_access_blocked_during_generation": blocked,
                   "registry": registry,
                   "admission": {"status": "pending", "note": PENDING_ADMISSION},
                   "rollbacks": checkpoints},
        "live_smoke": live,
        "final_test_evaluation": final,
        "pending": pending,
    }


# ── Markdown ─────────────────────────────────────────────────────────────

def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _ci(stat: Mapping, fmt: str = ".3f") -> str:
    return f"{stat['mean']:{fmt}} [{stat['ci_low']:{fmt}}, {stat['ci_high']:{fmt}}]"


def _iv(iv: Mapping) -> str:
    return f"{iv['point']:.3f} [{iv['low']:.3f}, {iv['high']:.3f}]"


def format_markdown(report: Mapping) -> str:
    cmp_ = report["policy_comparison"]
    cfg = cmp_["config"]
    s = report["settings"]
    L = [f"# {report['title']}", "", f"> **{report['notice']}**", ""]
    if report.get("generated_at"):
        L += [f"Generated: {report['generated_at']}", ""]
    L += ["## Headline findings", ""] + [f"- {h}" for h in report["headline"]] + [""]

    L += ["## Policy comparison (gate 4, synthetic simulator)", "",
          f"Benchmark `{cmp_['benchmark']}`, executor `{cmp_['executor_backend']}`. "
          f"{len(cfg['seeds'])} seeds x {cfg['train_episodes']} online train episodes, then "
          f"one pass over {cfg['validation_items']} validation tasks with learning off. "
          f"Mean over seeds with a {int(cfg['bootstrap']['confidence'] * 100)}% bootstrap CI "
          f"({cfg['bootstrap']['resamples']} resamples, unit: seed-level run mean).", "",
          "### Quality and utility", "",
          "| Policy | Train utility | Validation utility | Validation quality |",
          "|---|---|---|---|"]
    for name, p in cmp_["policies"].items():
        L.append(f"| {name} | {_ci(p['train']['utility'])} | {_ci(p['validation']['utility'])}"
                 f" | {_ci(p['validation']['quality'])} |")
    L += ["", "Utility = verified quality - 0.10 x min(cost ratio, 2) - 0.05 x min(latency "
          "ratio, 2); a critical violation scores -1 (`core.score_outcome`, unchanged).", "",
          "### Paired utility difference", "",
          f"Best fixed baseline, chosen by train utility: `{cmp_['best_fixed_baseline']}`.", "",
          "| Policy | Versus | Phase | Difference [CI] | Verdict |", "|---|---|---|---|---|"]
    L += [f"| {c['policy']} | {c['baseline']} | {c['phase']} | {c['mean_diff']:+.3f} "
          f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['verdict']} |"
          for c in cmp_["comparisons"]]
    L += ["", "## Cost and latency (simulated)", "",
          "Invented per-strategy costs (USD) and latencies (seconds) from `synthetic.py`.", "",
          "| Policy | Validation cost USD | Validation latency s | Train cost USD |",
          "|---|---|---|---|"]
    for name, p in cmp_["policies"].items():
        L.append(f"| {name} | {_ci(p['validation']['cost_usd'], '.4f')} | "
                 f"{_ci(p['validation']['latency_s'], '.2f')} | "
                 f"{_ci(p['train']['cost_usd'], '.4f')} |")

    vs = report["verifier_scores"]
    pv = report["provenance"]
    L += ["", "## Verifier scores on the open benchmark splits (gates 2 and 3)", "",
          f"Benchmark `{pv['benchmark']['name']}` {pv['benchmark']['version']} "
          f"(manifest `{(pv['benchmark']['manifest_sha256'] or '?')[:12]}`), split sizes "
          + ", ".join(f"{k} {v}" for k, v in vs["split_sizes"].items())
          + f". Verifier `{pv['verifier']['verifier_version']}`, rule "
          f"`{pv['verifier']['match_rule']}`, source `{pv['verifier']['verifier_sha256'][:12]}`. "
          f"Case-level bootstrap CI, seed {s['verifier_seed']}, "
          f"{s['verifier_resamples']} resamples.", "",
          "These reviews are references, not models: `ground_truth` reports exactly the "
          "labelled defects (ceiling), `empty` reports nothing (floor), and `noisy_reference` "
          "is an invented, seeded perturbation of the labels (keeps each defect with p=0.7, "
          "shifts lines by up to 1, adds a false alarm with p=0.25).", "",
          "| Review | Split | Cases | TP / FP / FN | Micro precision | Micro recall | "
          "Micro F1 | Macro F1 |", "|---|---|---|---|---|---|---|---|"]
    for name, splits in vs["reviewers"].items():
        for split, r in splits.items():
            t = r["totals"]
            L.append(f"| {name} | {split} | {r['n_cases']} | {t['true_positives']} / "
                     f"{t['false_positives']} / {t['false_negatives']} | "
                     f"{_iv(r['micro']['precision'])} | {_iv(r['micro']['recall'])} | "
                     f"{_iv(r['micro']['f1'])} | {_iv(r['macro']['f1'])} |")
    L += ["", "The empty review's macro F1 is above 0 because clean control cases (no "
          "defects) score 1.0 when nothing is reported."]

    L += ["", "## Uncertainty", "",
          "- Policy comparison: percentile bootstrap over seed-level means; paired "
          "per-seed differences for verdicts (`improvement` only if the whole CI is above "
          "0). Validation intervals for fixed policies have zero width because those runs "
          "are deterministic.",
          "- Verifier: percentile bootstrap over cases (micro pools TP/FP/FN, macro "
          "averages per-case scores).",
          "- Same code and seeds reproduce this report byte for byte.", ""]

    sf = report["safety"]
    cv = sf["critical_violations"]
    L += ["## Safety", "",
          f"- Critical violations (tool call outside the strategy allowlist, simulated): "
          f"{cv['total_violations']} in {cv['total_steps']} simulated policy decisions "
          "(all seeds and phases). Each scores -1 and disqualifies the outcome.",
          f"- Sealed test split: {sf['sealed_access_blocked_during_generation']} blocked "
          "access attempts while generating this report (everything ran inside "
          "`optimisation_phase()`).", "",
          "| Policy | Train violations / steps | Validation violations / steps |",
          "|---|---|---|"]
    for name, phases in cv["by_policy"].items():
        cells = [f"{phases[ph]['violations']} / {phases[ph]['steps']}" if ph in phases else "-"
                 for ph in ("train", "validation")]
        L.append(f"| {name} | {cells[0]} | {cells[1]} |")
    reg = sf["registry"]
    L += ["", "### Candidate registry (gate 5)", ""]
    if reg["status"] == "supplied":
        L += ["| Source | Status | Candidates |", "|---|---|---|"]
        L += [f"| {c['source']} | {c['status']} | {c['count']} |" for c in reg["candidates"]]
        L += ["", f"Rejected candidates: {len(reg['rejected'])}. Retired: "
              f"{len(reg['retired'])}."]
        L += [f"- `{m['candidate_id']}` {m['from']} -> {m['to']} by {m['decided_by']}: "
              f"{m['reason']}" for m in reg["rejected"] + reg["retired"]]
    else:
        L.append(f"Not supplied: {reg['note']}.")
    L += ["", "### Admission (gate 6)", "", _sentence(sf["admission"]["note"]) + ".", "",
          "### Checkpoints and rollbacks (gate 7)", ""]
    rb = sf["rollbacks"]
    if rb["status"] == "supplied":
        L += ["Checkpoints by status: " + (", ".join(f"{k} {v}" for k, v in
                                                     rb["checkpoints"].items()) or "none")
              + ". Log events: " + (", ".join(f"{k} {v}" for k, v in rb["events"].items())
                                    or "none") + f". Rollbacks: {len(rb['rollbacks'])}."]
        L += [f"- checkpoint {r['from']} rolled back to {r['to']}: {r['reason']}"
              for r in rb["rollbacks"]]
        L.append("")
    else:
        L += [_sentence(rb["note"]) + ".", ""]

    live = report["live_smoke"]
    L += ["## Live Hermes smoke run", ""]
    if live["status"] == "supplied":
        def num(v, f):
            return "n/a" if v is None else format(v, f)
        L += [f"**{live['label']}.** {live['note']}", "",
              f"Source: {live['source']}. Model: {live['model']}. Split: {live['split']}.",
              "", "| Case | Completed | Quality (F1) | Est. cost USD | Latency s | Violation |",
              "|---|---|---|---|---|---|"]
        L += [f"| {c['case_id']} | {c['completed']} | {num(c['quality'], '.2f')} | "
              f"{num(c['cost_usd'], '.4f')} | {num(c['latency_s'], '.1f')} | "
              f"{c['violation']} |" for c in live["cases"]]
        L += ["",
              f"- Completed: {live['completed']} / {live['n_cases']}; violations: "
              f"{live['violations']}.",
              f"- Mean quality {num(live['mean_quality'], '.3f')}, mean cost USD "
              f"{num(live['mean_cost_usd'], '.4f')}, mean latency s "
              f"{num(live['mean_latency_s'], '.1f')}.", ""]
    else:
        L += [_sentence(live["note"]) + ".", ""]

    fe = report["final_test_evaluation"]
    L += ["## Final test evaluation (one-time, recorded)", ""]
    if fe["status"] == "not run":
        L += [_sentence(fe["note"]) + ". It is run by "
              "`evolution.adaptive.final_evaluation.run_final_evaluation`, which refuses a "
              "second run.", ""]
    else:
        L += [f"Status **{fe['status']}**, started {fe.get('started_at')}, finished "
              f"{fe.get('finished_at')}, reviewer `{fe.get('reviewer_id')}`, synthetic: "
              f"{fe.get('synthetic')}."]
        if fe["status"] == "completed":
            L += ["", f"{fe['n_cases']} cases. Micro F1 {_iv(fe['micro']['f1'])}, micro "
                  f"precision {_iv(fe['micro']['precision'])}, micro recall "
                  f"{_iv(fe['micro']['recall'])}."]
        elif fe.get("error"):
            L += ["", f"Error: {fe['error']}"]
        L.append("")

    L += ["## Pending", ""] + [f"- {_sentence(p)}" for p in report["pending"]] + [""]
    L += ["## Reproduce", "", "```bash",
          f"python -m evolution.adaptive.report --seeds {s['seeds']} --train-episodes "
          f"{s['train_episodes']} --benchmark-seed {s['benchmark_seed']} --resamples "
          f"{s['comparison_resamples']} --verifier-seed {s['verifier_seed']} "
          f"--verifier-resamples {s['verifier_resamples']}", "```", ""]
    return "\n".join(L)


def to_json(report: Mapping) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_report(report: Mapping, out_dir: Path = DEFAULT_OUT) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "report.md", out_dir / "report.json"
    md.write_text(format_markdown(report), encoding="utf-8")
    js.write_text(to_json(report), encoding="utf-8")
    return md, js


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the KAFO-EVO-001 sprint report "
                                     "(offline, seeded, synthetic).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--verifier-seed", type=int, default=0)
    parser.add_argument("--verifier-resamples", type=int, default=1000)
    parser.add_argument("--registry", help="candidate registry SQLite file (read-only)")
    parser.add_argument("--checkpoints", help="gate 7 checkpoint store SQLite file (read-only)")
    parser.add_argument("--live-smoke", default=str(DEFAULT_LIVE_SMOKE),
                        help="live Hermes smoke results JSON ('' to omit)")
    parser.add_argument("--final-ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--generated-at", help="timestamp to print (omitted by default "
                        "so the output is reproducible)")
    args = parser.parse_args(argv)
    if min(args.seeds, args.train_episodes, args.resamples, args.verifier_resamples) < 1:
        parser.error("--seeds, --train-episodes and resample counts must be positive")
    report = generate(seeds=args.seeds, train_episodes=args.train_episodes,
                      benchmark_seed=args.benchmark_seed, resamples=args.resamples,
                      verifier_seed=args.verifier_seed,
                      verifier_bootstrap=args.verifier_resamples,
                      registry_path=args.registry, checkpoint_path=args.checkpoints,
                      live_smoke_path=args.live_smoke or None,
                      final_ledger_path=Path(args.final_ledger),
                      generated_at=args.generated_at)
    md, js = write_report(report, Path(args.out))
    print(f"wrote {_rel(md)} and {_rel(js)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
