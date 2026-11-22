# KAFO-EVO-001 — Standalone Adaptive Hermes: Research Sprint

## Objective

Demonstrate a reproducible, headless learning loop that selects between
strategies based on task context and independently scored outcomes.
Preserve the existing Nous Research GEPA skill evolution pipeline.

## Architecture and terminology

- **Monitor:** capture task outcome, cost, time, external verifier result.
- **Analyse:** identify quality, constraints and failure type.
- **Plan:** contextual epsilon-greedy policy (first baseline).
- **Legitimise:** fixed reward calculation and critical-violation handling.
- **Execute:** `evolution/adaptive/executor.py` — `Executor` protocol,
  `HermesExecutor` (documented Hermes Python API, opt-in) and deterministic
  `FakeExecutor`. See "Hermes executor (gate 1)" below.
- **Knowledge:** SQLite outcomes, context/action counts and estimated reward.
- **Slow evolution:** upstream GEPA, later connected through candidate validation.

This is a MAPE-K-inspired implementation with an explicit adaptation gate,
not a new RL training framework. The initial bandit is **not LinUCB** and
does not update any LLM model weights.

## Delivered in initial branch

- Pure Python contextual epsilon-greedy strategy selector.
- Explicit action-selection propensity for future policy evaluation.
- Utility = verified quality - 0.10 * min(relative cost, 2) - 0.05 * min(relative latency, 2).
- Critical observed violation yields a disqualifying -1 reward, strictly below any valid outcome.
- Every strategy tried once per context before epsilon-greedy selection.
- NaN, +/-inf, booleans and non-numeric rewards rejected before estimates change.
- Local SQLite experience records with policy version, skill version and UTC timestamp;
  older databases migrated in place (legacy rows keep NULL provenance).
- Dependency-free, seeded synthetic demonstration.
- Unit tests for policy, reward, invalid actions and persistence.

## Reward range

| Outcome | Reward |
|---|---|
| Valid (no violation) | `quality - 0.10*min(cost_ratio, 2) - 0.05*min(time_ratio, 2)`, in **[-0.30, 1.0]** |
| Critical violation | **-1.0** (`VIOLATION_REWARD`), always below every valid outcome |

`RATIO_CAP = 2.0`. Cost or latency beyond twice the reference adds no further
penalty, so an expensive success can never score as low as a violation.
`score_outcome` rejects non-finite or boolean fields, and rejects weights for
which `(cost_weight + time_weight) * RATIO_CAP >= 1` (a valid outcome could
then tie a violation).

## Run a synthetic smoke test

From repository root:

```bash
python -m evolution.adaptive.cli --episodes 300 --seed 7
python -m pytest tests/test_adaptive_loop.py -q
```

The simulated reward distributions in the CLI are **hypothetical**.
Any observed learning in this simulator is NOT evidence that Hermes
improves in real code-review tasks.

## Validation evidence (2026-11-18, synthetic only)

| Check | Result |
|---|---|
| Targeted `tests/test_adaptive_loop.py`, initial branch (5 tests) | 5 passed (Python 3.14, pytest 9.0.3) |
| Initial CLI, `--episodes 300 --seed 7` | average utility 0.771 |
| Targeted suite after corrections (40 tests) | 40 passed (Python 3.12.13, isolated venv) |
| Full repository suite, `pytest -q` | 185 passed (Python 3.12.13, dspy 3.5.0, isolated venv) |
| CLI after corrections, `--episodes 300 --seed 7` | average utility 0.760, exit 0 |

The 0.771 -> 0.760 change comes from the per-context initial sweep altering the
random sequence; it is not a quality regression measurement. In the `complex`
context the post-correction run settled on `tool_assisted` (n=75) after a poor
single early sample of `structured` (n=3), although `structured` has the higher
simulated success rate. This is a known weakness of plain epsilon-greedy and is
what the pending LinUCB/random/fixed comparison must measure.

**These are results from a hypothetical simulator. They are not evidence that
Hermes improves on any real task.**

## Remaining Sprint 01 acceptance gates

- [x] Verify upstream dependencies and run the complete repository test suite
      (185 passed, Python 3.12 + dspy 3.5.0; Python 3.14 has no dspy install here).
- [x] Implement a Hermes executor adapter using an actually supported API
      (`AIAgent.run_conversation`, contract-checked against hermes-agent source;
      offline tests only — **no live Hermes/LLM run yet**, see "Hermes executor (gate 1)").
- [ ] Build labelled Python-review benchmark with ground-truth defect IDs.
- [ ] Build independent verifier with reproducible precision/recall scoring.
- [ ] Compare fixed, random, epsilon-greedy and LinUCB policies.
- [ ] Preserve train / validation / sealed test separation.
- [ ] Connect GEPA skill-candidate generation to the strategy registry.
- [ ] Require candidate-vs-baseline validation before candidate admission.
- [ ] Implement policy/skill checkpoint and rollback.
- [ ] Publish quality, utility, cost, latency, uncertainty and safety report.

## Hermes executor (gate 1)

### Supported Hermes interfaces (verified, not invented)

Checked in NousResearch/hermes-agent at `caafe5389a` (2026-11-19):

| Interface | Where it is defined / documented | Used? |
|---|---|---|
| `run_agent.AIAgent(...)` + `run_conversation(user_message, task_id)` -> dict | `run_agent.py` (`class AIAgent`), `agent/turn_facade.py` (`TurnFacadeMixin.run_conversation`), result keys built in `agent/turn_finalizer.py`; guide `website/docs/guides/python-library.md` | **Yes** |
| `hermes -z PROMPT --usage-file PATH` | `website/docs/reference/cli-commands.md`, `hermes_cli/oneshot.py` | No: loads all tools and auto-bypasses approvals, routes through the gateway daemon |
| ACP, TUI gateway JSON-RPC, OpenAI-compatible API server | `website/docs/developer-guide/programmatic-integration.md` | No: long-lived servers, more surface than one turn needs |

### How a run works

`HermesExecutor.execute(task, strategy)` writes a JSON request to a temp dir
and runs `evolution/adaptive/hermes_bridge.py` as
`<hermes python> -c <bridge> request.json result.json` with the hermes-agent
checkout as working directory (as the Python-library guide requires). The
bridge calls `AIAgent(**kwargs).run_conversation(...)`, then `close()`, and
writes a JSON summary. The project itself gains no dependency.

Fixed by the bridge, not settable by a strategy:
`quiet_mode=True`, `skip_memory=True`, `skip_context_files=True`,
`skip_background_review=True`, `save_trajectories=False`,
`checkpoints_enabled=False`. Tools: `enabled_toolsets` is the strategy's
allowlisted toolsets (default `[]`, which Hermes treats as *no tools*; only
`None` means all). Only `todo` is allowlisted. Terminal, code execution, file,
browser, skills, memory, delegation, cron, kanban, session search and
connections are always passed as `disabled_toolsets`.

Budgets (`Budget`: iterations, run seconds, max tokens, max cost) belong to
the executor, so every compared strategy gets the same limits. The subprocess
is killed at `timeout_s + startup_grace_s`.

### Outcome record

`ExecutionOutcome`: output, completed, latency (agent turn), wall time,
`cost_usd` (Hermes `estimated_cost_usd`; **`None` when `cost_status` is
`unknown`**, never silently 0), tokens, API calls, tool calls, error, and
`violation_reasons`. Violations: a tool call outside the strategy allowlist,
a Hermes guardrail `block`/`halt`, or known cost above `max_cost_usd`.
`to_reward(outcome, quality=..., reference_cost_usd=..., reference_latency_s=...)`
builds `core.Reward`; quality must come from the independent verifier, and
an unknown cost is refused. `loop_executor(...)` plugs an executor into
`choose_strategy`. Reward logic in `core.py` is unchanged.

### Limits and what is still pending

- `HermesExecutor` refuses to run unless constructed with `allow_live=True`;
  a live run spends LLM credit from the local Hermes configuration.
- **No live run against real Hermes and a real model has been done yet.**
  Evidence so far: offline unit tests, a real subprocess run of the bridge
  against a stub `run_agent`, and an AST contract test against the local
  hermes-agent source (skipped on CI, which has no checkout).
- Cost covers the main agent loop only; Hermes reports auxiliary calls
  (title generation, compression, …) separately and they are not included.
- `python` must be an interpreter that can import hermes-agent
  (`HERMES_PYTHON`); Hermes' launcher picks a PM-managed environment, so this
  is set by the operator.

## Scientific acceptance criteria

1. Compared policies receive equivalent task distributions and resource budgets.
2. Task context features are observable before the decision.
3. Record action probabilities and policy/skill versions.
4. No access to sealed test cases during optimisation.
5. Identify if improvement is absent; never infer generalisation from synthetic data.
6. Evaluate policy adaptation separately from GEPA skill evolution.

## Safety constraints

Keep CLI demonstrations offline and read-only by default. Do not permit
unrestricted code execution or let the agent modify reward logic, evaluation
data, permissions or budgets. Treat violations as hard disqualifiers for
candidate promotion. Do not automatically deploy evolved skills.

## Integration note

Existing `python -m evolution.skills.evolve_skill` is a separate GEPA
pipeline. It is **not yet invoked** by `evolution.adaptive.cli`.
The initial PR is the bandit/MAPE-K foundation, not a completed
Hermes + RL + GEPA end-to-end claim.
