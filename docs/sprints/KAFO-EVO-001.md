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
- **Execute:** adapter to Hermes (not yet connected in this PR).
- **Knowledge:** SQLite outcomes, context/action counts and estimated reward.
- **Slow evolution:** upstream GEPA, later connected through candidate validation.

This is a MAPE-K-inspired implementation with an explicit adaptation gate,
not a new RL training framework. The initial bandit is **not LinUCB** and
does not update any LLM model weights.

## Delivered in initial branch

- Pure Python contextual epsilon-greedy strategy selector.
- Explicit action-selection propensity for future policy evaluation.
- Utility = verified quality - 0.10 * relative cost - 0.05 * relative latency.
- Critical observed violation yields a disqualifying -1 reward.
- Local SQLite experience records.
- Dependency-free, seeded synthetic demonstration.
- Unit tests for policy, reward, invalid actions and persistence.

## Run a synthetic smoke test

From repository root:

```bash
python -m evolution.adaptive.cli --episodes 300 --seed 7
python -m pytest tests/test_adaptive_loop.py -q
```

The simulated reward distributions in the CLI are **hypothetical**.
Any observed learning in this simulator is NOT evidence that Hermes
improves in real code-review tasks.

## Remaining Sprint 01 acceptance gates

- [ ] Verify upstream dependencies and run the complete repository test suite.
- [ ] Implement a Hermes executor adapter using an actually supported API.
- [ ] Build labelled Python-review benchmark with ground-truth defect IDs.
- [ ] Build independent verifier with reproducible precision/recall scoring.
- [ ] Compare fixed, random, epsilon-greedy and LinUCB policies.
- [ ] Preserve train / validation / sealed test separation.
- [ ] Connect GEPA skill-candidate generation to the strategy registry.
- [ ] Require candidate-vs-baseline validation before candidate admission.
- [ ] Implement policy/skill checkpoint and rollback.
- [ ] Publish quality, utility, cost, latency, uncertainty and safety report.

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
