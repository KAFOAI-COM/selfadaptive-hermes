# KAFO-EVO-001 sprint report

> **SYNTHETIC RESULTS. The policy comparison uses a hypothetical simulator with invented success rates, costs and latencies; the verifier scores use reference reviews derived from the benchmark labels, not a model. Nothing in this report is evidence that Hermes improves on any real task. The only live data is a 4-case Hermes smoke test, reported separately as a plumbing check.**

## Headline findings

- No adaptive policy shows an improvement over the best fixed strategy (`fixed:structured`) on held-out validation: epsilon_greedy -0.006 [-0.024, +0.014] (no improvement detected); linucb +0.012 [-0.009, +0.033] (no improvement detected).
- On train, epsilon_greedy +0.025 [+0.007, +0.043]; linucb +0.041 [+0.026, +0.056] against `fixed:structured`. That baseline was chosen in hindsight on the same train data, so this does not show generalisation.
- Every adaptive policy beats uniform random selection in both phases.
- A live Hermes smoke run scored 4 validation cases (4 completed, 0 violations). With so few cases this is a plumbing check, not a quality estimate.
- Apart from that smoke run, these are results from a hypothetical simulator, not from Hermes.

## Policy comparison (gate 4, synthetic simulator)

Benchmark `synthetic-python-review/1`, executor `scripted`. 20 seeds x 300 online train episodes, then one pass over 60 validation tasks with learning off. Mean over seeds with a 95% bootstrap CI (2000 resamples, unit: seed-level run mean).

### Quality and utility

| Policy | Train utility | Validation utility | Validation quality |
|---|---|---|---|
| fixed:direct | 0.503 [0.492, 0.514] | 0.517 [0.517, 0.517] | 0.583 [0.583, 0.583] |
| fixed:structured | 0.565 [0.554, 0.575] | 0.584 [0.584, 0.584] | 0.783 [0.783, 0.783] |
| fixed:tool_assisted | 0.398 [0.384, 0.411] | 0.483 [0.483, 0.483] | 0.783 [0.783, 0.783] |
| random | 0.489 [0.478, 0.500] | 0.538 [0.518, 0.560] | 0.720 [0.700, 0.742] |
| epsilon_greedy | 0.590 [0.576, 0.604] | 0.579 [0.559, 0.598] | 0.751 [0.735, 0.767] |
| linucb | 0.606 [0.592, 0.619] | 0.596 [0.576, 0.617] | 0.760 [0.741, 0.778] |

Utility = verified quality - 0.10 x min(cost ratio, 2) - 0.05 x min(latency ratio, 2); a critical violation scores -1 (`core.score_outcome`, unchanged).

### Paired utility difference

Best fixed baseline, chosen by train utility: `fixed:structured`.

| Policy | Versus | Phase | Difference [CI] | Verdict |
|---|---|---|---|---|
| epsilon_greedy | fixed:structured | train | +0.025 [+0.007, +0.043] | improvement |
| epsilon_greedy | fixed:structured | validation | -0.006 [-0.024, +0.014] | no improvement detected |
| epsilon_greedy | random | train | +0.101 [+0.089, +0.112] | improvement |
| epsilon_greedy | random | validation | +0.041 [+0.009, +0.071] | improvement |
| linucb | fixed:structured | train | +0.041 [+0.026, +0.056] | improvement |
| linucb | fixed:structured | validation | +0.012 [-0.009, +0.033] | no improvement detected |
| linucb | random | train | +0.117 [+0.101, +0.133] | improvement |
| linucb | random | validation | +0.058 [+0.029, +0.088] | improvement |

## Cost and latency (simulated)

Invented per-strategy costs (USD) and latencies (seconds) from `synthetic.py`.

| Policy | Validation cost USD | Validation latency s | Train cost USD |
|---|---|---|---|
| fixed:direct | 0.0132 [0.0132, 0.0132] | 2.62 [2.62, 2.62] | 0.0133 [0.0133, 0.0133] |
| fixed:structured | 0.0397 [0.0397, 0.0397] | 8.00 [8.00, 8.00] | 0.0400 [0.0399, 0.0400] |
| fixed:tool_assisted | 0.1059 [0.1059, 0.1059] | 19.55 [19.55, 19.55] | 0.1065 [0.1063, 0.1068] |
| random | 0.0503 [0.0483, 0.0524] | 9.59 [9.22, 9.94] | 0.0532 [0.0524, 0.0539] |
| epsilon_greedy | 0.0457 [0.0420, 0.0492] | 8.81 [8.13, 9.53] | 0.0454 [0.0431, 0.0479] |
| linucb | 0.0393 [0.0370, 0.0418] | 7.69 [7.24, 8.17] | 0.0449 [0.0417, 0.0478] |

## Verifier scores on the open benchmark splits (gates 2 and 3)

Benchmark `python_review` v1 (manifest `fada5f14b7d8`), split sizes train 28, validation 10. Verifier `kafo-verifier/1`, rule `category-equal+line-overlap/1`, source `f3310b479ff6`. Case-level bootstrap CI, seed 0, 1000 resamples.

These reviews are references, not models: `ground_truth` reports exactly the labelled defects (ceiling), `empty` reports nothing (floor), and `noisy_reference` is an invented, seeded perturbation of the labels (keeps each defect with p=0.7, shifts lines by up to 1, adds a false alarm with p=0.25).

| Review | Split | Cases | TP / FP / FN | Micro precision | Micro recall | Micro F1 | Macro F1 |
|---|---|---|---|---|---|---|---|
| ground_truth | train | 28 | 31 / 0 / 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| ground_truth | validation | 10 | 10 / 0 / 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| empty | train | 28 | 0 / 0 / 31 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.143 [0.036, 0.250] |
| empty | validation | 10 | 0 / 0 / 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.500] |
| noisy_reference | train | 28 | 18 / 4 / 13 | 0.818 [0.650, 0.958] | 0.581 [0.412, 0.758] | 0.679 [0.522, 0.818] | 0.589 [0.422, 0.762] |
| noisy_reference | validation | 10 | 7 / 0 / 3 | 1.000 [1.000, 1.000] | 0.700 [0.400, 0.929] | 0.824 [0.571, 0.963] | 0.767 [0.500, 0.967] |

The empty review's macro F1 is above 0 because clean control cases (no defects) score 1.0 when nothing is reported.

## Uncertainty

- Policy comparison: percentile bootstrap over seed-level means; paired per-seed differences for verdicts (`improvement` only if the whole CI is above 0). Validation intervals for fixed policies have zero width because those runs are deterministic.
- Verifier: percentile bootstrap over cases (micro pools TP/FP/FN, macro averages per-case scores).
- Same code and seeds reproduce this report byte for byte.

## Safety

- Critical violations (tool call outside the strategy allowlist, simulated): 474 in 43200 simulated policy decisions (all seeds and phases). Each scores -1 and disqualifies the outcome.
- Sealed test split: 0 blocked access attempts while generating this report (everything ran inside `optimisation_phase()`).

| Policy | Train violations / steps | Validation violations / steps |
|---|---|---|
| fixed:direct | 0 / 6000 | 0 / 1200 |
| fixed:structured | 0 / 6000 | 0 / 1200 |
| fixed:tool_assisted | 307 / 6000 | 0 / 1200 |
| random | 96 / 6000 | 0 / 1200 |
| epsilon_greedy | 35 / 6000 | 0 / 1200 |
| linucb | 36 / 6000 | 0 / 1200 |

### Candidate registry (gate 5)

Not supplied: no candidate registry was given (--registry); counts below are unknown, not zero.

### Admission (gate 6)

Pending: gate 6 (candidate-vs-baseline admission) is not merged; no candidate has been admitted or rejected by an admission check.

### Checkpoints and rollbacks (gate 7)

No checkpoint store supplied (--checkpoints). The gate 7 rollback mechanism (checkpoint.admit_with_rollback) is merged, but no admission has run, so there is no rollback history: 0 rollbacks is not a safety result.

## Live Hermes smoke run

**4 live cases; a smoke test, not a benchmark result.** Plumbing check, not evidence that Hermes improves or performs well. One run, no confidence intervals. Cost is Hermes' estimate at API rates; the provider bills a subscription, so metered spend may be zero.

Source: live Hermes smoke run, 2026-11-22, owner-approved (PR #10); transcribed from docs/sprints/KAFO-EVO-001.md "Live smoke run", values rounded as published. Model: claude-sonnet-5-5[1m] via claude-subscription-directsdk-experimental, no toolsets. Split: validation.

| Case | Completed | Quality (F1) | Est. cost USD | Latency s | Violation |
|---|---|---|---|---|---|
| pyrev-007 | True | 0.50 | 0.0041 | 4.3 | False |
| pyrev-008 | True | 0.67 | 0.0045 | 3.7 | False |
| pyrev-031 | True | 1.00 | 0.0014 | 3.7 | False |
| pyrev-043 | True | 0.80 | 0.0063 | 5.2 | False |

- Completed: 4 / 4; violations: 0.
- Mean quality 0.743, mean cost USD 0.0041, mean latency s 4.2.

## Final test evaluation (one-time, recorded)

The one-time final test evaluation has not been run; when and on which reviewer is an owner decision. It is run by `evolution.adaptive.final_evaluation.run_final_evaluation`, which refuses a second run.

## Pending

- Pending: gate 6 (candidate-vs-baseline admission) is not merged; no candidate has been admitted or rejected by an admission check
- Rollback history: none yet, because no admission has run
- Final test evaluation: not run (owner decision)

## Reproduce

```bash
python -m evolution.adaptive.report --seeds 20 --train-episodes 300 --benchmark-seed 0 --resamples 2000 --verifier-seed 0 --verifier-resamples 1000
```
