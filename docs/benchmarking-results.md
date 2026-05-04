# Benchmarking & Results Interpretation

> **STATUS — pre-restart, retained for historical reference (Phase 10, D10.2).**
> This document predates the Phase-1 restart and refers to the dead
> `src/benchmarking/` package, which was removed in Phase 10 (2026-05-04).
> The canonical, gate-passing benchmark chapter is now
> [`docs/results/06_benchmark/RESULTS.md`](results/06_benchmark/RESULTS.md);
> the Phase-7 ablation/OOD chapter is
> [`docs/results/07_ablation/RESULTS.md`](results/07_ablation/RESULTS.md).
> The live evaluation runners are `src/benchmark/{baseline_policies,
> eval_runner,latency}.py` (Phase 6) and `scripts/benchmark/run_test_eval.py`,
> not the modules referenced below.

## Purpose

Benchmarking evaluates how well the Blue Team policy mitigates attacks while minimizing false positives and preserving availability.

The evaluation pipeline lives in:

- `src/benchmarking/benchmark_runner.py`
- `src/benchmarking/metrics_collector.py`
- `src/benchmarking/benchmark_analyzer.py`

## Benchmark flow

1. Load trained model(s).
2. Create evaluation environment with same generator + dataset.
3. Run $N$ episodes (default: 20).
4. Collect episode metrics.
5. Aggregate security metrics and generate plots.

## Core security metrics

| Metric | Meaning | Desired Direction |
|:--|:--|:--|
| Attack Mitigation Rate | % episodes not reaching IMPACT | Higher ↑ |
| False Positive Rate | % benign steps with active defense | Lower ↓ |
| Mean Time to Contain | Steps to reset to BENIGN | Lower ↓ |
| Availability Score | $1/(1+\text{total action cost})$ | Higher ↑ |

## Stage detection metrics

Actions are projected to stages for recall/F1 reporting:

- OBSERVE → BENIGN
- LOG → RECON
- THROTTLE → ACCESS
- BLOCK → MANEUVER
- ISOLATE → IMPACT

This enables per-stage recall and macro-F1 even for RL policies.

## Outputs

Results are saved to `results/benchmark/`:

- `comparison.json` — raw metrics by run
- `benchmark_evaluation_report.json` — summarized report
- `analysis/` — plots

Plots include:

- Reward distributions
- Security metrics comparison
- Defense heatmap
- Stage-action confusion matrices

## Interpreting false positives

A high false positive rate means the policy takes active defense during BENIGN steps. Tune:

- `penalty_overreact_benign`
- `penalty_block_benign`
- `reward_benign_passive`

## Interpreting mitigation

Low mitigation rate typically indicates:

- insufficient escalation reward
- too-low impact penalty
- short episodes with low stage coverage

## Key files

- `src/benchmarking/benchmark_runner.py`
- `src/benchmarking/metrics_collector.py`
- `src/benchmarking/benchmark_analyzer.py`
