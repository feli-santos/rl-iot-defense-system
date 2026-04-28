# RL-IoT Defense System Documentation

Welcome to the technical documentation for the **Adversarial IoT Defense System**. This doc set targets internal engineers and is aligned with the current implementation on branch `feature/reward-shaping`.

## What this system does

The pipeline builds an adversarial training loop over **real CICIoT2023 traffic**:

- **Red Team (LSTM)** learns a Kill Chain grammar and generates attack stage sequences.
- **Blue Team (RL)** learns defensive actions using realized network features sampled from CICIoT2023.
- **Benchmarking** evaluates mitigation success, false positives, and availability trade-offs.

## How to read these docs

Start with architecture and data flow, then drill into the subsystems and math.

1. **Architecture** → `architecture.md`
2. **Data pipeline** → `data-pipeline.md`
3. **Kill Chain mapping** → `kill-chain-mapping.md`
4. **Generator (LSTM)** → `generator.md`
5. **Environment** → `environment.md`
6. **Reward shaping** → `reward-shaping.md`
7. **RL training** → `rl-training.md`
8. **Benchmarking** → `benchmarking-results.md`
9. **Configuration & CLI** → `configuration.md`
10. **Experiments & MLflow** → `experiments-mlflow.md`
11. **Metrics glossary** → `metrics-glossary.md`
12. **Design decisions** → `decisions.md`
13. **Reproducibility** → `reproducibility.md`
14. **Step-by-step walkthrough** → `walkthrough.md`

## Source of truth

All descriptions are derived from these entry points and modules:

- `main.py`
- `config.yml`, `config_dry_run.yml`
- `src/utils/*` (dataset processing, label mapping, realization engine)
- `src/generator/*` (episode generator, LSTM, transition mask)
- `src/environment/adversarial_env.py`
- `src/algorithms/adversarial_algorithm.py`
- `src/training/*` (training managers and callbacks)
- `src/benchmarking/*` (benchmark runner, analyzer, metrics)

If you spot a mismatch between docs and code, treat code as canonical and flag the discrepancy.
