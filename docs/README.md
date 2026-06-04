# RL-IoT Defense System Documentation

Welcome to the technical documentation for the **Adversarial IoT
Defense System**. This doc set targets internal engineers and the
thesis defense committee, and is aligned with the current
implementation on the `main` branch.

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
8. **Configuration & CLI** → `configuration.md`
9. **Experiments & MLflow** → `experiments-mlflow.md`
10. **Design decisions** → `decisions.md`
13. **Reproducibility** → `reproducibility.md`
14. **Step-by-step walkthrough** → `walkthrough.md`

## Source of truth

All descriptions are derived from these entry points and modules:

- `main.py`
- `config.yml`
- `src/utils/*` (dataset processing, label mapping, realization engine)
- `src/generator/*` (episode generator, LSTM, transition mask)
- `src/environment/adversarial_env.py`
- `src/algorithms/adversarial_algorithm.py`
- `src/training/*` (training managers and callbacks)
- `src/benchmark/*` (Phase-6 baselines, eval runner, latency bench)
- `src/blue_team/*` (Phase-5 env factory, callbacks, run config)
- `src/detector/*` (Phase-4 supervised stage detector: RF + 1D-CNN)

If you spot a mismatch between docs and code, treat code as canonical and flag the discrepancy.
