# Experiments & MLflow

> **Scope (Step-5 F5 / Step-8 doc-fix).** This document describes the
> MLflow setup that was *available* during the early-Phase exploratory
> work. **Phase 5 onwards intentionally does NOT use MLflow** —
> Phase-5 D5.6 locks the per-run logging format as schema-v1.0
> JSONL files (``runs/phase5/<algo>/seed_<k>/{episodes,eval}.jsonl``)
> with side-car ``run_manifest.json`` records, and Phase 6 / Phase 7
> follow the same convention via
> ``scripts.benchmark.run_test_eval`` and the ``run_*`` ablation
> drivers. The rationale is reproducibility-by-hash-chain (see
> ``docs/results/<phase>/manifest.json::input_hashes``) — the JSONL
> format is what the per-figure manifest SHA-pins, not the
> MLflow-tracking-server URI.
>
> The earlier Step-4 mentor handoff (`04_HANDOFF.md`) incorrectly
> forecast Phase 5 as "the first phase with MLflow runs". That
> forecast was retired by the actual Phase-5 implementation; this
> document should not be read as a current-Phase mandate.
> Step 9 (LaTeX rebuild) reads its run telemetry from the JSONL
> records + per-figure manifests, not from MLflow.
>
> **What is actually tracked under `mlruns/`** today: only the
> generator-trainer (`scripts/red_team/train_lstm.py` via
> `src.training.generator_trainer.GeneratorTrainer`) emits MLflow
> entries; the runs are local-only, not committed, and Phase 2's
> canonical numerical record is `F1_summary.json` (see
> `docs/results/02_red_team/RESULTS.md` §2). The RL-side
> `MLflowCallback` described below is unused in Phase 5 — the
> Phase-5 callback chain is defined in `src.blue_team.callbacks`
> with `EpisodeJSONLCallback` as the canonical log.

## Overview

Both generator training and RL training **can** be tracked in MLflow
in principle, but the production phases (5/6/7) use JSONL +
manifest-based logging instead.

- **Generator** uses `GeneratorTrainer` with MLflow logging.
  *(Active in Phase 2, but the canonical numerical record for the
  defense is `F1_summary.json`, not the MLflow run.)*
- **RL training** could use `TrainingManager` with `MLflowCallback`.
  *(Phase 5 chose `EpisodeJSONLCallback` instead — see Phase-5
  RESULTS.md and PLAN D5.6 for the rationale.)*

Tracking directory defaults to `mlruns/` (local-only, not committed).

## Generator MLflow logging

Logged parameters:

- LSTM architecture (embedding size, hidden size, layers)
- Training hyperparameters (epochs, batch size, LR)
- Episode statistics (min/mean/max length)

Logged metrics:

- Training/validation loss
- Macro-F1 and per-stage recalls
- Early stopping indicators

Artifacts:

- `attack_sequence_generator.pth`
- `loss_curves.png`
- config files

## RL MLflow logging

`MLflowCallback` logs:

- Reward statistics (recent mean, std, best, worst)
- Episode lengths
- Action distribution probabilities
- Losses (policy, value, entropy)
- FPS and step time
- Training stability metrics

Artifacts:

- Reward curve (`rl_training_curve.png`)
- Best model (from `EvalCallback`)

## Run organization

Each training run creates:

- `artifacts/rl/<algorithm>_<timestamp>_<id>/`
- `models/`, `logs/`, `plots/` subdirectories

## Interpreting MLflow metrics

- **rewards/episode_mean_recent**: stability of learning
- **actions/*_probability**: policy aggressiveness
- **training/explained_variance**: value function fit (PPO/A2C)
- **training/clip_fraction**: PPO update magnitude

## Key files

- `src/training/training_manager.py`
- `src/training/generator_trainer.py`
- `mlruns/` directory
