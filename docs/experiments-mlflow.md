# Experiments & MLflow

## Overview

Both generator training and RL training can be tracked in MLflow.

- **Generator** uses `GeneratorTrainer` with MLflow logging.
- **RL training** uses `TrainingManager` with `MLflowCallback`.

Tracking directory defaults to `mlruns/`.

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
