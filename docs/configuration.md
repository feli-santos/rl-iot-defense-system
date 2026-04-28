# Configuration & CLI

## Configuration files

- `config.yml` — full training configuration
- `config_dry_run.yml` — quick plumbing test

## CLI entrypoint

`main.py` parses arguments and dispatches to modes:

- `process-data`
- `train-generator`
- `train-rl`
- `train-all-rl`
- `train-all`
- `evaluate`

Common flags:

- `--config` (default: `config.yml`)
- `--data-path` (processed dataset)
- `--generator-path`
- `--rl-path`
- `--algorithm {dqn,ppo,a2c}`
- `--timesteps`
- `--eval-episodes`
- `--device {cpu,cuda,mps}`
- `--force`

## Config structure (top-level)

- `dataset` — raw path, sampling, feature selection
- `episode_generation` — synthetic episode grammar
- `attack_generator` — LSTM architecture + training
- `adversarial_environment` — observation window, action space, rewards
- `rl` — algorithm + hyperparameters
- `models` — artifact paths
- `mlflow` — tracking config
- `benchmark` — evaluation config

## Overrides

CLI arguments override values in `config.yml`, e.g.:

- `--timesteps` overrides `rl.training.total_timesteps`
- `--generator-epochs` overrides `attack_generator.training.epochs`

## Dry run

`config_dry_run.yml` uses tiny PPO timesteps (2048) for end-to-end validation.
