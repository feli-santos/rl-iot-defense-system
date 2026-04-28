# RL Training (Blue Team)

## Overview

The Blue Team uses Stable Baselines3 algorithms (DQN, PPO, A2C) with **MlpPolicy** to learn from the Box observation space. The training wrapper is `AdversarialAlgorithm` with orchestration in `TrainingManager`.

## Algorithms

- **DQN** (value-based)
- **PPO** (policy gradient)
- **A2C** (actor-critic)

All are configured via `config.yml` under `rl.algorithms`.

## Observation and policy

The environment emits a flattened feature window of real CICIoT2023 data. Because observations are Box vectors, `MlpPolicy` is used for all algorithms.

## Training loop

1. Create environment (`AdversarialIoTEnv`)
2. Instantiate model via `AdversarialAlgorithm.create_model`
3. Train using `TrainingManager.train_algorithm`
4. Save best and final models
5. Optionally run quick evaluation

## MLflow logging

`TrainingManager` attaches a custom `MLflowCallback` that logs:

- Rewards (mean, std, min, max)
- Episode lengths
- Action distribution (OBSERVE…ISOLATE)
- Performance (FPS, step time)
- Losses (policy/value/entropy)
- Exploration rate (DQN)

## Key config parameters

From `config.yml`:

- `rl.training.total_timesteps`
- `rl.training.eval_freq`
- `rl.training.n_eval_episodes`
- `rl.training.save_freq`

Algorithm-specific:

- `rl.algorithms.dqn.*`
- `rl.algorithms.ppo.*`
- `rl.algorithms.a2c.*`

## Key files

- `src/algorithms/adversarial_algorithm.py`
- `src/training/training_manager.py`
- `src/environment/adversarial_env.py`

## Notes

- Training uses **real** observations; the agent never sees the true attack stage.
- Environment generator inference is forced to CPU for stability on Apple MPS.
- Action costs and reward shaping can dominate learning behavior—tune carefully.
