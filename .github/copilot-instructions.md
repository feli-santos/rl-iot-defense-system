# Copilot Instructions

## Project Overview
This is an **IoT Defense System** using **Reinforcement Learning** to defend against cyberattacks. The system combines LSTM attack prediction trained on **CICIoT2023** dataset with RL algorithms (DQN, PPO, A2C) for adaptive defense policy learning.

## Key Technologies
- **Python 3.12** with virtual environment
- **Stable Baselines3** for RL algorithms  
- **PyTorch** for LSTM attack predictor
- **Gymnasium** for custom IoT environment
- **MLflow** for experiment tracking
- **CICIoT2023** dataset for real attack data

## Coding Standards
- Use **type hints** for all function parameters and returns
- Follow **Google docstring** format
- Use **pathlib.Path** for file paths, not os.path
- Prefer **f-strings** for string formatting
- Add comprehensive **error handling** with descriptive messages
- Use **dataclasses** for configuration objects

## Key Technical Constraints
- **Discrete action space** (4 defense actions: monitor, rate_limit, block_ips, shutdown_services)
- **Dict observation space** with keys: `current_state`, `state_history`, `action_history`, `attack_prediction`
- **MultiInputPolicy** required for all SB3 algorithms due to Dict observations
- **Real data only** - CICIoT2023 dataset, no synthetic data generation
- **Stable Baselines3** - use SB3 implementations, not custom RL algorithms

## Common Patterns
- Use `AlgorithmFactory.create_algorithm()` to instantiate RL algorithms
- Use `TrainingManager` for MLflow experiment tracking and model artifacts
- Use `EnhancedAttackPredictor` to bridge LSTM predictions with RL environment
- Main training via `main.py` with modes: `lstm`, `rl`, `both`
- Configuration loading via `ConfigLoader.load_config()`

## Data Flow
```
CICIoT2023 → RealDataLSTMPredictor → EnhancedAttackPredictor → IoTEnv → RL Agent
```

## Debugging Tips
- Verify **observation/action space compatibility** between environment and algorithm
- Ensure **MultiInputPolicy** is used for Dict observation spaces
- Check **config parameter names** match between algorithms (dqn/ppo/a2c sections)
- Test **LSTM model loading** before RL training (use `--force-retrain-lstm` if needed)
- Use **--log-level DEBUG** for detailed troubleshooting
- Validate **preprocessing artifacts** exist for dataset loading

## Critical Implementation Notes
- Environment uses `EnhancedAttackPredictor` internally for attack predictions
- All RL algorithms must handle 4-element Dict observation space
- LSTM predictions are integrated into RL observation via `attack_prediction` key
- Training modes can be run independently or together via main.py CLI
- MLflow tracking is automatic for both LSTM and RL training phases