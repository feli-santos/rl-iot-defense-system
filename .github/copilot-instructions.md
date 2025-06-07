# Copilot Instructions

## Project Overview
This is an **IoT Defense System** using **Reinforcement Learning** to defend against trigger-action attacks. The system uses LSTM for attack prediction and RL algorithms (DQN, PPO, A2C) for defense policy learning.

## Key Technologies
- **Python 3.12** with virtual environment
- **Stable Baselines3** for RL algorithms  
- **PyTorch** for LSTM attack predictor
- **Gymnasium** for custom IoT environment
- **MLflow** for experiment tracking

## Project Structure
```
src/
├── algorithms/          # RL algorithms (DQN, PPO, A2C)
├── benchmarking/        # Benchmark runner and analysis
├── environment.py       # Custom IoT gym environment
├── models/             # LSTM attack predictor
├── utils/              # Training manager, config loader
└── config.yml          # Main configuration
```

## Coding Standards
- Use **type hints** for all function parameters and returns
- Follow **Google docstring** format
- Use **pathlib** for file paths, not os.path
- Prefer **f-strings** for string formatting
- Add **error handling** with descriptive messages

## Key Constraints
- **Discrete action space** (4 defense actions) - no continuous actions
- **Dict observation space** - use `MultiInputPolicy` for all SB3 algorithms
- **Consistent interfaces** - all algorithms must implement BaseAlgorithm
- **Stable Baselines3** - use SB3 for all RL algorithms, not custom implementations

## Common Patterns
- Use `AlgorithmFactory` to create algorithm instances
- Use `TrainingManager` for experiment tracking and artifact management
- Use `BenchmarkRunner` for multi-algorithm comparisons
- Environment observations: `{'current_state', 'state_history', 'action_history'}`

## Debugging Tips
- Check observation/action space compatibility first
- Use `MultiInputPolicy` for Dict observations
- Verify config parameter names match between algorithms
- Test single algorithm before running full benchmarks