# IoT Defense System powered by Reinforcement Learning

## Overview

This project implements a comprehensive IoT defense system that uses **Reinforcement Learning** to defend against trigger-action attacks in smart IoT environments. The system combines LSTM-based attack prediction with RL-based defense policy learning, providing an adaptive defense mechanism that learns optimal strategies through experience.

## Architecture

The system consists of two main components:

1. **LSTM Attack Predictor**: Predicts potential attack sequences based on historical data
2. **RL Defense Agent**: Learns optimal defense policies using state-of-the-art RL algorithms

### Key Features

- **Custom Gymnasium Environment**: Simulates IoT network defense with Dict observation spaces including current state, state history, and action history
- **Multiple RL Algorithms**: Supports DQN, PPO, and A2C with consistent Stable Baselines3 implementations
- **LSTM Attack Prediction**: Preprocesses attack patterns to guide defense decisions
- **Comprehensive Benchmarking**: Built-in comparison framework for algorithm performance analysis
- **MLflow Integration**: Complete experiment tracking with artifacts, metrics, and model versioning
- **Discrete Action Space**: Four defense actions (monitoring, blocking, quarantine, allow)

## Project Structure

```
src/
├── algorithms/              # RL algorithm implementations
│   ├── algorithm_factory.py # Factory pattern for algorithm creation
│   ├── base_algorithm.py    # Common interface for all algorithms
│   ├── dqn_algorithm.py     # Deep Q-Network implementation
│   ├── ppo_algorithm.py     # Proximal Policy Optimization
│   └── a2c_algorithm.py     # Advantage Actor-Critic
├── benchmarking/            # Algorithm comparison framework
│   ├── benchmark_runner.py  # Orchestrates multi-algorithm training
│   ├── metrics_collector.py # Collects and manages performance metrics
│   └── benchmark_analyzer.py # Generates comparison reports and plots
├── models/                  # Neural network models
│   └── attack_predictor.py  # LSTM-based attack prediction model
├── utils/                   # Utilities and helpers
│   ├── training_manager.py  # MLflow integration and artifact management
│   ├── config_loader.py     # Configuration management
│   └── data_generator.py    # Realistic attack data generation
├── environment.py           # Custom IoT defense Gymnasium environment
├── training.py             # Main training script with CLI interface
├── evaluation.py           # Model evaluation utilities
└── config.yml             # Main configuration file
```

## Requirements

- **Python 3.12**
- **PyTorch** (for LSTM models)
- **Stable Baselines3** (for RL algorithms)
- **MLflow** (for experiment tracking)
- **Gymnasium** (for RL environment)
- **NumPy, Matplotlib, Pandas** (for data processing and visualization)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd rl-iot-defense-system
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Train a single algorithm (PPO):
```bash
cd src
python training.py --algorithm PPO
```

### Benchmarking Multiple Algorithms

Compare all three algorithms with multiple runs:
```bash
cd src
python training.py --algorithm ALL --runs 3
```

Or specify specific algorithms:
```bash
cd src
python training.py --algorithms DQN PPO A2C --runs 5
```

### Advanced Options

```bash
# Skip LSTM retraining (use existing model)
python training.py --algorithm ALL --skip-lstm

# Analyze existing benchmark results without retraining
python training.py --analyze-only

# Train with custom configuration
python training.py --algorithm PPO --runs 1
```

### Configuration

Modify `config.yml` to customize:
- **Algorithm hyperparameters** (learning rates, network architectures)
- **Environment parameters** (number of devices, actions, states)
- **Training settings** (timesteps, evaluation episodes)
- **Benchmarking options** (number of runs, metrics to track)

## Environment Details

### Observation Space
```python
Dict({
    'current_state': Box(0.0, 1.0, (12,), float32),     # Current IoT network state
    'state_history': Box(0.0, 1.0, (5, 12), float32),  # Historical states
    'action_history': Box(0.0, 3.0, (5,), float32)     # Previous defense actions
})
```

### Action Space
```python
Discrete(4)  # Defense actions: [monitor, block, quarantine, allow]
```

### Reward Function
- **Positive rewards** for successful defense actions
- **Negative penalties** for allowing attacks to progress
- **Adaptive scoring** based on attack proximity and action appropriateness

## Algorithms Supported

### 1. DQN (Deep Q-Network)
- **Value-based** learning with experience replay
- **Epsilon-greedy** exploration strategy
- **Target network** for stable learning

### 2. PPO (Proximal Policy Optimization)
- **Policy-based** learning with clipped surrogate objective
- **On-policy** algorithm with advantage estimation
- **Robust** and stable training

### 3. A2C (Advantage Actor-Critic)
- **Actor-critic** architecture with advantage estimation
- **Faster** training compared to PPO
- **Good baseline** for policy gradient methods

## Benchmarking and Analysis

The system automatically generates:

### Performance Metrics
- **Average reward ± standard deviation** across runs
- **Training time** comparison
- **Convergence analysis** and learning curves
- **Statistical significance** testing

### Visualization
- **Performance comparison** plots (bar charts and box plots)
- **Training time** comparison
- **Reward distributions** histograms
- **Learning curves** with moving averages

### Output Structure
```
benchmark_analysis/
├── performance_comparison.png
├── training_time_comparison.png
├── reward_distributions.png
├── convergence_analysis.png
├── summary_table.csv
└── summary_report.txt

artifacts/
└── iot_defense_system_YYYYMMDD_HHMMSS_xxxxxx/
    ├── models/          # Trained model files
    ├── logs/           # Training logs and TensorBoard data
    └── plots/          # Generated visualizations
```

## Experiment Tracking

All experiments are automatically tracked with **MLflow**:
- **Hyperparameters** logging
- **Training metrics** (rewards, losses, episode lengths)
- **Model artifacts** with versioning
- **Reproducible results** with seed control

Access MLflow UI:
```bash
mlflow ui
```

## Example Results

After training, you'll get comprehensive analysis including:

```
ALGORITHM BENCHMARK SUMMARY
============================
Algorithm  | Avg Reward      | Training Time   | Runs
DQN        | 6.45 ± 0.12    | 8.2 ± 0.5s     | 3
PPO        | 6.58 ± 0.08    | 7.1 ± 0.3s     | 3
A2C        | 6.36 ± 0.15    | 8.1 ± 0.4s     | 3
```

## Extending the System

### Adding New Algorithms
1. Implement `BaseAlgorithm` interface
2. Add to `AlgorithmFactory`
3. Update configuration file

### Customizing the Environment
- Modify `environment.py` for different IoT scenarios
- Adjust observation/action spaces as needed
- Update reward function for specific use cases

### Custom Metrics
- Extend `MetricsCollector` for additional metrics
- Modify `BenchmarkAnalyzer` for custom visualizations

## Contributing

1. Follow the coding standards in `.github/copilot-instructions.md`
2. Use **type hints** for all functions
3. Follow **Google docstring** format
4. Ensure all algorithms use **Stable Baselines3**
5. Test with the benchmarking framework

## Citation

If you use this work in your research, please cite:

```bibtex
@software{iot_defense_rl,
  title={IoT Defense System powered by Reinforcement Learning},
  author={Felipe Santos},
  year={2025},
  url={https://github.com/feli-santos/rl-iot-defense-system}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Keywords**: IoT Security, Reinforcement Learning, Deep Q-Network, PPO, A2C, Cybersecurity, Attack Prediction, Defense Systems