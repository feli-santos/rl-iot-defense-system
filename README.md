# IoT Defense System powered by Reinforcement Learning

## Overview

This project implements a comprehensive IoT defense system that uses **Reinforcement Learning** to defend against cyberattacks in smart IoT environments. The system combines LSTM-based attack prediction trained on real **CICIoT2023** dataset with RL-based defense policy learning, providing an adaptive defense mechanism that learns optimal strategies through experience.

## Architecture

The system consists of two main components:

1. **LSTM Attack Predictor**: Trained on real CICIoT2023 IoT attack data to predict attack types and risk levels
2. **RL Defense Agent**: Learns optimal defense policies using state-of-the-art RL algorithms (DQN, PPO, A2C)

### Key Features

- **Real IoT Attack Data**: Uses CICIoT2023 dataset with 33 attack types for realistic training
- **Custom Gymnasium Environment**: Simulates IoT network defense with Dict observation spaces
- **Multiple RL Algorithms**: Supports DQN, PPO, and A2C with consistent Stable Baselines3 implementations
- **LSTM Attack Prediction**: 98%+ accuracy on real IoT attack classification
- **Comprehensive Benchmarking**: Built-in comparison framework for algorithm performance analysis
- **MLflow Integration**: Complete experiment tracking with artifacts, metrics, and model versioning
- **Unified Training Pipeline**: Single entry point for both LSTM and RL training

## Project Structure

```
rl-iot-defense-system/
.
├── config.yml
├── docs
│   ├── attack_prediction.md
│   ├── environment.md
│   ├── evolution_roadmap.md
│   ├── mathematical_foundations.md
│   ├── overview.md
│   ├── papers
│   │   ├── A Survey for Deep Reinforcement Learning Based Network Intrusion Detection.pdf
│   │   ├── CICIoT2023- A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.pdf
│   │   ├── Deep Reinforcement Learning for Internet of Things- A Comprehensive Survey.pdf
│   │   ├── Deep Reinforcement Learning for Intrusion Detection in IoT- A Survey.pdf
│   │   ├── Enhancing IoT Intelligence- A Transformer-based Reinforcement Learning Methodology.pdf
│   │   ├── HoneyIoT- Adaptive High-Interaction Honeypot for IoT Devices Through Reinforcement Learning.pdf
│   │   ├── IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System to Mitigate Trigger-action IoT Attacks.pdf
│   │   ├── Reinforcement Learning for IoT Security- A Comprehensive Survey.pdf
│   │   └── Wireless Communications and Mobile Computing - 2022 - Tharewal - Intrusion Detection System for Industrial Internet of.pdf
│   ├── project_proposal.md
│   ├── rl_defense_agents.md
│   └── visual_architecture.md
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── src
    ├── algorithms
    │   ├── a2c_algorithm.py
    │   ├── algorithm_factory.py
    │   ├── base_algorithm.py
    │   ├── dqn_algorithm.py
    │   └── ppo_algorithm.py
    ├── benchmarking
    │   ├── benchmark_analyzer.py
    │   ├── benchmark_runner.py
    │   └── metrics_collector.py
    ├── environment
    │   └── environment.py
    ├── predictors
    │   ├── lstm_attack_predictor.py
    │   └── rl_predictor_interface.py
    ├── scripts
    │   └── dataset_exploratory_analysis.py
    ├── training
    │   ├── lstm_trainer.py
    │   ├── rl_trainer.py
    │   └── training_manager.py
    └── utils
        ├── config_loader.py
        ├── dataset_loader.py
        └── dataset_processor.py

11 directories, 40 files
```

## Requirements

- **Python 3.12+**
- **PyTorch** (for LSTM models)
- **Stable Baselines3** (for RL algorithms)
- **MLflow** (for experiment tracking)
- **Gymnasium** (for RL environment)
- **NumPy, Pandas, Scikit-learn** (for data processing)
- **Matplotlib, Seaborn** (for visualization)

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

4. **Download CICIoT2023 Dataset:**
   ```bash
   # Download from: https://www.unb.ca/cic/datasets/iotdataset-2023.html
   # Extract to: data/raw/CICIoT2023/
   ```

## Usage

### Quick Start

The system provides a unified training pipeline through the main entry point:

```bash
# Train both LSTM and RL agents (recommended for first run)
python main.py --mode both

# Train only LSTM attack predictor
python main.py --mode lstm

# Train only RL agents (requires existing LSTM model)
python main.py --mode rl
```

### Training Modes

#### 1. Complete Training Pipeline
Train both LSTM attack predictor and RL defense agents:
```bash
python main.py --mode both --lstm-epochs 50 --rl-timesteps 100000
```

#### 2. LSTM Training Only
Train attack predictor on CICIoT2023 dataset:
```bash
python main.py --mode lstm --lstm-epochs 30 --lstm-batch-size 64
```

#### 3. RL Training Only
Train defense agents (requires existing LSTM model):
```bash
python main.py --mode rl --rl-algorithm dqn --rl-timesteps 50000
```

### Advanced Usage

#### Custom Configuration
Use custom configuration file:
```bash
python main.py --config custom_config.yml --mode both
```

#### Force Retrain LSTM
Retrain LSTM even if model exists:
```bash
python main.py --mode both --force-retrain-lstm
```

#### Algorithm-Specific Training
Train with specific RL algorithm:
```bash
python main.py --mode rl --rl-algorithm ppo --rl-timesteps 100000
python main.py --mode rl --rl-algorithm a2c --rl-timesteps 75000
```

#### Override Hyperparameters
Override specific parameters:
```bash
python main.py --mode both --learning-rate 0.001 --lstm-batch-size 32
```

### Configuration

The [`config.yml`](config.yml) file contains all system parameters.

### Data Processing

#### Exploratory Data Analysis
Analyze the dataset:
```bash
python src/scripts/dataset_exploratory_analysis.py
```

## Algorithms Supported

### 1. DQN (Deep Q-Network)
- **Value-based** learning with experience replay
- **Target network** for stable learning
- **Best for**: Discrete action spaces with complex state representations

### 2. PPO (Proximal Policy Optimization)
- **Policy-based** learning with clipped surrogate objective
- **Stable** and robust training
- **Best for**: Consistent performance across different environments

### 3. A2C (Advantage Actor-Critic)
- **Actor-critic** architecture
- **Faster** training compared to PPO
- **Best for**: Quick baseline results


## Experiment Tracking

### MLflow Integration
All experiments are tracked automatically:
```bash
# Start MLflow UI
mlflow ui

# Access at: http://localhost:5000
```

### Key Metrics Tracked
- **LSTM Performance**: Accuracy, F1-score, confusion matrix
- **RL Performance**: Episode rewards, training time, convergence
- **Model Artifacts**: Trained models, configuration, plots


## Contributing

1. **Follow Coding Standards**:
   - Use **type hints** for all functions
   - Follow **Google docstring** format
   - Use **pathlib** for file paths
   - Add comprehensive **error handling**

2. **Testing Requirements**:
   - Add tests for new features
   - Ensure backward compatibility
   - Test with all supported algorithms

3. **Documentation**:
   - Update README for new features
   - Add docstrings to all functions
   - Include usage examples

## Citation

If you use this work in your research, please cite:

```bibtex
@software{iot_defense_rl_2025,
  title={IoT Defense System powered by Reinforcement Learning},
  author={Felipe Santos},
  year={2025},
  note={Real CICIoT2023 dataset integration with LSTM attack prediction},
  url={https://github.com/feli-santos/rl-iot-defense-system}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **CICIoT2023 Dataset**: University of New Brunswick Cyber Security Research Group
- **Stable Baselines3**: High-quality RL algorithm implementations
- **PyTorch**: Deep learning framework for LSTM implementation

---

**Keywords**: IoT Security, Reinforcement Learning, CICIoT2023, Deep Q-Network, PPO, A2C, Cybersecurity, Attack Prediction, Defense Systems, LSTM, Real Dataset