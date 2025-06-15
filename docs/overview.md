# RL-IoT Defense System: Overview

## System Architecture

The RL-IoT Defense System is an intelligent cybersecurity solution that leverages reinforcement learning to protect IoT networks from adaptive attacks. The system's architecture consists of four primary components:

1.  **IoT Environment Simulation**: A customizable network of IoT devices, attackable nodes, and defensive capabilities, implemented as a Gymnasium environment (`src/environment.py`).
2.  **Attack Prediction Model**: An LSTM-based sequence predictor (`src/models/lstm_attack_predictor.py`) trained on the CICIoT2023 dataset to anticipate attacker behavior. The `src/models/predictor_interface.py` provides a standardized way for the environment to use these predictions.
3.  **RL Defense Agents (DQN, PPO, A2C)**: Reinforcement learning agents that learn and select optimal defensive actions using Stable Baselines3. Managed by `src/algorithms/algorithm_factory.py`.
4.  **Experiment Tracking & Management**: MLflow integration via a `TrainingManager` (`src/training/training_manager.py`) for monitoring, versioning models, and managing experiments. The overall training is orchestrated by `main.py` using `src/training/lstm_trainer.py` and `src/training/rl_trainer.py`.

## Workflow

The system operates through a continuous cycle of:

1.  **Environment Observation**: Monitoring the current state of the IoT network (current device states, history of states, past actions, and attack predictions).
2.  **Defense Selection**: Applying the trained RL policy (DQN, PPO, or A2C) to select the optimal defensive countermeasure.
3.  **Environment Update**: Implementing the defense and observing the new network state and received reward. The environment internally uses the LSTM predictor to inform its state and reward calculations.

### Data Flow

```
┌───────────────────────────┐       ┌───────────────────────────┐      ┌─────────────────────────┐
│                           │       │                           │      │                         │
│  IoT Network              │───▶   │  Attack                   │───▶  │  RL Defense             │
│  Environment (Gymnasium)  │       │  Predictor (Interface)    │      │  Policy (DQN,           │
│  (src/environment.py)     │       │  (src/models/predictor_   │      │   PPO, or A2C)          │
│                           │       │   interface.py)           │      │  (src/algorithms/*)     │
└───────────┬───────────────┘       └───────────┬───────────────┘      └───────────┬─────────────┘
            │ (State, Reward)                   │ (LSTM Predictions)               │ (Action)
            │ (Includes Attack Prediction)      │                                  │
            └───────────────────────────────────◀──────────────────────────────────┘
                                    Action Implementation & State Update
```

## Key Innovations

1.  **Modular RL Agents**: Support for multiple state-of-the-art RL algorithms (DQN, PPO, A2C) using Stable Baselines3, allowing for comparative analysis.
2.  **Real Data Attack Prediction**: LSTM-based attack predictions trained on the CICIoT2023 dataset are integrated into the RL agent's observation space, enabling proactive defense.
3.  **Adaptive Defense**: RL agents learn to counter evolving attack strategies through interaction with the simulated environment that incorporates realistic attack patterns.
4.  **Comprehensive Experimentation**: Unified training script (`main.py`) with CLI, robust benchmarking framework, and MLflow integration for reproducible research.
5.  **Rich Observation Space**: `Dict` observation space including current state, state history, action history, and real-time attack predictions for informed decision-making by agents.

## Performance Metrics

The system's effectiveness is measured using:

1.  **Average Episode Reward**: Primary metric for RL agent performance.
2.  **Training Time**: Efficiency of different algorithms.
3.  **Convergence Speed**: How quickly agents reach optimal policies.
4.  **Attack Prediction Accuracy/F1-score**: Performance of the LSTM model.
5.  **Network Integrity & Attack Mitigation Rate** (via environment logs/custom metrics): Overall health and security of the IoT network during evaluation.

## Components Integration

The system integrates these components through:

1.  **State Representation**: Unified `Dict` observation space for the RL agents, including LSTM predictions.
2.  **Action Space Mapping**: Consistent discrete action space for defense countermeasures.
3.  **Reward Signal Design**: Crafted to incentivize both short-term threat mitigation (informed by predictions) and long-term network security.
4.  **Configuration Management**: Centralized `config.yml` for all parameters, loaded by `src/utils/config_loader.py`.
5.  **Algorithm Factory**: Simplifies creation and swapping of RL algorithms.

The following documents provide detailed explanations of each component:
- [IoT Environment Simulation](./environment.md)
- [CICIoT2023 Dataset](./ciciot2023_dataset.md)
- [LSTM Attack Prediction](./attack_prediction.md)
- [RL Defense Agents](./rl_defense_agents.md)
- [Mathematical Foundations](./mathematical_foundations.md)
- [Visual Architecture](./visual_architecture.md)