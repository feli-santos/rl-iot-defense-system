# RL-IoT Defense System: Overview

## System Architecture

The RL-IoT Defense System is an intelligent cybersecurity solution that leverages reinforcement learning to protect IoT networks from adaptive attacks. The system's architecture consists of four primary components:

1.  **IoT Environment Simulation**: A customizable network of IoT devices, attackable nodes, and defensive capabilities, implemented as a Gymnasium environment.
2.  **Attack Prediction Model**: An LSTM-based sequence predictor that anticipates attacker behavior based on historical event patterns.
3.  **RL Defense Agents (DQN, PPO, A2C)**: Reinforcement learning agents that learn and select optimal defensive actions using Stable Baselines3.
4.  **Experiment Tracking & Management**: MLflow integration via a `TrainingManager` for monitoring, versioning models, and managing experiments.

## Workflow

The system operates through a continuous cycle of:

1.  **Environment Observation**: Monitoring the current state of the IoT network (current device states, history of states, and past actions).
2.  **Attack Prediction**: Using the LSTM model to predict the next likely attack target or sequence.
3.  **Defense Selection**: Applying the trained RL policy (DQN, PPO, or A2C) to select the optimal defensive countermeasure.
4.  **Environment Update**: Implementing the defense and observing the new network state and received reward.

### Data Flow

```
┌─────────────────┐       ┌─────────────────┐      ┌─────────────────┐
│                 │       │                 │      │                 │
│  IoT Network    │───▶   │  LSTM Attack    │───▶  │  RL Defense     │
│  Environment    │       │  Predictor      │      │  Policy (DQN,   │
│ (Gymnasium)     │       │                 │      │   PPO, or A2C)  │
└────────┬────────┘       └─────────────────┘      └────────┬────────┘
         │ (State, Reward)                              │ (Action)
         │                                              │
         └──────────────────────◀───────────────────────┘
                         Action Implementation
```

## Key Innovations

1.  **Modular RL Agents**: Support for multiple state-of-the-art RL algorithms (DQN, PPO, A2C) using Stable Baselines3, allowing for comparative analysis.
2.  **Predictive Defense Capabilities**: Potential to integrate LSTM-based attack predictions to inform RL agent decisions for proactive defense.
3.  **Adaptive Defense**: RL agents learn to counter evolving attack strategies through interaction with the simulated environment.
4.  **Comprehensive Experimentation**: Unified training script with CLI, robust benchmarking framework, and MLflow integration for reproducible research.
5.  **Rich Observation Space**: `Dict` observation space including current state, state history, and action history for informed decision-making by agents.

## Performance Metrics

The system's effectiveness is measured using:

1.  **Average Episode Reward**: Primary metric for RL agent performance.
2.  **Training Time**: Efficiency of different algorithms.
3.  **Convergence Speed**: How quickly agents reach optimal policies.
4.  **Attack Success Rate** (via environment logs/custom metrics): Percentage of successful attacks during evaluation.
5.  **Network Integrity** (via environment logs/custom metrics): Overall health of the IoT network.

## Components Integration

The system integrates these components through:

1.  **State Representation**: Unified `Dict` observation space for the RL agents.
2.  **Action Space Mapping**: Consistent discrete action space for defense countermeasures.
3.  **Reward Signal Design**: Crafted to incentivize both short-term threat mitigation and long-term network security.
4.  **Configuration Management**: Centralized `config.yml` for all parameters.
5.  **Algorithm Factory**: Simplifies creation and swapping of RL algorithms.

The following documents provide detailed explanations of each component:
- [IoT Environment Simulation](./environment.md)
- [LSTM Attack Prediction](./attack_prediction.md)
- [RL Defense Agents](./rl_defense_agents.md)
- [Experiment Tracking](./experiment_tracking.md)
- [Benchmarking Setup](./benchmarking_setup.md)
- [Mathematical Foundations](./mathematical_foundations.md)