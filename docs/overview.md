# RL-IoT Defense System: Overview

## System Architecture

The RL-IoT Defense System is an intelligent cybersecurity solution that leverages reinforcement learning to protect IoT networks from adaptive attacks. The system's architecture consists of four primary components:

![System Architecture](./images/system_architecture.png)

1. **IoT Environment Simulation**: A customizable network of IoT devices with attackable nodes and defensive capabilities
2. **Attack Prediction Model**: An LSTM-based sequence predictor that anticipates attacker behavior
3. **DQN Defense Policy**: A reinforcement learning agent that selects optimal defensive actions
4. **Experiment Tracking System**: MLflow-based monitoring and versioning of models and experiments

## Workflow

The system operates through a continuous cycle of:

1. **Environment Observation**: Monitoring the current state of the IoT network
2. **Attack Prediction**: Using the LSTM model to predict the next likely attack target
3. **Defense Selection**: Applying the DQN policy to select the optimal defensive countermeasure
4. **Environment Update**: Implementing the defense and observing the new network state

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  IoT Network    │───▶│  LSTM Attack    │───▶│  DQN Defense    │
│  Environment    │    │  Predictor      │    │  Policy         │
│                 │    │                 │    │                 │
└────────┬────────┘    └─────────────────┘    └────────┬────────┘
         │                                              │
         │                                              │
         │                                              │
         └──────────────────────◀───────────────────────┘
                         Action Implementation
```

## Key Innovations

1. **Dual-Model Approach**: Combining predictive modeling (LSTM) with decision-making (DQN) for proactive defense
2. **Adaptive Defense**: Learning to counter evolving attack strategies through reinforcement learning
3. **Realistic Attack Simulation**: Graph-based attack propagation with intelligent attacker behavior

## Performance Metrics

The system's effectiveness is measured using:

1. **Attack Success Rate**: Percentage of successful attacks (lower is better)
2. **Network Integrity**: Overall health of the IoT network during attacks
3. **Defense Efficiency**: Ratio of successful defenses to defensive actions taken

## Components Integration

The system integrates these components through:

1. **State Representation**: Unified representation of network state for both models
2. **Action Space Mapping**: Translation between predicted attacks and defensive countermeasures
3. **Reward Signal Design**: Crafted to incentivize both short-term threat mitigation and long-term network health

The following documents provide detailed explanations of each component:
- [IoT Environment Simulation](./environment.md)
- [LSTM Attack Prediction](./attack_prediction.md)
- [DQN Defense Policy](./defense_policy.md)
- [Experiment Tracking](./experiment_tracking.md)
- [Mathematical Foundations](./mathematical_foundations.md)