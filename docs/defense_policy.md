# DQN Defense Policy

## Reinforcement Learning Approach

The defense policy is implemented using Deep Q-Network (DQN), a reinforcement learning algorithm that learns to select optimal defensive actions in response to the current network state and predicted attack patterns.

## DQN Architecture

### Network Structure

The DQN agent is based on the `stable_baselines3` implementation with a custom architecture:

```
MultiInput Policy
    ┌─ State Features ─┐    ┌─ Attack Prediction ─┐
    │                  │    │                     │
    ▼                  ▼    ▼                     ▼
┌───────────┐      ┌───────────┐      ┌───────────────┐
│ Feature   │      │ Feature   │      │ Feature       │
│ Extractor │      │ Extractor │      │ Extractor     │
└─────┬─────┘      └─────┬─────┘      └───────┬───────┘
      │                  │                    │
      ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────┐
│                  Concatenation Layer                 │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Hidden Layer 1: [256 neurons, ReLU]                 │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Hidden Layer 2: [128 neurons, ReLU]                 │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ Output Layer: [4 neurons, Linear]                   │
└─────────────────────────────────────────────────────┘
                (Q-values for each action)
```

## Mathematical Foundations

### Q-Learning

DQN is based on Q-learning, which aims to learn an action-value function $Q(s, a)$ representing the expected future rewards when taking action $a$ in state $s$ and following the optimal policy thereafter.

The optimal Q-function satisfies the Bellman equation:

$$Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$

Where:
- $r$ is the immediate reward
- $\gamma$ is the discount factor (between 0 and 1)
- $s'$ is the next state
- $\max_{a'} Q^*(s', a')$ is the maximum Q-value achievable in the next state

### DQN Algorithm

DQN approximates the Q-function using a neural network and introduces two key innovations:

1. **Experience Replay**: Storing and randomly sampling transitions to break correlations in the observation sequence
2. **Target Network**: Using a separate network for generating targets to stabilize learning

The loss function minimized during training is:

$$L(\theta) = \mathbb{E}_{(s,a,r,s')} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

Where:
- $\theta$ are the parameters of the online network
- $\theta^-$ are the parameters of the target network
- $Q(s, a; \theta)$ is the predicted Q-value
- $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ is the target Q-value

### Epsilon-Greedy Exploration

To balance exploration and exploitation, DQN uses an epsilon-greedy policy:

$$a = 
\begin{cases} 
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1-\epsilon
\end{cases}$$

The epsilon value is annealed from `exploration_initial_eps` to `exploration_final_eps` over `exploration_fraction` of the total training steps.

## Implementation Details

The DQN policy is implemented using the Stable Baselines 3 library:

```python
def train_dqn_policy(lstm_model, training_manager):
    # Create environment
    env = IoTEnv(config)
    env = Monitor(env, monitor_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Define DQN model
    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=config.DQN_LEARNING_RATE,
        buffer_size=config.DQN_BUFFER_SIZE,
        learning_starts=1000,
        batch_size=config.DQN_BATCH_SIZE,
        tau=config.DQN_TAU,
        gamma=config.DQN_GAMMA,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config.DQN_TARGET_UPDATE_FREQ,
        exploration_fraction=0.1,
        exploration_initial_eps=config.EXPLORATION_EPS_START,
        exploration_final_eps=config.EXPLORATION_EPS_END,
        policy_kwargs={
            "net_arch": config.NETWORK_HIDDEN_LAYERS,
            "activation_fn": torch.nn.ReLU
        },
        verbose=config.TRAINING_VERBOSE,
        seed=config.TRAINING_SEED,
        device=config.TRAINING_DEVICE
    )
    
    # Train model
    model.learn(
        total_timesteps=config.DQN_TOTAL_EPISODES * config.DQN_EPOCHS_PER_EPISODE,
        callback=mlflow_callback,
        log_interval=10
    )
```

## Hyperparameters

Key hyperparameters for the DQN agent include:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `learning_rate` | Learning rate for optimizer | 1e-4 |
| `buffer_size` | Size of replay buffer | 100,000 |
| `batch_size` | Minibatch size for updates | 64 |
| `gamma` | Discount factor | 0.99 |
| `tau` | Soft update coefficient | 1e-3 |
| `target_update_interval` | Steps between target network updates | 500 |
| `exploration_fraction` | Fraction of training to anneal epsilon | 0.1 |
| `exploration_initial_eps` | Initial exploration rate | 1.0 |
| `exploration_final_eps` | Final exploration rate | 0.1 |
| `net_arch` | Neural network architecture | [256, 128] |

These parameters are configurable in the `config.py` file.

## Training Process

The DQN agent is trained for a specified number of episodes (`DQN_TOTAL_EPISODES`) with each episode consisting of multiple steps (`DQN_EPOCHS_PER_EPISODE`).

During training, the agent:

1. Observes the current state of the environment
2. Uses the LSTM model's predictions to anticipate future attacks
3. Selects an action based on its current policy (with exploration)
4. Applies the action to the environment
5. Observes the reward and next state
6. Stores the transition in the replay buffer
7. Periodically samples from the buffer and updates its policy

## Performance Metrics

Training progress is monitored using several metrics:

1. **Episode Reward**: Average reward per episode
2. **Episode Length**: Average number of steps per episode
3. **Exploration Rate**: Current epsilon value
4. **Loss**: Current Q-network loss value

Typical metrics from recent training:
- Episode Reward Mean: ~7.0
- Episode Length Mean: ~3.5
- Loss: ~0.1

## Inference Process

During deployment, the trained DQN model selects actions using a deterministic policy:

```python
def evaluate_model(model_path, num_episodes=10):
    env = DummyVecEnv([lambda: IoTEnv(config)])
    model = DQN.load(model_path, env=env)
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action deterministically
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action
            obs, reward, done, info = env.step(action)
            total_reward += reward
```

## Integration with LSTM Predictor

The DQN defense policy integrates with the LSTM attack predictor by:

1. Including LSTM predictions in the state representation
2. Using prediction confidence to weight defensive priorities
3. Adjusting the reward function based on prediction accuracy

This integration allows the defense policy to act proactively rather than merely reactively, anticipating attacks before they materialize.