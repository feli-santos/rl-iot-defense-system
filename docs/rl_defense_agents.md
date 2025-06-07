# Reinforcement Learning Defense Agents

The defense policy is implemented using various Reinforcement Learning (RL) algorithms provided by the Stable Baselines3 library. This allows for flexibility in choosing the best agent for the IoT defense task and facilitates comparative analysis. The currently supported algorithms are Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C).

## Common Architecture

All RL agents utilize the `MultiInputPolicy` from Stable Baselines3 due to the `Dict` observation space of the IoT environment. The general network architecture for the policy and/or value functions consists of:
1.  **Feature Extraction**: Separate small MLPs for each component of the `Dict` observation space (`current_state`, `state_history`, `action_history`).
2.  **Concatenation**: The extracted features are concatenated.
3.  **Shared/Separate Layers**: The concatenated features are then passed through one or more hidden layers (defined by `config.NETWORK_HIDDEN_LAYERS`, e.g., `[128, 64]`) before producing the final output (Q-values for DQN, policy and value for PPO/A2C).

## 1. Deep Q-Network (DQN)

### Approach
DQN is a value-based, off-policy algorithm that learns an action-value function (Q-function) to estimate the expected return for taking an action in a given state.

### Key Features
-   **Experience Replay**: Stores transitions in a replay buffer and samples mini-batches to break correlations and improve learning stability.
-   **Target Network**: Uses a separate target Q-network (with parameters $\theta^-$) that is periodically updated with the online Q-network's parameters ($\theta$) to provide stable targets for the Bellman updates.
-   **Epsilon-Greedy Exploration**: Balances exploration and exploitation by choosing a random action with probability $\epsilon$ (annealed over time) and the greedy action (with respect to current Q-values) with probability $1-\epsilon$.

### Mathematical Foundations (DQN)
The core idea is to learn a Q-function $Q(s, a; \theta)$ that approximates the optimal action-value function $Q^*(s, a)$.
The Bellman optimality equation for $Q^*$ is:
$$Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$
DQN minimizes the Mean Squared Error (MSE) loss:
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]$$
Where the target $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$, and $D$ is the replay buffer.

### Implementation Details
Instantiated via `AlgorithmFactory` and configured in `config.yml` under the `dqn` section.
```python
# Example from dqn_algorithm.py
model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=self.config.DQN_LEARNING_RATE,
    buffer_size=self.config.DQN_BUFFER_SIZE,
    # ... other DQN specific parameters ...
    policy_kwargs={"net_arch": self.config.NETWORK_HIDDEN_LAYERS}
)
```

## 2. Proximal Policy Optimization (PPO)

### Approach
PPO is a policy-based, on-policy algorithm that aims to make the largest possible improvement step on a policy without stepping too far and causing performance collapse. It's known for its stability and good performance across a wide range of tasks.

### Key Features
-   **Clipped Surrogate Objective**: PPO optimizes a surrogate objective function that includes a clipping mechanism to limit the change in the policy at each update step.
-   **Actor-Critic Architecture**: Typically uses an actor-critic setup where an actor network learns the policy and a critic network learns a value function to estimate expected returns.
-   **Generalized Advantage Estimation (GAE)**: Often used to reduce variance in advantage estimates.

### Mathematical Foundations (PPO)
PPO's objective function (simplified for the clipped version):
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
Where:
-   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.
-   $\hat{A}_t$ is an estimator of the advantage function at timestep $t$.
-   $\epsilon$ is a hyperparameter (e.g., 0.2) defining the clipping range.
The full objective also includes terms for value function loss and an entropy bonus to encourage exploration.

### Implementation Details
Instantiated via `AlgorithmFactory` and configured in `config.yml` under the `ppo` section.
```python
# Example from ppo_algorithm.py
model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=self.config.PPO_LEARNING_RATE,
    n_steps=self.config.PPO_N_STEPS,
    # ... other PPO specific parameters ...
    policy_kwargs={
        "net_arch": dict(pi=self.config.NETWORK_HIDDEN_LAYERS, vf=self.config.NETWORK_HIDDEN_LAYERS)
    }
)
```

## 3. Advantage Actor-Critic (A2C)

### Approach
A2C is a synchronous, deterministic version of Asynchronous Advantage Actor-Critic (A3C). It's an on-policy, actor-critic algorithm that updates the policy (actor) and value function (critic) using a batch of experiences collected from parallel environments (or a single environment over multiple steps).

### Key Features
-   **Actor-Critic**: Explicitly maintains and updates a policy (actor) and a value function (critic).
-   **Advantage Function**: Uses the advantage function $A(s,a) = Q(s,a) - V(s)$ to reduce variance in policy gradient updates. The critic estimates $V(s)$.
-   **Synchronous Updates**: Unlike A3C, A2C waits for all actors to finish their segment of experience before updating, which can be more efficient on GPUs.

### Mathematical Foundations (A2C)
The actor (policy) is updated in the direction of $\nabla_\theta \log \pi_\theta(a_t|s_t) \hat{A}_t$.
The critic (value function) is updated to minimize the MSE between its predictions $V(s_t)$ and the observed returns.
The loss function typically combines policy loss, value loss, and an entropy bonus:
$$L(\theta) = L_{policy}(\theta) + c_1 L_{value}(\theta) - c_2 H(\pi_\theta(\cdot|s_t))$$
Where $H$ is the entropy of the policy.

### Implementation Details
Instantiated via `AlgorithmFactory` and configured in `config.yml` under the `a2c` section.
```python
# Example from a2c_algorithm.py
model = A2C(
    "MultiInputPolicy",
    env,
    learning_rate=self.config.A2C_LEARNING_RATE, # Assuming A2C_LEARNING_RATE exists
    n_steps=self.config.A2C_N_STEPS, # Assuming A2C_N_STEPS exists
    # ... other A2C specific parameters ...
    policy_kwargs={
        "net_arch": dict(pi=self.config.NETWORK_HIDDEN_LAYERS, vf=self.config.NETWORK_HIDDEN_LAYERS)
    }
)
```

## Hyperparameters and Training

Key hyperparameters for each agent (e.g., learning rate, batch size, discount factor $\gamma$, network architecture) are defined in `config.yml`. The training process for each agent is managed by the `train_single_algorithm` function in `training.py`, which leverages the `learn()` method from Stable Baselines3. Progress is tracked using MLflow via the `TrainingManager` and custom callbacks.

## Inference Process

During evaluation or deployment, the trained models select actions deterministically (or stochastically if exploration is desired) based on the learned policy:
```python
# General inference example
action, _ = model.predict(observation, deterministic=True)
```
This is handled within the evaluation loops in the benchmarking framework or custom evaluation scripts.