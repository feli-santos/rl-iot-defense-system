# Mathematical Foundations

This document provides a detailed examination of the mathematical concepts underpinning the RL-IoT Defense System, focusing on the Reinforcement Learning framework and the specific algorithms used.

## Reinforcement Learning Framework

### Markov Decision Process (MDP)
The system is modeled as an MDP, defined by $(S, A, P, R, \gamma)$:
-   $S$: State space (observations from the IoT environment, a `Dict` space).
-   $A$: Action space (discrete defensive actions, `spaces.Discrete(4)`).
-   $P(s'|s,a)$: State transition probability $P(S_{t+1}=s' | S_t=s, A_t=a)$.
-   $R(s,a,s')$: Reward function $R_t = R(S_t=s, A_t=a, S_{t+1}=s')$.
-   $\gamma \in [0, 1]$: Discount factor (e.g., `gamma` hyperparameter for RL algorithms).

### Value Functions
-   **State-Value Function $V^\pi(s)$**: Expected return from state $s$ following policy $\pi$.
    $$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s \right]$$
-   **Action-Value Function $Q^\pi(s,a)$**: Expected return from state $s$, taking action $a$, then following policy $\pi$.
    $$Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s, A_t = a \right]$$

### Bellman Equations
The Bellman optimality equation for $Q^*(s,a)$ (the optimal action-value function) is central:
$$Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} \left[ R(s,a,s') + \gamma \max_{a' \in A} Q^*(s', a') \right]$$

## 1. Deep Q-Network (DQN)

### Function Approximation
DQN uses a neural network $Q(s, a; \theta)$ with parameters $\theta$ to approximate $Q^*(s, a)$.

### Loss Function
The loss is typically the Mean Squared Bellman Error (MSBE):
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( (r + \gamma \max_{a'} Q(s', a'; \theta^-)) - Q(s, a; \theta) \right)^2 \right]$$
-   $D$: Replay buffer.
-   $\theta^-$: Parameters of a separate, periodically updated target network, enhancing stability.

### Gradient Descent Update
Parameters $\theta$ are updated via gradient descent on $L(\theta)$.

## 2. Proximal Policy Optimization (PPO)

PPO is a policy gradient method that aims for stable and reliable policy updates. It's an actor-critic approach.

### Surrogate Objective Function
PPO optimizes a "surrogate" objective. The clipped version is commonly used:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
-   $\theta$: Policy parameters.
-   $\hat{\mathbb{E}}_t$: Empirical average over a batch of timesteps.
-   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: Probability ratio of the current policy to the old policy (that collected the data).
-   $\hat{A}_t$: An estimator of the advantage function at timestep $t$. $A(s,a) = Q(s,a) - V(s)$.
-   $\epsilon$: A small hyperparameter (e.g., 0.2) that clips the probability ratio $r_t(\theta)$, constraining how much the new policy can diverge from the old one.

### Full Objective Function
The actual objective function in PPO often includes:
1.  The clipped surrogate policy objective $L^{CLIP}$.
2.  A value function error term $L^{VF}(\theta) = (V_\theta(s_t) - V_t^{target})^2$, where $V_\theta(s_t)$ is the output of the critic.
3.  An entropy bonus $S[\pi_\theta](s_t)$ to encourage exploration.
$$L(\theta) = \hat{\mathbb{E}}_t [L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)]$$
-   $c_1, c_2$: Coefficients.

### Generalized Advantage Estimation (GAE)
For $\hat{A}_t$, PPO often uses GAE:
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error. $\lambda \in [0,1]$ controls the bias-variance trade-off.

## 3. Advantage Actor-Critic (A2C)

A2C is a synchronous version of the Asynchronous Advantage Actor-Critic (A3C). It's an on-policy actor-critic algorithm.

### Actor (Policy) Update
The actor learns the policy $\pi(a|s; \theta_\pi)$. Its objective is to maximize expected rewards. The gradient is typically:
$$\nabla_{\theta_\pi} J(\theta_\pi) = \hat{\mathbb{E}}_t [\nabla_{\theta_\pi} \log \pi(A_t|S_t; \theta_\pi) \hat{A}_t]$$
-   $\hat{A}_t = R_t - V(S_t; \theta_v)$ is the advantage estimate, where $R_t$ is an estimate of the return (e.g., n-step return) and $V(S_t; \theta_v)$ is the value estimated by the critic.

### Critic (Value Function) Update
The critic learns the value function $V(s; \theta_v)$. It's updated by minimizing a loss function, often MSE, between $V(S_t; \theta_v)$ and the target returns (e.g., n-step returns):
$$L(\theta_v) = \hat{\mathbb{E}}_t [(R_t - V(S_t; \theta_v))^2]$$

### Combined Loss
Similar to PPO, an entropy bonus for the policy is often added to encourage exploration. The overall update involves gradients from both policy loss and value loss.
$$L(\theta_\pi, \theta_v) = \hat{\mathbb{E}}_t [-\log \pi(A_t|S_t; \theta_\pi) \hat{A}_t + c_1 (R_t - V(S_t; \theta_v))^2 - c_2 H(\pi(\cdot|S_t; \theta_\pi))]$$
(Note: The sign for the policy term depends on whether maximizing reward or minimizing loss).

## LSTM Network (for Attack Prediction - `RealDataLSTMPredictor`)
The LSTM network (`src/models/lstm_attack_predictor.py`) processes sequences of network features from the CICIoT2023 dataset to predict attack probabilities.
-   **Input**: Sequences of feature vectors $x_1, x_2, ..., x_T$, where each $x_t \in \mathbb{R}^{num\_features}$.
-   **LSTM Cells**: Standard LSTM cells (as described in `docs/attack_prediction.md`) process the input sequence.
-   **Output**: The final LSTM hidden state (or a combination of hidden states) is passed through dense layers.
-   **Softmax Output Layer**: Produces a probability distribution $P(y=j|X)$ over $K$ attack classes (including benign), where $X = (x_1, ..., x_T)$.
    $$P(y=j|X) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$
    The model is trained using Cross-Entropy Loss.

## State Representation (for RL Agents)
The state $s_t$ provided to the RL agent is a dictionary from `IoTEnv`:
$$s_t = \{ \text{'current_state'}, \text{'state_history'}, \text{'action_history'}, \text{'attack_prediction'} \}$$
-   `current_state`: $\mathbf{cs}_t \in \mathbb{R}^{N_{cs}}$ (e.g., $N_{cs}=22$ features)
-   `state_history`: $\mathbf{SH}_t \in \mathbb{R}^{L_s \times N_{cs}}$ (history of $L_s$ past states)
-   `action_history`: $\mathbf{AH}_t \in \mathbb{Z}^{L_a}$ (history of $L_a$ past discrete actions)
-   `attack_prediction`: $\mathbf{ap}_t \in \mathbb{R}^{N_{ap}}$ (e.g., $N_{ap}=6$ features from LSTM predictor)

The `MultiInputPolicy` in Stable Baselines3 processes these inputs, typically with separate small networks (feature extractors) for each key. The outputs of these extractors are then concatenated and fed into further shared or policy/value-specific layers, as defined by `net_arch` in the configuration.