# Mathematical Foundations

This document provides a detailed examination of the mathematical concepts underpinning the RL-IoT Defense System.

## Reinforcement Learning Framework

### Markov Decision Process

The system is modeled as a Markov Decision Process (MDP), formalized as a tuple $(S, A, P, R, \gamma)$ where:

- $S$ is the state space representing all possible configurations of the IoT network
- $A$ is the action space of all possible defensive countermeasures
- $P: S \times A \times S \rightarrow [0, 1]$ is the state transition probability function
- $R: S \times A \times S \rightarrow \mathbb{R}$ is the reward function
- $\gamma \in [0, 1]$ is the discount factor balancing immediate vs. future rewards

### Value Functions

The state-value function $V^\pi(s)$ represents the expected return starting from state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \Big| S_0 = s \right]$$

The action-value function $Q^\pi(s, a)$ represents the expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \Big| S_0 = s, A_0 = a \right]$$

### Bellman Equations

The Bellman expectation equation for $V^\pi$ is:

$$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

The Bellman expectation equation for $Q^\pi$ is:

$$Q^\pi(s, a) = \sum_{s' \in S} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s', a') \right]$$

The Bellman optimality equation for $Q^*$ is:

$$Q^*(s, a) = \sum_{s' \in S} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a' \in A} Q^*(s', a') \right]$$

## Deep Q-Network (DQN)

### Function Approximation

DQN uses a neural network to approximate the Q-function:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

Where $\theta$ represents the parameters of the neural network.

### Loss Function

The loss function used to train the DQN is the mean squared error between the predicted Q-values and the target Q-values:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]$$

Where:
- $D$ is the replay buffer containing transition tuples $(s, a, r, s')$
- $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ is the target Q-value
- $\theta^-$ are the parameters of the target network

### Gradient Descent Update

The parameters are updated using gradient descent:

$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

Where $\alpha$ is the learning rate and:

$$\nabla_\theta L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

## LSTM Network

### Sequence Modeling

The LSTM models the attack sequence as a time series:

$$X = (x_1, x_2, ..., x_T)$$

Where each $x_t$ represents an attack event at time $t$.

### LSTM Cell Equations

The LSTM cell updates its internal state using the following equations:

**Input Gate**:
$$i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$$

**Forget Gate**:
$$f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$$

**Cell State**:
$$\tilde{c}_t = \tanh(W_{ic} x_t + b_{ic} + W_{hc} h_{t-1} + b_{hc})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output Gate**:
$$o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$$
$$h_t = o_t \odot \tanh(c_t)$$

Where:
- $\sigma$ is the sigmoid function
- $\odot$ represents element-wise multiplication
- $W$ and $b$ are weight matrices and bias vectors
- $h_t$ is the hidden state at time $t$
- $c_t$ is the cell state at time $t$

### Bidirectional LSTM

The bidirectional LSTM combines forward and backward passes:

$$\overrightarrow{h}_t = \text{LSTM}_{\text{forward}}(x_t, \overrightarrow{h}_{t-1}, \overrightarrow{c}_{t-1})$$
$$\overleftarrow{h}_t = \text{LSTM}_{\text{backward}}(x_t, \overleftarrow{h}_{t+1}, \overleftarrow{c}_{t+1})$$
$$h_t = [\overrightarrow{h}_t, \overleftarrow{h}_t]$$

### Softmax Output Layer

The output probability distribution for next attack prediction:

$$p(y_t = j | x_{1:t}) = \frac{\exp(W_j^T h_t + b_j)}{\sum_{k=1}^C \exp(W_k^T h_t + b_k)}$$

Where:
- $C$ is the number of possible attack classes
- $W_j$ and $b_j$ are the weights and bias for class $j$

## Graph-Based Attack Modeling

### Attack Graph

The attack graph is represented as a directed graph $G = (V, E)$ where:
- $V$ is the set of nodes (devices)
- $E$ is the set of edges (connections)

Each node $v \in V$ has attributes:
- $v_{\text{critical}} \in \{0, 1\}$ indicates if the node is critical
- $v_{\text{compromised}} \in \{0, 1\}$ indicates if the node is compromised
- $v_{\text{vulnerability}} \in [0, 1]$ represents the vulnerability level

### Path Selection

The probability of an attacker selecting a path $P = (v_1, v_2, ..., v_k)$ is:

$$p(P) = \frac{\exp(-\beta \cdot c(P))}{\sum_{P' \in \mathcal{P}} \exp(-\beta \cdot c(P'))}$$

Where:
- $c(P)$ is the cost of path $P$, calculated as $c(P) = \sum_{i=1}^{k-1} c(v_i, v_{i+1})$
- $c(v_i, v_{i+1})$ is the cost of moving from node $v_i$ to $v_{i+1}$
- $\beta" is a parameter controlling the rationality of the attacker
- $\mathcal{P}$ is the set of all possible paths

### Adaptive Attacker Learning

The adaptive attacker updates node values based on defense actions:

$$v_{\text{value}}(t+1) = v_{\text{value}}(t) \cdot (1 - \alpha \cdot \mathbb{I}[v \in D_t])$$

Where:
- $v_{\text{value}}(t)$ is the value of node $v$ at time $t$
- $\alpha$ is the learning rate
- $\mathbb{I}[v \in D_t]$ is an indicator function equal to 1 if node $v$ was defended at time $t$
- $D_t$ is the set of nodes defended at time $t$

## Reward Function Engineering

### Component-Based Reward

The reward function is decomposed into:

$$R(s, a, s') = w_1 R_{\text{defense}}(s, a, s') + w_2 R_{\text{compromise}}(s') + w_3 R_{\text{critical}}(s') + w_4 R_{\text{health}}(s')$$

Where:
- $R_{\text{defense}}(s, a, s')$ rewards successful defenses
- $R_{\text{compromise}}(s')$ penalizes compromised nodes
- $R_{\text{critical}}(s')$ heavily penalizes critical compromised nodes
- $R_{\text{health}}(s')$ rewards overall network health
- $w_i$ are weight parameters

### Defense Success Reward

$$R_{\text{defense}}(s, a) = \alpha \cdot \mathbb{I}[\text{attack\_prevented}](s, a) \cdot \text{value}(\text{target})$$

Where:
- $\mathbb{I}[\text{attack\_prevented}](s, a)$ is 1 if the action prevented an attack
- $\text{value}(\text{target})$ is the value of the targeted node

## Network Health Penalty

$$R_{\text{health}}(s) = -\beta \cdot \sum_{i=1}^{N} \mathbb{I}[\text{node\_compromised}](i) \cdot \text{criticality}(i)$$

Where:
- $N$ is the total number of nodes
- $\mathbb{I}[\text{node\_compromised}](i)$ is 1 if node $i$ is compromised
- $\text{criticality}(i)$ represents the importance of node $i$

## Action Cost

$$R_{\text{cost}}(a) = -\gamma \cdot \text{cost}(a)$$

Where:
- $\text{cost}(a)$ is the computational/resource cost of action $a$
- $\gamma$ weights the importance of efficiency

## Total Reward Function

$$R(s, a, s') = R_{\text{defense}}(s, a) + R_{\text{health}}(s') + R_{\text{cost}}(a)$$

## State Representation

### Feature Encoding

The state $s_t$ at time $t$ is represented as a feature vector combining:

$$s_t = [s_{\text{network}}, s_{\text{history}}, s_{\text{prediction}}]$$

Where:
- $s_{\text{network}}$ encodes the current network status
- $s_{\text{history}}$ encodes recent attack history
- $s_{\text{prediction}}$ encodes LSTM predictions

### Network Status Encoding

$$s_{\text{network}} = [v_1^{\text{status}}, v_2^{\text{status}}, ..., v_n^{\text{status}}]$$

Where $v_i^{\text{status}}$ is a multi-dimensional encoding of node $i$'s status.

### History Encoding

$$s_{\text{history}} = [a_{t-H}, a_{t-H+1}, ..., a_{t-1}]$$

Where $a_{t-i}$ is the attack at time $t-i$ and $H$ is the history length.

### Prediction Encoding

$$s_{\text{prediction}} = [p(a_{t+1}=1), p(a_{t+1}=2), ..., p(a_{t+1}=C)]$$

Where $p(a_{t+1}=j)$ is the LSTM-predicted probability of attack $j$ at the next step.

## Experience Replay

### Transition Storage

The replay buffer $D$ stores transitions $(s_t, a_t, r_t, s_{t+1})$.

### Prioritized Sampling

Transitions are sampled according to priority:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

Where:
- $p_i = |\delta_i| + \epsilon$ is the priority
- $\delta_i = r_i + \gamma \max_a Q(s_{i+1}, a; \theta^-) - Q(s_i, a_i; \theta)$ is the TD error
- $\alpha$ controls the sampling bias
- $\epsilon$ is a small constant ensuring non-zero probability

### Importance Sampling Correction

To correct for sampling bias, updates are weighted:

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$

Where $\beta$ is annealed from an initial value to 1 during training.

These mathematical formulations provide the foundation for the learning and decision-making processes in the RL-IoT Defense System.