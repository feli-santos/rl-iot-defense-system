# IoT Environment Simulation

## Environment Design

The IoT environment is modeled as a Markov Decision Process (MDP) and implemented as a custom Gymnasium environment in `src/environment.py`. It simulates an IoT network where an RL agent learns to apply defensive actions against potential attacks.

-   **State Space ($S$)**: Represents the current security status of the network, historical states, and past actions.
-   **Action Space ($A$)**: A discrete set of defensive measures the agent can take.
-   **Transition Function ($P(s'|s,a)$)**: Defines how the state changes when the agent takes an action $a$ in state $s$, and how simulated attacks progress.
-   **Reward Function ($R(s,a,s')$)**: Provides feedback to the agent based on the outcomes of its actions and the security status of the network.

## Environment Implementation (`src/environment.py`)

### State Representation
The observation space is a `gymnasium.spaces.Dict`:
```python
self.observation_space = spaces.Dict({
    'current_state': spaces.Box(
        low=0, high=1, 
        shape=(config.ENVIRONMENT_NUM_STATES,), # Represents features of current device states
        dtype=np.float32
    ),
    'state_history': spaces.Box(
        low=0, high=1,
        shape=(config.ENVIRONMENT_HISTORY_LENGTH, config.ENVIRONMENT_NUM_STATES),
        dtype=np.float32
    ),
    'action_history': spaces.Box(
        low=0, high=config.ENVIRONMENT_NUM_ACTIONS-1, # Actions are discrete, encoded as floats here
        shape=(config.ENVIRONMENT_HISTORY_LENGTH,),
        dtype=np.float32
    )
})
```
-   `current_state`: Features describing the current status of `config.ENVIRONMENT_NUM_DEVICES` (e.g., vulnerability levels, compromise status). `ENVIRONMENT_NUM_STATES` defines the total number of features.
-   `state_history`: A rolling window of the past `config.ENVIRONMENT_HISTORY_LENGTH` states.
-   `action_history`: A rolling window of the past `config.ENVIRONMENT_HISTORY_LENGTH` actions taken by the agent.

### Action Space
A discrete action space with `config.ENVIRONMENT_NUM_ACTIONS` possible defensive actions. For example:
```python
self.action_space = spaces.Discrete(config.ENVIRONMENT_NUM_ACTIONS)
# Example: 0: Monitor, 1: Block, 2: Quarantine, 3: Allow/No Action
```

### Attack Simulation
The environment simulates attacks, potentially based on predefined patterns, probabilities, or an adaptive attacker model if implemented within the `step` method. The original `docs/environment.md` mentioned an `AdaptiveAttacker` and graph-based attacks. If this is part of `IoTEnv.step()`, its logic determines how the environment evolves apart from the agent's actions.

### Reward Function
The reward function is crucial for guiding the agent's learning. It's designed to:
1.  Reward actions that successfully mitigate or prevent attacks.
2.  Penalize states where devices are compromised.
3.  Potentially penalize the cost of actions.
4.  Encourage maintaining overall network security.

The `config.yml` includes parameters like `reward: injection_threshold` and `goal_reward` which influence the reward calculation. The specific logic is within the `IoTEnv.step()` method.

A conceptual reward function might be:
$$R_t = (\text{reward for defense}) - (\text{penalty for compromises}) - (\text{cost of action})$$

### Episode Termination
An episode can terminate under several conditions:
1.  A critical security breach occurs (e.g., specific target compromised).
2.  The network reaches a "lost" state.
3.  A maximum number of steps per episode is reached.

## Environment-Agent Interaction

The interaction loop is standard for RL:
1.  Agent observes current state $s_t$ from the environment.
2.  Agent selects an action $a_t$ based on its policy $\pi(a_t|s_t)$.
3.  The environment executes $a_t$, simulates attacks, and transitions to a new state $s_{t+1}$.
4.  The environment returns the reward $r_t$ and a `done` signal to the agent.
5.  This process repeats until the episode terminates.

## Environment Configuration (`config.yml`)
Key parameters influencing the environment are found in `config.yml`:
```yaml
environment:
  num_devices: 12       # Number of simulated IoT devices
  num_actions: 4        # Number of distinct defensive actions
  num_states: 12        # Dimensionality of the 'current_state' vector
  history_length: 5     # Length of state and action histories

reward:
  injection_threshold: 0.25 # Example parameter for reward logic
  goal_reward: -100         # Example parameter for reward logic
```
These parameters allow for simulating different network sizes, complexities, and defense capabilities. The `create_training_environment` function in `src/training.py` wraps this base environment with `Monitor` (for logging episode statistics) and `DummyVecEnv` (for compatibility with Stable Baselines3).