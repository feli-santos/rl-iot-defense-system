# IoT Environment Simulation

## Environment Design

The IoT environment is modeled as a Markov Decision Process (MDP) and implemented as a custom Gymnasium environment in `src/environment.py` (class `IoTEnv`). It simulates an IoT network where an RL agent learns to apply defensive actions against potential attacks, with attack likelihood and characteristics informed by a real-data LSTM predictor.

-   **State Space ($S$)**: Represents the current security status of the network, historical states, past actions, and real-time attack predictions.
-   **Action Space ($A$)**: A discrete set of defensive measures the agent can take.
-   **Transition Function ($P(s'|s,a)$)**: Defines how the state changes when the agent takes an action $a$ in state $s$. Attack occurrences and their effects are simulated within this function.
-   **Reward Function ($R(s,a,s')$)**: Provides feedback to the agent based on the outcomes of its actions, the accuracy of predictions versus actual occurrences, and the security status of the network.

## Environment Implementation (`src/environment.py`)

### State Representation
The observation space is a `gymnasium.spaces.Dict`:
```python
self.observation_space = spaces.Dict({
    'current_state': spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(22,), # Example: 22 features representing current network telemetry
        dtype=np.float32
    ),
    'state_history': spaces.Box(
        low=-np.inf, high=np.inf, 
        shape=(config.ENVIRONMENT_STATE_HISTORY_LENGTH, 22), # History of telemetry
        dtype=np.float32
    ),
    'action_history': spaces.Box(
        low=0, high=3, # Max action index (0,1,2,3 for 4 actions)
        shape=(config.ENVIRONMENT_ACTION_HISTORY_LENGTH,), 
        dtype=np.int32
    ),
    'attack_prediction': spaces.Box(
        low=0.0, high=1.0, 
        shape=(6,), # Example: 6 features from LSTM predictor (risk, confidence, is_attack, severity, category, max_prob)
        dtype=np.float32
    )
})
```
-   `current_state`: Features describing the current network status (e.g., traffic statistics, protocol information). The number of features (e.g., 22) is based on the chosen representation for the RL agent.
-   `state_history`: A rolling window of the past `config.ENVIRONMENT_STATE_HISTORY_LENGTH` states.
-   `action_history`: A rolling window of the past `config.ENVIRONMENT_ACTION_HISTORY_LENGTH` actions taken by the agent.
-   `attack_prediction`: A vector summarizing the output from the `EnhancedAttackPredictor`, providing insights about potential threats.

### Action Space
A discrete action space with 4 possible defensive actions:
```python
self.action_space = spaces.Discrete(4)
# Actions: 0: No action (monitor), 1: Rate limiting, 2: Block suspicious IPs, 3: Shutdown affected services
```

### Attack Simulation
The environment simulates attacks based on `config.ENVIRONMENT_ATTACK_PROBABILITY`. When an attack occurs:
-   `current_attack_active` is set to `True`.
-   An `attack_type` (e.g., 'ddos', 'botnet') is chosen.
-   The `current_network_state` is modified to reflect the attack's impact.
-   `attack_severity` is set.

The `EnhancedAttackPredictor` (`src/models/predictor_interface.py`) is used internally by the environment to get predictions based on the `state_history`. These predictions are then included in the observation provided to the RL agent.

### Reward Function
The reward function in `_calculate_reward` (or `_calculate_fallback_reward`) is crucial. It considers:
1.  **Prediction Accuracy**:
    -   Correct attack prediction (True Positive): Positive reward, scaled by risk score.
    -   Correct benign prediction (True Negative): Positive reward.
    -   Missed attack (False Negative): Negative reward, scaled by risk score.
    -   False alarm (False Positive): Negative reward.
2.  **Defense Action Effectiveness**:
    -   If an attack occurred: Bonus for effective actions, penalty for ineffective ones.
    -   If no attack occurred: Penalty for unnecessarily aggressive actions.
The specific reward values and scaling factors are defined within the `IoTEnv` class, potentially influenced by `config.ENVIRONMENT_REWARD_SCALE`.

### Episode Termination
An episode can terminate if:
1.  `current_step` reaches `config.ENVIRONMENT_MAX_STEPS` (truncated).
2.  A critical condition is met, e.g., a severe attack occurs and the agent takes no defensive action (terminated).

## Environment-Agent Interaction

The interaction loop is standard for RL:
1.  Agent observes current state $s_t$ (a `Dict` including `attack_prediction`) from the environment.
2.  Agent selects an action $a_t$ based on its policy $\pi(a_t|s_t)$.
3.  The environment executes $a_t$, simulates network evolution and potential attacks (updating `current_network_state`), gets a new prediction via `EnhancedAttackPredictor`, and transitions to a new state $s_{t+1}$.
4.  The environment returns the reward $r_t$ and `terminated`/`truncated` signals to the agent.
5.  This process repeats.

## Environment Configuration (`config.yml`)
Key parameters influencing the environment are found in `config.yml` under the `environment` section:
```yaml
environment:
  max_steps: 1000
  attack_probability: 0.3
  state_history_length: 10   # Corresponds to ENVIRONMENT_STATE_HISTORY_LENGTH
  action_history_length: 5  # Corresponds to ENVIRONMENT_ACTION_HISTORY_LENGTH
  reward_scale: 1.0         # Corresponds to ENVIRONMENT_REWARD_SCALE
  # The number of features for current_state (e.g., 22) and attack_prediction (e.g., 6)
  # are implicitly defined by the IoTEnv implementation.
```
The `AlgorithmFactory` creates the `IoTEnv` instance using an `EnvironmentConfig` dataclass, which sources its values from the main `config.yml`.