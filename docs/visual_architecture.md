# Visual Architecture and Workflows

This document provides a more detailed visual representation of the RL-IoT Defense System's architecture, data flow, and operational workflows.

## 1. Overall System Architecture

This diagram illustrates the main components and their high-level interactions.

```
+---------------------------------------------------------------------------------+
| Experimentation & Management                                                    |
| +---------------------------+   +-------------------------------------------+ |
| | training.py (Main Script) |   | Benchmarking Framework                    | |
| | (CLI Interface)           |   |  - BenchmarkRunner                        | |
| +-----------+---------------+   |  - MetricsCollector                       | |
|             | (Uses)            |  - BenchmarkAnalyzer                      | |
|             v                   +-----------------------+-------------------+ |
| +---------------------------+                           | (Results/Plots)     |
| | config.yml (Configuration)|                           |                     |
| +---------------------------+                           v                     |
|             | (Used by)                       +---------------------+         |
|             v                                 | TrainingManager     |         |
| +------------------------------------------+  | (MLflow Integration)|         |
| | Intelligence Core                        |<--+ (Logs Metrics,    |         |
| |                                          |  |  Models, Params)    |         |
| | +--------------------------------------+ |  +---------------------+         |
| | | LSTM Attack Predictor (Optional)     | |                                  |
| | +------------------+-------------------+ |                                  |
| |                    | (Predictions)       |                                  |
| |                    v                     |                                  |
| | +--------------------------------------+ |                                  |
| | | RL Defense Agents (DQN, PPO, A2C)  | |                                  |
| | | (via AlgorithmFactory)             | |                                  |
| | | (Uses Stable Baselines3            | |                                  |
| | |  MultiInputPolicy)                 | |                                  |
| | +------------------+-------------------+ |                                  |
| +--------------------|---------------------+                                  |
|                      | (Action)                                               |
|                      v (Observation, Reward)                                  |
| +----------------------------------------------------------------------------+ |
| | IoT Environment (Gymnasium)                                                | |
| |  - IoT Devices & Network Simulation                                        | |
| |  - Attack Simulation Logic                                                 | |
| |  - State & Reward Calculation                                              | |
| +----------------------------------------------------------------------------+ |
+---------------------------------------------------------------------------------+
```
*   **IoT Environment**: Simulates the network, attacks, and calculates state/rewards.
*   **Intelligence Core**: Contains the LSTM for attack prediction and the RL agents for defense.
*   **Experimentation & Management**: Handles configuration, training orchestration, benchmarking, and MLflow logging.

## 2. Detailed Data and Control Flow

This diagram expands on the interactions, particularly during training and benchmarking.

```
                               +---------------------+
                               |     config.yml      |
                               +----------+----------+
                                          | (Loads)
                                          v
+---------------------------------------+------------------------------------------+
| training.py (Main Script with CLI)                                               |
|                                       +-----------------+                        |
|                                       | TrainingManager |----(MLflow Logging)----> MLflow UI & Server
|                                       +--------+--------+                        |
|                                                ^                                 |
|  (Manages Artifacts, Logs Metrics & Params)    |                                 |
|                                                |                                 |
|  +----------------------+   (Trains)   +---------------------+                  |
|  | LSTM Attack          |<--------------| LSTM Training Logic |                  |
|  | Predictor (models/)  |------------->| (attack_predictor.py)|                  |
|  +----------------------+ (Predictions)|                     |                  |
|          (Optional)                    +---------------------+                  |
|             |                                  ^                                 |
|             | (Input)                          |(Logs Metrics, Saves Model)      |
|             v                                  |                                 |
|  +----------------------+              +-------+---------+                       |
|  | AlgorithmFactory     |--(Creates)-->| RL Agent (DQN,  |                       |
|  | (algorithms/)        |              | PPO, or A2C)    |                       |
|  +----------------------+              +-------+---------+                       |
|                                                |         ^                       |
|                               (Action)         v         |(Observation, Reward)  |
|  +------------------------------------------------------+-----------+           |
|  | IoT Environment (environment.py)                                 |           |
|  | - Observation Space (Dict: current_state, state_history, etc.)   |           |
|  | - Action Space (Discrete)                                        |           |
|  | - Reward Logic                                                   |           |
|  +------------------------------------------------------------------+           |
|                                                                                  |
|  IF BENCHMARK MODE:                                                              |
|  +--------------------------+  (Orchestrates) +--------------------------------+ |
|  | BenchmarkRunner          |---------------->| Multiple RL Agent Training Runs| |
|  | (benchmarking/)          |                 | (DQN, PPO, A2C x N times)      | |
|  +----------+---------------+                 +-----------------+--------------+ |
|             | (Collects Metrics)                                | (Metrics)      |
|             v                                                   v                |
|  +--------------------------+                      +--------------------------+ |
|  | MetricsCollector         |                      | Individual Run Metrics   | |
|  | (benchmarking/)          |                      +--------------------------+ |
|  +----------+---------------+                                                   |
|             | (Analyzes)                                                       |
|             v                                                                  |
|  +--------------------------+  (Generates)  +---------------------------------+ |
|  | BenchmarkAnalyzer        |-------------->| Reports & Plots                 | |
|  | (benchmarking/)          |               | (./benchmark_analysis/)         | |
|  +--------------------------+               +---------------------------------+ |
+----------------------------------------------------------------------------------+
```

## 3. RL Agent - Environment Interaction Loop

This visualizes the core cycle for an RL agent.

```
+---------------------------------+      +---------------------------------+
|        RL Defense Agent         |      |        IoT Environment          |
| (e.g., PPO with MultiInputPolicy)|      |        (Gymnasium)            |
+---------------------------------+      +---------------------------------+
              |                                        ^
(1. Observe)  | current_state, state_history,          | (4. Return)
              | action_history, reward, done           | reward, next_state, done
              v                                        |
+---------------------------------+      +---------------------------------+
| Policy Network (π(a|s))         |----->| Action Selection (a_t)          |
| Value Network (V(s)) (if appl.) |      | (e.g., argmax Q or sample policy)|
+---------------------------------+      +---------------------------------+
              ^                                        |
              |                                        | (2. Execute Action a_t)
              | (3. Update Policy/Value based on experience) |
              | (e.g., using collected trajectory/batch)     |
              +----------------------------------------------+
```
1.  **Observe**: Agent receives the current state (a `Dict` observation) and reward from the environment.
2.  **Act**: Agent selects an action based on its current policy.
3.  **Update (Training)**: Agent's internal model (policy/value function) is updated based on the experience (state, action, reward, next state).
4.  **Environment Step**: Environment processes the action, simulates attacks, transitions to a new state, and calculates the reward.

## 4. Observation Space Structure (`Dict`)

Visual breakdown of the `Dict` observation space fed to the `MultiInputPolicy`.

```
Observation (Dict)
├── 'current_state': Box(shape=(NUM_STATES,))
│   └─ [feature_1, feature_2, ..., feature_NUM_STATES]  (e.g., device status, vulnerability levels)
│
├── 'state_history': Box(shape=(HISTORY_LENGTH, NUM_STATES))
│   └─ [
│        [hist_state_t-H_1, ..., hist_state_t-H_NUM_STATES],  // Oldest in history
│        ...
│        [hist_state_t-1_1, ..., hist_state_t-1_NUM_STATES]   // Most recent in history
│      ]
│
└── 'action_history': Box(shape=(HISTORY_LENGTH,))
    └─ [action_t-H, ..., action_t-1]  (Sequence of past discrete actions taken)

---------------------------------------------------------------------> To MultiInputPolicy
                                                                         (Feature Extractors for each key, then Concatenation)
```
-   `NUM_STATES`: As defined in `config.yml` (e.g., 12).
-   `HISTORY_LENGTH`: As defined in `config.yml` (e.g., 5).

## 5. Training & Benchmarking Workflow (`training.py`)

A simplified flowchart of what happens when `training.py` is executed.

```
[ Start training.py ]
        |
        v
[ Parse CLI Args? ] --(No)--> [ Use config.yml Defaults ]
        | (Yes)
        v
[ Use CLI Args ]
        |
        v
[ Setup TrainingManager (MLflow) ]
        |
        v
[ Analyze Only Mode? ] --(Yes)--> [ Load Existing Benchmark Results ] --> [ Run BenchmarkAnalyzer ] --> [ End Script ]
        | (No)
        v
[ Skip LSTM Training? ] --(Yes)--> [ Load/Assume Existing LSTM Model ]
        | (No)                                      ^
        v                                           |
[ Train LSTM Attack Predictor ] --> [ Log LSTM Metrics/Model via TM ]
        |
        v
[ Algorithm Mode? (Single / ALL / List) ]
        |
        |--(Single Algorithm)--> [ Train Single RL Algorithm ] --> [ Log RL Metrics/Model via TM ] --> [ End Script ]
        |
        |--(ALL or List of Algorithms)--> [ Execute BenchmarkRunner ]
                                                    |
                                                    v
                                     [ MetricsCollector Gathers Data ]
                                                    |
                                                    v
                                     [ Run BenchmarkAnalyzer on Collected Data ]
                                                    |
                                                    v
                                     [ Log Benchmark Summary/Plots via TM (Optional) ]
                                                    |
                                                    v
                                            [ End Script ]
```

This document aims to provide clearer visual insights into your project's structure and operation. You can expand these diagrams or add more specific ones as needed.
