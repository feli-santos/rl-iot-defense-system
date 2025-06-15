# Visual Architecture and Workflows

This document provides a more detailed visual representation of the RL-IoT Defense System's architecture, data flow, and operational workflows, reflecting the refactored project structure.

## 1. Overall System Architecture

This diagram illustrates the main components and their high-level interactions.

```
+---------------------------------------------------------------------------------+
| Experimentation & Management                                                    |
| +---------------------------+   +-------------------------------------------+ |
| | main.py (Main Script)     |   | Benchmarking Framework (Optional)         | |
| | (CLI Interface)           |   |  - src/benchmarking/*                     | |
| +-----------+---------------+   +-----------------------+-------------------+ |
|             | (Uses)            |  (Results/Plots)      |                     |
|             v                   |                       v                     |
| +---------------------------+   |             +---------------------+         |
| | config.yml (Configuration)|   |             | TrainingManager     |         |
| +---------------------------+   |             | (src/training/      |         |
|             | (Used by)         |             |  training_manager.py)         |
|             v                     +---------->| (MLflow Integration)|         |
| +------------------------------------------+  | (Logs Metrics,      |         |
| | Training Orchestration                   |<--+  Models, Params)    |         |
| |                                          |  +---------------------+         |
| | +--------------------------------------+ |                                  |
| | | LSTMTrainer (src/training/           | |                                  |
| | |             lstm_trainer.py)         | |                                  |
| | |  - Uses RealDataLSTMPredictor &      | |                                  |
| | |    RealDataTrainer (src/models/      | |                                  |
| | |    lstm_attack_predictor.py)         | |                                  |
| | +------------------+-------------------+ |                                  |
| |                    | (Trained LSTM Model Path)                               |
| |                    v                     |                                  |
| | +--------------------------------------+ |                                  |
| | | RLTrainer (src/training/rl_trainer.py)| |                                  |
| | |  - Uses AlgorithmFactory (src/algos) | |                                  |
| | |  - Uses IoTEnv (src/environment.py)  | |                                  |
| | |    (IoTEnv uses EnhancedAttackPredictor| |                                  |
| | |     from src/models/predictor_if.py) | |                                  |
| | +--------------------------------------+ |                                  |
| +------------------------------------------+                                  |
+---------------------------------------------------------------------------------+
```
*   **IoT Environment (`src/environment.py`)**: Simulates the network, attacks, and calculates state/rewards. Internally uses `EnhancedAttackPredictor`.
*   **`EnhancedAttackPredictor` (`src/models/predictor_interface.py`)**: Loads and uses the trained `RealDataLSTMPredictor`.
*   **`RealDataLSTMPredictor` & `RealDataTrainer` (`src/models/lstm_attack_predictor.py`)**: The LSTM model and its specific trainer.
*   **RL Agents (DQN, PPO, A2C)**: Instantiated by `AlgorithmFactory` within `RLTrainer`.
*   **Training Orchestration (`main.py`, `LSTMTrainer`, `RLTrainer`)**: Manages the overall training process.
*   **`TrainingManager`**: Handles MLflow logging for RL training. LSTM training logs to MLflow via `RealDataTrainer`.
*   **`config.yml`**: Central configuration.

## 2. Detailed Data and Control Flow (Training Focus)

```
                               +---------------------+
                               |     config.yml      |
                               +----------+----------+
                                          | (Loads via ConfigLoader)
                                          v
+---------------------------------------+------------------------------------------+
| main.py (Main Script with CLI)                                                   |
|                                                                                  |
| IF mode == 'lstm' or 'both':                                                     |
|  +--------------------------+ (Instantiates) +--------------------------------+ |
|  | LSTMTrainer              |--------------->| RealDataTrainer (from            | |
|  | (src/training/           |                |  src/models/lstm_attack_pred.py)| |
|  |  lstm_trainer.py)        |                |  - Loads CICIoTDataLoader        | |
|  +----------+---------------+                |  - Trains RealDataLSTMPredictor  | |
|             |                                |  - Saves model (e.g., .pth)      | |
|             | (Returns Model Path)           |  - Logs to MLflow directly       | |
|             v                                +-----------------+--------------+ |
|  (lstm_model_path)                                                               |
|                                                                                  |
| IF mode == 'rl' or 'both':                                                       |
|  +--------------------------+ (Instantiates) +--------------------------------+ |
|  | RLTrainer                |<--------------| TrainingManager                  | |
|  | (src/training/           |                | (src/training/training_manager.py| |
|  |  rl_trainer.py)          |                |  - Handles MLflow for RL         | |
|  |  - Uses AlgorithmFactory |                +-----------------+--------------+ |
|  |  - Creates IoTEnv        |                                  ^                |
|  +----------+---------------+                                  | (Logs RL Metrics) |
|             | (Trains)                                         |                |
|             v                                +-----------------+--------------+ |
|  +----------------------------------------+  | RL Agent (DQN, PPO, A2C)       | |
|  | IoTEnv (src/environment.py)            |<--Policy Update & Action Selection| |
|  |  - Uses EnhancedAttackPredictor        +--------------------------------+ |
|  |    (which loads trained LSTM model                                        | |
|  |     using lstm_model_path)                                               | |
|  |  - Generates Observation (incl. preds)                                   | |
|  |  - Calculates Reward                                                     | |
|  +--------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------+
```

## 3. RL Agent - Environment Interaction Loop

This visualizes the core cycle for an RL agent.

```
+---------------------------------+      +---------------------------------+
|        RL Defense Agent         |      |        IoT Environment          |
| (e.g., PPO with MultiInputPolicy)|      |        (src/environment.py)     |
| (Manages by RLTrainer)          |      |  (Uses EnhancedAttackPredictor) |
+---------------------------------+      +---------------------------------+
              |                                        ^
(1. Observe)  | current_state, state_history,          | (4. Return)
              | action_history, attack_prediction,     | reward, next_state, done
              | reward, done                           |
              v                                        |
+---------------------------------+      +---------------------------------+
| Policy Network (π(a|s))         |----->| Action Selection (a_t)          |
| Value Network (V(s)) (if appl.) |      | (e.g., argmax Q or sample policy)|
+---------------------------------+      +---------------------------------+
              ^                                        |
              |                                        | (2. Execute Action a_t)
              | (3. Update Policy/Value based on experience) |
              | (e.g., using collected trajectory/batch by SB3 algo) |
              +----------------------------------------------+
```
1.  **Observe**: Agent receives the current state (`Dict` observation, including `attack_prediction`) and reward.
2.  **Act**: Agent selects an action.
3.  **Update (Training)**: Agent's internal model is updated by the SB3 algorithm.
4.  **Environment Step**: Environment processes action, simulates network, gets new LSTM prediction, transitions state, calculates reward.

## 4. Observation Space Structure (`Dict`) in `IoTEnv`

```
Observation (Dict from IoTEnv)
├── 'current_state': Box(shape=(22,)) // Example shape
│   └─ [network_feature_1, ..., network_feature_22]
│
├── 'state_history': Box(shape=(STATE_HISTORY_LENGTH, 22)) // Example shape
│   └─ Sequence of past 'current_state' vectors
│
├── 'action_history': Box(shape=(ACTION_HISTORY_LENGTH,))
│   └─ [action_t-L_a, ..., action_t-1] (Sequence of past discrete actions)
│
└── 'attack_prediction': Box(shape=(6,)) // Example shape
    └─ [risk_score, confidence, is_attack_bool, severity_encoded, category_encoded, max_prob]

---------------------------------------------------------------------> To MultiInputPolicy
                                                                         (Feature Extractors for each key, then Concatenation)
```

## 5. `main.py` Workflow

```
[ Start main.py ]
        |
        v
[ Parse CLI Args (mode, config path, overrides) ]
        |
        v
[ Load config.yml via ConfigLoader ]
        |
        v
[ IF mode == 'lstm' or 'both' ]
        |
        +--[ Instantiate LSTMTrainer ]
        |         |
        |         +--[ LSTMTrainer.train() ] --> (Trains RealDataLSTMPredictor, logs to MLflow, saves .pth)
        |         |
        |         +-- (lstm_model_path obtained)
        v
[ IF mode == 'rl' or 'both' ]
        |
        +-- (Ensure lstm_model_path is available, load from config if mode=='rl' only)
        |
        +--[ Instantiate RLTrainer (with lstm_model_path) ]
        |         |
        |         +--[ RLTrainer.train() ] --> (Creates Env & Algo, uses TrainingManager for MLflow, trains RL agent)
        v
[ End Script ]
```
This provides a clearer view of the refactored system's operation.
