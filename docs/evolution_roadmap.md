# Project Evolution Roadmap

## 1. Objective
To develop and refine the RL-IoT Defense System by deeply integrating and leveraging the real-world CICIoT2023 dataset. This roadmap focuses on maximizing the system's robustness, adaptability, and realism by grounding its learning and evaluation in observed attack patterns.

## Phase 1: Real Data Integration & Baseline (Completed)
This foundational phase focused on adapting the system to work exclusively with the CICIoT2023 real-world dataset. The core training pipeline is now stable and robust.

### 1.1. Data Exploration & Preprocessing (Completed)
- **Action:** Acquired and performed thorough EDA on the CICIoT2023 dataset.
- **Task:** Analyzed features, documented class distribution, and established a preprocessing pipeline (`src/utils/dataset_processor.py`, `src/utils/dataset_loader.py`).

### 1.2. Environment Adaptation for Real Data (Completed)
- **Action:** Updated the `IoTEnv` class (`src/environment.py`) to use features and simulate scenarios reflective of CICIoT2023.
- **Task:**
    - **State Representation:** Uses features compatible with CICIoT2023 for network state.
    - **Attack Simulation:** Attack occurrences and types are simulated, with effects on the network state.
    - **LSTM Integration:** The environment incorporates predictions from an LSTM model trained on CICIoT2023 via `EnhancedAttackPredictor`.

### 1.3. LSTM Attack Predictor Training on Real Data (Completed)
- **Action:** Implemented and trained the `RealDataLSTMPredictor` (`src/models/lstm_attack_predictor.py`) solely on the CICIoT2023 dataset.
- **Task:** Achieved high accuracy in predicting attack types based on real network flow sequences.

### 1.4. Redefined Reward Function (Completed)
- **Action:** Updated the reward logic within `IoTEnv` to be informed by the LSTM's predictions on real data patterns and the effectiveness of defenses against simulated attacks.

### 1.5. Train Baseline RL Models with Real Data Integration (Completed)
- **Action:** Used the `RLTrainer` and `TrainingManager` to train and evaluate RL agents (DQN, PPO, A2C) in the environment that uses the CICIoT2023-trained LSTM.
- **Outcome:** Established baseline performance metrics for RL agents operating with real-data-informed predictions.

## Phase 2: Advanced Analysis & Refinement
This phase focuses on in-depth analysis of the current system and iterative improvements.

### 2.1. In-depth Performance Analysis
- **Action:** Conduct detailed analysis of both the LSTM predictor and the RL agents.
- **Task:**
    - **LSTM:** Analyze per-class performance, confusion matrices, and identify challenging attack types.
    - **RL Agents:** Examine learning curves, policy behavior under specific predicted threats, and sensitivity to reward function components.
    - Use MLflow extensively for tracking and comparing experiments.

### 2.2. Feature Engineering & State Representation Refinement
- **Action:** Explore alternative feature sets or transformations from CICIoT2023 for both LSTM input and RL state.
- **Task:** Investigate if different feature subsets or engineered features can improve LSTM prediction or RL agent learning. Evaluate impact on model complexity and performance.

### 2.3. Hyperparameter Optimization
- **Action:** Systematically tune hyperparameters for both the LSTM model and the RL algorithms.
- **Task:** Employ tools like Optuna (integrated with MLflow) or grid search to find optimal configurations.

### 2.4. Robustness Testing
- **Action:** Evaluate the system's performance against variations in attack patterns or simulated network conditions (e.g., varying `attack_probability` in `IoTEnv`).
- **Task:** Assess how well the learned policies generalize.

## Phase 3: Documentation, Benchmarking & Finalization
This phase focuses on preparing the project for final presentation and potential open-sourcing.

### 3.1. Comprehensive Benchmarking
- **Action:** Run final, extensive benchmarks comparing DQN, PPO, and A2C under optimized configurations.
- **Task:** Generate detailed reports, plots, and statistical comparisons of algorithm performance. Utilize the `src/benchmarking/` tools if further developed.

### 3.2. Finalize Documentation
- **Action:** Ensure all README files, code comments (docstrings), and architectural documents are up-to-date, clear, and comprehensive.
- **Task:** Create a final project report summarizing objectives, methods, results, and conclusions.

### 3.3. Codebase Polish & Release Preparation
- **Action:** Refactor code for clarity, efficiency, and adherence to coding standards.
- **Task:** Ensure all dependencies are correctly listed in `requirements.txt`. Prepare for a potential code release.

## Future Considerations (Beyond Current Scope)

### Synthetic Data Augmentation (GANs)
-   **Original Idea:** Implement a GAN (e.g., cGAN) trained on CICIoT2023 to generate synthetic attack data, particularly for rare attack types, to augment training datasets.
-   **Status:** Currently deferred to focus on maximizing performance with the real dataset. Could be revisited if specific data augmentation needs are identified (e.g., for extremely rare but critical attack types not well-represented).

### Advanced Attacker Modeling
-   Explore more sophisticated, adaptive attacker models within the environment rather than probabilistic attack occurrences.

### Multi-Agent RL Systems
-   Investigate using multiple RL agents for different defensive roles or to protect different network segments.
This roadmap prioritizes a deep understanding and robust utilization of the CICIoT2023 dataset as the core of the defense system.