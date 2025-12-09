# Copilot Instructions

## Project Overview
Adversarial **IoT Defense System** on **CICIoT2023**. The LSTM is the **Attack Sequence Generator (Red Team)** that learns Kill Chain grammar (5 stages) and drives stochastic attack transitions. RL agents (**DQN/PPO/A2C**, Blue Team) learn to interrupt these sequences using realized traffic samples. End-to-end pipeline runs via `main.py` modes: `process-data` → `train-generator` → `train-rl` / `train-all-rl` / `train-all` → `evaluate`.

## Key Technologies
- **Python 3.12** (venv)
- **PyTorch** for the LSTM Attack Sequence Generator
- **Stable Baselines3** (DQN/PPO/A2C) for RL defense
- **Gymnasium** custom adversarial environment
- **MLflow** for experiment tracking
- **CICIoT2023** (real data only)

## Coding Standards
- Type hints for all params/returns
- Google docstring format
- Prefer `pathlib.Path`, f-strings
- Add descriptive error handling
- Use `dataclasses` for configs

## Key Technical Constraints
- **Kill Chain stages:** 5 (BENIGN, RECON, ACCESS, MANEUVER, IMPACT) via `AbstractStateLabelMapper`.
- **Action space:** Discrete(5) Force Continuum (OBSERVE, LOG, THROTTLE, BLOCK, ISOLATE) with tunable costs.
- **Observations:** Realized CICIoT2023 feature vectors (windowed) scaled by `scaler.joblib`; hidden state is the abstract stage.
- **Real data only**; no synthetic traffic. Episodes come from real samples mapped to stages.
- **Stable Baselines3** only; use appropriate policy for observation shape (current env emits Box/Dict per implementation).

## Common Patterns
- Run end-to-end via `main.py` modes: `process-data`, `train-generator`, `train-rl`, `train-all-rl`, `train-all`, `evaluate`.
- Data prep uses `CICIoTProcessor.process_for_adversarial_env` → outputs `features.npy`, `labels.npy`, `state_indices.json`, `scaler.joblib`, `metadata.json` (includes stage_distribution and feature_selection flag).
- LSTM training uses `EpisodeGenerator` + `AttackSequenceGenerator`; inference provides next-stage logits with temperature sampling.
- RL uses `AdversarialIoTEnv` and `AdversarialAlgorithm` wrappers; configure via `config.yml` and CLI overrides (timesteps, algorithm, device, seeds).
- Use MLflow experiments implicitly via training managers where applicable.

## Data Flow
```
CICIoT2023 → dataset_processor (scaler, stage_indices) → AdversarialIoTEnv realization → RL Agent
LSTM Attack Sequence Generator → adversarial env state transitions → sampled CICIoT2023 rows → RL defense
```

## Debugging Tips
- Ensure processed data exists: `features.npy`, `labels.npy`, `state_indices.json`, `scaler.joblib`, `metadata.json` in `data/processed/ciciot2023` (or CLI `--data-path`).
- Check stage distribution via metadata; imbalance may require `sampling_strategy: balanced` and `feature_selection: true` in `config.yml`.
- Use `--force` to reprocess or retrain generator/RL when artifacts exist.
- For RL, confirm environment observation shape matches chosen policy (Box vs Dict). Use proper SB3 policy (e.g., `MlpPolicy`/`MultiInputPolicy`).
- Run with `--log-level DEBUG` for detailed traces; validate scaler loading in env.

## Critical Implementation Notes
- Label mapping is PRD-compliant; unknown labels default to BENIGN with warning.
- Process-data now supports variance-based feature selection and safe inf/NaN handling.
- `state_indices.json` powers stage → row sampling; keep scaler and features in sync.
- Stage distribution exposed in processing results and metadata.
- RL uses 5-action space per PRD Force Continuum; costs configured in `config.yml` adversarial_environment.actions.
- Use CLI overrides for seeds/timesteps/device to generate comparison runs.