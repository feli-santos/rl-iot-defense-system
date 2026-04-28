# Step-by-Step Walkthrough

This walkthrough guides you through the full pipeline and how to evaluate results.

## 1) Process CICIoT2023

- Mode: `process-data`
- Outputs: `features.npy`, `labels.npy`, `state_indices.json`, `scaler.joblib`, `metadata.json`
- Check `metadata.json` for stage distribution and imbalance ratio.

## 2) Train the Red Team (LSTM)

- Mode: `train-generator`
- Generates episodes via `EpisodeGenerator`
- Trains `AttackSequenceGenerator` on next-token prediction
- Outputs: model + loss curves

## 3) Train the Blue Team (RL)

- Mode: `train-rl`
- Environment uses generator + RealizationEngine
- Trains DQN/PPO/A2C with MlpPolicy
- Outputs: RL model zip files, MLflow logs

## 4) Evaluate and benchmark

- Mode: `evaluate`
- Single-model or multi-algorithm comparison
- Produces reports and plots in `results/benchmark/`

## 5) Interpret results

- Prefer **attack mitigation rate** and **false positive rate** as primary security signals.
- Use reward as a secondary signal; it is sensitive to shaping.
- Use per-stage appropriate rate and heatmaps to validate policy behavior.

## 6) Iterate safely

- If false positives are high, increase benign penalties or lower escalation rewards.
- If mitigation is low, increase `correct_escalation_reward` or `impact_penalty`.
- Re-run `evaluate` after each change to keep analysis consistent.

## Recommended order for reading

1. `architecture.md`
2. `data-pipeline.md`
3. `kill-chain-mapping.md`
4. `generator.md`
5. `environment.md`
6. `reward-shaping.md`
7. `rl-training.md`
8. `benchmarking-results.md`
