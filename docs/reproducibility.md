# Reproducibility Guide

## Environment setup

- Python 3.12 (venv)
- Install dependencies via `requirements.txt`

## Determinism controls

- `EpisodeGenerator` accepts a seed.
- `GeneratorTrainer` uses a seed for split reproducibility.
- `AdversarialIoTEnv.reset(seed=...)` seeds numpy + torch.

## Recommended workflow

1. **Process data**
   - Run with `--mode process-data`
   - Confirm `metadata.json` contains stage counts

2. **Train generator**
   - Use fixed seed in config if reproducibility required

3. **Train RL**
   - Run with a fixed seed for environment reset (if desired)

4. **Evaluate**
   - Use the same generator and dataset as training

## Artifact integrity

Ensure these are aligned per run:

- `features.npy`
- `scaler.joblib`
- `state_indices.json`
- `attack_sequence_generator.pth`

## Recommended versioning

- Keep a run folder per experiment under `artifacts/rl/`
- Store `config.json` and `training_config.json` alongside the generator
- Snapshot `config.yml` used for each run
