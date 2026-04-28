# Data Pipeline

## Objectives

The dataset processor converts raw CICIoT2023 CSVs into artifacts used by both the LSTM generator and the RL environment. The pipeline focuses on:

- **Leakage avoidance**: splits happen before scaling and feature selection.
- **Real data only**: no synthetic traffic generation.
- **Stage mapping**: all labels map to 5 Kill Chain stages via `AbstractStateLabelMapper`.

## Key artifacts

Produced under `data/processed/ciciot2023` (or `--data-path`):

- `features.npy` — normalized feature matrix for environment.
- `features_raw.npy` — optional raw features (no scaling).
- `labels.npy` — original CICIoT2023 labels (strings).
- `state_indices.json` — mapping from stage ID to row indices.
- `scaler.joblib` — `StandardScaler` fitted on train split only.
- `metadata.json` — split counts, stage distribution, feature selection diagnostics.

## Pipeline steps

### 1) Load raw data

All CSVs under `data/raw/CICIoT2023` are loaded and concatenated.

### 2) Sampling strategy

The processor uses a smart balancing strategy when configured:

- **Benign quota** reserved first.
- Remaining budget allocated to attack classes with a derived per-class cap.
- Optionally enforces per-stage minimums to prevent ACCESS starvation.

Config keys (from `config.yml`):

- `dataset.sample_size`
- `dataset.sampling_mode` (`default` | `smart_balanced`)
- `dataset.sampling_strategy` (`balanced` for severe imbalance)
- `dataset.benign_target_count`
- `dataset.max_samples_per_attack_class`

### 3) Split before scaling

The raw data is split **by label** into train/val/test before any scaling to avoid leakage.

### 4) Feature processing

- Categorical columns are encoded as category codes.
- Infinite values are replaced with large finite values.
- NaNs are filled with median values.

### 5) Feature selection (optional)

Three-stage filtering is applied when `feature_selection=true`:

1. **Zero variance removal**
2. **Low variance removal** (threshold via `variance_threshold`)
3. **High correlation removal** (threshold via `correlation_threshold`)

Features matching `feature_keep_keywords` are protected but still de-duplicated if perfectly correlated.

### 6) Scaling

`StandardScaler` is fit on **train split only** and then applied to val/test.

### 7) Stage mapping and state indices

Labels are mapped to stages via `AbstractStateLabelMapper`. The resulting indices populate `state_indices.json` for fast sampling in the environment.

## Output metadata

`metadata.json` includes:

- `stage_counts`, `stage_percentages`, `imbalance_ratio`
- `split_info`: train/val/test sample counts
- `feature_selection_info` (dropped features + thresholds)
- `sampling_info` (smart-balanced quota/cap details)

## Key implementation locations

- `src/utils/dataset_processor.py`
- `src/utils/label_mapper.py`
- `src/utils/realization_engine.py`

## Common pitfalls

- **Stage sparsity:** ACCESS is rare; use `min_samples_per_stage` and balanced sampling.
- **Leakage risk:** scaling before split inflates metrics; this code avoids it.
- **Feature selection drift:** ensure `scaler.joblib`, `features.npy`, and `state_indices.json` come from the same run.
