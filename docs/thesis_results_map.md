# Thesis Results Map

This document is the **single canonical mapping** between thesis figures /
tables and the code, data, and MLflow runs that produce them. Every figure
that ships in the thesis must appear here, and every entry must be
reproducible from the corresponding `produced_by` script.

> **Convention:** Figures are labeled `F1, F2, …` and tables `T1, T2, …`.
> The labels match the filenames committed under `docs/results/<phase>/`.

## Tier 1 — Must-have (thesis-blocking)

| ID  | Title                                              | Phase | Produced by                                | Aligned with                |
|-----|----------------------------------------------------|-------|--------------------------------------------|-----------------------------|
| F1  | LSTM Red Team learning curves (loss + token-acc) on synthetic episodes | 2 | `scripts/red_team/train_lstm.py` | IoTWarden Fig. 3(a)         |
| F2  | LSTM empirical 5×5 transition matrix vs ground truth | 2  | `scripts/red_team/train_lstm.py` | our contribution             |
| F3  | RL episodic reward curves (DQN/PPO/A2C × 5 seeds)  | 5     | `scripts/blue_team/plot_learning_curves.py`| IoTWarden Fig. 4(a)         |
| F4  | Action-distribution evolution over training        | 5     | `scripts/blue_team/plot_action_dist.py`    | IoTWarden Fig. 5            |
| F5  | Final security metrics table                       | 6     | `scripts/benchmark/build_summary_table.py` | extension                    |
| F6  | Stage × Action confusion matrices per algorithm    | 6     | `scripts/benchmark/plot_stage_action_cm.py`| our contribution             |
| F7  | Computation overhead (latency CDF + training time) | 6     | `scripts/benchmark/plot_overhead.py`       | IoTWarden Fig. 4(b)         |
| F15 | OOD-class robustness (held-out attack-class eval) [audit-AF1, 2026-04-30] | 7 | `scripts/ablation/plot_ood_robustness.py` | our contribution (Phase-4 → Phase-7 link) |

## Tier 2 — Strongly recommended

| ID  | Title                                              | Phase | Produced by                                | Aligned with                |
|-----|----------------------------------------------------|-------|--------------------------------------------|-----------------------------|
| F8  | RL vs random / always-OBSERVE / always-BLOCK / RF / recommended-action | 6 | `scripts/benchmark/plot_baselines.py`      | extension                    |
| F9  | Reward-component ablation                          | 7     | `scripts/ablation/plot_reward_ablation.py` | our contribution             |
| F10 | Sensitivity to attack aggressiveness               | 7     | `scripts/ablation/plot_aggressiveness.py`  | IoTWarden Fig. 6            |
| F11 | Per-stage detection recall (detector + RF + 1D-CNN)| 4     | `scripts/detector/plot_per_stage_recall.py`| Tharewal et al.             |
| F12 | Pareto: security gain vs availability cost         | 7     | `scripts/ablation/plot_pareto.py`          | our contribution             |

## Tier 3 — Nice-to-have

| ID  | Title                                              | Phase | Produced by                                | Aligned with                |
|-----|----------------------------------------------------|-------|--------------------------------------------|-----------------------------|
| F13 | Robustness to observation noise / drift            | 8     | `scripts/robustness/plot_drift.py`         | extension                    |
| F14 | Generalization to held-out attack class (training-time augmentation) | 8 | `scripts/robustness/plot_oo_attack.py` | extension (F15 covers eval-time; F14 if it ships covers train-time) |
| T1  | Hyperparameters per algorithm                      | 5     | `scripts/blue_team/dump_hparams.py`        | IoTWarden Tab. I            |

## Per-figure manifest

Each figure directory under `docs/results/<phase>/` contains a
`manifest.json` referencing the MLflow run id(s) that produced its underlying
data. Figures with no manifest are not considered ready for the thesis.

## Update protocol

When a figure is regenerated:

1. Delete old PNG and `manifest.json`.
2. Re-run the producing script (it must read seed and config from CLI).
3. Commit the new artifacts together with the script change.
4. Update this table only if the figure title or scope changes.
