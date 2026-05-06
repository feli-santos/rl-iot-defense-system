# Thesis Results Map

This document is the **single canonical mapping** between thesis figures /
tables and the code, data, and MLflow runs that produce them. Every
figure that ships in the thesis must appear here, and every entry must
be reproducible from the corresponding `produced_by` script.

> **Convention.** Figures are labeled `F1, F2, …` and tables `T1, T2, …`
> The labels match the filenames committed under
> `docs/results/<phase>/`. Chapter and section identifiers refer to the
> 5-chapter structure locked in
> [`mentor_review/00_framing.md`](mentor_review/00_framing.md).

## Thesis-blocking figures (must ship)

| ID  | Title                                                                  | Phase | Chapter | Section | Produced by |
|-----|------------------------------------------------------------------------|-------|---------|---------|-------------|
| F0a | Class distribution after rebalancing (per CICIoT2023 class)            | 1     | Ch. 4   | §4.1    | `scripts/data/plot_dataset_overview.py` |
| F0b | Kill-Chain stage distribution per split                                | 1     | Ch. 4   | §4.1    | `scripts/data/plot_dataset_overview.py` |
| F1  | LSTM Red Team learning curves (loss + token-acc) on synthetic episodes | 2     | Ch. 4   | §4.1    | `scripts/red_team/train_lstm.py` |
| F2  | LSTM empirical 5×5 transition matrix vs ground truth                   | 2     | Ch. 4   | §4.1    | `scripts/red_team/train_lstm.py` |
| F11 | Per-stage detection recall (Random Forest + 1D-CNN)                    | 4     | Ch. 4   | §4.2    | `scripts/detector/train_detector.py` (+ plotter) |
| F3  | RL episodic reward curves (DQN/PPO/A2C × 5 seeds)                      | 5     | Ch. 4   | §4.3    | `scripts/blue_team/plot_learning_curves.py` |
| F4  | Action-distribution evolution over training                            | 5     | Ch. 4   | §4.3    | `scripts/blue_team/plot_action_dist.py` |
| F5  | Final security metrics table (8 policies, bootstrap CIs)               | 6     | Ch. 4   | §4.4    | `scripts/benchmark/build_summary_table.py` |
| F6  | Stage × action confusion matrices per algorithm                        | 6     | Ch. 4   | §4.4    | `scripts/benchmark/plot_stage_action_cm.py` |
| F7  | Computation overhead (latency CDF + training time)                     | 6     | Ch. 4   | §4.4    | `scripts/benchmark/plot_overhead.py` |
| F8  | RL vs random / always-OBSERVE / always-BLOCK / RF-acting / oracle      | 6     | Ch. 4   | §4.4    | `scripts/benchmark/plot_baselines.py` |
| F9  | Reward-component ablation (5 components × {0.5×,1×,2×} + impact_terminal) | 7  | Ch. 4   | §4.5    | `scripts/ablation/plot_reward_ablation.py` |
| F10 | Sensitivity to attack aggressiveness (`p_defender_de-escalation` sweep) | 7    | Ch. 4   | §4.5    | `scripts/ablation/plot_aggressiveness.py` |
| F12 | Pareto: security gain vs availability cost                             | 7     | Ch. 4   | §4.5    | `scripts/ablation/plot_pareto.py` |
| F15 | OOD-class robustness (held-out attack-class evaluation)                | 7     | Ch. 4   | §4.6    | `scripts/ablation/plot_ood_robustness.py` |
| T1  | Hyperparameters per algorithm                                          | 5     | App. C  | —       | `scripts/blue_team/dump_hparams.py` |

All Tier-1 + Tier-2 figures from earlier planning rounds are present
in this list and committed under `docs/results/<phase>/`.

## Future-work figures (Chapter 5 only — not blocking)

These were originally scoped as Phase-8 deliverables but are reframed as
**future work** in the conclusions chapter. They are *not* required for
the defense and **no producing script ships in this release**.

| ID  | Title                                                          | Status |
|-----|----------------------------------------------------------------|--------|
| F13 | Robustness to observation noise / drift                        | Future work (Ch. 5) |
| F14 | Generalisation to held-out attack class via training-time augmentation | Future work (Ch. 5) |

## Per-figure manifest

Each figure directory under `docs/results/<phase>/` contains a
`manifest.json` referencing the input JSONL / model / split SHA-256
hashes and the producing git SHA. Figures with no manifest are not
considered ready for the thesis.

## Update protocol

When a figure is regenerated:

1. Delete the old PNG and `manifest.json`.
2. Re-run the producing script (it must read seed and config from CLI).
3. Commit the new artifacts together with the script change.
4. Update this table only if the figure title or scope changes.

## Note on prior IoTWarden alignment claims

Earlier revisions of this file annotated several figures (F1 / F3 /
F4 / F7 / F10 / T1) as "aligned with IoTWarden Fig. X" or "Tab. I". Per
[`mentor_review/00_framing.md`](mentor_review/00_framing.md), IoTWarden
is now scoped as inspiration only and no head-to-head visual or
numerical comparison is part of the thesis contract. Those alignment
annotations have been removed; the figures stand on their own as
direct empirical results on CICIoT2023.
