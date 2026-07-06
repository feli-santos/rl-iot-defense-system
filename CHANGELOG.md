# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the
research artefact versions recorded in `CITATION.cff`.

## [Unreleased]

### Changed
- Rewrote `requirements.txt` as a minimal direct-dependency list (10 runtime
  packages) and added `requirements-dev.txt` (lint, format, test, pre-commit).
- Removed the unused `mlflow` integration and the dead `src/training/` package,
  dropping ~100 transitive dependencies (docker, fastapi, flask, sqlalchemy,
  alembic, opentelemetry-*, …).
- Removed stale `MLP` detector references from `README.md` and
  `docs/ARCHITECTURE.md` (the MLP detector was dropped in `5d2bb2c`; only the
  `RandomForest` stage detector remains).
- `Makefile` `install-dev` now installs from `requirements-dev.txt`.

### Added
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- GitHub Actions CI (`.github/workflows/ci.yml`): Python 3.9, `make lint` +
  `pytest -q --cov`.

### Removed
- Dead code: `src/training/training_manager.py`, `src/training/__init__.py`.
- Orphan/personal artifacts: `scripts/review/redesign_smoke.py`,
  `scripts/run_alpha_sweep_5M_det.sh`, `tex/cover_letter.md`,
  `data/img/cic-{topology,diagram3}-2023.jpg`, `tex/figs/F15b_recall_vs_advantage.png`.


## [0.2.0] — 2026-06-21

First open-source release. Adversarial-RL IoT defense system: a reactive
tug-of-war attacker walks the kill chain against a POMDP blue-team agent
(DQN/PPO/A2C) that never observes the true attack stage.

### Headline empirical findings (deterministic-5M regime)

- **PPO dominates across the adversarial-strength sweep (α=0.0/0.2/0.4/0.6):**
  +138.6 → +113.3 vs tuned RF +137.5 → +73.6. Tie at α=0 (overlapping CIs);
  disjoint CIs from α=0.4 (PPO +121.3 vs RF +80.9, significant by +40.3).
  Oracle ceiling +194.9. Source: `docs/results/ablation/Falpha_summary.json`.
- **OOD robustness (10 held-out zero-day classes):** PPO prevents 0.30–0.59
  vs RF 0.00–0.15 on every class, with no detectable dependence on detector
  recall (Spearman ρ=0.16, p=0.66; Pearson r=0.38, p=0.28; OLS slope CI
  spans zero).
- **Reward-coupling ablation:** coupled reward best DQN +274.8 (gap −128.0 vs
  RF +146.8); outcome reward best PPO +123.8 (gap −42.9 vs RF +80.9). The
  on-policy advantage is training reliability, not peak return — across-seed
  sd PPO≈15 / DQN≈52 / A2C≈38.
- **447 tests passed** (synthetic-only default; real-data tests auto-skip).

### Locked contracts

- Primary reward: `reward_mode=outcome` (sparse, outcome-only) for training +
  benchmark. `coupled` used only in the reward-coupling ablation.
- 10 seeds `{0..9}` for DRL; baselines/oracle run 1 seed. n=300 episodes for
  all policies. Tug-of-war `p_down=0.90` (ISOLATE 0.98) / `p_up=0.90`;
  BENIGN onset `p_onset=0.35`, `p_onset_access=0.10`. Prevention bonus +50.
- Canonical checkpoints: `runs/redesign_5M_det/alpha_{00,02,04,06,08,10}/<algo>/seed_<n>/best_model.zip`.
