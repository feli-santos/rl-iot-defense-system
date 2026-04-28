# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 0: Mentor-restart hygiene

### Added
- `Makefile` with `help`, `lint`, `test`, `train-*`, `evaluate`, and `reproduce-thesis`
  targets as the canonical developer entrypoint.
- `pyproject.toml` configuring black, isort, ruff, pytest, mypy, coverage.
- `.pre-commit-config.yaml` with ruff/black/isort and standard hygiene hooks.
- GitHub Actions CI (`.github/workflows/ci.yml`) running lint + tests on
  Python 3.9 / 3.10 / 3.11.
- `CITATION.cff` for proper academic citation, referencing IoTWarden and
  CICIoT2023.
- `docs/results/` directory as the canonical home for thesis-quality figures.
- `docs/thesis_results_map.md` mapping every planned thesis figure → script →
  MLflow run.
- `CHANGELOG.md` (this file).
- Git tag `pre-mentor-restart` snapshotting the project state before the
  mentor-driven restart.

### Changed
- (Pending) Reconciled README mode names with `main.py` actual choices.

### Removed
- Orphan run directories under `artifacts/rl/` (10 runs from 2026-03-12/13).
- Dead artifact directories `artifacts/rl_agent/` and
  `artifacts/tmp_processor_validation/`.
- Legacy `results/benchmark/` and `results/logs/` from the pre-restart era.
- All removed content was archived to `.archive/pre_mentor_artifacts_<TS>.tgz`
  before deletion (not committed).

### Notes on results
- The pre-restart benchmark (`avg_reward = -6.67 ± 88`,
  `false_positive_rate = 0.79`, `macro_f1 = 0.29`) and the pre-restart LSTM
  (`macro_f1 = 0.59`, IMPACT-biased confusion matrix) are NOT considered
  thesis-quality and will be regenerated in Phases 2–7.
- Root-causes documented in `docs/results/00_phase0_diagnosis.md` (to be added
  during Phase 1).
