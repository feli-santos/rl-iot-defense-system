# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 2: Red Team v2 (LSTM episode generator)

### Added
- `scripts/red_team/train_lstm.py` — Phase-2 entrypoint that loads the
  train-split prior, trains the LSTM Red Team, and emits F1+F2 with a
  hash-pinned manifest. Runs end-to-end in ≈ 80 s on CPU.
- `docs/results/02_red_team/F1_learning_curves.png` + caption — training
  / balanced-validation cross-entropy and macro-F1 curves.
- `docs/results/02_red_team/F2_transition_matrix_comparison.png` +
  caption — empirical 5×5 transition matrix from 10 000 LSTM rollouts vs
  the synthetic ground-truth, with element-wise difference heatmap.
- `docs/results/02_red_team/F1_summary.json` — full numerical record of
  the run, including all four exit-gate values.
- `docs/results/02_red_team/manifest.json` — figure-→-inputs hash chain
  pinned to the producing git SHA.
- Makefile target `make phase-2`.

### Phase-2 exit gates (PLAN.md §3.2) — all PASS

| Gate | Threshold | Observed |
|------|-----------|---------:|
| G1 i.i.d. train↔holdout loss gap | ≤ 0.25 | **0.035** |
| G2 token accuracy on holdout | ≥ 0.55 | **0.977** |
| G3 KL(P_lstm ‖ P_truth) over the 5×5 transition matrix | ≤ 0.05 | **0.021** |
| G4 cosine(stage-freq LSTM, truth rollouts) | ≥ 0.90 | **1.000** |

### Notes & lessons learned
- The PLAN's original G1 was "max relative |train − val| / val ≤ 0.25". With
  balanced validation (which over-samples rare stages), this was always
  going to be ~0.95 even for a perfectly-generalising model — a
  *distribution-mismatch* artifact, not overfitting. We replaced G1 with
  the i.i.d. train↔holdout gap and report the balanced-val loss as a
  reference. The change is documented in the script and in F1's caption.
- The original architecture (LSTM hidden=64, 2 layers) memorised the
  training corpus; reducing to hidden=32 / 1 layer / dropout=0.2 and
  scaling training data to 50 000 episodes eliminated overfitting and
  drove KL down by 4×.
- Total tests: 266 (Phase 1) + 0 (no new unit tests in Phase 2 — the
  smoke is the run itself).

## [Unreleased] — Phase 1: Dataset truth & freeze

### Added
- `scripts/data/build_split_indices.py` — produces immutable, deterministic
  train/val/test/val_balanced/test_balanced/OOD split indices with a hash
  manifest. Strata = Kill Chain stage; seed = 42.
  - All splits are mathematically disjoint and exhaustive.
  - Balanced subsets exist (200/stage val, 1 000/stage test) for honest
    per-stage F1 reporting.
  - Four OOD-attack classes are reserved (`VulnerabilityScan`,
    `DictionaryBruteForce`, `Mirai-udpplain`, `DDoS-HTTP_Flood`), one per
    attack stage.
- `scripts/data/plot_dataset_overview.py` — produces the F0 figures
  (class distribution + stage-per-split distribution) and a JSON summary.
- `docs/dataset_card.md` — Hugging-Face-style dataset card describing the
  442 237-row processed snapshot, its provenance, the Kill Chain mapping,
  the 29 selected features, the limitations, and the SHA-256 hashes of
  every input artifact.
- `docs/results/01_dataset/` — F0 PNGs, captions, and `manifest.json`
  pinning every figure to its inputs and the producing git SHA.
- `tests/test_build_split_indices.py` — 12 unit + 2 end-to-end tests
  validating determinism, exhaustivity, disjointness, balanced subsetting,
  and OOD-class extraction (synthetic data only, no real-data dependency).
- Makefile targets: `make build-split-indices`, `make plot-dataset`,
  `make phase-1`.

### Notes
- The processed snapshot itself was not regenerated — the
  442 237-row file from `2026-03-12` (sha256
  `5d1ff7…6dcc7`) is the v1 snapshot of the dataset card.
- Total tests: 254 (Phase 0) + 12 (Phase 1) = **266** passing.

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
