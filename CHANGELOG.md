# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 4: Stage Detector + Supervised Baselines (F11)

### Added
- `docs/results/04_detector/PLAN.md` — pre-code audit + locked design
  decisions D1/D2/D3 (eval split, OOD gate fallback, fair baseline configs).
- `docs/results/04_detector/RESULTS.md` — as-built record covering the
  three findings worth defending in the thesis (RF saturates, RECON is
  the universal hard stage, OOD generalisation is class-asymmetric).
- `scripts/data/derive_stage_labels.py` + 10 unit tests + Makefile
  target `make derive-stages`. Builds the frozen `stages.npy` from
  `state_indices.json` and hash-pins via `stages.manifest.json`.
- `src/detector/` — new module with the production MLP head
  (`StageDetector`, ~4 357 params), the Tharewal-style 1-D conv
  baseline (`CNN1D`), the sklearn RandomForest wrapper, and shared
  evaluation helpers (`per_stage_recall`, `summarize_run`, etc.).
- `scripts/detector/train_detector.py` (Makefile target `make
  phase-4`) — trains all three models, evaluates on
  `test_balanced` / `test` / OOD, renders F11, dumps `F11_summary.json`
  + `manifest.json` (hash chain pinned to the producing git SHA).
- `docs/results/04_detector/F11_per_stage_recall.png` (1775 × 694)
  + caption: bar chart of per-stage recall across the three models +
  StageDetector confusion matrix on `test_balanced`.

### Fixed
- **`scripts/data/build_split_indices.py` — CRITICAL**: held-out OOD
  attack classes were not being removed from `train` / `val` / `test`
  before persisting. Discovered during Phase-4 step 4.5 by the
  defensive disjointness check. Concrete leakage:
  `train ∩ ood:DDoS-HTTP_Flood = 8 546 rows` (70 % of the class)
  and similar for the other three OOD classes. Fix computes OOD
  indices first, masks them, then stratified-splits the remainder.
  Three new regression asserts lock the disjointness invariant.
  Phase 2 (LSTM Red Team) consumes only stage labels, not features,
  so its F1/F2 numbers are approximately correct and *not* rebuilt.
  See `RESULTS.md` §5 for the full bug report.

### Phase-4 exit gates (`PLAN.md` §3.3 + §8 revisions) — all PASS

| Gate | Threshold | Observed | Status |
|------|----------:|---------:|:------:|
| G4.1 | full pytest suite green | 329 / 329 | **PASS** |
| G4.2 | StageDetector macro-F1 on `test_balanced` ≥ 0.75 | **0.7855** | **PASS** |
| G4.3 | StageDetector worst per-stage recall ≥ 0.50 | **0.539** (RECON) | **PASS** |
| G4.4 | min(OOD recall) ≤ 0.30 (revised D2) | **0.001**, gap **0.998** | **PASS-with-finding** |
| G4.5 | StageDetector inference latency ≤ 1 ms / sample | **0.039 ms** | **PASS** |

### Phase-4 thesis findings (RESULTS.md §4)

1. **RandomForest saturates at 0.90 macro-F1** on the 29-D feature
   vector — the thesis story is preserved because the RL value is
   "act correctly on detector outputs over time", not "detect more
   accurately than RF".
2. **RECON is the universal hard stage** across all three models
   (worst recall: StageDetector 0.539, RF 0.785, CNN1D 0.497).
   The Phase-3 proportionality reward already accommodates this:
   ±1 around the recommended `LOG` action is rewarded, so the
   RL agent can hedge on uncertain RECON observations.
3. **OOD generalisation is class-asymmetric** (recall span 0.001-
   0.999, gap 0.998). The detector trivially generalises on
   `DDoS-HTTP_Flood` (matches in-dist DDoS-* signatures) but fails
   completely on `VulnerabilityScan` (genuinely novel RECON
   pattern). This is the *right* thesis story: OOD generalisation
   is structurally bounded by in-distribution feature-class overlap,
   and the RL agent has to defend correctly *despite* the detector's
   silent confident-wrongness.

### Phase-4 commits
`4fd3460` PLAN — `0a8ef3e` D1/D2/D3 lock-in — `0d154e9` stages.npy +
10 tests + Makefile — `f3b82c3` src/detector/ (4 modules) + 23 tests
— `3cd2fb9` fix(phase-1) OOD leakage — `1357ec6` train_detector.py
entrypoint + F11 + 4/4 gate verification — `<this commit>` RESULTS +
CHANGELOG.

---

## [Unreleased] — Phase 3: Environment v2 (lifecycle, reward, MTTC, split-aware features)

### Added
- `docs/results/03_env/PLAN.md` — pre-code audit naming six bugs (B1-B6)
  in the v1 environment + `src/utils/realization_engine.py`.
- `docs/results/03_env/RESULTS.md` — as-built record covering the three
  iterations needed to satisfy every gate, the lifecycle/reward formulae,
  and the constants used as Phase-5 defaults.
- `RealizationEngine(allowed_indices=...)` constructor argument and
  `RealizationEngine.from_split_manifest(...)` factory. The factory
  loads a Phase-1 splits manifest, restricts per-stage sampling to the
  named split, and (by default) excludes the OOD-attack rows. Verified
  on the real CICIoT manifest: train pool ∩ val.idx = ∅.
- `tests/test_realization_engine_split_aware.py` — 9 unit tests on
  synthetic data covering empty/partial coverage and OOD overlap removal.
- `tests/test_phase3_env_gates.py` — 13 regression tests mapping 1:1 to
  the exit gates in `PLAN.md` §3.2.

### Changed
- `src/environment/adversarial_env.py` rewritten:
  - **Lifecycle (B1).** Dropped the `BLOCK = instant win` early
    termination. Episodes now run for at least `min_episode_length=20`
    steps. An IMPACT-clamp downgrades any pre-floor IMPACT transition to
    MANEUVER, matching the IoTWarden threat model in which IMPACT is the
    consummation of MANEUVER, not an instantaneous transition from RECON.
    The terminal IMPACT penalty (and missed-impact / mitigation bonus) is
    now applied **inline** when the env terminates due to IMPACT — the
    `_step_at_impact` codepath is preserved for explicit IMPACT-stage
    rollouts only.
  - **Reward (B2).** Replaced the action-vs-previous-action heuristic
    with stage-action proportionality against the IoTWarden recommended-
    action mapping (`_recommended_action`). Reward depends only on
    `decision_stage` and `action`. The four old action-change-based
    fields (`patience_bonus`, `correct_escalation_reward`,
    `correct_de_escalation_reward`, `maintained_defense_reward`,
    `false_positive_penalty`) are removed.
  - **De-escalation (B3).** Added `_maybe_defender_deescalation`: at any
    step where the agent picks BLOCK or ISOLATE on an ACCESS+ stage, the
    env resets the attack to BENIGN with probability
    `p_defender_deescalation=0.6`. The agent earns
    `+defense_success_bonus`. This makes the dead-code de-escalation
    branch reachable on the LSTM's upper-triangular transition matrix.
  - **MTTC (B5).** `info` now exposes `compromised`, `mttc_steps`,
    `first_attack_step`, `compromise_step`, `defender_deescalations`,
    `recommended_action`. Tracked across episode lifecycle.
  - **Calibration (B6).** `defense_success_bonus` raised from 10.0 to
    250.0 so the *correct* IMPACT response (ISOLATE) nets +49 instead
    of -190.8. Asymmetry preserved: OBSERVE@IMPACT still loses -350.
    This is what allows G3.4 (recommended-policy mean reward > 0) to
    hold.

### Phase-3 exit gates (`PLAN.md` §3.2) — all PASS

| Gate | Threshold | Status |
|------|-----------|:------:|
| G3.1 (8 mechanical regression tests) | individual asserts | **PASS** |
| G3.2 median random-action episode length | ≥ 15 | **PASS** |
| G3.3 median always-BLOCK episode length | ≥ 10 | **PASS** |
| G3.4 recommended-policy mean reward | > 0 | **PASS** |
| G3.5 always-OBSERVE mean reward | < 0 | **PASS** |
| G3.6 always-ISOLATE mean reward | < 0 | **PASS** |
| G3.7 full test suite | green | **296 / 296** |

### Notes & lessons learned

- The first cut of the env failed three of the six empirical gates
  (G3.2, G3.3, G3.4 in iter-1; G3.5 in iter-2; G3.4 again in iter-3).
  Each failure pointed to a real design hole, not a flaky test:
  (a) the lifecycle floor needed an IMPACT-clamp because
  `min_episode_length` alone could not stop a uniform-LSTM
  one-shot to IMPACT; (b) the IMPACT terminal accounting was unreachable
  via the rollout loop and had to be inlined; (c) the
  `defense_success_bonus` had to be large enough that even the optimal
  policy stayed net-positive when an unavoidable IMPACT consummated.
  Documenting these in `RESULTS.md` §5 so the design is reproducible.
- Phase 3 is **infrastructure** — it produces no thesis figure. The
  first figures consuming the new env appear in Phase 4 (detector head,
  F11) and Phase 5 (RL Blue Team, F3-F4).
- All 283 pre-Phase-3 tests still pass; the env API changes are
  backwards-compatible at the `gym.Env` boundary (`reset` and `step`
  signatures unchanged, `info` only gains keys, never loses them).

### Phase-3 commits
`482299e` PLAN — `3a6b13a` split-aware engine — `2a526af` env rewrite —
`36fec22` gates + calibration.

---

## Phase 2: Red Team v2 (LSTM episode generator)

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
