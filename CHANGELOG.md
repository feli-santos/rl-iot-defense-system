# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the
research artefact versions recorded in `CITATION.cff`.

## [Unreleased]

## [0.8.1] — 2026-07-10

Elsevier _Internet of Things_ (ISSN 2542-6605) journal submission package.

### Added
- `paper/` submission package: a condensed, self-contained journal version of
  the thesis built on the Elsevier `elsarticle` double-column template
  (`paper/manuscript.tex`, ~9 pages, `\bibliographystyle{elsarticle-num}`),
  reusing the thesis' macro-driven numbers (`paper/numbers.tex`) and vector
  figures (`paper/figs/`). New title: *Partially Observable Kill-Chain Defense:
  Deep Reinforcement Learning for Autonomous IoT Security*.
- `paper/Makefile` (targets `build`, `draft`, `numbers`, `wordcount`, `verify`,
  `clean`), building the paper in the same Podman/TeXLive container as the
  thesis.
- Guide-mandated side files: `paper/highlights.tex` (5 bullets ≤85 chars),
  `paper/cover-letter.md`, `paper/README.md` (pre-submission checklist), and
  `paper/declarations/` (CRediT, competing-interest `.md` + separately-uploaded
  `.docx`, funding, generative-AI, and data-availability statements).
- `[dataset]` reference `CICIoT2023Dataset` in `paper/refs.bib` for the
  Elsevier Research-Data Option-C data-availability statement (raw CICIoT2023
  not redistributed under the CIC license; code + hash-chain manifests deposited
  via the public GitHub release and a Zenodo DOI).

### Notes
- **Open pre-submission blocker:** the Zenodo DOI placeholder
  `10.5281/zenodo.XXXXXXX` (Data availability) must be replaced with the minted
  DOI, then `make -C paper build` re-run, before submission.

## [0.8.0] — 2026-07-06

Post-`v0.7.0` consolidation around the A2C on-policy result and the canonical
deterministic-5M numbers.

### Added
- F10 (aggressiveness) and F17 (evasion) robustness sweeps extended to all
  three DRL agents (PPO/A2C/DQN), loading the fixed det-5M α=0.4 checkpoints
  (no retraining).
- OOD train/eval parity guard and canonical eval defaults in the eval scripts.

### Changed
- Consolidated the A2C result paths and re-ran the matched-contract RF eval;
  regenerated the F3 learning-curve and F4 per-stage action-distribution
  figures as a 3-row A2C/PPO/DQN grid (stale A2C series purged).
- Synced headline numbers and the test count to canonical post-A2C truth
  (**462 tests**); removed math notation from the resumo/abstract prose.

### Fixed
- Isolated the environment RNG, raise on invalid actions, reset the session
  cursor on `reset()`, and made the latency file write exception-safe.
- Deterministic dataset labels and pandas-3.0-safe `fillna`; dropped dead
  branches. Corrected stale docstrings, the eval-parity fallback, the
  reproducibility-smoke target, and canonical test wiring.

## [0.7.0] — 2026-07-06

Open-source hardening pass plus a post-hoc environment correctness fix, on top
of the deterministic-5M A2C consolidation.

### Added
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- GitHub Actions CI (`.github/workflows/ci.yml`): Python 3.9, `make lint` +
  `pytest -q --cov`.
- Evasive-persistence (post-detection hardening) attacker coupling used by the
  F17 sweep.

### Changed
- Rewrote `requirements.txt` as a minimal direct-dependency list (10 runtime
  packages) and added `requirements-dev.txt` (lint, format, test, pre-commit);
  `Makefile` `install-dev` now installs from it. Fixed the dev toolchain to be
  venv-resolved (lint/format, pre-commit hook, isort cleanup).
- Removed the unused `mlflow` integration and dead `src/training/` package,
  dropping ~100 transitive dependencies (docker, fastapi, flask, sqlalchemy,
  alembic, opentelemetry-*, …).
- Removed stale `MLP` detector references from `README.md` and
  `docs/ARCHITECTURE.md`; repointed checkpoints to `runs/redesign_5M_det`.
- Fixed the A2C `n_steps` clamp that silently trained at 5 instead of 256, and
  regenerated all figures + manifests after the retrain.

### Fixed
- **Environment reward-accounting edge case:** `AdversarialIoTEnv.step()`
  previously let an episode escape terminal IMPACT accounting when the attacker
  reached `IMPACT` on the *exact* step that exhausted the horizon
  (`impact_is_terminal=False`). Such a tail-end compromise returned
  `truncated=True`/`terminated=False` with a stale outcome and **no impact
  penalty**. Terminal accounting now also fires at the horizon boundary. Added
  three regression tests (`TestImpactAtTruncationBoundary`); a 20k-episode
  measurement puts the boundary case at ≈0.16 % (random policy), so canonical
  numbers are unchanged and **no re-run is required**.
- Digit-free LaTeX macro names for A2C (`ATwoC`); removed dead `BestAgentName`.

### Removed
- Dead code: `src/training/training_manager.py`, `src/training/__init__.py`.
- Orphan/personal artifacts: `scripts/review/redesign_smoke.py`,
  `scripts/run_alpha_sweep_5M_det.sh`, `tex/cover_letter.md`, MLP-era orphan
  figures, and stale dataset images.

## [0.6.7] — 2026-06-30

### Fixed
- Detector macro-F1 split error; made the affected thesis numbers macro-driven.

## [0.6.6] — 2026-06-30

### Changed
- Phase-1 journal framing tightened; thesis prose/figure polish.

## [0.6.5] — 2026-06-28

### Changed
- Journal-quality prose revisions across the thesis chapters.

## [0.6.4] — 2026-06-25

### Changed
- Journal-revision pass: citation integrity, threats-to-validity section,
  honest framing, figure refinements, and cover letter.

## [0.6.3] — 2026-06-25

### Fixed
- Incorporated all three-review fixes into the thesis.

## [0.6.2] — 2026-06-24

### Fixed
- Resolved all 7 reviewer defects; OOD eval re-run at 10 seeds × 300 episodes;
  updated reported ranges; added RESTRAIN / RL-IoTIDS references; documented the
  durable-checkpoint note.

## [0.6.1] — 2026-06-24

### Changed
- Updates to `Makefile`, docs, and thesis files following the redesign.

## [0.6.0] — 2026-06-23

Redesign for genuine partial observability (the POMDP central thesis).

### Added
- Session-coherent feature sampling, adjacent-stage observation aliasing at
  configurable rate α, proximity-coupled escalation (replacing the finite
  intrusion budget), outcome-only reward mode, and no post-transition feature
  leakage. Comprehensive tests for the redesigned environment.
- Alpha-curve and reward-coupling ablation scripts; redesigned eval pipeline
  and Makefile targets.

### Changed
- Aligned tooling to Python 3.9; consolidated pytest config; rewrote README,
  AGENTS, and architecture docs for the redesign; regenerated thesis figures
  under the locked outcome contract.
- Replaced Phase-N identifiers with semantic stage names throughout.

### Removed
- RecurrentPPO (LSTM belief-state) agent; retired the finite intrusion budget
  and dead `generator_path`; removed dead code and legacy LSTM remnants.

## [0.5.0] — 2026-06-17

Tug-of-war attacker dynamics.

### Added
- Reactive **tug-of-war** kill-chain attacker: signed proportionality rule
  `d = action − rec(stage)` — proportionate (`d==0`) de-escalates
  (`p_down=0.90`, ISOLATE 0.98), under-force escalates (`p_up=0.90`),
  over-force holds; BENIGN autonomous multi-rung onset (`p_onset=0.35`,
  `p_onset_access=0.10`). `prevention_rate` promoted to primary KPI.

### Changed
- `THROTTLE` action renamed `RESTRICT` (index unchanged). Re-ran all
  ablation/benchmark/training/detector results under the new dynamics; rewrote
  thesis prose and docs to match. Suite updated to 428 tests.

### Removed
- CNN1D detector (MLP + RandomForest only at this point).

## [0.4.0] — 2026-06-05

Prevention pivot.

### Added
- Finite attacker budget (prevention model), evasion-before-commit reactive
  attacker, and outcome-only reward mode. Phase-D ablation suite (F9/F10/F12/
  F15/F16) and F17 evasion-reactive sweep.
- Machine-readable `feature_provenance.json` (29-col provenance).

### Changed
- Pivoted thesis prose from the LSTM red-team to a finite-budget Markov
  attacker + prevention spine; centralized PNG→PDF figure export with F-named
  figure assets; drove benchmark/latency/benign-FPR tables from generated
  macros. Consolidated docs to AGENTS/README + `docs/{ARCHITECTURE,ENVIRONMENT,
  RESULTS,STATUS}.md`.

### Removed
- LSTM red-team modules and training pipeline; dangling G1–G5 tooling; dead
  `tex/figs` JSON sidecars.

## [0.3.0] — 2026-06-04

Pre-pivot baseline: thesis revision complete (86 pp, 459 tests). Snapshot
before the finite-budget / Markov-attacker re-centering.

### Added
- First-order Markov attacker (replacing the LSTM red-team); JSON→`.tex`
  generator with anti-drift macros + tables; freshness gate, results index, and
  resume/skip support in the blue-team sweep driver.
- Env ablations: Lagrangian FPR penalty, non-monotonic attacker retreat, RF
  tree-count sweep, and detector-in-obs.

### Changed
- Full thesis prose rewrite to canonical 10-seed / 300-episode numbers; migrated
  to `\citeonline` (abnTeX2cite); locked Phase-3 contract decisions.

## [0.2.0] — 2026-05-25

FEEC CCPG 001-2015 (abnTeX2) template migration.

### Changed
- Migrated the dissertation to the FEEC CCPG 001-2015 (abnTeX2) template,
  resolving the Overleaf compile-loop/timeout (font/math fixes, PNG→PDF,
  `\clearpage`). Phase-5 full manuscript rewrite (PPO-primary 10-seed, 81 %
  oracle, 141× latency); Phase-6 QA release.

### Removed
- Deprecated/dead code, unused one-shot scripts, and Copilot instructions.

## [0.1.0] — 2026-05-12

First thesis release: *Adversarial Reinforcement Learning for Kill-Chain-Aware
IoT Defense* (MSc Thesis, FEEC/UNICAMP, 2026).

### Added
- CICIoT2023-backed pipeline: 442,237 rows, 29 features, 5-stage kill-chain
  abstraction with immutable stratified train/val/test/OOD split-index
  manifests (SHA-256 hash chain). Figures F0a/F0b.
- Eight-phase audit chain (dataset → red team → environment → detector →
  blue-team training → benchmark → ablations), closed via the ten-step
  thesis-mentor walkthrough in `docs/mentor_review/`.
- `--impact-is-terminal` / `--reward-overrides` passthrough and the R1
  smoke-reproducibility harness.

[Unreleased]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.8.1...HEAD
[0.8.1]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.7...v0.7.0
[0.6.7]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.6...v0.6.7
[0.6.6]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.5...v0.6.6
[0.6.5]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.4...v0.6.5
[0.6.4]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/feli-santos/rl-iot-defense-system/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/feli-santos/rl-iot-defense-system/releases/tag/v0.1.0
