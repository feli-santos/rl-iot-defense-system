# AGENTS.md

Adversarial-RL IoT defense research codebase (MSc thesis). Python, CPU-first.
`src/` is importable library code; `scripts/` holds phase-pinned runners;
`main.py` is a thin CLI for the core train pipeline. Use the **Makefile as the
source of truth for commands**.

## Commands

```bash
make help              # canonical target list (always trust this over README)
make test              # pytest -q  (this is what CI runs)
make test-cov          # pytest with coverage
make lint              # ruff check + black --check on src tests scripts main.py
make format            # black + ruff --fix + isort (run before committing)
pytest tests/test_adversarial_env.py -q          # single file
pytest tests/test_adversarial_env.py::TestX::test_y   # single test
```

Order that matters for commits/CI: **format -> lint -> test**. CI (`.github/workflows/ci.yml`)
runs `ruff check`, `black --check`, then `pytest -q --cov` on Python 3.10 and 3.11.
Pre-commit hooks (ruff, ruff-format, black, isort) run on commit; `make install-dev` installs them.

## Architecture (phase = chapter)

Adversarial loop: an LSTM **Red Team** (`src/generator/`) emits kill-chain stage
sequences -> `RealisationEngine` (`src/utils/`) samples a real CICIoT2023 feature
row for that stage -> `AdversarialIoTEnv` (`src/environment/`, Gymnasium API, 29-feat
obs, 5 actions OBSERVE/LOG/THROTTLE/BLOCK/ISOLATE) -> **Blue Team** SB3 agent
(DQN/PPO/A2C via `src/algorithms/`) acts -> kill-chain reward.

`src/` modules map to phases: `generator/`=Phase2, `environment/`=Phase3,
`detector/`=Phase4, `blue_team/`=Phase5, `benchmark/`=Phase6, ablations in
`scripts/ablation/`=Phase7. Each `scripts/<area>/` subdir is owned by one phase.

Kill-chain stages (5): `0 BENIGN, 1 RECON, 2 ACCESS, 3 MANEUVER, 4 IMPACT`
(canonical map in `tests/conftest.py`).

## Conventions / gotchas

- **`config.yml` is the single source of hyperparameters.** `main.py` reads it and
  passes values into dataclass configs; there are many `.get(key, default)` fallbacks,
  so a missing key silently uses the code default — grep both places when changing a param.
- **Reproducibility = hash chains.** Every thesis figure under `docs/results/<NN>_*/`
  ships a sibling `manifest.json` (git SHA + input/output SHA-256). A figure without a
  reconciling manifest is not "defense-ready". Verify with
  `python -m scripts.benchmark.run_test_eval --verify-manifests` and
  `python -m scripts.ablation.close_phase7 --verify-manifests`.
- **Gitignored / machine-local:** `data/`, `runs/`, `artifacts/`, `mlruns/`, `results/`.
  Raw CICIoT2023 CSVs are NOT in the repo (CIC license); they go in `data/raw/ciciot2023/`.
- **Generator path auto-detection:** `main.py:get_generator_path` finds the latest
  timestamped run under `artifacts/generator/`. Pass `--generator-path` to pin one.
- Lint tolerates scientific naming (`X_train`, `y_test`) and SB3/Gym default-arg patterns;
  `data/ mlruns/ artifacts/ runs/ results/ notebooks/` are excluded from all tooling.
- `notebooks/` is exploratory and NOT on the thesis reproduction path.

## Testing notes

- Default run is **synthetic-only**. Real-data tests are guarded with
  `@pytest.mark.skipif(not (data/processed/ciciot2023/...).exists())` — they auto-skip
  on a fresh checkout. Markers: `slow`, `integration`, `gpu` (registered in `pyproject.toml`).
- Pipeline smoke targets are the canary for env/config drift; run them after touching
  the env, reward, or training code: `make blue-team-smoke` (~20s), `make benchmark-smoke`,
  `make ablation-ood-smoke`. If a smoke fails on a clean checkout, treat it as a bug.
- Full sweeps are expensive and CPU-bound (hours): `make blue-team`, `make benchmark`,
  `make ablation`. Don't run these to "verify" a code change — use smoke targets.

## Thesis (separate toolchain)

LaTeX under `tex/` builds via Podman or Docker: `make thesis` (wraps `bash tex/build.sh`).
Podman is auto-detected and preferred; Docker is the fallback. Main file is `tex/principal.tex`
(abnTeX2/FEEC template), NOT `thesis.tex`. See `memory-bank/activeContext.md` for the build
details and history.

## More context

`docs/architecture.md`, `docs/environment.md`, `docs/reward-shaping.md`,
`docs/reproducibility.md`, `docs/decisions.md`; per-phase `docs/results/<NN>_*/PLAN.md`
+ `RESULTS.md`.

## Thesis Revision Progress (active)

Full plan: `docs/review/REVISION_PLAN.md` (read it before touching the thesis).

**Locked decisions:** 10 seeds (`{0..9}`); primary contract `impact_is_terminal=False`;
all 3 new ablations (stage_pred-in-training, RF tree-count sweep, non-monotonic attacker);
full Lagrangian FPR penalty (`-beta*FPR`); title revised (no "Proactive Prediction").

**Rules:** pytest green every commit; thesis numbers from JSON macros only (never
hand-typed); NEVER write "Phase N" in thesis prose — use semantic stage names
(Red-Team LSTM / Adversarial Environment / Stage Detector / Blue-Team Training /
Held-Out Benchmark / Ablation & Robustness stage).

**Baseline (pre-revision):** HEAD=`b2644fc`, pytest=445 passed, 2 warnings.

**Current step:** Phase 4 complete — all pipeline stages done, manifests verified, thesis builds (81 pages)

**Phase 3 decisions (locked):**
- Primary contract: `impact_is_terminal=False` for training + benchmark
- Baseline case-study: `impact_is_terminal=True` retained as reward-hacking case study
- n=300 episodes for ALL policies (fixed BENCHMARK_N_DET_EPISODES=300, ABLATION_OOD_N_DET_EPISODES=300)
- p_de_esc=0.6 default carried through

| Task | Status | Commit SHA | Notes |
|---|---|---|---|
| 0.1 plan file | ✅ | 3f97a22 | written to docs/review/REVISION_PLAN.md |
| 0.2 AGENTS.md ledger | ✅ | 3f97a22 | this section added |
| 0.3 baseline | ✅ | b2644fc | 445 passed, 2 warnings |
| 0.4 commit | ✅ | 3f97a22 | plan + 4 review files committed |
| 1.1 SUB→subsection | ✅ | 315f8c2 | 25 occ fixed across 3 files |
| 1.2 rm dup inputenc | ✅ | 315f8c2 | principal.tex:15 removed |
| 1.3 thesis.pdf→principal.pdf | ✅ | 315f8c2 | apendice.tex 80,86 fixed |
| 1.4 thesis→dissertation | ✅ | 315f8c2 | background.tex + 2 stray fixes |
| 1.5 ToC depth | ✅ | — | SUB fix restores hierarchy; no 0.0.0.0 nesting remains |
| 1.6 tese.bib | ✅ | fe43557 | titles wrapped, +6 refs, IoTWarden preprint note |
| 1.7 make thesis | ✅ | — | Podman auto-detect added to tex/build.sh; compile verification TBD after Phase 4 |
| 2.1 stage_pred plumbing | ✅ | b0156e8 | env obs + classifier injection, 3 tests |
| 2.2 RF tree-count sweep | ✅ | 721c777 | --n-estimators CLI, config pass-through, 2 tests |
| 2.3 non-monotonic attacker | ✅ | 0a64e07 | retreat_prob param, 2 tests + flaky test fix |
| 2.4 Lagrangian FPR penalty | ✅ | 16fc1ea | episode accumulator + terminal -beta*FPR, 2 tests |
| 2.5 JSON→.tex generator | ✅ | e358e07 | anti-drift macros + tables, 5 tests |
| 2.6 Makefile updates | ✅ | f6766ce | 10 seeds, sync-figures, stale-check, render-tables, full reproduce-thesis |
| 3.1 primary=False contract | ✅ | — | decision locked in AGENTS.md |
| 3.2 n=300 all policies | ✅ | f6766ce | BENCHMARK_N_DET_EPISODES=300, ABLATION_OOD_N_DET_EPISODES=300 |
| 3.3 p_de_esc=0.6 confirmed | ✅ | — | default preserved across all configs |
| 4.1 backup + clean | ✅ | — | runs/ + artifacts/ + results/ + mlruns/ → .archive/ |
| 4.2 dataset | ✅ | 23ee841 | build-split-indices + plot-dataset (seed=0, F0 regenerated) |
| 4.3 red-team | ✅ | 23ee841 | LSTM generator trained, all G1-G4 passed |
| 4.4 detector | ✅ | 23ee841 | MLP+RF+CNN1D trained, RF macro-F1 0.9077, all G4 passed |
| 4.5 blue-team@False | ✅ | — | 30 runs complete (3 algos × 10 seeds, impact_is_terminal=false) |
| 4.6 benchmark | ✅ | — | eval + figures F5/F6/F7/F8 done; best=recommended_action (1684.8); a2c best RL (1336.6) |
| 4.7 ablation | ✅ | 52a67c3 | OOD+reward+aggressiveness+pareto done; 7 PASS / 2 FAIL-WITH-FINDING; thesis builds (81pp) |
| **Bug fixes (pre-run):** argparse dest mismatches in 10 scripts + render_tables FPR nesting | ✅ | — | `args.blue_team_*` → `args.phase5_*`, `args.benchmark_*` → `args.phase6_*`; benign_fpr.json nested structure |
| **Next:** Commit Phase 4 results + update thesis prose with ablation findings | ☐ | |
