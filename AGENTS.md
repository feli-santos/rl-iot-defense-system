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
(abnTeX2/FEEC template), **NOT** `thesis.tex`. Output is `tex/principal.pdf` (committed);
`tex/tese.pdf` is a stray untracked duplicate — ignore it. See `memory-bank/activeContext.md`
for build history.

**Numbers in the thesis are macro-driven, never hand-typed.** Canonical experiment JSONs
live under `docs/results/<NN>_*/` (e.g. `06_benchmark/F5_summary.json`). `make render-tables`
(`scripts/thesis/render_tables.py`) regenerates `tex/generated/{numbers,tables}.tex` from them;
`tex/generated/` is **gitignored** (rebuild before any thesis edit/build). `tex/preambulo.tex`
must `\input{generated/numbers}` and `\input{generated/tables}` for the macros to resolve in
the abstract/resumo — without it you get "Undefined control sequence". `docs/results/test_count.json`
feeds the `\NumTests` macro; bump it when the suite count changes. `make verify-fresh` (a CI gate)
fails if any derived artifact is older than its source JSON; `make verify-fresh-fix` regenerates.

**LaTeX gotcha:** macro names cannot contain digits — a digit ends the control-sequence name.
Use spelled-out forms (`\FPRatc` not `\FPRa2c`, `\FnineStructuralReward` not `\F9...`).
`render_tables.py` already maps digits to letters when emitting macro names; keep it that way.

**Thesis prose rule:** NEVER write "Phase N" in `tex/` prose — use semantic stage names
(Red-Team LSTM / Adversarial Environment / Stage Detector / Blue-Team Training /
Held-Out Benchmark / Ablation & Robustness). "Phase N" is fine in dev docs and `docs/results/`.

## More context

`docs/architecture.md`, `docs/environment.md`, `docs/reward-shaping.md`,
`docs/reproducibility.md`, `docs/decisions.md`; per-phase `docs/results/<NN>_*/PLAN.md`
+ `RESULTS.md`. Full thesis-revision plan: `docs/review/REVISION_PLAN.md`.

## Locked experiment decisions

These are fixed contracts; do not silently change them.

- **Primary reward contract: `impact_is_terminal=False`** for training + benchmark.
  `impact_is_terminal=True` is retained only as a reward-mis-specification case study.
  Default in code is `True` (`run_config.py`); the eval/training scripts pass the flag
  explicitly — the eval manifest does not record it, so correctness relies on the flag,
  not the manifest.
- **10 seeds `{0..9}`** for DRL; baselines/oracle run 1 seed. **n=300 episodes for ALL
  policies** (`BENCHMARK_N_DET_EPISODES=300`, `ABLATION_OOD_N_DET_EPISODES=300`).
  `p_de_esc=0.6` default. `make reproduce-thesis` overrides `BLUE_TEAM_IMPACT_TERM=false`.
- **Canonical headline numbers** (from `06_benchmark/F5_summary.json`): best deployable RL =
  **A2C +1336.6**; oracle ceiling +1684.8 (79.3% capture); RF-Acting +1516.0 @ 13.83 ms p50
  (~146× slower than A2C); benign FPR DQN 6.1% / PPO 10.2% / A2C 11.5%; `compromise_rate=1.0`
  for every policy (reactive mitigation only). The reward ablation's mit-rate 0.840 is a
  PPO-only n=30 *probe* — it does NOT replicate at benchmark scale (A2C mit-rate 0.317).
  Test suite: **459 passed**, 2 warnings (pre-revision baseline was 445).

**Status:** thesis revision complete — prose rewritten, builds clean (86 pages, 0 LaTeX
errors). `make lint` exits non-zero on ~21 pre-existing UP031/F401 findings; that is the
known baseline, not a regression.
