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
obs, 5 actions OBSERVE/ALERT/ISOLATE/RATE-LIMIT/BLOCK) -> **Blue Team** SB3 agent
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

LaTeX under `tex/` builds via Docker only: `make thesis` (wraps `bash tex/build.sh`).
Main file is `tex/principal.tex` (abnTeX2/FEEC template), NOT `thesis.tex`. See
`memory-bank/activeContext.md` for the build details and history.

## More context

`docs/architecture.md`, `docs/environment.md`, `docs/reward-shaping.md`,
`docs/reproducibility.md`, `docs/decisions.md`; per-phase `docs/results/<NN>_*/PLAN.md`
+ `RESULTS.md`.
