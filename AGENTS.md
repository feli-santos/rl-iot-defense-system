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
The dev lint/format tools (ruff, black, pre-commit) are **not** in the project `.venv` — they are
system-level only. Run `make install-dev` to install them into the dev environment.

## Architecture (phase = chapter)

Adversarial loop: a fixed 5x5 first-order **Markov attacker** (`MarkovAttacker`,
`src/generator/`) walks the kill chain under a finite intrusion budget -> emits a
stage -> `RealisationEngine` (`src/utils/`) samples a real CICIoT2023 feature
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
- **Reproducibility = hash chains.** Every thesis figure under `docs/results/<area>/`
  ships a sibling `manifest.json` (git SHA + input/output SHA-256). A figure without a
  reconciling manifest is not "defense-ready". Verify the chain end-to-end via the
  reproducibility-smoke harness: `python -m scripts.reproducibility_smoke`
  (`--strict` to exit 1 on any hash miss).
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

LaTeX under `tex/` builds with **Podman** (Docker is a fallback): `make thesis` (wraps
`bash tex/build.sh`, which auto-detects and prefers Podman). Main file is `tex/main.tex`
(abnTeX2/FEEC template), **NOT** `thesis.tex`. Output is `tex/main.pdf`.

**Numbers in the thesis are macro-driven, never hand-typed.** Canonical experiment JSONs
live under `docs/results/<area>/` (e.g. `benchmark/F5_summary.json`). `make render-tables`
(`scripts/thesis/render_tables.py`) regenerates `tex/generated/{numbers,tables}.tex` from them;
`tex/generated/` is **gitignored** (rebuild before any thesis edit/build). `tex/preamble.tex`
must `\input{generated/numbers}` and `\input{generated/tables}` for the macros to resolve in
the abstract/resumo — without it you get "Undefined control sequence". `docs/results/test_count.json`
feeds the `\NumTests` macro; bump it when the suite count changes. `make verify-fresh` (a CI gate)
fails if any derived artifact is older than its source JSON; `make verify-fresh-fix` regenerates.

**LaTeX gotcha:** macro names cannot contain digits — a digit ends the control-sequence name.
Use spelled-out forms (`\FPRatc` not `\FPRa2c`, `\FnineStructuralReward` not `\F9...`).
`render_tables.py` already maps digits to letters when emitting macro names; keep it that way.

**Thesis prose rule:** NEVER write "Phase N" in `tex/` prose — use semantic stage names
(Markov Attacker / Adversarial Environment / Stage Detector / Blue-Team Training /
Held-Out Benchmark / Ablation & Robustness). "Phase N" is fine in dev docs and `docs/results/`.

## More context

`docs/ARCHITECTURE.md` (module map + adversarial loop + config flow),
`docs/ENVIRONMENT.md` (obs/actions/reward/budget mechanics), `docs/RESULTS.md`
(budget=40 headline + gate scoreboard), `docs/STATUS.md` (live status, locked
decisions, caveat dispositions, commit journal); per-area
`docs/results/<area>/PLAN.md` + `RESULTS.md`. Dataset provenance:
`docs/dataset_card.md`, `docs/kill-chain-mapping.md`.

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
- **Canonical headline numbers** (budget=40, from `benchmark/F5_summary.json`): best deployable
  RL = **PPO +1034.7** (CI [+998.1, +1069.8]; DQN +1028.9, A2C +973.1); oracle ceiling
  `recommended_action` **+1393.8** (CI [+1366.9, +1420.6], 74.2% capture); RF-Acting +1323.0 @
  13.692 ms p50 (latency-ratio ~142.8× vs best-agent 0.096 ms); benign FPR DQN 7.5% / PPO 8.7% /
  A2C 7.7%. With the finite attacker budget, `compromise_rate` now varies by policy (PPO 0.68 /
  always_block 0.36 / oracle 0.47) instead of the pre-budget 1.0-for-everything. The reward
  ablation's structural mit-rate 0.867 is an F9 *strand* result, separate from the F5 benchmark.
  Test suite: **432 passed**, 1 warning.

**Status:** thesis revision complete — prose rewritten, builds clean (86 pages, 0 LaTeX
errors). `make lint` exits non-zero on ~21 pre-existing UP031/F401 findings; that is the
known baseline, not a regression.
