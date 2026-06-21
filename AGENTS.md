# AGENTS.md

Adversarial-RL IoT defense research codebase (MSc thesis). Python, CPU-first.
`src/` is importable library code; `scripts/` holds runners; `main.py` is a
thin CLI. Use the **Makefile as the source of truth for commands**.

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

Order that matters for commits/CI: **format → lint → test**. CI runs `ruff
check`, `black --check`, then `pytest -q --cov` on Python 3.10 and 3.11.
Pre-commit hooks (ruff, ruff-format, black, isort) run on commit; `make
install-dev` installs them. The dev lint/format tools (ruff, black, pre-commit)
are **not** in the project `.venv` — they are system-level only.

## Architecture

Adversarial loop: a reactive **tug-of-war attacker** (`MarkovAttacker`,
`src/generator/`) walks the kill chain. Escalation pressure is **coupled to the
attacker's proximity to the impact stage** (no fixed intrusion budget). It
reacts to the defender's force via a signed proportionality rule on
`d = action - recommended(stage)`: proportionate (`d==0`) de-escalates one
stage (p_down=0.90; ISOLATE 0.98), under-forced (`d<=-1`) advances (p_up=0.90),
over-forced (`d>=1`) holds. The attacker emits a stage → `RealizationEngine`
(`src/utils/`) samples a **session-coherent, aliasing-aware** CICIoT2023
feature row for that stage → `AdversarialIoTEnv` (`src/environment/`,
Gymnasium API, 290-dim windowed obs, 5 actions) → **Blue Team** SB3 agent
(DQN/PPO/A2C via `src/algorithms/`) acts → kill-chain reward. The defender
**never observes the true stage** (POMDP = central thesis).

Kill-chain stages (5): `0 BENIGN, 1 RECON, 2 ACCESS, 3 MANEUVER, 4 IMPACT`
(canonical map in `tests/conftest.py`).

`src/` modules: `generator/`=attacker, `environment/`=Gymnasium env,
`detector/`=supervised stage detector, `blue_team/`=env factory + run config,
`benchmark/`=baselines + eval runner, `algorithms/`=SB3 wrappers,
`utils/`=dataset + realization engine.

## Conventions / gotchas

- **`config.yml` is the single source of hyperparameters.** `main.py` reads it
  and passes values into dataclass configs; there are many `.get(key, default)`
  fallbacks, so a missing key silently uses the code default — grep both places
  when changing a param.
- **Reproducibility = hash chains.** Every thesis figure under
  `docs/results/<area>/` ships a sibling `manifest.json` (git SHA + input/output
  SHA-256). Verify via `python -m scripts.reproducibility_smoke` (`--strict` to
  exit 1 on any hash miss).
- **Gitignored / machine-local:** `data/`, `runs/`, `artifacts/`, `mlruns/`,
  `results/`. Raw CICIoT2023 CSVs are NOT in the repo (CIC license); they go in
  `data/raw/ciciot2023/`.
- Lint tolerates scientific naming (`X_train`, `y_test`) and SB3/Gym
  default-arg patterns; `data/ mlruns/ artifacts/ runs/ results/ notebooks/`
  are excluded from all tooling.
- `notebooks/` is exploratory and NOT on the thesis reproduction path.

## Testing notes

- Default run is **synthetic-only**. Real-data tests are guarded with
  `@pytest.mark.skipif(not (data/processed/ciciot2023/...).exists())` — they
  auto-skip on a fresh checkout. Markers: `slow`, `integration`, `gpu`
  (registered in `pyproject.toml`).
- Pipeline smoke targets are the canary for env/config drift; run them after
  touching the env, reward, or training code: `make blue-team-smoke` (~20s),
  `make benchmark-smoke`, `make ablation-ood-smoke`. If a smoke fails on a
  clean checkout, treat it as a bug.
- Full sweeps are expensive and CPU-bound (hours): `make blue-team`, `make
  benchmark`, `make ablation`. Don't run these to "verify" a code change — use
  smoke targets.

## Thesis (separate toolchain)

LaTeX under `tex/` builds with **Podman** (Docker is a fallback): `make thesis`
(wraps `bash tex/build.sh`). Main file is `tex/main.tex` (abnTeX2/FEEC
template), **NOT** `thesis.tex`. Output is `tex/main.pdf`.

**Numbers in the thesis are macro-driven, never hand-typed.** Canonical
experiment JSONs live under `docs/results/<area>/`. `make render-tables`
(`scripts/thesis/render_tables.py`) regenerates `tex/generated/{numbers,tables}.tex`
from them; `tex/generated/` is **gitignored** (rebuild before any thesis
edit/build). `docs/results/test_count.json` feeds the `\NumTests` macro; bump
it when the suite count changes.

**LaTeX gotcha:** macro names cannot contain digits — a digit ends the
control-sequence name. Use spelled-out forms (`\FnineStructuralReward` not
`\F9...`). `render_tables.py` already maps digits to letters when emitting
macro names; keep it that way.

**Thesis prose rule:** NEVER write "Phase N" in `tex/` prose — use semantic
stage names (Markov Attacker / Adversarial Environment / Stage Detector /
Blue-Team Training / Held-Out Benchmark / Ablation & Robustness).

## Locked experiment decisions

These are fixed contracts; do not silently change them.

- **Primary reward contract: `reward_mode=outcome`** (sparse, outcome-only) for
  training + benchmark. The `coupled` (proportionality-shaped) reward is used
  only in the reward-coupling ablation. `impact_is_terminal=False` for training
  + benchmark; `True` retained only as a reward-mis-specification case study.
- **10 seeds `{0..9}`** for DRL; baselines/oracle run 1 seed. **n=300 episodes
  for ALL policies.** Tug-of-war probabilities: `p_down=0.90` (ISOLATE 0.98) /
  `p_up=0.90`; BENIGN onset `p_onset=0.35`, `p_onset_access=0.10`.
  Proximity-coupled escalation: `sigma_min=0.4`,
  `p_up_eff = p_up * (sigma_min + (1-sigma_min) * lambda)`, `lambda = stage/4`.
  Prevention bonus `b_prevent = +50`.
- **Canonical headline numbers** (from `docs/results/ablation/Falpha_summary.json`):
  PPO flat +131.3→+117.4 vs tuned RF +134.8→+67.7 across α=0.0/0.2/0.4/0.6;
  tie at α=0 (overlapping CIs), disjoint CIs from α=0.4. Oracle ceiling +194.8.
  Reward-coupling: coupled gap −63.1 (DQN +226.2 best), outcome gap −43.1 (PPO
  best; DQN collapses to −8.6). OOD: PPO prevents 0.37–0.65 vs RF 0.00–0.18 on
  all 10 held-out zero-day classes. Test suite: **473 passed**.
