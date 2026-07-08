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
check`, `black --check`, then `pytest -q --cov` on Python 3.9.
Pre-commit hooks (ruff, ruff-format, black, isort) run on commit; `make
install-dev` installs them into `.venv` via `requirements-dev.txt` (versions
pinned to match `.pre-commit-config.yaml`). The `lint`/`format` Makefile
targets invoke the tools as `$(PYTHON) -m ruff|black|isort` so they always
resolve through `.venv` — no system-level install is needed.

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

**Canonical blue-team checkpoints (current contract): `runs/redesign_5M_det/alpha_<NN>/<algo>/seed_<n>/best_model.zip`** (`alpha_00/02/04/06/08/10`; headline = `alpha_04`,
matching HeadlineAlpha=0.4). The legacy `runs/blue_team/` path is **pre-redesign
and absent on fresh checkouts** — do NOT point eval at it. The OOD eval target
`make ablation-ood-eval` is wired to `ABLATION_OOD_BLUE_TEAM_RUNS ?=
runs/redesign_5M_det/alpha_04` and passes the partial-observability flags
(`--aliasing-rate 0.4 --session-coherent --no-post-transition-leak
--proximity-coupled --proximity-min-escalation 0.4`) via
`ABLATION_OOD_POMDP_FLAGS`; these flags are **mandatory** — omitting them makes
`run_ood_eval.py` fall back to a fully-observable MDP (aliasing=0, no session
coherence) that mismatches the trained checkpoints and trips the train/eval
parity assertion.

## Conventions / gotchas

- **`config.yml` `dataset:` section is live** (consumed by `main.py
  process-data`); all RL/env/reward sections are **documentation-only** (the
  training pipeline uses `DEFAULT_HPARAMS` in `scripts/blue_team/train_agent.py`
  + Makefile overrides + dataclass defaults in `run_config.py`). There are many
  `.get(key, default)` fallbacks, so a missing key silently uses the code
  default — grep both places when changing a param.
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
- **Canonical headline numbers (deterministic-5M regime)** (from
  `docs/results/ablation/Falpha_summary.json`): PPO flat +138.6→+113.3 vs tuned
  RF +136.5→+64.0 across α=0.0/0.2/0.4/0.6 (RF continues to −29.3 at α=1.0);
  tie at α=0 (overlapping CIs), disjoint CIs from α=0.4 (PPO +121.3 vs RF +94.4,
  sig +26.9). Oracle ceiling +194.8.
  Reward-coupling: coupled best DQN +226.2 (gap −63.1 vs RF +163.1), outcome
  best A2C +146.1 (gap −63.0 vs RF +83.1; DQN −8.6). **On-policy advantage is
  training reliability and defensive doctrine**: at α=0.4 all three clear the
  negative regime on best checkpoint (DQN +72.5, PPO +121.3, A2C +138.7) and
  A2C matches or exceeds PPO at every aliasing rate; across-seed sd PPO≈15 /
  DQN≈52 / A2C≈9; the same DQN's sd inflates ≈17→≈52 switching coupled→outcome.
  OOD (10 held-out zero-day classes): best RL (A2C) prevents 0.71–0.85 vs RF
  0.00–0.15 on every class, with **no detectable dependence** of RL advantage
  on detector recall (Spearman ρ=0.22 p=0.54, Pearson r=−0.02 p=0.95, OLS slope
  CI spans zero).
  F10 (aggressiveness) and F17 (evasion) load the fixed det-5M α=0.4 PPO and
  evaluate across the swept knob (no retraining); F17 uses the **evasive-
  persistence** (post-detection hardening) attacker coupling. Test suite:
  **462 passed**.
