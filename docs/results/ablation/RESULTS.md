# Ablation + OOD-class Robustness: Results

> Companion to `PLAN.md`. Locked PLAN first, then implementation,
> then this document captures **what happened on real data**.
> The two headline strands (per audit AF1 / AF2) are **Fcoupling**
> (does the RL advantage survive stripping dense per-step reward
> shaping?) and **F15** (does trained RL recover
> the supervised detector's `VulnerabilityScan` blind spot?).

## 1 — Headline numbers

**Fcoupling — reward-coupling ablation (D7.1):**
PASS: under the sparse outcome reward the best RL agent (a2c, +146.1) outperforms the memoryless RF-Acting baseline (+80.9) by +65.2 points — the RL advantage is not an artefact of dense per-step shaping.

  - Outcome gap (RL − RF): **-65.2**
  - Coupled gap (RL − RF): **-79.4**
  - Gap reduction (coupled → outcome): **-14.2**

**F15 — OOD-class robustness (audit-AF1, HEADLINE):**
Trained RL recovers some of the supervised RF blind spot on VulnerabilityScan.

  - On `VulnerabilityScan` (RF detector recall = 0.001):
    - Best trained RL: `a2c` mean = +157.5
      (CI [154.98493333333333, 160.35998333333333])
    - RF-Acting mean = +90.1
      (CI [76.60926666666667, 102.57432499999997])
    - Δ = **+67.4**

**F10 — attack-aggressiveness (fixed-policy sweep; conceptually aligned with IoTWarden Fig. 6):**
PASS: PPO benefits from a more lenient environment (higher p_down ⇒ higher reward) by ≥ 1σ between p_down=0.0 and p_down=0.6, and the rule curve is monotone non-decreasing in p_down — the value function shifts with environment difficulty as expected (conceptually aligned with IoTWarden Fig. 6).

## 2 — Gate scoreboard

| Gate | Threshold | Status | Value / Notes |
|---|---|:---:|---|
| **G7.1** | pytest -q ≥ 428 passed; zero new skips | **PASS** | ======================= 455 passed in 18.12s ======================== |
| **G7.2** | Under outcome (sparse) reward, best RL agent outperforms RF-Acting (gap_outcome < 0) | **PASS** | outcome: best_rl=a2c (+146.1), RF=+80.9, gap=-65.2 |
| **G7.3** | PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone | **PASS** | p=0.0 CI=(-64.7, -37.3); p=0.6 CI=(93.5, 107.7) |
| **G7.5** | Environment-design frozen tests pass with impact_is_terminal=True | **PASS** | G7.1 carries this through (full pytest green ⇒ environment-design contract preserved) |
| **G7.6** | No regression on environment-design/detector/Blue-Team/benchmark frozen tests overall | **PASS** | G7.1 carries this through |
| **G7.7** | Fcoupling/F10/F15/F17 manifest.json all present + SHA-pinned | **PASS** | all 4 manifests present |
| **G7.8** | F15 4-class × 8-policy matrix complete, no NaN means | **PASS** | 80/80 cells; n_missing=0; n_nan=0 |
| **G7.9** | On VulnerabilityScan, best trained RL CI_low > RF-Acting CI_high (≥1σ separation, RL > RF) | **PASS** | best_rl=a2c (+157.5), RF=(+90.1), Δ=+67.4 |
| **G7.10** | F17 max-evasion (0.75) mean test reward within robust_tol=0.25 of evasion=0 reference (graceful degradation, no collapse) | **PASS** | ref(e=0)=+123.8, max(e=0.75)=+96.4, ci_low_degradation=30.9 (tol_abs=30.9) |

Tally: **9 PASS / 0 FAIL-WITH-FINDING**.
Source of record: `G7_scoreboard.json` next to this file.

## 3 — Deliverables (figures + tables)

| Artefact | Path | Description |
|---|---|---|
| **Fcoupling** (Tier 2) | `Fcoupling_reward_gap.png` + `Fcoupling_summary.json` | Reward-coupling ablation: coupled vs outcome reward gap between best RL agent and RF-Acting. |
| **F10** (Tier 2) | `F10_aggressiveness.png` + `F10_summary.json` | Fixed det-5M α=0.4 PPO and oracle-rule mean test reward as a function of `p_defender_deescalation` (re-evaluated, not retrained); conceptually aligned with IoTWarden Fig. 6. |
| **F15** (Tier 1, audit-AF1) | `F15_ood_robustness.png` + `F15_summary.json` | 10 OOD class × 8 policy grouped bar chart with bootstrap CIs. |
| Captions | `F10_caption.md`, `F15_caption.md` | Thesis-paper captions per figure. |
| Manifests | `Fcoupling_manifest.json`, `F10_manifest.json`, `F15_manifest.json` | SHA-256 hash chain over input JSONLs + Blue-Team Training sweep manifest + Held-Out Benchmark eval manifest + git SHA at production time. |
| Scoreboard | `G7_scoreboard.json` | Per-gate threshold + value + status + finding-id. |
| Run artefacts (gitignored) | `runs/ablation/{ood,reward_sweep,aggressiveness}/.../eval_test.jsonl` | The schema-v1.0 input data for every figure. |

## 4 — Code summary

| File | Purpose |
|---|---|
| `src/environment/adversarial_env.py` | Added `impact_is_terminal: bool = False` (primary training+benchmark contract; `True` retained only as a reward-mis-specification case study). |
| `src/blue_team/run_config.py` | `EnvConfigSerializable` extended from 7 → 18 fields (all reward coefficients + `impact_is_terminal`). |
| `src/blue_team/env_factory.py` | `_build_env_config` now forwards full reward field set. |
| `scripts/blue_team/train_agent.py` | Added `--reward-overrides JSON`, `--p-defender-deescalation FLOAT`, `--impact-is-terminal BOOL` CLI args. |
| `scripts/ablation/run_ood_eval.py` | F15 OOD eval driver with hybrid realiser (in-distribution train pool + OOD overlay at the OOD class's stage). |
| `scripts/ablation/plot_ood_robustness.py` | F15 plotter + G7.8 / G7.9 evaluators. |
| `scripts/ablation/run_aggressiveness_sweep.py` | F10 6-p-value PPO sweep + oracle-rule reference rolls. |
| `scripts/ablation/plot_aggressiveness.py` | F10 plotter + G7.3 evaluator. |
| `scripts/ablation/close_ablation.py` | This file: assembles `G7_scoreboard.json` + `RESULTS.md` + CHANGELOG. |
| `tests/test_env_impact_terminal.py` | 8 synthetic tests pinning the `impact_is_terminal` codepath. |
| `tests/test_train_agent_reward_overrides.py` | 14 synthetic tests pinning the CLI override plumbing. |

Total tests: 455 (no run-time-data tests added; G7.2/G7.3/G7.8/G7.9 are real-data acceptance tests).

## 5 — Cross-step findings discovered during the ablation evaluation

(Hand-fill — examples: hybrid OOD realiser was needed because each OOD class is single-stage; train-eval window-shape mismatch under `--smoke` surfaced by smoke run; etc.)

## 6 — Ablation findings worth defending in the thesis

### 6.1 The reward-coupling ablation result (G7.2)

(Hand-fill from G7.2 above — the outcome reward gap shows whether the RL advantage survives stripping dense per-step shaping.)

### 6.2 The OOD-class robustness result (D7.9.1 if needed; audit-AF1 HEADLINE)

(Hand-fill from G7.9 above — either trained RL beats RF-Acting on `VulnerabilityScan` by ≥1σ (RL closes the OOD gap), or it does not (RL is *robust to* not *better at* the OOD class). Either outcome is defensible.)

### 6.3 The fixed-policy difficulty sensitivity sweep (G7.3; conceptually aligned with IoTWarden Fig. 6)

(Hand-fill from G7.3 above.)

## 7 — Future work hand-offs

Post-thesis work includes:

1. **F13 — Robustness to observation noise / drift** (Tier 3).
2. **F14 — Generalisation training to held-out attack class** (Tier 3 if it ships); F15 covered the eval-time complement, F14 would be the train-time augmentation.

The ablation evaluation does NOT defer:

- The reward-coupling gap. Fcoupling either shows the RL advantage survives under the sparse outcome reward (G7.2 PASS) or characterises the
  closure attempt as the limit of the environment-design reward formulation (D7.1.1).
- The OOD-class robustness claim. F15 either delivered it (G7.9 PASS) or narrowed it to "robust to (not better at)"
  per D7.9.1.

## 8 — Reproducibility

Every ablation figure ships a `manifest.json` with:

- SHA-256 hashes of every input JSONL under
  `runs/ablation/{ood,reward_sweep,aggressiveness}/.../eval_test.jsonl`.
- SHA-256 of the upstream `runs/redesign_5M_det/alpha_04/sweep_manifest.json`
  (trained checkpoints) and `runs/benchmark/eval_manifest.json` (benchmark
  baselines).
- Git SHA at production time.

To regenerate from scratch on a fresh checkout::

    make blue-team-sweep BLUE_TEAM_TIMESTEPS=5000000  # deterministic-5M regime
    make benchmark                                    # ~10 min CPU
    make ablation                                     # ~7.5 h CPU (walk-away)
    python -m scripts.ablation.close_ablation         # assemble G7 scoreboard + RESULTS

The `runs/redesign_5M_det/`, `runs/benchmark/`, `runs/ablation/` dirs are all
gitignored; all derived figures + summaries + manifests live under
`docs/results/<area>/`.

## 9 — Test count history

Dataset prep 254 → Dataset prep 266 → Markov Attacker 283 → Env design 296 → Detector 329
→ Blue-Team 376 → Benchmark 420 → Ablation 447 → A2C-retrain + contract audit 452 → **M1 truncation-boundary fix 455**.
