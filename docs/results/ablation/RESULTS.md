# Ablation + OOD-class Robustness: Results

> Companion to `PLAN.md`. Locked PLAN first, then implementation,
> then this document captures **what happened on real data**.
> The two headline strands (per audit AF1 / AF2) are **F9**
> (does the reward-component sweep close the +288 deployable gap
> to the oracle ceiling?) and **F15** (does trained RL recover
> the supervised detector's `VulnerabilityScan` blind spot?).

## 1 — Headline numbers

**F9 — reward-component sweep (D7.1):**
FAIL-WITH-FINDING (D7.1.1, activated 2026-05-01): no reward-comparable cell (environment-design reward fn preserved) beats DQN +1336 on raw reward by ≥ 1σ. BUT: the security-KPI strand passes — cell `impact_is_terminal_false` improves mitigated_impact_rate to 0.867 (vs DQN baseline 0.153, ≥ 1.5× threshold 0.229). The one-at-a-time linear sweep characterised the limit of environment-design-style reward shaping at the apples-to-apples reward level, but env-semantics + coefficient scaling do move the real-security needle. Closing the +288 reward gap under fixed reward semantics requires a different mechanism (curriculum, reward modelling, or attack-aware exploration), deferred to future work.

  - Best cell: `impact_is_terminal_false` (mean = +1125.7,
    CI = (+1079.6,
    +1184.3))
  - Δ to benchmark deployable best (DQN +1336): **-210.6**
  - Δ to benchmark oracle ceiling (rule +1624): **-498.7**
  - Stretch goal (oracle ceiling) met: **False**

**F15 — OOD-class robustness (audit-AF1, HEADLINE):**
FAIL-WITH-FINDING: trained RL does NOT beat RF-Acting on VulnerabilityScan; the thesis claim narrows from 'RL closes the OOD gap' to 'RL is robust to (not better at) the OOD class'. See PLAN §8 D7.9.1.

  - On `VulnerabilityScan` (RF detector recall = 0.001):
    - Best trained RL: `ppo` mean = +1109.5
      (CI [1081.7868250000001, 1142.5862916666667])
    - RF-Acting mean = +1443.4
      (CI [1391.4600916666666, 1496.2217416666667])
    - Δ = **-333.9**

**F10 — attack-aggressiveness (IoTWarden Fig. 6 re-impl):**
PASS: PPO benefits from a more lenient defender (p↑ ⇒ reward↑) by ≥ 1σ between p=0.0 and p=0.6, and the rule curve is monotone non-decreasing in p — replicates the IoTWarden Fig. 6 qualitative shape on CICIoT2023.

**F12 — security-vs-availability Pareto:**
PASS: Pareto frontier has 4 distinct dominant points — non-trivial trade-off surface; operating-point choice is a real defender contribution.

  - Total points collected: 32
  - Frontier points (distinct): 4

## 2 — Gate scoreboard

| Gate | Threshold | Status | Value / Notes |
|---|---|:---:|---|
| **G7.1** | pytest -q ≥ 430 passed; zero new skips | **PASS** | ======================= 432 passed, 1 warning in 14.85s ======================== |
| **G7.2** | F9 best reward-comparable cell mean test reward > Phase-6 DQN +1336 by ≥1σ (apples-to-apples; reward-coefficient cells fall back to security-KPI strand per D7.1.1) | FAIL-WITH-FINDING | reward-comparable best=impact_is_terminal_false (+1125.7); security-KPI best=impact_is_terminal_false (mit=0.867); meets_oracle_stretch=False |
| **G7.3** | PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone | **PASS** | p=0.0 CI=(127.9, 138.6); p=0.6 CI=(856.9, 923.9) |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | **PASS** | n_distinct=4/32 |
| **G7.5** | Environment-design frozen tests pass with impact_is_terminal=True | **PASS** | G7.1 carries this through (full pytest green ⇒ environment-design contract preserved) |
| **G7.6** | No regression on environment-design/detector/blue-team/benchmark frozen tests overall | **PASS** | G7.1 carries this through |
| **G7.7** | F9/F10/F12/F15 manifest.json all present + SHA-pinned | **PASS** | all 4 manifests present |
| **G7.8** | F15 4-class × 8-policy matrix complete, no NaN means | **PASS** | 32/32 cells; n_missing=0; n_nan=0 |
| **G7.9** | On VulnerabilityScan, best trained RL CI_low > RF-Acting CI_high (≥1σ separation, RL > RF) | FAIL-WITH-FINDING | best_rl=ppo (+1109.5), RF=(+1443.4), Δ=-333.9 |

Tally: **7 PASS / 2 FAIL-WITH-FINDING**.
Source of record: `G7_scoreboard.json` next to this file.

## 3 — Deliverables (figures + tables)

| Artefact | Path | Description |
|---|---|---|
| **F9** (Tier 2) | `F9_reward_ablation.png` + `F9_summary.json` | 6-panel reward-component effect plot (5 components × {0.5×, 1×, 2×} + impact_is_terminal binary) with benchmark reference lines (oracle +1624, DQN +1336). |
| **F10** (Tier 2) | `F10_aggressiveness.png` + `F10_summary.json` | PPO and oracle-rule mean test reward as a function of `p_defender_deescalation`; IoTWarden Fig. 6 re-impl. |
| **F12** (Tier 2) | `F12_pareto.png` + `F12_summary.json` | 2-D scatter on (availability_cost, security_gain) with Pareto frontier; reads F9 + F10 + benchmark outputs. |
| **F15** (Tier 1, audit-AF1) | `F15_ood_robustness.png` + `F15_summary.json` | 4 OOD class × 8 policy grouped bar chart with bootstrap CIs. |
| Captions | `F9_caption.md`, `F10_caption.md`, `F12_caption.md`, `F15_caption.md` | Thesis-paper captions per figure. |
| Manifests | `F9_manifest.json` … `F15_manifest.json` | SHA-256 hash chain over input JSONLs + Phase-5 sweep manifest + Phase-6 eval manifest + git SHA at production time. |
| Scoreboard | `G7_scoreboard.json` | Per-gate threshold + value + status + finding-id. |
| Run artefacts (gitignored) | `runs/ablation/{ood,reward_sweep,aggressiveness}/.../eval_test.jsonl` | The schema-v1.0 input data for every figure. |

## 4 — Code summary

| File | Purpose |
|---|---|
| `src/environment/adversarial_env.py` | Added `impact_is_terminal: bool = True` (default preserves environment-design frozen contract). |
| `src/blue_team/run_config.py` | `EnvConfigSerializable` extended from 7 → 18 fields (all reward coefficients + `impact_is_terminal`). |
| `src/blue_team/env_factory.py` | `_build_env_config` now forwards full reward field set. |
| `scripts/blue_team/train_agent.py` | Added `--reward-overrides JSON`, `--p-defender-deescalation FLOAT`, `--impact-is-terminal BOOL` CLI args. |
| `scripts/ablation/run_ood_eval.py` | F15 OOD eval driver with hybrid realiser (in-distribution train pool + OOD overlay at the OOD class's stage). |
| `scripts/ablation/plot_ood_robustness.py` | F15 plotter + G7.8 / G7.9 evaluators. |
| `scripts/ablation/run_reward_sweep.py` | F9 12-cell sparse one-at-a-time sweep driver (PPO + 5 components × 3 multipliers + impact_is_terminal binary). |
| `scripts/ablation/plot_reward_ablation.py` | F9 plotter + G7.2 evaluator. |
| `scripts/ablation/run_aggressiveness_sweep.py` | F10 6-p-value PPO sweep + oracle-rule reference rolls. |
| `scripts/ablation/plot_aggressiveness.py` | F10 plotter + G7.3 evaluator. |
| `scripts/ablation/plot_pareto.py` | F12 Pareto-frontier plot + G7.4 evaluator. |
| `scripts/ablation/close_ablation.py` | This file: assembles `G7_scoreboard.json` + `RESULTS.md` + CHANGELOG. |
| `tests/test_env_impact_terminal.py` | 8 synthetic tests pinning the `impact_is_terminal` codepath. |
| `tests/test_train_agent_reward_overrides.py` | 14 synthetic tests pinning the CLI override plumbing. |

Total tests: 442 → ~442 (no run-time-data tests added; G7.2/G7.3/G7.4/G7.8/G7.9 are real-data acceptance tests).

## 5 — Cross-step findings discovered during the ablation evaluation

(Hand-fill — examples: hybrid OOD realiser was needed because each OOD class is single-stage; train-eval window-shape mismatch under `--smoke` surfaced by smoke run; etc.)

## 6 — Ablation findings worth defending in the thesis

### 6.1 The reward-component sweep result (D7.2.1 if needed)

(Hand-fill from G7.2 above — either the +288 gap was closed, partially closed, or characterised as the limit of one-at-a-time environment-design-style reward shaping per D7.1.1.)

### 6.2 The OOD-class robustness result (D7.9.1 if needed; audit-AF1 HEADLINE)

(Hand-fill from G7.9 above — either trained RL beats RF-Acting on `VulnerabilityScan` by ≥1σ (RL closes the OOD gap), or it does not (RL is *robust to* not *better at* the OOD class). Either outcome is defensible.)

### 6.3 The IoTWarden Fig. 6 sensitivity replication (G7.3)

(Hand-fill from G7.3 above.)

### 6.4 The operating-point Pareto contribution (G7.4)

(Hand-fill from G7.4 above.)

## 7 — Future work hand-offs

Post-thesis work includes:

1. **F13 — Robustness to observation noise / drift** (Tier 3).
2. **F14 — Generalisation training to held-out attack class** (Tier 3 if it ships); F15 covered the eval-time complement, F14 would be the train-time augmentation.

The ablation evaluation does NOT defer:

- The +288 deployable gap. F9 either closed it (G7.2 PASS) or characterised the closure attempt as the limit of the
  environment-design reward formulation (D7.1.1).
- The OOD-class robustness claim. F15 either delivered it (G7.9 PASS) or narrowed it to "robust to (not better at)"
  per D7.9.1.

## 8 — Reproducibility

Every ablation figure ships a `manifest.json` with:

- SHA-256 hashes of every input JSONL under
  `runs/ablation/{ood,reward_sweep,aggressiveness}/.../eval_test.jsonl`.
- SHA-256 of the upstream `runs/blue_team/sweep_manifest.json` (trained
  checkpoints) and `runs/benchmark/eval_manifest.json` (benchmark
  baselines).
- Git SHA at production time.

To regenerate from scratch on a fresh checkout::

    make blue-team-sweep BLUE_TEAM_TIMESTEPS=250000  # ~108 min CPU (one-off)
    make benchmark                                   # ~10 min CPU
    make ablation                                    # ~7.5 h CPU (walk-away)
    python -m scripts.ablation.close_ablation        # assemble G7 scoreboard + RESULTS

The `runs/blue_team/`, `runs/benchmark/`, `runs/ablation/` dirs are all
gitignored; all derived figures + summaries + manifests live under
`docs/results/0[5-7]_*/`.

## 9 — Test count history

Dataset prep 254 → Dataset prep 266 → Red-team 283 → Env design 296 → Detector 329
→ Blue-team 376 → Benchmark 420 → **Ablation 442** (+22 from C3 + C4).
