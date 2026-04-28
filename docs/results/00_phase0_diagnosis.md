# Phase 0 — Diagnosis of Pre-Restart Results

> **Purpose.** Before generating any new thesis-quality results, this note
> documents the state of the project at git tag `pre-mentor-restart`,
> identifies the technical problems that prevent the existing artifacts from
> being thesis-ready, and explains why each subsequent phase is necessary.
> Negative results live forever in the appendix; this is ours.

## 1. Snapshot

| Item                       | Value                                                             |
|----------------------------|-------------------------------------------------------------------|
| Git tag                    | `pre-mentor-restart`                                              |
| Branch                     | `feature/reward-shaping`                                          |
| Last commit                | `80939a3` — *docs: add project documentation …*                    |
| LSTM artifact              | `artifacts/generator/attack_sequence_generator.pth` (preserved)   |
| Pre-restart RL runs        | 10 PPO/SB3 runs from 2026-03-12/13 (archived, then deleted)       |
| Pre-restart benchmark      | `results/benchmark/` (archived, then deleted)                     |
| Archive bundle             | `.archive/pre_mentor_artifacts_<TS>.tgz`                          |

## 2. What the existing numbers actually say

### 2.1 LSTM Red Team

From `artifacts/generator/generator_training_report.json`:

```
accuracy             = 0.9630
perplexity           = 1.115
transition_accuracy  = 0.9630
macro_f1             = 0.5972      ← weak
```

Confusion matrix (rows = true stage, cols = predicted):

|             | B    | R    | A    | M    | I       |
|-------------|------|------|------|------|---------|
| **BENIGN**  | 132  | 0    | 0    | 0    | 0       |
| **RECON**   | 49   | 105  | 0    | 0    | 0       |
| **ACCESS**  | 24   | 168  | 325  | 11   | 0       |
| **MANEUVER**| 20   | 32   | 508  | 796  | 0       |
| **IMPACT**  | 43   | 21   | 106  | 1314 | 58 498  |

**Diagnosis.** Accuracy 0.96 is the trivial all-IMPACT solution (~96 % of
samples are IMPACT). The model has *not* learned the kill chain; it has
learned to predict IMPACT or one stage back. This invalidates any thesis
claim that "the LSTM models attack progression". The recall gates in
`config.yml` (`min_recall_stage_1 = 0.3`, `min_recall_stage_2 = 0.3`) are
so loose that they triggered on every epoch (`recall_gate_pass_count = 30`),
making the early-stopping criterion non-functional.

### 2.2 RL Blue Team

From `results/benchmark/benchmark_evaluation_report.json` and
`results/benchmark/comparison.json`:

| Metric                  | Value             | Comment                              |
|-------------------------|-------------------|--------------------------------------|
| `avg_reward`            | -6.67 ± 87.92     | Bimodal: ~+25 or ~-200               |
| `attack_mitigation_rate`| 0.80              | Inflated by 2-step episodes          |
| `false_positive_rate`   | 0.79              | **Catastrophic**                     |
| `availability_score`    | 0.05              | Agent intervenes constantly          |
| `mean_time_to_contain`  | 0.00              | **Metric is structurally broken**    |
| `macro_f1`              | 0.29              | Worse than chance for several stages |

Episode lengths are mostly 2–3 steps (`max_steps = 100` in `config.yml`).
Most episodes look like `attack_stages = [0, 4, 4]` — BENIGN jumps directly
to IMPACT in one transition.

## 3. Root causes (code-level evidence)

### 3.1 Environment terminates on the first non-trivial action

In `src/environment/adversarial_env.py`:

- **L313–332** — when stage == IMPACT, the step returns
  `terminated = True`. Any episode that reaches IMPACT is over.
- **L339–348** — when the agent picks BLOCK or ISOLATE during an active
  attack, `terminated = True`. **The agent cannot ever take a second
  strong action.**

Combined with §3.2, this guarantees ~2-step episodes, making sustained
kill-chain reasoning structurally impossible. The bimodal reward
distribution (≈ +25 / ≈ -200) is the direct symptom: a single BLOCK
either lands during an attack and ends with a +bonus, or it doesn't and
the next step lands in IMPACT and ends with -200.

### 3.2 Red Team biases the env toward IMPACT

Because the LSTM (§2.1) predicts IMPACT on essentially every non-BENIGN
input, the env's `_advance_attack` jumps to IMPACT after one step. The
agent never observes the intermediate stages it is supposedly learning
to defend against.

### 3.3 `mean_time_to_contain` is silently always zero

Defined as "steps until stage returns to BENIGN", but the env never
includes a BENIGN-reset transition. MTTC == 0 by construction.

### 3.4 Reward double-counting on BENIGN

Across `_calculate_reward`, OBSERVE/LOG on BENIGN can grant
`reward_benign_passive` (+10) and `maintained_defense` (+0.2) and
`patience_bonus` simultaneously. The +25 cluster of the reward
distribution is mostly accumulated benign passive bonuses, not actual
defense quality.

### 3.5 `max_steps` from YAML is not threaded through

`AdversarialEnvConfig` (`adversarial_env.py:103`) defaults to
`max_steps = 500`; `config.yml` declares `max_steps = 100`. The wiring
between YAML and the dataclass needs verification — either way, episodes
end far short of either cap due to §3.1.

### 3.6 Action↔stage projection inflates F1

The benchmarking code projects actions onto stages via
`OBSERVE→BENIGN, LOG→RECON, …, ISOLATE→IMPACT` to compute a "macro-F1".
This is not a real IDS metric — it bakes a perfect bijection assumption
into the score. Replaced by a dedicated detector head in Phase 4.

## 4. Methodological gaps (what an examiner will ask)

| # | Gap                                                                              | Resolved in |
|---|----------------------------------------------------------------------------------|-------------|
| 1 | No multi-seed runs (single seed per algorithm in pre-restart benchmarks)         | Phase 5     |
| 2 | No baselines (random / always-OBSERVE / always-BLOCK / supervised IDS)           | Phase 4 + 7 |
| 3 | No held-out test environment / no OOD evaluation                                 | Phase 1 + 7 |
| 4 | No statistical comparison between algorithms                                     | Phase 7     |
| 5 | No latency / overhead measurement (IoTWarden Fig. 4(b) equivalent missing)       | Phase 7     |
| 6 | Reward shaping not justified by ablation                                         | Phase 8     |
| 7 | No direct comparison against IoTWarden's own DQN setup                           | Phase 6     |
| 8 | LSTM trained only on synthetic grammar episodes, not real attack timelines       | Phase 2     |

## 5. What we keep, what we redo

**Keep (with care):**
- The kill-chain abstraction (BENIGN → RECON → ACCESS → MANEUVER → IMPACT).
- The force-continuum action space (OBSERVE / LOG / THROTTLE / BLOCK / ISOLATE).
- The `RealizationEngine` design (stage-conditioned sampling from real CICIoT2023).
- The MLflow tracking + benchmarking layer scaffolding.
- The 179-test suite and existing module boundaries.

**Redo:**
- LSTM training pipeline (Phase 2): real-data sequences, balanced
  test set, label smoothing, real recall gates.
- Environment dynamics (Phase 3): no premature termination on BLOCK,
  episode ≥ 20 steps, working MTTC, normalized rewards, single-source-of-truth
  config.
- All RL agents (Phase 5): retrained on new env with multi-seed.
- All benchmark figures (Phase 7): baselines, statistical tests, latency.

## 6. Acceptance gates carried into later phases

| Phase | Gate                                                                                      |
|-------|-------------------------------------------------------------------------------------------|
| 2     | macro-F1 ≥ 0.75 on balanced held-out test; all per-stage recalls ≥ 0.5; no zero rows.     |
| 3     | Random policy reward distribution is roughly Gaussian (not bimodal); MTTC > 0 in rollouts.|
| 5     | PPO mean reward monotonic ↑ then plateau; std band shrinks; FPR < 0.1; mitigation > 0.9.  |
| 7     | RL ≥ best baseline on ≥ 2 of 4 metrics with statistical significance (p < 0.05).          |

The current run does not meet any of these gates. That is what makes a
**Phase 0** necessary.
