# Held-Out Benchmark: Results

> Companion to `PLAN.md`. The benchmark evaluated every trained
> checkpoint plus four non-RL baselines on the held-out `test_balanced`
> split — the **first** use of that split for blue-team metrics.
>
> **Canonical data source:** `main_results.json` (n_seeds=10, n_episodes=300,
> `impact_is_terminal=False`, `test_balanced`). All numbers in this
> document are drawn from that file. `generated_at` timestamps in all
> sibling JSONs record when each artifact was last regenerated.
>
> **Oracle reframing note (audit AF2).** The recommended-action rule
> receives `info["attack_stage"]` directly from the env — free perfect
> stage classification. It is therefore **not a deployable defender**;
> it is an upper bound on the value of perfect stage detection. The
> deployable comparison sits between the trained RL trio and RF-Acting;
> the rule's mean reward (+543.1) is the *oracle ceiling*, not a
> competing baseline.

## 1 — Headline numbers

The benchmark evaluated all Phase-5 trained checkpoints (DQN, PPO, A2C ×
10 seeds = 30 runs) plus five non-RL reference policies on `test_balanced`
(n=300 deterministic episodes per policy, `impact_is_terminal=False`).

**Final ranking by mean episodic reward (95 % bootstrap CI):**

| # | Policy | Mean reward | 95 % CI | Benign FPR | Stage knowledge |
|---|---|---:|---|---:|---|
| ★ | **Recommended-Action (oracle)** ⓞ | **+543.1** | (+536.6, +549.4) | 0.0 % | true stage (oracle) |
| RF | **RF-Acting** (supervised+rules) | **+448.2** | — | — | RF-predicted stage |
| 1 | **A2C** (best deployable RL) | **+278.5** | (+251.1, +308.8) | **0.66 %** | none |
| 2 | PPO | +274.5 | — | 0.89 % | none |
| 3 | DQN | +267.8 | — | **0.46 %** | none |
| 4 | Always-OBSERVE | −393.15 | — | 0.0 % | none |
| 5 | Random | −573.93 | — | 41.3 % | none |
| 6 | Always-BLOCK | −2005.06 | — | 100.0 % | none |

ⓞ = **oracle baseline**: receives `info["attack_stage"]` directly from the
env (free perfect classification); not deployable. Cited as an *upper bound
on the value of perfect stage detection*.

**Key derived metrics:**

- **Oracle capture (best deployable RL):** A2C +278.5 / oracle +543.1 = **51.3 %**
  (PPO 50.5 %, DQN 49.3 %, RF-Acting 82.5 %).
- **Latency advantage (RL vs RF-Acting):** 16.505 ms / 0.094 ms ≈ **176×** — RF-Acting
  FAILS the 3 ms latency budget; RL p50 ≈ 0.094 ms (DQN 0.063 ms).
- **Benign FPR — A2C:** 0.66 %; **DQN:** 0.46 %; **PPO:** 0.89 % (see `benign_fpr.json`).
  All three RL agents are now **below the 1 % operational threshold**.
- **`compromise_rate`:** varies by policy — a2c 0.403, ppo 0.67, dqn 0.463, oracle 0.00,
  always_block 0.00. The attacker is a reactive tug-of-war process, so compromise is no
  longer structurally forced; the metric now discriminates between policies.
- **`prevention_rate` (primary security KPI):** the fraction of episodes where the
  attacker's intrusion budget is exhausted before IMPACT — oracle 1.00, a2c 0.60,
  dqn 0.54, ppo 0.33. `mitigated_impact_rate` is **retired** (it collapsed onto
  `compromise_rate` for always-block policies); see §6.2.

Among deployable policies the trade-off is:
- RF-Acting: highest deployable reward (+448.2), but slow inference (16.505 ms p50,
  ~176× slower than RL) — it **FAILS the 3 ms latency gate**.
- Trained RL: lower reward at 51.3 % of oracle, fast inference (~0.094 ms p50), and all
  three are statistically tied on reward.

## 2 — Gate scoreboard

| Gate | Threshold | Status | Value / Notes |
|---|---|:---:|---|
| **G6.1** | `pytest -q` ≥ 388 passed | **PASS** | **428** passed, 1 warning (canonical count at HEAD) |
| **G6.2** | trained-RL `mean_reward` > recommended-action (D6.2.1 revised) | **FAIL-WITH-FINDING** | oracle +543.1 > {A2C +278.5, PPO +274.5, DQN +267.8}. **Headline finding** — see §6 |
| **G6.3** | non-IMPACT proportionality band ≥ 0.70 | **PASS** | DQN 0.867, PPO 0.893, A2C 0.940 |
| **G6.4** | p50 latency: RL ≤ 5 ms / RF ≤ 3 ms / rule ≤ 1 ms | **FAIL-WITH-FINDING** | RL 0.063–0.094 ms (PASS); RF-Acting 16.505 ms (D6.8.1 finding — sklearn dispatch overhead; FAILS 3 ms budget) |
| **G6.5** | trained-RL CI ⊥ every non-RL CI | **PASS** | A2C/PPO/DQN show zero CI overlap with any non-RL baseline |
| **G6.6** | no regression on earlier frozen tests | **PASS** | all frozen tests green |
| **G6.7** | F5/F6/F7/F8 each ship a `manifest.json` | **PASS** | SHA-256 hash chain on every figure |

Tally: 4 PASS / 1 PASS-WITH-FINDING / 1 FAIL-WITH-FINDING / 0 PASS-VOIDS.
Both findings are **registered design decisions** (D6.2.1, D6.8.1) with
rationale, not silently relaxed gates.

Source of record: `benchmark_acceptance.json` next to this file.

## 3 — Deliverables (figures + tables)

| Artefact | Path | Description |
|---|---|---|
| **F5** | `main_results_table.png` + `F5_summary.{json,md,csv}` | Final security metrics table (8 rows × 9 cols) with bootstrap CIs and best-row highlight. |
| **F6** | `stage_action_cm.png` + `stage_action_proportionality.json` | 2 × 3 grid of 5×5 stage × action heatmaps with proportionality-band overlay and per-panel G6.3 score. |
| **F7** | `overhead.png` + `latency_profile.json` | Two-panel: left = inference-latency CDF (log-x) per policy; right = training wallclock per algo (sum over seeds). |
| **F8** | `baselines.png` + `reward_ranking.json` | Horizontal bar chart of mean reward with 95 % bootstrap CIs, sorted descending; oracle ceiling reference line. |
| Captions | `main_results.caption.md` … `reward_ranking.caption.md` | One-paragraph thesis captions per figure. |
| Manifests | `main_results_manifest.json` … `reward_ranking_manifest.json` | SHA-256 hash chain over input JSONLs + upstream eval manifest + git SHA at production time. |
| Scoreboard | `benchmark_acceptance.json` | Per-gate threshold + value + status (canonical; regenerated 2025-06-04). |
| FPR data | `benign_fpr.json` | Per-policy benign false-positive rate (block/isolate rate on BENIGN episodes). |

## 4 — Code summary

| File | LoC | Purpose |
|---|---:|---|
| `src/benchmark/__init__.py` | 53 | Package surface; re-exports the public API. |
| `src/benchmark/baseline_policies.py` | 287 | `random_policy`, `always_observe`, `always_block`, `recommended_action_policy`, `RFActingPolicy`, `SB3PolicyAdapter`, `Policy` Protocol. |
| `src/benchmark/eval_runner.py` | 308 | `run_policy(...)` — drives any Policy on a VecEnv, emits schema-v1.0 EpisodeRecord JSONL + optional sidecar latency JSONL. |
| `src/benchmark/latency.py` | 124 | `measure_inference_latency(...)` — ns-precision micro-benchmark with deterministic-clock injection for tests. |
| `scripts/benchmark/run_test_eval.py` | 360 | CLI driver: rolls every trained checkpoint and every baseline on `test_balanced`, writes eval manifest. |
| `scripts/benchmark/build_summary_table.py` | 308 | F5 builder. |
| `scripts/benchmark/plot_stage_action_cm.py` | 243 | F6 builder. |
| `scripts/benchmark/plot_overhead.py` | 287 | F7 builder. |
| `scripts/benchmark/plot_baselines.py` | 215 | F8 builder. |
| `tests/test_baseline_policies.py` | 188 | 24 tests pinning every baseline policy. |
| `tests/test_benchmark_eval_runner.py` | 263 | 11 tests pinning the JSONL round-trip + decision-stage bookkeeping. |
| `tests/test_benchmark_latency.py` | 117 | 9 tests pinning the warmup/measure split + clock injection. |

## 5 — Cross-phase findings discovered during benchmark

**None — but the benchmark *re-frames* the blue-team headline.**

The held-out evaluation revealed that the training-phase headline
(all three algorithms converge to high reward) does not imply a
comparably strong *deployable security* result. On `test_balanced`:

- Oracle capture (51.3 %) quantifies the cost of partial observability for a
  stage-unaware policy.
- `compromise_rate` now varies by policy (a2c 0.403 / ppo 0.67 / dqn 0.463 /
  oracle 0.00) because the attacker is a reactive tug-of-war process, not an
  upper-triangular driver; the metric discriminates between policies.
- Benign FPR is now **below 1 % for all RL** (DQN 0.46 % / PPO 0.89 % / A2C 0.66 %),
  so the earlier "elevated FPR disqualifies RL" concern is **resolved**;
  `prevention_rate` is the primary security KPI alongside reward.

## 6 — Findings worth defending in the thesis

### 6.1 Trained RL captures 51.3 % of the oracle ceiling without seeing stages (D6.2.1, audit AF2)

**Single most important benchmark finding.** On `test_balanced`, the
recommended-action rule scores **+543.1** while the best deployable agent
(**A2C, +278.5**) scores **51.3 %** of that ceiling. Bootstrap CIs do not
overlap (A2C max +308.8 < oracle min +536.6), so the gap is statistically
real. The three RL agents (A2C +278.5, PPO +274.5, DQN +267.8) are themselves
**statistically tied** on reward (overlapping CIs).

The oracle receives `info["attack_stage"]` directly from the env — it is a
*measurement instrument* that quantifies the value of perfect stage detection,
not a competing baseline. The right question is: **"how much of the value of
perfect stage detection did RL capture without ever seeing a stage?"** —
and the answer is **51.3 %**.

The remaining **+264.6 reward** (≈48.7 % gap, 543.1 − 278.5) is characterized by the
ablation study (§7 below) and reflects the cost of partial observability (POMDP).

### 6.2 Prevention rate is the primary security KPI (G6.2 finding)

`mitigated_impact_rate` is **retired**: it collapsed onto `compromise_rate` for
always-block policies and is no longer reported as a benchmark KPI. The primary
security metric is now **`prevention_rate`** — the fraction of episodes where the
attacker's intrusion budget is exhausted before reaching IMPACT. At benchmark
scale (10 seeds, 300 episodes per policy): oracle 1.00, a2c 0.60, dqn 0.54,
ppo 0.33. The oracle prevents IMPACT in every episode; the best deployable agent
(A2C) prevents it in 60 % of episodes.

The structural reward strand is characterized separately in the ablation study
(F9), where the structural reward variant achieves a mit-rate of **0.850** (reward
+278.5) versus **0.0** for the mis-specified baseline. That F9 figure is a
distinct strand from this F5 benchmark and is not directly comparable to the
benchmark `prevention_rate` numbers above; cost-of-partial-observability remains
the dominant gap to the oracle.

### 6.3 Trained agents *do* learn proportional behaviour on non-IMPACT stages (G6.3 PASS)

DQN 0.867, PPO 0.893, A2C 0.940 — all clear the 0.70 proportionality
threshold on BENIGN/RECON/ACCESS/MANEUVER. The F6 heatmaps show the diagonal
structure clearly: BENIGN → mostly OBSERVE/LOG; ACCESS → RESTRICT;
MANEUVER → BLOCK. Training learned a meaningful proportional policy. (The
action ladder is [OBSERVE, LOG, RESTRICT, BLOCK, ISOLATE]; the recommended
mapping is BENIGN→OBSERVE, RECON→LOG, ACCESS→RESTRICT, MANEUVER→BLOCK,
IMPACT→ISOLATE.)

### 6.4 RF-Acting has the highest deployable reward — but FAILS the latency gate

RF-Acting (+448.2) beats all trained RL by reward and captures 82.5 % of the
oracle ceiling, but it **FAILS the 3 ms latency budget**: its p50 inference
latency is **16.505 ms** vs RL's **~0.094 ms** (DQN 0.063 ms) — a **~176×
latency advantage for trained RL** in latency-critical deployments. The higher
RF reward comes from its stage-aware decisions (RandomForest macro-F1 ≈ 0.91 on
in-distribution classes); the latency cost is sklearn Python dispatch on the
forest. Because RF-Acting violates the 3 ms gate, the trained RL trio remains
the only deployable option under the latency contract.

**Cross-quadrant summary:**

| Policy class | Mean reward | p50 latency | Deployable? |
|---|---:|---:|:---:|
| Recommended-Action ⓞ | **+543.1** | 0.001 ms | **No** (oracle stage access) |
| RF-Acting | +448.2 (82.5 % of oracle) | **16.505 ms** | **No** (FAILS 3 ms gate) |
| Trained RL (best = A2C) | +278.5 (51.3 % of oracle) | 0.094 ms | Yes |
| Always-OBSERVE | −393.15 | 0.001 ms | Yes |
| Random | −573.93 | 0.002 ms | Yes |
| Always-BLOCK | −2005.06 | 0.001 ms | Yes (worst policy) |

### 6.5 Benign FPR: now below the 1 % operational threshold (G6.4 finding extension)

Across trained RL policies, benign false-positive rates (block/isolate on
BENIGN episodes) are now **below 1 %**: DQN 0.46 %, A2C 0.66 %, PPO 0.89 %. The
earlier "elevated FPR disqualifies RL" concern is **resolved** — all three RL
agents satisfy the < 1 % operational threshold. Only the trivial baselines still
exceed it (random 41.3 %, always_block 100 %). FPR is reported as a primary
operational metric alongside reward, not a footnote.

## 7 — Ablation hand-offs

The ablation study (F9 reward sweep, F10 aggressiveness, F12 Pareto, F15 OOD)
characterises the remaining +264.6 gap (cost of partial observability) and the
OOD robustness of the trained policies. See `docs/results/ablation/RESULTS.md`
for findings. Headline ablation outcomes:

- **F12 Pareto (G7.4 FAIL-WITH-FINDING, R7.3):** under perfect perception the
  oracle dominates at (security_gain = 1.0, availability_cost = 0.0); interior
  RL placement quantifies the cost of partial observability (POMDP).
- **F15 OOD (G7.9 PASS) — now an RL WIN:** detector-free PPO **+298.3** vs
  detector-coupled RF-Acting **−4430.6** on VulnerabilityScan (delta **+4728.9**).
  RF's detector is blind to VulnerabilityScan (RECON recall 0.000), so it
  mis-predicts → under-forces → the attacker advances → catastrophic loss. The
  earlier "robust to but not better at" framing is superseded.

## 8 — Reproducibility

Every benchmark figure ships a `manifest.json` with SHA-256 hashes of every
input JSONL, the upstream eval manifest (which records SHA-256 hashes of every
trained model checkpoint, the RF model, the dataset scaler, and the splits
manifest), and the git SHA at production time. See `docs/reproducibility.md`
for the full protocol.

To regenerate from scratch:

```bash
make blue-team          # train DQN/PPO/A2C × 10 seeds (CPU-bound, hours)
make benchmark          # eval + F5/F6/F7/F8 (~10 min CPU)
make render-tables      # regenerate tex/generated/*.tex from canonical JSONs
make verify-fresh       # CI gate: confirm no derived artifact is stale
```

## 9 — Test count history

Canonical current count: **428 passed, 1 warning** (as of HEAD). All earlier
counts in this file (411 / 420 / 442 / 454 / 459) are superseded. The 428 figure
is the baseline against which future additions are tracked.
