# Phase 6 — RL Algorithm Benchmark: Plan

> Pre-code audit + design contract for Phase 6. Same protocol as
> Phases 3–5 (`docs/results/blue-team-training/PLAN.md`). The PLAN is committed
> *before* any implementation; once §8 is locked, every subsequent commit
> must cite the §-number it implements (`feat(phase-6,§3.1.2): ...`).

## 1 — Why Phase 6 exists

Phase 6 produces the thesis's **standalone empirical contribution**
beyond reproducing the IoTWarden claim: the **first head-to-head
benchmark of three model-free RL algorithms (DQN, PPO, A2C) plus four
non-RL baselines on the CICIoT2023 kill-chain defense problem**.

Phase 5 proved each of the trained trio converges above the random
baseline (G5 PASS). Phase 6 evaluates those frozen checkpoints on the
held-out `test_balanced` split (10 seeds × 300 episodes) and turns the
results into the four publishable comparisons
that anchor the empirical chapter:

- **F5** — final security metrics table (Tier 1; our contribution).
- **F6** — per-algorithm stage × action confusion matrices (Tier 1;
  our contribution).
- **F7** — computational overhead — latency CDF + training time
  (Tier 1; aligned with IoTWarden Fig. 4(b)).
- **F8** — RL vs random / always-OBSERVE / always-BLOCK / RF-acting /
  recommended-action (Tier 2; our contribution).

The thesis claim Phase 6 supports verbatim:

> *"DQN, PPO, and A2C, trained on the Phase-3 environment against the
> Phase-2 LSTM Red Team, all dominate four non-RL baselines (random,
> always-OBSERVE, always-BLOCK, RandomForest-acting-policy) and the
> hand-crafted IoTWarden recommended-action policy on (mean reward,
> MTTC, mitigated-impact rate) on the held-out test split, while
> staying within a 5 ms / step inference budget. The action histogram
> further reveals that a non-trivial fraction of reward is harvested
> from defender-driven de-escalation, motivating the Phase-7
> reward-component ablation."*

The IoTWarden head-to-head was officially **retired** as a phase
(`ecfb584`); the user's contribution is the cross-algorithm benchmark,
not a re-implementation of someone else's hparams. Phase 5's
`info["recommended_action"]` already serves as the IoTWarden
rule-based baseline.

## 2 — Audit findings (what we have / what is missing)

### A1. Phase-5 substrate is ready and frozen

`src/blue_team/` exposes everything Phase 6 reuses without modification:

- **`callbacks.py::EpisodeJSONLCallback`** — frozen JSONL schema v1.0
  written per episode. Phase 6 baseline rollouts must conform to this
  schema so `aggregation.py` aggregates them identically.
- **`callbacks.py::EpisodeRecord`** dataclass — 17 fields including
  `action_counts` (length-5 list) and **`action_counts_by_stage`**
  (`Dict[str, List[int]]` keyed `"0".."4"`). The latter is exactly the
  5×5 matrix F6 needs; no schema change required for F6.
- **`env_factory.py::make_eval_env(spec, generator_path, dataset_path,
  splits_manifest, seed)`** — split-aware eval env constructor; F5/F6/F7/F8
  call this with `spec.split="test_balanced"`.
- **`aggregation.py`** — already aggregates `mean_reward`, `mean_mttc`,
  `compromise_rate`, `mitigated_impact_rate`, `mean_episode_length`,
  per-action shares, per-stage action histograms. F5 reuses these names
  verbatim.
- **`run_config.py::EnvConfigSerializable`** — freezes the env config
  hash so eval-time env matches training-time env.

`runs/phase5/<algo>/seed_<k>/model.zip` — 15 trained checkpoints
(3 algos × 5 seeds), with `runs/phase5/sweep_manifest.json` recording
training wallclock per run (input to F7's right panel).

### A2. The Phase-3 env exposes everything Phase 6 baselines need

`info` dict (per step):

| Key | Type | Use in Phase 6 |
|---|---|---|
| `compromised` | `bool` | F5: `compromise_rate` |
| `mttc_steps` | `Optional[int]` | F5: `mean_mttc` |
| `first_attack_step` | `int` | (already aggregated) |
| `compromise_step` | `Optional[int]` | (already aggregated) |
| `defender_deescalations` | `int` | F5: G5.4 follow-up |
| `recommended_action` | `int` ∈ [0,4] | **The IoTWarden rule-based baseline policy** |
| `attack_stage` | `int` ∈ [0,4] | F6: row index of stage×action CM |
| `last_action` | `int` | (already aggregated) |
| `outcome` | `str` | F5: `mitigated_impact_rate` |

The `recommended_action` is the per-stage proportional mapping locked in
Phase 3: `{BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK,
IMPACT→ISOLATE}`. Phase 6's `recommended_action_policy(info)` is a
one-line wrapper around this field.

### A3. The Phase-4 RandomForest detector is ready

`artifacts/detector/random_forest.joblib` — sklearn `RandomForestClassifier`
loaded by `src/detector/random_forest.py`. Macro-F1 ≈ 0.79 on
`test_balanced` (per Phase-4 RESULTS.md), per-stage recall RECON 0.785,
IMPACT 1.000. Phase 6 wraps it as the **"RF-acting-policy" baseline**:
the RF predicts the kill-chain stage from the per-step features, then
the recommended-action mapping converts predicted stage → action. This
tests "supervised stage classifier + rules" vs. "learned RL", which is
exactly the thesis claim that justifies RL over straight supervised
defense.

### A4. The Phase-5 frozen checkpoints are on disk

```
runs/phase5/
├── a2c/seed_{0..4}/model.zip
├── dqn/seed_{0..4}/model.zip
├── ppo/seed_{0..4}/model.zip
└── sweep_manifest.json   # input hash anchor; carries training wallclock
```

These are gitignored. If absent on a fresh checkout, re-run
`make phase-5-sweep PHASE5_TIMESTEPS=250000` (~108 min CPU). Phase 6
treats them as **read-only inputs**; no retraining.

### A5. Gaps Phase 6 must fill

1. **No `src/benchmark/` package.** Phase 6 creates it.
2. **No `scripts/benchmark/` directory.** Phase 6 creates it.
3. **No baseline-policy rollout harness.** Phase 6's `eval_runner.py`
   plays the role `train_agent.py` plays for trained models: it rolls
   the policy on the eval env and emits schema-v1.0 JSONL.
4. **No per-step inference latency capture.** JSONL schema v1.0 has no
   latency field. Phase 6 introduces a **sidecar `latency.jsonl`**
   (one row per step or one row per (run_id, n_measure) batch). Schema
   v1.0 stays frozen (D6.4).
5. **Phase-5 eval rollouts were on `val_balanced`.** F5/F6/F7/F8 are the
   thesis's headline numbers and should report on the held-out
   `test_balanced` split (D6.2). Phase 6 re-rolls every checkpoint and
   every baseline on `test_balanced` — ~10 min CPU total.

### A6. What is **not** missing (avoid scope creep)

- Reward-component ablation → Phase 7.
- Attack-aggressiveness sweep → Phase 7.
- OOD-class evaluation → Phase 8.
- `impact_is_terminal` env-config flag → Phase 7 (D6.6).
- Re-implementing IoTWarden's DQN → retired (`ecfb584`).

## 3 — Concrete deliverables

### 3.1 Code

| Path | Purpose | LoC est. |
|---|---|---|
| `src/benchmark/__init__.py` | package init; re-export public symbols | 10 |
| `src/benchmark/baseline_policies.py` | Five `(obs, info) -> int` callables: `random_policy`, `always_observe`, `always_block`, `recommended_action_policy`, `RFActingPolicy(rf_path)`. Plus a `Policy` Protocol and a thin `SB3PolicyAdapter(model)` that exposes the SB3 `model.predict` under the same call signature. | ~150 |
| `src/benchmark/eval_runner.py` | `run_policy(policy, env, n_episodes, jsonl_path, run_id, *, latency_path=None, deterministic=True, seed=None)` — rolls the policy, writes EpisodeRecord-v1.0 JSONL, optional sidecar latency JSONL. | ~120 |
| `src/benchmark/latency.py` | `measure_inference_latency(policy_callable, obs_pool, *, n_warmup=100, n_measure=1000) -> np.ndarray` (ns-precision via `time.perf_counter_ns`). | ~60 |
| `scripts/benchmark/__init__.py` | empty | 1 |
| `scripts/benchmark/run_test_eval.py` | CLI: rolls the 15 trained checkpoints + 5 baselines on `test_balanced` (30 episodes × 5 seeds for non-deterministic baselines, or 1 seed × 30 episodes for deterministic ones); writes to `runs/phase6/<policy>/seed_<k>/eval_test.jsonl` (+ `latency.jsonl`); produces `runs/phase6/eval_manifest.json` with SHA-256 of every input + git SHA. | ~180 |
| `scripts/benchmark/build_summary_table.py` (**F5**) | reads all eval JSONLs + latency JSONLs → `docs/results/benchmark/main_results.{json,md,csv,png}` + `main_results_manifest.json`. Columns: `mean_reward`, `mean_mttc`, `compromise_rate`, `mitigated_impact_rate`, `mean_episode_length`, `mean_inference_latency_ms`, `p95_inference_latency_ms`. | ~150 |
| `scripts/benchmark/plot_stage_action_cm.py` (**F6**) | one 5×5 row-normalised heatmap per algo from `action_counts_by_stage` aggregated across 5 seeds × 30 episodes; → `stage_action_cm.png` + `stage_action_proportionality.json` + `stage_action_proportionality_manifest.json`. | ~120 |
| `scripts/benchmark/plot_overhead.py` (**F7**) | left panel: per-step inference latency CDF (one curve per policy) from `latency.jsonl`; right panel: training-time bar from `runs/phase5/sweep_manifest.json` (sum-over-seeds wallclock per algo). → `overhead.png` + `latency_profile.json` + `latency_profile_manifest.json`. | ~140 |
| `scripts/benchmark/plot_baselines.py` (**F8**) | bar of `mean_reward` across all 8 policies with 95 % bootstrap CI; horizontal line at `recommended_action` floor for visual reference. → `baselines.png` + `reward_ranking.json` + `reward_ranking_manifest.json`. | ~110 |

### 3.2 OPTIONAL — Phase-3.1 patch (`impact_is_terminal`) — DEFERRED

**Decision (D6.6): defer to Phase 7.**

Rationale: G5.4 PASS-WITH-FINDING (de-escalation farming) is best
diagnosed through a *coordinated* reward-component + termination-rule
sweep (Phase 7's natural scope), not through a one-off env-config flag
shipped inside Phase 6. Including it now bloats the phase, requires
risking a fresh re-train, and risks contaminating F5/F6/F7/F8 numbers.
Gate G6.6 forbids regressing the Phase-3 frozen tests anyway. Phase 7
already owns this story; it stays there.

### 3.3 Tests (synthetic only — no real-data dependencies)

| Test file | What it pins |
|---|---|
| `tests/test_baseline_policies.py` | `random_policy(obs, info, rng)` returns int ∈ [0,4]; `always_observe`/`always_block` return their constants; `recommended_action_policy(info)` returns `info["recommended_action"]`; `RFActingPolicy` with a stub RF (mock predict→stage) returns the recommended action for the predicted stage; `SB3PolicyAdapter` round-trips a stub SB3 model. |
| `tests/test_benchmark_eval_runner.py` | Round-trip of a 1-episode rollout on a stub gym env emits a JSONL line that loads cleanly through `aggregation.py`'s loader (schema v1.0 compliance); manifest hash chain reconstructs. |
| `tests/test_benchmark_latency.py` | `measure_inference_latency` returns array of length `n_measure` with median > 0; deterministic with a monkey-patched clock; n_warmup samples are not in the returned array. |

Target: **376 → 388–392 passed** (12–16 new tests).

### 3.4 Exit gates G6.1 .. G6.7

| ID | Gate | Threshold |
|---|---|---|
| **G6.1** | Full pytest suite | green; ≥ 388 passed (376 baseline + ≥ 12 new) |
| **G6.2** | F5 table populated for {dqn, ppo, a2c, random, always_observe, always_block, recommended_action, rf_acting} on every metric column | every cell non-NaN; trained-RL row `mean_reward` **strictly exceeds** the `recommended_action` row's `mean_reward` *(threshold revised — see D6.2.1 below)* |
| **G6.3** | F6 per-algo proportionality band, **non-IMPACT stages only** (BENIGN/RECON/ACCESS/MANEUVER) | for each of {dqn, ppo, a2c}, `(Σ over s∈non-IMPACT, a with |a−rec(s)|≤1) / (Σ over s∈non-IMPACT, a)` ≥ **0.70** |
| **G6.4** | F7 latency budget (median per-step inference) | RL ≤ **5 ms**, RF ≤ **3 ms**, rule-based ≤ **1 ms** |
| **G6.5** | F8 separation | trained-RL `mean_reward` 95 % bootstrap CI does **not overlap** any of {random, always_observe, always_block, rf_acting, recommended_action}'s CIs |
| **G6.6** | No regression on Phase-3/4/5 frozen tests | `tests/test_phase3_env_gates.py`, `test_adversarial_env.py`, `test_blue_team_*.py`, `test_detector.py` all unchanged & green |
| **G6.7** | Reproducibility | F5/F6/F7/F8 each ship a `manifest.json` with SHA-256 of input JSONLs + RF model SHA + git SHA at production time |

### 3.5 Figures + caption sketches

- **F5** — table + value heatmap. *Caption:* "Final security metrics on
  the held-out `test_balanced` split. Trained RL (DQN/PPO/A2C, 5 seeds ×
  30 deterministic episodes/seed) vs. four non-RL baselines (random
  policy seeded 5 ways; deterministic always-OBSERVE / always-BLOCK /
  recommended-action; RF-acting-policy = Phase-4 RandomForest stage
  classifier composed with the recommended-action mapping). Bold = best
  per column; † = within 95 % bootstrap CI of the best. No retraining
  was performed; checkpoints are the Phase-5 `runs/phase5/` artefacts."
- **F6** — 1×3 grid of 5×5 row-normalised heatmaps (rows: kill-chain
  stage 0=BENIGN..4=IMPACT; cols: action 0=OBSERVE..4=ISOLATE). Diagonal
  band (|a−rec(s)|≤1) highlighted; G6.3 score printed as a sub-caption
  per algo.
- **F7** — left: per-step inference latency CDF, log-x; right: training
  wallclock bar (sum-over-seeds, h) per algo. Annotation: "RL forward
  pass + env step on macOS / Apple silicon, single-process. CPU-only."
- **F8** — horizontal bar chart, mean reward ± 95 % bootstrap CI, sorted
  descending; horizontal line at recommended-action floor.

## 4 — Sequencing table (commits)

| # | Commit message | Files | Wallclock |
|---|---|---|---|
| C1 | `docs(phase-6): audit & PLAN — F5+F6+F7+F8 RL benchmark` | `docs/results/benchmark/PLAN.md` | this commit |
| C2 | `feat(phase-6,§3.1.1-3): src/benchmark/{baseline_policies,eval_runner,latency}.py + tests` | code + 3 test files | ~30 min author / <1 s tests |
| C3 | `feat(phase-6,§3.1.4): scripts/benchmark/run_test_eval.py + Makefile target` | sweeper + `make phase-6-eval` | ~15 min author / ~10 min CPU |
| C4 | `feat(phase-6,§3.1.5): F5 final security metrics table` | `build_summary_table.py` + outputs | ~15 min |
| C5 | `feat(phase-6,§3.1.6): F6 stage×action CMs` | `plot_stage_action_cm.py` + outputs | ~15 min |
| C6 | `feat(phase-6,§3.1.7): F7 latency CDF + training-time` | `plot_overhead.py` + outputs | ~15 min |
| C7 | `feat(phase-6,§3.1.8): F8 cross-policy reward bars` | `plot_baselines.py` + outputs | ~15 min |
| C8 | `docs(phase-6): close — RESULTS + CHANGELOG + G6 scoreboard` | `RESULTS.md`, `CHANGELOG.md`, `benchmark_acceptance.json` | ~30 min |

Total: ~1 day author time; <30 min CPU wallclock.

## 5 — What we are NOT doing

- ❌ Re-training PPO / A2C / DQN — Phase-5 checkpoints are frozen by
  contract (D6.1).
- ❌ Reward-component ablation (R1..R5 sweep) — Phase 7.
- ❌ Attack-aggressiveness sweep — Phase 7.
- ❌ OOD-class evaluation — Phase 8.
- ❌ `impact_is_terminal` env-config flag — Phase 7 (D6.6).
- ❌ Modifying any Phase 1/2/3/4/5 artefact, schema, gate, or test.
  JSONL schema v1.0 is frozen.
- ❌ Re-implementing IoTWarden's DQN — officially retired
  (`ecfb584`).
- ❌ Detector-augmented observations as a Phase-6 baseline (Phase 9).

## 6 — Risks tracked

| ID | Risk | Mitigation |
|---|---|---|
| **R6.1** | Test-split rollouts produce numbers that disagree with Phase-5 val-split numbers in a way that confuses the thesis narrative. | F5 / F8 captions explicitly state "test_balanced split, 30 deterministic episodes/seed"; manifest records both splits' SHA hashes; Phase-5 numbers stay reported as "validation-split selection metrics". |
| **R6.2** | RF-acting-policy looks artificially weak because RF was trained on per-flow features but the RL agent observes engineered window state — comparison framed unfairly. | Document explicitly in F8 caption + RESULTS §5; comparison's purpose is "supervised stage classifier + rules vs. learned RL on the same env", not a feature-level detector battle. |
| **R6.3** | Latency measurement is noisy on macOS (no isolated CPUs, JIT warmup). | `measure_inference_latency` uses n_warmup=100 + n_measure=1000; figures use **CDF + median + p95 + p99** (robust to outliers); accept 2–3× pessimism vs. server hardware; document the measurement environment in F7 manifest (`platform.platform()`, `platform.processor()`). |
| **R6.4** | F6 G6.3 fails because trained agents over-de-escalate (G5.4 finding) and the action histogram lives outside the proportionality band. | Gate G6.3 already excludes IMPACT stages by construction (D6.7). If still fails, escalate to `PASS-WITH-FINDING` like G5.4 — narrow gate further to BENIGN-only and document in §8 as `D6.3.1`. Do **not** silently relax. |
| **R6.5** | Phase-3 frozen tests start failing because something Phase 6 touched in `src/environment/` regressed the contract. | Hard-stop: G6.6 forbids any change to `src/environment/`. Phase 6 is purely a consumer. CI: `pytest -q tests/test_phase3_env_gates.py tests/test_adversarial_env.py` runs after every Phase-6 commit. |
| **R6.6** | `runs/phase5/` is missing on the machine (gitignored), blocking the C3 sweep. | C3 will refuse to run with a clear error message pointing to `make phase-5-sweep PHASE5_TIMESTEPS=250000` (~108 min CPU). Does not block C2 (synthetic-only tests). |
| **R6.7** | The RF model loaded from `artifacts/detector/random_forest.joblib` was fitted on a different feature scaler than the one the env uses. | The Phase-4 RF was trained on the same `data/processed/ciciot2023/` scaler that Phase-3 env consumes; Phase 6's `RFActingPolicy` reads features through the env's emitted obs slice (the per-step row of the window) without re-scaling. Tested in `tests/test_baseline_policies.py` with a stub RF + stub env emitting a known feature vector. If the SHA of the scaler changes between training and eval, F5_manifest catches it. |

## 7 — Cross-references to thesis chapter outline

- F5 → Chapter "Empirical Results" §6.1 (Security Metrics Across
  Algorithms and Baselines).
- F6 → Chapter "Empirical Results" §6.2 (Per-Stage Action Behaviour).
- F7 → Chapter "Empirical Results" §6.3 (Computational Cost).
- F8 → Chapter "Empirical Results" §6.4 (vs. Non-RL Baselines).

All four feed the chapter's headline claim (see §1).

Phase 7 follow-ups already enumerated:
- Reward-component ablation explaining the G5.4 / R6.4 de-escalation
  finding (F9, F10).
- `impact_is_terminal` swept as a termination-rule ablation (D6.6).
- Attack-aggressiveness sweep (F12).

## 8 — Locked design decisions

These are locked at PLAN-commit time. Subsequent `feat(phase-6,…)`
commits cite §-numbers; revisions of these decisions get explicit
`D6.X.1` follow-up entries with date + rationale, mirroring the
Phase-5 protocol (D5.3.1, D5.4.1, D5.10.1).

| ID | Decision | Rationale |
|---|---|---|
| **D6.1** | Phase 6 consumes the **frozen Phase-5 checkpoints**; no retraining. | HANDOFF mandate; G5 already passed. Retraining would invalidate the F3/F4/T1 numbers Phase 5 already shipped. |
| **D6.2** | Eval split for F5/F6/F7/F8 is **`test_balanced`** (the held-out split, untouched by training-time decisions). Phase-5 val-split numbers stay reported as model-selection metrics. | Honest use of the held-out split; correct for the thesis's headline numbers. Both splits' hashes pinned in `main_results_manifest.json`. |
| **D6.3** | **300 deterministic eval episodes per policy** (non-deterministic: 10 seeds × 30 ep; deterministic baselines: 1 seed × 300 ep). | Updated from original 5-seed/150-ep design; 10-seed/300-ep run locked in commit f6766ce and sweep_manifest confirmed impact_is_terminal=false. |
| **D6.4** | Per-step inference latency captured in a **sidecar `latency.jsonl`**, not by extending `EpisodeRecord` v1.0. | Schema v1.0 stays frozen (Phase-5 contract); sidecar evolves independently. |
| **D6.5** | "RF-acting-policy" baseline = `recommended_action_policy(predicted_stage = rf.predict(features))`, where `features` is the **last row of the env's observation window** (the most recent step). | Tests "supervised stage classifier + rules vs. learned RL" cleanly; reuses two existing Phase-3 / Phase-4 contracts (`recommended_action` mapping + `RandomForest.predict`) without inventing new ones. |
| **D6.6** | `impact_is_terminal` env-config flag deferred to Phase 7. | Avoids Phase-6 scope creep; Phase 7 already owns reward / termination ablations. |
| **D6.7** | F6 proportionality gate G6.3 evaluated on **non-IMPACT stages only** (BENIGN, RECON, ACCESS, MANEUVER). | Direct consequence of the G5.4 PASS-WITH-FINDING; Phase 6 should not regress that finding by silently relaxing the threshold for all stages. The IMPACT-stage exclusion is the *finding*, not a workaround, and is documented as a Phase-7 hand-off. |
| **D6.8** | Latency budget G6.4 thresholds: **RL ≤ 5 ms / step, RF ≤ 3 ms / step, rule-based ≤ 1 ms / step**, all measured at p50. | Matches Phase-4 G4.5 detector-head budget (≤ 1 ms) plus the SB3 policy-network forward-pass cost; aligned with IoTWarden Fig. 4(b)'s order of magnitude. CPU-only on macOS / Apple silicon. |
| **D6.9** | Manifests use the SHA-256 hash chain convention from Phase 1 / 4 / 5. Each Phase-6 figure ships `<F>_manifest.json` containing: input JSONL SHAs, RF model SHA, scaler SHA (if used), splits-manifest SHA, git SHA at production time. | Reproducibility contract; matches existing `docs/results/blue-team-training/{F3,F4}_manifest.json` schema. |
| **D6.10** | The "best algo" sticker shown in F5 / F8 is selected by **mean reward** on `test_balanced` (tie-breaker: lower p95 latency). | Consistent with Phase-5's `best_algo` field in `G5_scoreboard.json`. Empirical result (10-seed/300-ep): A2C +1336.6 is best deployable RL. |

### Follow-up D-decisions (post-PLAN-lock)

| ID | Decision | Date | Rationale |
|---|---|---|---|
| **D6.2.1** | **G6.2 threshold revised on first contact with `test_balanced` evidence.** Original threshold ("trained-RL `mean_reward` ≥ 5× recommended-action") was anchored on Phase-5's val-split estimate that the recommended-action floor was ~+50. The held-out `test_balanced` split revealed that the oracle ceiling is **+1684.8** (10 seeds × 300 ep), *higher* than every trained RL algo (A2C +1336.6, PPO +1320.2, DQN +1313.0). Phase 5's "25× over baseline" claim was therefore a **selection-bias artefact of the val-split rollouts**: the trained agents converged on a de-escalation-farming strategy (G5.4 PASS-WITH-FINDING) that scored well on val but does not generalise to the held-out split. **Revised G6.2 threshold:** trained-RL `mean_reward` strictly exceeds recommended-action `mean_reward` (i.e., a 1× floor). The original 5× threshold is **declared FAIL on `test_balanced`** and the failure is the headline finding of Phase 6, not a gate-tuning issue. Best deployable = A2C +1336.6, capturing 79.3 % of oracle. | 2026-04-30 (updated 2026-06-04) | The honest move is to surface the finding, not relax the gate silently. Phase 7's reward-component ablation now has a sharply-defined target: "show why the oracle outperforms trained RL on the held-out split, and identify a structural change that closes the gap." |
| **D6.8.1** | **G6.4 declared FAIL-WITH-FINDING for the RF-Acting baseline only.** Empirical p50 inference latency on `test_balanced` (Apple silicon CPU, single process): RL trio 0.07–0.10 ms ✓ (≥ 30× headroom on the 5 ms budget), rule baselines ≤ 0.002 ms ✓ (≥ 500× headroom on the 1 ms budget), **RF-Acting 13.83 ms ✗** (budget 3 ms; ~4.6× over). RF-Acting trades inference cost for **higher reward than the RL trio** (+1516.0 vs. +1313..+1336.6); any deployment would naturally batch or compile the forest. The remaining seven policies all pass G6.4; the headline RL latency claim (RL ≈ 146× faster than RF-Acting) is unaffected. | 2026-04-30 (updated 2026-06-04) | RF-Acting wins reward but loses inference cost; RL wins inference cost but loses reward — the cross-quadrant trade-off motivating Phase-7. |
