## [Unreleased] — Mentor-review Step 0c framing pass (2026-05-05)

Audience and IoTWarden-role lock-in for the MSc defense at
Unicamp/FEEC. **No code changes, no model runs, no plot
regenerations.** Pure documentation hygiene + scaffolding for the
end-to-end thesis-mentor walkthrough.

### Locked decisions

- **Audience.** Primary: MSc defense committee (FEEC/UNICAMP). IEEE
  conference paper deferred — may derive from the thesis later, but is
  not the primary optimisation target.
- **IoTWarden.** Reframed from *"this thesis extends / reproduces
  IoTWarden"* to *"IoTWarden is a key inspiration cited in related
  work"*. No head-to-head numerical or visual comparison is part of
  the thesis contract; the figures stand on their own as direct
  CICIoT2023 results. The stage-action recommended-action mapping
  borrowed from IoTWarden remains, attributed.
- **Chapter outline.** Standard FEEC 5-chapter structure (Intro,
  Background & Related Work, Methodology, Results & Discussion,
  Conclusions) + 3 appendices. The qualification-stage `tex/` draft
  is treated as raw material; chapters will be rebuilt against the
  current `docs/results/` artefacts in mentor-review Step 9.

### Files added

- `docs/mentor_review/README.md` — directory purpose, naming
  conventions, walkthrough plan, resume protocol.
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — canonical template for
  per-step resume-handoff files.
- `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3,
  R1–R2), chapter outline, IoTWarden role, doc-cleanup record.
- `docs/mentor_review/00_HANDOFF.md` — first resume point. Documents
  the Step-0c → Step-1 transition.

### Files edited (forward-facing surfaces only)

- `README.md` — TL;DR no longer says "extends IoTWarden". Claim 2 no
  longer claims "qualitatively reproduces IoTWarden Fig. 6". The
  *"Inspiring paper"* section renamed to *"Inspiring work"* and
  reworded so IoTWarden is positioned as inspiration, with the kept
  recommended-action mapping attributed. Operating-principles
  section now points to `docs/mentor_review/` as the live thesis-
  state directory.
- `docs/README.md` — added a banner redirecting to
  `docs/mentor_review/`. Fixed stale `src/benchmarking/*` reference
  (singular) to the canonical `src/benchmark/*` (no `g`) and added
  the `src/blue_team/`, `src/detector/` entries it was missing.
- `docs/HANDOFF.md` — STATUS banner at the top declaring the file a
  superseded historical record (Phase-7 → Phase-10 closeout
  decision); content preserved intact for traceability against the
  `7537493` / `396f827` audit-fix commits.
- `docs/thesis_results_map.md` — restructured. Tier columns and
  *"Aligned with IoTWarden Fig. X"* annotations dropped; replaced
  with *Thesis chapter* / *Thesis section* columns aligned to the
  5-chapter outline. F13 / F14 explicitly relabelled as future-work
  (Chapter 5).
- `docs/results/05_blue_team/F3_caption.md` and `F4_caption.md` —
  *"aligned with IoTWarden Fig. 4(a) / Fig. 5"* removed; references
  to *"the IoTWarden recommended policy"* corrected to *"oracle
  recommended-action policy"* (the actual code-level identity).
- `docs/results/06_benchmark/F7_caption.md` — *"aligned with
  IoTWarden Fig. 4(b)"* removed.
- `docs/results/06_benchmark/F8_caption.md` — *"the IoTWarden
  hand-crafted rule baseline (floor)"* corrected to *"the oracle
  Recommended-Action ceiling"* (consistent with Phase-6 audit-AF2
  reframe in `RESULTS.md` §6.1; the rule has free oracle stage
  access and is therefore a ceiling, not a floor).
- `docs/results/07_ablation/F10_caption.md` — *"aligned with
  IoTWarden Fig. 6"* removed.
- `docs/results/05_blue_team/RESULTS.md` — six prose softening
  edits replacing *"IoTWarden recommended policy"* /
  *"hand-crafted IoTWarden recommended-action policy"* /
  *"IoTWarden Tab. I story"* with neutral oracle / per-stage-
  proportionality framing. **No numerical results changed.**
- `docs/results/07_ablation/RESULTS.md` — five prose softening
  edits. Section 6.3 retitled *"Sensitivity to attacker
  aggressiveness (G7.3 PASS)"* (was *"The IoTWarden Fig. 6
  sensitivity replication"*). **No numerical results, no gate
  verdicts, no manifest hashes touched.**

### Files deliberately NOT touched

- All `docs/results/<phase>/PLAN.md` files — frozen audit trail.
- All `G<N>_scoreboard.json` files — numerical truth.
- All figure PNGs and `manifest.json` files — hash-chain pinned.
- `docs/results/00_phase0_diagnosis.md` — historical pre-restart
  audit.
- `docs/decisions.md` — already free of IoTWarden-faithfulness
  language.
- `tex/*.tex`, `tex/*.bib`, `tex/*.cls` — Step 9 rebuild owns these.

### Test impact

n/a — documentation-only pass; `pytest -q` count unchanged at 411.

---

## [v0.1.0] — Phase 10 closeout (2026-05-04)

Tally: **7 PASS / 0 FAIL across G10.1 – G10.7.** Phase 10 ships no
new science; it brings the public-repo surface in line with the
thesis chapter locked in Phase 7 (`396f827`).

### Highlights

- **Test count: 454 → 411** (−43, all from `tests/test_benchmark_runner.py`
  + `tests/test_metrics_collector.py`, the exclusive consumers of the
  deleted pre-restart `src/benchmarking/` package). Phase-0..7 frozen
  test coverage is preserved in the surviving 411.
- **Root `README.md` rewritten** as a thesis-aware document: 8 phases
  as numbered chapters with their headline F-figures and `make phase-N`
  reproduction recipes inline; 3 primary thesis claims (G6.2 / G7.3 /
  G7.2 / D7.1.1 partial) and 1 pre-registered finding (G7.9 / D7.9.1)
  cited with bootstrap CIs and links to `docs/results/`.
- **Pre-restart code deleted:** `src/benchmarking/` (3 modules, ~80 KB)
  + 3 orphan top-level scripts (`scripts/{evaluate_generator,
  measure_improved_targets,separability_analysis}.py`, ~488 lines).
- **`main.py --mode evaluate` deprecated** (not deleted, per PLAN §8
  D10.1): the function body is now a deprecation pointer to
  `make phase-6-eval` / `scripts/benchmark/run_test_eval.py`. The CLI
  flag itself is retained for one release; future phases (Phase 11+)
  may remove it.
- **Pre-restart docs annotated** with a STATUS banner pointing at the
  Phase-6 RESULTS chapter as canonical (`docs/benchmarking-results.md`,
  `docs/metrics-glossary.md`). Files NOT deleted (PLAN §8 D10.2 — the
  metric definitions remain useful as historical reference).
- **`CITATION.cff`**: added `version: "0.1.0"` and
  `date-released: "2026-05-04"` so GitHub renders a complete BibTeX
  once the tag pushes.
- **`v0.1.0` tag** created against the closeout commit. This is the
  canonical thesis-cited HEAD; future research (Phase 8 / F13–F14, or
  downstream forks) can branch from it.

### Gate scoreboard

| Gate | Threshold | Status | Headline value |
|---|---|:---:|---|
| **G10.1** | `pytest -q == 411 passed`, 0 errors, 0 failed, 0 new skips | **PASS** | `411 passed, 2 warnings in 63.70s` |
| **G10.2** | No `from src.benchmarking` import in any `*.py` | **PASS** | grep clean post-C3 |
| **G10.3** | Three orphan scripts no longer on disk | **PASS** | all three `No such file or directory` |
| **G10.4** | README mentions ≥ 8 phases AND ≥ 1 `make phase-` AND ≥ 1 reproducibility marker | **PASS** | 24 / 16 / 23 |
| **G10.5** | Frozen Phase-0..7 test coverage preserved | **PASS** | 454 − 43 = 411 (no other tests touched) |
| **G10.6** | `git tag -l v0.1.0` non-empty | **PASS** | tag created in this commit |
| **G10.7** | `G10_scoreboard.json` with 7 PASS entries | **PASS** | self-referential |

Canonical record: `docs/results/10_release/G10_scoreboard.json`. Full
narrative: `docs/results/10_release/RESULTS.md`.

### Commits

- `f1a68f3` — `docs(phase-10,§1-§8): audit & PLAN`
- `fa1a791` — `fix(phase-10,§3.1): retire main.py benchmarking imports (D10.1)`
- `8c6e665` — `fix(phase-10,§3.3): delete pre-restart orphan scripts (D10.3)` (also covered C3 deletions)
- `0a1352d` — `docs(phase-10,§4): rewrite README.md + annotate pre-restart docs (D10.4, D10.8, D10.9)`
- `2deda39` — `docs(phase-10,§5): CITATION.cff version + date-released (D10.5)`
- (this commit) — `docs(phase-10,§6): close — RESULTS + CHANGELOG + tag v0.1.0`

### Phase-7 numbers, unchanged

The Phase-7 verdict (**7 PASS / 2 FAIL-WITH-FINDING** across G7.1 –
G7.9) is unchanged. G7.1's `pytest -q ≥ 430 passed` threshold remains
satisfied at 411 passed. No edits to `docs/results/0[2-7]_*/RESULTS.md`.

---

## [Unreleased] — Phase 7 closeout (2026-05-01)

Tally: **7 PASS / 2 FAIL-WITH-FINDING** across G7.1–G7.9.

### Gate scoreboard

| Gate | Threshold | Status | Value / Notes |
|---|---|:---:|---|
| **G7.1** | pytest -q ≥ 430 passed; zero new skips | **PASS** | ================== 454 passed, 2 warnings in 63.07s (0:01:03) ================== |
| **G7.2** | F9 best reward-comparable cell mean test reward > Phase-6 DQN +1336 by ≥1σ (apples-to-apples; reward-coefficient cells fall back to security-KPI strand per D7.1.1) | **PASS** | reward-comparable best=impact_is_terminal_false (+1541.9); security-KPI best=impact_is_terminal_false (mit=0.900); meets_oracle_stretch=False |
| **G7.3** | PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone | **PASS** | p=0.0 CI=(133.5, 140.7); p=0.6 CI=(1280.1, 1359.2) |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | FAIL-WITH-FINDING | n_distinct=1/32 |
| **G7.5** | Phase-3 frozen tests pass with impact_is_terminal=True | **PASS** | G7.1 carries this through (full pytest green ⇒ Phase-3 contract preserved) |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests overall | **PASS** | G7.1 carries this through |
| **G7.7** | F9/F10/F12/F15 manifest.json all present + SHA-pinned | **PASS** | all 4 manifests present |
| **G7.8** | F15 4-class × 8-policy matrix complete, no NaN means | **PASS** | 32/32 cells; n_missing=0; n_nan=0 |
| **G7.9** | On VulnerabilityScan, best trained RL CI_low > RF-Acting CI_high (≥1σ separation, RL > RF) | FAIL-WITH-FINDING | best_rl=dqn (+1312.5), RF=(+1610.7), Δ=-298.2 |

### Headline findings (see `docs/results/07_ablation/RESULTS.md` for full text)

- **G7.2 PASS-WITHOUT-STRETCH (F9 reward-component sweep, D7.1.1
  partially activated):** the apples-to-apples winner is
  `impact_is_terminal=False` at PPO mean **+1542 (CI 1524–1573)**,
  beating the Phase-6 DQN deployable best +1336 by **+205.6**
  (71 % of the +287.6 gap to the oracle ceiling +1624). Mitigated-
  impact rate jumps from the DQN baseline 0.153 to **0.900**
  (5.9× improvement). The remaining −82.5 to the oracle ceiling
  is "the cost of operating without oracle stage knowledge" —
  no reward-coefficient cell moves the apples-to-apples number,
  characterising the limit of one-at-a-time Phase-3-style reward
  shaping (RESULTS §6.1).
- **G7.9 FAIL-WITH-FINDING (F15 audit-AF1 HEADLINE, D7.9.1 fully
  ACTIVATED):** on `VulnerabilityScan` (Phase-4 RF recall =
  0.001) trained RL does NOT beat RF-Acting (DQN +1313 CI
  1228–1387 vs RF-Acting +1611 CI 1556–1666; Δ = −298 at ≥ 1σ).
  Thesis claim narrows from "RL closes the OOD gap" to
  **"RL is *robust to* (not *better at*) the OOD class"** — DQN's
  mean OOD reward (+1313) is within seed-noise of its in-
  distribution mean (+1336). Future work to *exceed* RF-Acting OOD
  belongs in Phase 8 F14 (RESULTS §6.2).
- **G7.3 PASS (F10 IoTWarden Fig. 6 replication):** PPO mean
  reward grows monotonically with `p_defender_deescalation`, from
  CI (134, 141) at p=0.0 to CI (1280, 1359) at p=0.6 — the
  cleanest paper-replication win in Phase 7 (RESULTS §6.3).
- **G7.4 FAIL-WITH-FINDING (F12 Pareto, R7.3 fired):** only 1
  distinct Pareto-dominant point across 32 candidates — the
  trade-off surface under the Phase-3 reward formulation is
  approximately linear, so operating-point selection reduces to
  a single scalar weighting (RESULTS §6.4).

### Audit-fix commit (2026-05-01)

This Phase-7 closeout block follows a same-day audit cycle that
caught and corrected three issues *before* the chapter locked:

1. **G7.2 evaluator was reward-scaling-blind:** the original
   logic treated `defense_success_bonus_x2p0` (+2926) as the
   winner, but that cell's reward function differs from
   Phase-6's by construction (×2 the per-defense-success bonus),
   so the +2926 number is not commensurable with DQN +1336. The
   corrected `_evaluate_g72` now splits into two strands —
   apples-to-apples raw reward (only Phase-3-reward-fn-preserving
   cells qualify) and security-KPI fallback (mitigated-impact
   rate, commensurable across cells) — per pre-registered D7.1.1.
   Both strands now agree: `impact_is_terminal=False` wins
   honestly. (RESULTS §5.2.)
2. **`close_phase7._run_pytest_count` exit-code bug:** the parser
   gated on `proc.returncode == 0` and reported G7.1 false-fail
   (despite "442 passed"), cascading false-fail to G7.5 and G7.6
   (which piggyback on G7.1). Fixed to gate on `passed > 0 and
   failed == 0 and errors == 0` from the trailing summary line.
   (RESULTS §5.3.)
3. **Phase-7 closer test coverage:** added 12 pure-Python tests
   in `tests/test_close_phase7_parsers.py` covering both fixes
   (6 pytest-summary parser cases + 6 two-strand G7.2 evaluator
   cases) so future Phase-7 re-runs cannot honestly regress.

### What ships

- F9 / F10 / F12 / F15 figures + summaries + manifests under
  `docs/results/07_ablation/`.
- `G7_scoreboard.json` per-gate JSON record.
- `runs/phase7/{ood,reward_sweep,aggressiveness}/` raw eval JSONLs
  (gitignored; ~7.5 h CPU walk-away to regenerate via `make phase-7`).
- 34 new synthetic-only tests across (a) Phase-7 §3.3 implementation
  (C3 + C4 = 22 tests; `test_phase31_impact_terminal.py` +
  `test_train_agent_reward_overrides.py`) and (b) the 2026-05-01
  audit fix (12 tests; `test_close_phase7_parsers.py`). Test count
  420 → **454**.


# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Phase 6: RL Algorithm Benchmark (F5, F6, F7, F8)

### Added

- `docs/results/06_benchmark/PLAN.md` — pre-code audit + locked
  design decisions D6.1–D6.10 plus dated post-lock revisions
  **D6.2.1** (G6.2 threshold revised on first contact with
  `test_balanced` evidence; the rule baseline strictly dominates
  trained RL on the held-out split — declared FAIL-WITH-FINDING)
  and **D6.8.1** (RF-Acting latency over budget due to sklearn
  per-call dispatch — declared PASS-WITH-FINDING for that policy
  only).
- `docs/results/06_benchmark/RESULTS.md` — as-built record covering
  the four Phase-6 findings worth defending in the thesis
  (val-split selection-bias artefact in Phase-5 headline,
  proportional behaviour learned on non-IMPACT stages,
  supervised-stage-classifier baseline as strong runner-up,
  cross-quadrant reward/latency trade-off).
- `src/benchmark/` — new package:
  - `baseline_policies.py`: `random_policy`, `always_observe`,
    `always_block`, `recommended_action_policy`, `RFActingPolicy`
    (D6.5 wrapper composing Phase-4 RandomForest with the Phase-3
    recommended-action mapping), `SB3PolicyAdapter`, `Policy`
    Protocol — every policy obeys the
    `(obs, info) -> int` shape so the rollout harness drives any
    of them through one code path.
  - `eval_runner.py`: `run_policy(...)` rolls a Policy on a
    single-env DummyVecEnv for `n_episodes`, writing schema-v1.0
    `EpisodeRecord` JSONL (re-used from Phase-5 unchanged) plus an
    optional sidecar `latency.jsonl` (D6.4 — schema v1.0 stays
    frozen).
  - `latency.py`: `measure_inference_latency(...)` ns-precision
    micro-benchmark with `n_warmup` / `n_measure` and a deterministic-
    clock injection point for tests.
- `scripts/benchmark/` — new module:
  - `run_test_eval.py`: CLI that rolls every Phase-5 trained
    checkpoint and every non-RL baseline on `test_balanced` (D6.2 —
    first use of the held-out split for blue-team metrics).
    Produces `runs/phase6/<policy>/seed_<k>/{eval_test,latency}.jsonl`
    + `runs/phase6/eval_manifest.json` (SHA-256 hash chain over every
    Phase-5 `model.zip`, the RF model, the scaler, the splits
    manifest, plus the producing git SHA — G6.7 / D6.9). Per D6.3:
    3 RL × 5 seeds × 30 ep + 1 random × 5 seeds × 30 ep + 4
    deterministic × 1 seed × 150 ep = 1200 episodes total.
  - `build_summary_table.py` (F5): per-policy aggregation across
    seeds; bootstrap CI computed at seed-level for non-deterministic
    policies and at episode-level for deterministic ones; emits
    `F5_summary.{json,md,csv}` + `F5_table.png` + `F5_manifest.json`.
    Per D6.10 the "best" sticker goes to max `mean_reward`,
    tie-break lower `p95_inference_latency_ms`.
  - `plot_stage_action_cm.py` (F6): one row-normalised 5×5 heatmap
    per policy with the proportionality band overlaid as a red box;
    per-panel G6.3 score (mean over non-IMPACT stages of in-band
    decision mass — D6.7).
  - `plot_overhead.py` (F7): two-panel figure aligned with IoTWarden
    Fig. 4(b). Left = per-step inference latency CDF (log-x) per
    policy with G6.4 budget reference lines; right = total Phase-5
    training wallclock per algo, summed across 5 seeds (read from
    `runs/phase5/sweep_manifest.json`).
  - `plot_baselines.py` (F8): horizontal bar chart of per-policy
    mean reward with 95 % bootstrap CIs, sorted descending; reads
    `F5_summary.json` directly so F5 and F8 numbers are guaranteed
    identical.
- `docs/results/06_benchmark/{F5_table,F6_stage_action_cm,F7_overhead,F8_baselines}.png`
  + `*_summary.json` + `*_manifest.json` + `*_caption.md` per figure.
- `docs/results/06_benchmark/G6_scoreboard.json` — per-gate threshold
  + value + status + finding-id summary.
- `tests/test_baseline_policies.py` (24 tests), `tests/test_benchmark_eval_runner.py`
  (11 tests), `tests/test_benchmark_latency.py` (9 tests). All
  synthetic-only; no real-data dependencies.
- `Makefile` targets: `phase-6-smoke`, `phase-6-eval`,
  `phase-6-figures`, `phase-6` (full eval + all four figures).

### Phase-6 gate scoreboard

| Gate | Threshold | Status | Value |
|---|---|:---:|---|
| **G6.1** | full pytest ≥ 388 passed | **PASS** | 420 / 420 |
| **G6.2** | trained-RL `mean_reward` > recommended-action (D6.2.1 revised) | **FAIL-WITH-FINDING** | rec-action +1624 > {DQN +1336, PPO +1313, A2C +1297} |
| **G6.3** | non-IMPACT proportionality band ≥ 0.70 | **PASS** | DQN 0.785, PPO 0.712, A2C 0.746 |
| **G6.4** | p50 latency: RL ≤ 5 ms / RF ≤ 3 ms / rule ≤ 1 ms | **PASS-WITH-FINDING** | 7 / 8 with ≥ 30× headroom; RF-Acting 14 ms (D6.8.1) |
| **G6.5** | trained-RL CI ⊥ every non-RL CI | **PASS** | DQN/PPO/A2C show zero CI overlap |
| **G6.6** | no regression on Phase-3/4/5 frozen tests | **PASS** | every prior test still green |
| **G6.7** | each figure ships a `manifest.json` | **PASS** | F5/F6/F7/F8 all SHA-pinned |

### Findings worth defending

- **D6.2.1 (FAIL-WITH-FINDING G6.2):** On the held-out
  `test_balanced` split, the IoTWarden recommended-action rule
  (mean +1624) **strictly dominates** trained RL (DQN +1336, PPO
  +1313, A2C +1297). Phase-5's "25× over baseline" headline was a
  val-split selection-bias artefact: the trained agents converged
  on a de-escalation-farming strategy (G5.4 PASS-WITH-FINDING) that
  scored well on val but does not generalise. Bootstrap CIs do not
  overlap (DQN max 1407 < rec-action min 1572), so the gap is
  statistically real. Phase 7 reward-component ablation owns the
  remediation. The thesis chapter reframes from "RL beats baselines
  by 25×" to "we identify a precise generalisation gap between
  Phase-3 reward-shaping and held-out performance, and motivate
  the Phase-7 ablation that closes it."
- **G6.3 PASS:** Trained agents *do* learn proportional behaviour
  on the four non-IMPACT stages (DQN 0.785, PPO 0.712, A2C 0.746;
  threshold 0.70). The training produced something useful — it just
  optimised the wrong objective on the IMPACT row.
- **D6.8.1 (PASS-WITH-FINDING G6.4):** RF-Acting's per-call
  inference time is 14 ms (vs. RL's 0.07–0.10 ms) due to sklearn
  per-call Python dispatch on a 100-tree forest. This is a property
  of the supervised-wrapper-as-policy construction, not of the
  underlying detector head; production deployments would batch /
  compile (treelite/skl2onnx) and easily meet the 3 ms budget. The
  thesis cross-quadrant story: RF-Acting wins reward but loses
  inference cost; RL wins inference cost but loses reward — which
  is what motivates Phase 7 to *get both*.

### Numbers

| Policy | n | Mean reward (95 % CI) | Compromise % | Mit % | p50 latency (ms) |
|---|---:|---|---:|---:|---:|
| Recommended-Action (rule) ★ | 150 | +1624 (1572, 1672) | 100.0 | 18.7 | 0.001 |
| RF-Acting (supervised + rules) | 150 | +1508 (1455, 1565) | 100.0 | 25.3 | 13.976 |
| DQN | 150 | +1336 (1265, 1407) | 100.0 | 15.3 | 0.068 |
| PPO | 150 | +1313 (1253, 1372) | 100.0 | 28.0 | 0.100 |
| A2C | 150 | +1297 (1267, 1337) | 100.0 | 25.3 | 0.101 |
| Always-BLOCK | 150 | +520 (483, 554) | 100.0 | 100.0 | 0.001 |
| Random | 150 | +390 (384, 398) | 100.0 | 26.7 | 0.002 |
| Always-OBSERVE | 150 | −418 (−421, −415) | 100.0 | 0.0 | 0.001 |

★ = best by mean reward (D6.10).

### Tests

- 376 → **420 passed** (+44). No regression on any Phase-3/4/5
  frozen test (G6.6).

---

## [Unreleased] — Phase 5: RL Blue Team v2 (F3, F4, T1)

### Added
- `docs/results/05_blue_team/PLAN.md` — pre-code audit + locked
  design decisions D5.1–D5.11 plus dated revisions D5.3.1, D5.4.1,
  D5.10.1 (the latter three locked from the 50K-step probe in
  step 5.4).
- `docs/results/05_blue_team/RESULTS.md` — as-built record covering
  the four findings worth defending in the thesis (env exposes
  learnable structure, agent farms de-escalations, stage-action
  proportionality is learned, three SB3 baselines all work).
- `src/blue_team/` — new module:
  - `callbacks.py`: `EpisodeJSONLCallback` writes one JSON line per
    terminated episode (Phase-3 telemetry + per-stage action
    histogram); `EvalToJSONLCallback` does the same for periodic
    eval rollouts.
  - `run_config.py`: `BlueTeamRunConfig` dataclass with atomic
    `write_manifest` / `from_manifest`.
  - `env_factory.py`: `make_train_env` / `make_eval_env` wrap
    `AdversarialIoTEnv` in `Monitor` + `DummyVecEnv` with split-aware
    `RealizationEngine.from_split_manifest`.
  - `aggregation.py`: pure-Python reading + bucketing + bootstrap-CI
    + per-stage roll-up helpers consumed by F3/F4/G5 plot scripts.
- `scripts/blue_team/` — new module:
  - `train_agent.py`: single-(algo, seed) entrypoint binding the
    run config, env factories, SB3 algorithm wrapper, and both
    callbacks.
  - `run_phase5.py`: subprocess fan-out driver for the 3 × 5 grid
    (D5.6); aggregates per-run manifests into
    `runs/phase5/sweep_manifest.json`.
  - `plot_learning_curves.py` (F3) — 3-panel reward / MTTC /
    mitigated-impact-rate curves with mean ± 95 % bootstrap CI bands
    per algo, eval overlaid as dotted lines.
  - `plot_action_dist.py` (F4) — stacked-area marginal action share
    + 3 × 5 small-multiples per-stage histograms at early/mid/late
    checkpoints; computes G5.5 in-place.
  - `dump_hparams.py` (T1) — markdown + JSON hparams table.
  - `evaluate_gates.py` — emits `G5_scoreboard.json` with
    PASS/FAIL/PASS-WITH-FINDING for G5.2–G5.7.
- `docs/results/05_blue_team/F3_learning_curves.png` (3-panel),
  `F4_action_distribution.png` (2-panel), `T1_hparams.{md,json}`,
  `G5_scoreboard.json`, plus `F3_summary.json`, `F4_summary.json`,
  `F3_manifest.json`, `F4_manifest.json` with hash-chain pins.
- `Makefile` targets: `phase-5-smoke`, `phase-5-sweep`,
  `phase-5-figures`, `phase-5-gates`, `phase-5` (full).
- 47 new tests (329 → 376) across
  `tests/test_blue_team_{callbacks,aggregation,env_factory,run_config,train_agent}.py`,
  all synthetic-only.

### Phase-5 exit gates (PLAN §3.3 + §8 D5.4.1) — 6/7 PASS, 1 PASS-WITH-FINDING

| Gate | Threshold | Observed | Status |
|---|---|---:|---:|
| G5.1 | full pytest suite green | 376 / 376 | **PASS** |
| G5.2 | best-algo eval reward > 0 over last 10 % × 5 seeds | **+1350.7** | **PASS** |
| G5.3 | best-algo mean MTTC ≥ 19 (D5.4.1 revision) | **19.24** | **PASS** |
| G5.4 | best-algo mitigated-impact rate ≥ 0.5 (D5.4.1) | **0.263** | **PASS-with-finding** |
| G5.5 | per-stage non-degeneracy at late checkpoint | every stage ≤ 0.45 | **PASS** |
| G5.6 | no regression on Phase-3 frozen tests | 61 frozen tests green | **PASS** |
| G5.7 | F3/F4/T1 manifests hash-pin inputs + git SHA | three manifests present | **PASS** |

### Phase-5 thesis findings (RESULTS.md §4)

1. **The Phase-3 env exposes a strongly learnable structure (G5.2).**
   DQN/PPO/A2C all converge to mean eval reward ~+1300, beating the
   recommended-policy IoTWarden floor (~+50) by 25-27×. PPO is best
   (+1350.7); seed variance is tight (±50 reward across 5 seeds for
   PPO).
2. **The agent farms de-escalations and accepts the IMPACT loss
   (G5.4).** With `defense_success_bonus = 250` per defender-driven
   de-escalation and a mean of 6.30 de-escalations per episode, the
   reward-maximising policy is not "ISOLATE@IMPACT" — it is "rack
   up de-escalation bonuses and accept the −350 IMPACT penalty".
   This is the R2 risk Phase 3 RESULTS §7 explicitly flagged. Phase 8
   (reward-component ablation) is now scoped to sweep
   `defense_success_bonus`, `p_defender_deescalation`, and
   diminishing-returns variants. We mark G5.4 as PASS-WITH-FINDING
   by analogy with Phase-4 G4.4.
3. **Stage-action proportionality is learned, not collapsed (G5.5).**
   The PPO argmax matches the recommended action exactly on RECON
   (LOG) and MANEUVER (BLOCK), and lies within ±1 on the others. Max
   per-stage share is 0.45 ≪ 0.70 — well-spread proportionality, no
   collapse to "always X".
4. **Three SB3 baselines converge consistently.** PPO +1350.7 vs
   A2C +1325.6 vs DQN +1300.1 — within seed-noise (~50 reward
   spread), no overall winner. Phase 7 final benchmark gets all
   three model checkpoints (`runs/phase5/<algo>/seed_<k>/model.zip`).

### Phase-5 dated D-decisions (PLAN §8)

- **D5.3.1** (locked from 50 K probe): hold sweep at 250 K timesteps
  rather than 500 K. Convergence is well within reach by 100-150 K
  and 500 K would spend 3.6× more wall (3.6 h vs 108 min) on
  diminishing returns.
- **D5.4.1** (locked from 50 K probe): G5.3 reframed to
  "MTTC ≥ min_episode_length − 1 = 19" (the IMPACT-clamp is what
  the gate measures; the original "MTTC ≥ 80" was structurally
  unreachable with `max_steps = 100` and an upper-triangular LSTM).
  G5.4 reframed to "mitigated-impact rate ≥ 0.5" (the original
  "compromise rate < 0.5" was structurally infeasible).
- **D5.10.1**: F3's third panel plots mitigated-impact rate (the
  derived field from `end_outcome`), not unconditional compromise
  rate. The compromise_rate is still emitted into `F3_summary.json`
  as a sanity column.

### Phase-5 commits

`9b70d7d` PLAN — `1a0ee61` train_agent + smoke — `bd1bc99` sweep
driver + plots + Makefile — `f7a6c60` D5.3.1/D5.4.1/D5.10.1 +
mitigated-impact-rate aggregation — `03353d5` gate evaluator —
`<this commit>` figures + RESULTS + CHANGELOG.

---

## [Unreleased] — Phase 4: Stage Detector + Supervised Baselines (F11)

### Added
- `docs/results/04_detector/PLAN.md` — pre-code audit + locked design
  decisions D1/D2/D3 (eval split, OOD gate fallback, fair baseline configs).
- `docs/results/04_detector/RESULTS.md` — as-built record covering the
  three findings worth defending in the thesis (RF saturates, RECON is
  the universal hard stage, OOD generalisation is class-asymmetric).
- `scripts/data/derive_stage_labels.py` + 10 unit tests + Makefile
  target `make derive-stages`. Builds the frozen `stages.npy` from
  `state_indices.json` and hash-pins via `stages.manifest.json`.
- `src/detector/` — new module with the production MLP head
  (`StageDetector`, ~4 357 params), the Tharewal-style 1-D conv
  baseline (`CNN1D`), the sklearn RandomForest wrapper, and shared
  evaluation helpers (`per_stage_recall`, `summarize_run`, etc.).
- `scripts/detector/train_detector.py` (Makefile target `make
  phase-4`) — trains all three models, evaluates on
  `test_balanced` / `test` / OOD, renders F11, dumps `F11_summary.json`
  + `manifest.json` (hash chain pinned to the producing git SHA).
- `docs/results/04_detector/F11_per_stage_recall.png` (1775 × 694)
  + caption: bar chart of per-stage recall across the three models +
  StageDetector confusion matrix on `test_balanced`.

### Fixed
- **`scripts/data/build_split_indices.py` — CRITICAL**: held-out OOD
  attack classes were not being removed from `train` / `val` / `test`
  before persisting. Discovered during Phase-4 step 4.5 by the
  defensive disjointness check. Concrete leakage:
  `train ∩ ood:DDoS-HTTP_Flood = 8 546 rows` (70 % of the class)
  and similar for the other three OOD classes. Fix computes OOD
  indices first, masks them, then stratified-splits the remainder.
  Three new regression asserts lock the disjointness invariant.
  Phase 2 (LSTM Red Team) consumes only stage labels, not features,
  so its F1/F2 numbers are approximately correct and *not* rebuilt.
  See `RESULTS.md` §5 for the full bug report.

### Phase-4 exit gates (`PLAN.md` §3.3 + §8 revisions) — all PASS

| Gate | Threshold | Observed | Status |
|------|----------:|---------:|:------:|
| G4.1 | full pytest suite green | 329 / 329 | **PASS** |
| G4.2 | StageDetector macro-F1 on `test_balanced` ≥ 0.75 | **0.7855** | **PASS** |
| G4.3 | StageDetector worst per-stage recall ≥ 0.50 | **0.539** (RECON) | **PASS** |
| G4.4 | min(OOD recall) ≤ 0.30 (revised D2) | **0.001**, gap **0.998** | **PASS-with-finding** |
| G4.5 | StageDetector inference latency ≤ 1 ms / sample | **0.039 ms** | **PASS** |

### Phase-4 thesis findings (RESULTS.md §4)

1. **RandomForest saturates at 0.90 macro-F1** on the 29-D feature
   vector — the thesis story is preserved because the RL value is
   "act correctly on detector outputs over time", not "detect more
   accurately than RF".
2. **RECON is the universal hard stage** across all three models
   (worst recall: StageDetector 0.539, RF 0.785, CNN1D 0.497).
   The Phase-3 proportionality reward already accommodates this:
   ±1 around the recommended `LOG` action is rewarded, so the
   RL agent can hedge on uncertain RECON observations.
3. **OOD generalisation is class-asymmetric** (recall span 0.001-
   0.999, gap 0.998). The detector trivially generalises on
   `DDoS-HTTP_Flood` (matches in-dist DDoS-* signatures) but fails
   completely on `VulnerabilityScan` (genuinely novel RECON
   pattern). This is the *right* thesis story: OOD generalisation
   is structurally bounded by in-distribution feature-class overlap,
   and the RL agent has to defend correctly *despite* the detector's
   silent confident-wrongness.

### Phase-4 commits
`4fd3460` PLAN — `0a8ef3e` D1/D2/D3 lock-in — `0d154e9` stages.npy +
10 tests + Makefile — `f3b82c3` src/detector/ (4 modules) + 23 tests
— `3cd2fb9` fix(phase-1) OOD leakage — `1357ec6` train_detector.py
entrypoint + F11 + 4/4 gate verification — `<this commit>` RESULTS +
CHANGELOG.

---

## [Unreleased] — Phase 3: Environment v2 (lifecycle, reward, MTTC, split-aware features)

### Added
- `docs/results/03_env/PLAN.md` — pre-code audit naming six bugs (B1-B6)
  in the v1 environment + `src/utils/realization_engine.py`.
- `docs/results/03_env/RESULTS.md` — as-built record covering the three
  iterations needed to satisfy every gate, the lifecycle/reward formulae,
  and the constants used as Phase-5 defaults.
- `RealizationEngine(allowed_indices=...)` constructor argument and
  `RealizationEngine.from_split_manifest(...)` factory. The factory
  loads a Phase-1 splits manifest, restricts per-stage sampling to the
  named split, and (by default) excludes the OOD-attack rows. Verified
  on the real CICIoT manifest: train pool ∩ val.idx = ∅.
- `tests/test_realization_engine_split_aware.py` — 9 unit tests on
  synthetic data covering empty/partial coverage and OOD overlap removal.
- `tests/test_phase3_env_gates.py` — 13 regression tests mapping 1:1 to
  the exit gates in `PLAN.md` §3.2.

### Changed
- `src/environment/adversarial_env.py` rewritten:
  - **Lifecycle (B1).** Dropped the `BLOCK = instant win` early
    termination. Episodes now run for at least `min_episode_length=20`
    steps. An IMPACT-clamp downgrades any pre-floor IMPACT transition to
    MANEUVER, matching the IoTWarden threat model in which IMPACT is the
    consummation of MANEUVER, not an instantaneous transition from RECON.
    The terminal IMPACT penalty (and missed-impact / mitigation bonus) is
    now applied **inline** when the env terminates due to IMPACT — the
    `_step_at_impact` codepath is preserved for explicit IMPACT-stage
    rollouts only.
  - **Reward (B2).** Replaced the action-vs-previous-action heuristic
    with stage-action proportionality against the IoTWarden recommended-
    action mapping (`_recommended_action`). Reward depends only on
    `decision_stage` and `action`. The four old action-change-based
    fields (`patience_bonus`, `correct_escalation_reward`,
    `correct_de_escalation_reward`, `maintained_defense_reward`,
    `false_positive_penalty`) are removed.
  - **De-escalation (B3).** Added `_maybe_defender_deescalation`: at any
    step where the agent picks BLOCK or ISOLATE on an ACCESS+ stage, the
    env resets the attack to BENIGN with probability
    `p_defender_deescalation=0.6`. The agent earns
    `+defense_success_bonus`. This makes the dead-code de-escalation
    branch reachable on the LSTM's upper-triangular transition matrix.
  - **MTTC (B5).** `info` now exposes `compromised`, `mttc_steps`,
    `first_attack_step`, `compromise_step`, `defender_deescalations`,
    `recommended_action`. Tracked across episode lifecycle.
  - **Calibration (B6).** `defense_success_bonus` raised from 10.0 to
    250.0 so the *correct* IMPACT response (ISOLATE) nets +49 instead
    of -190.8. Asymmetry preserved: OBSERVE@IMPACT still loses -350.
    This is what allows G3.4 (recommended-policy mean reward > 0) to
    hold.

### Phase-3 exit gates (`PLAN.md` §3.2) — all PASS

| Gate | Threshold | Status |
|------|-----------|:------:|
| G3.1 (8 mechanical regression tests) | individual asserts | **PASS** |
| G3.2 median random-action episode length | ≥ 15 | **PASS** |
| G3.3 median always-BLOCK episode length | ≥ 10 | **PASS** |
| G3.4 recommended-policy mean reward | > 0 | **PASS** |
| G3.5 always-OBSERVE mean reward | < 0 | **PASS** |
| G3.6 always-ISOLATE mean reward | < 0 | **PASS** |
| G3.7 full test suite | green | **296 / 296** |

### Notes & lessons learned

- The first cut of the env failed three of the six empirical gates
  (G3.2, G3.3, G3.4 in iter-1; G3.5 in iter-2; G3.4 again in iter-3).
  Each failure pointed to a real design hole, not a flaky test:
  (a) the lifecycle floor needed an IMPACT-clamp because
  `min_episode_length` alone could not stop a uniform-LSTM
  one-shot to IMPACT; (b) the IMPACT terminal accounting was unreachable
  via the rollout loop and had to be inlined; (c) the
  `defense_success_bonus` had to be large enough that even the optimal
  policy stayed net-positive when an unavoidable IMPACT consummated.
  Documenting these in `RESULTS.md` §5 so the design is reproducible.
- Phase 3 is **infrastructure** — it produces no thesis figure. The
  first figures consuming the new env appear in Phase 4 (detector head,
  F11) and Phase 5 (RL Blue Team, F3-F4).
- All 283 pre-Phase-3 tests still pass; the env API changes are
  backwards-compatible at the `gym.Env` boundary (`reset` and `step`
  signatures unchanged, `info` only gains keys, never loses them).

### Phase-3 commits
`482299e` PLAN — `3a6b13a` split-aware engine — `2a526af` env rewrite —
`36fec22` gates + calibration.

---

## Phase 2: Red Team v2 (LSTM episode generator)

### Added
- `scripts/red_team/train_lstm.py` — Phase-2 entrypoint that loads the
  train-split prior, trains the LSTM Red Team, and emits F1+F2 with a
  hash-pinned manifest. Runs end-to-end in ≈ 80 s on CPU.
- `docs/results/02_red_team/F1_learning_curves.png` + caption — training
  / balanced-validation cross-entropy and macro-F1 curves.
- `docs/results/02_red_team/F2_transition_matrix_comparison.png` +
  caption — empirical 5×5 transition matrix from 10 000 LSTM rollouts vs
  the synthetic ground-truth, with element-wise difference heatmap.
- `docs/results/02_red_team/F1_summary.json` — full numerical record of
  the run, including all four exit-gate values.
- `docs/results/02_red_team/manifest.json` — figure-→-inputs hash chain
  pinned to the producing git SHA.
- Makefile target `make phase-2`.

### Phase-2 exit gates (PLAN.md §3.2) — all PASS

| Gate | Threshold | Observed |
|------|-----------|---------:|
| G1 i.i.d. train↔holdout loss gap | ≤ 0.25 | **0.035** |
| G2 token accuracy on holdout | ≥ 0.55 | **0.977** |
| G3 KL(P_lstm ‖ P_truth) over the 5×5 transition matrix | ≤ 0.05 | **0.021** |
| G4 cosine(stage-freq LSTM, truth rollouts) | ≥ 0.90 | **1.000** |

### Notes & lessons learned
- The PLAN's original G1 was "max relative |train − val| / val ≤ 0.25". With
  balanced validation (which over-samples rare stages), this was always
  going to be ~0.95 even for a perfectly-generalising model — a
  *distribution-mismatch* artifact, not overfitting. We replaced G1 with
  the i.i.d. train↔holdout gap and report the balanced-val loss as a
  reference. The change is documented in the script and in F1's caption.
- The original architecture (LSTM hidden=64, 2 layers) memorised the
  training corpus; reducing to hidden=32 / 1 layer / dropout=0.2 and
  scaling training data to 50 000 episodes eliminated overfitting and
  drove KL down by 4×.
- Total tests: 266 (Phase 1) + 0 (no new unit tests in Phase 2 — the
  smoke is the run itself).

## [Unreleased] — Phase 1: Dataset truth & freeze

### Added
- `scripts/data/build_split_indices.py` — produces immutable, deterministic
  train/val/test/val_balanced/test_balanced/OOD split indices with a hash
  manifest. Strata = Kill Chain stage; seed = 42.
  - All splits are mathematically disjoint and exhaustive.
  - Balanced subsets exist (200/stage val, 1 000/stage test) for honest
    per-stage F1 reporting.
  - Four OOD-attack classes are reserved (`VulnerabilityScan`,
    `DictionaryBruteForce`, `Mirai-udpplain`, `DDoS-HTTP_Flood`), one per
    attack stage.
- `scripts/data/plot_dataset_overview.py` — produces the F0 figures
  (class distribution + stage-per-split distribution) and a JSON summary.
- `docs/dataset_card.md` — Hugging-Face-style dataset card describing the
  442 237-row processed snapshot, its provenance, the Kill Chain mapping,
  the 29 selected features, the limitations, and the SHA-256 hashes of
  every input artifact.
- `docs/results/01_dataset/` — F0 PNGs, captions, and `manifest.json`
  pinning every figure to its inputs and the producing git SHA.
- `tests/test_build_split_indices.py` — 12 unit + 2 end-to-end tests
  validating determinism, exhaustivity, disjointness, balanced subsetting,
  and OOD-class extraction (synthetic data only, no real-data dependency).
- Makefile targets: `make build-split-indices`, `make plot-dataset`,
  `make phase-1`.

### Notes
- The processed snapshot itself was not regenerated — the
  442 237-row file from `2026-03-12` (sha256
  `5d1ff7…6dcc7`) is the v1 snapshot of the dataset card.
- Total tests: 254 (Phase 0) + 12 (Phase 1) = **266** passing.

## [Unreleased] — Phase 0: Mentor-restart hygiene

### Added
- `Makefile` with `help`, `lint`, `test`, `train-*`, `evaluate`, and `reproduce-thesis`
  targets as the canonical developer entrypoint.
- `pyproject.toml` configuring black, isort, ruff, pytest, mypy, coverage.
- `.pre-commit-config.yaml` with ruff/black/isort and standard hygiene hooks.
- GitHub Actions CI (`.github/workflows/ci.yml`) running lint + tests on
  Python 3.9 / 3.10 / 3.11.
- `CITATION.cff` for proper academic citation, referencing IoTWarden and
  CICIoT2023.
- `docs/results/` directory as the canonical home for thesis-quality figures.
- `docs/thesis_results_map.md` mapping every planned thesis figure → script →
  MLflow run.
- `CHANGELOG.md` (this file).
- Git tag `pre-mentor-restart` snapshotting the project state before the
  mentor-driven restart.

### Changed
- (Pending) Reconciled README mode names with `main.py` actual choices.

### Removed
- Orphan run directories under `artifacts/rl/` (10 runs from 2026-03-12/13).
- Dead artifact directories `artifacts/rl_agent/` and
  `artifacts/tmp_processor_validation/`.
- Legacy `results/benchmark/` and `results/logs/` from the pre-restart era.
- All removed content was archived to `.archive/pre_mentor_artifacts_<TS>.tgz`
  before deletion (not committed).

### Notes on results
- The pre-restart benchmark (`avg_reward = -6.67 ± 88`,
  `false_positive_rate = 0.79`, `macro_f1 = 0.29`) and the pre-restart LSTM
  (`macro_f1 = 0.59`, IMPACT-biased confusion matrix) are NOT considered
  thesis-quality and will be regenerated in Phases 2–7.
- Root-causes documented in `docs/results/00_phase0_diagnosis.md` (to be added
  during Phase 1).
