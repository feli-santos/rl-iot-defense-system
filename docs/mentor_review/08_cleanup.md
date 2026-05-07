# Step 8 — Cross-cutting Cleanup Wave: Mentor Memo

**Closed:** 2026-05-07 ~12:00 BRT
**Author (agent):** mentor agent (Step 8)
**Reviewed phase / scope:** cross-cutting cleanup of all deferred
items from Steps 1–7 + R1 smoke-reproducibility harness (R1 agreed
in the mentor-honest pushback discussion in lieu of a full-stack
re-run; R2 thesis-grade reproducibility appendix scoped for Step 9
LaTeX).
**Status:** **completed** — every Step-8 task in `07_HANDOFF.md §5`
shipped + R1 harness shipped + 9 Step-3/4/5/6 doc-fixes consolidated
into one commit; all gate verdicts preserved verbatim; pytest
411 / 411 at HEAD.

---

## 1. Headline

Step 8 closes the cross-cutting cleanup pile that the audit-first
protocol deferred from each per-phase review. **Eight commits
landed on `mentor-review/step-8-cleanup`** off `main = 99b2452`:

1. **Phase 0** (`99b2452` G2 of Step 7 + G1 of Step 8 already on
   `main`). Step-7 ablation audit merged in via
   `mentor-review/step-7-ablation` → `main`; new branch
   `mentor-review/step-8-cleanup` cut at `99b2452`.
2. **`364267b` — `fix(scoreboard): unify G4/G5/G7 to Phase-6-native
   status enum + finding_id`**. Step-8 task #2 (F3 schema
   unification, 07_HANDOFF.md §5).
3. **`3022e3d` — `fix(manifest,phase-7): pin upstream
   phase5/phase6/phase1 SHAs in F9/F10/F12/F15`**. Step-8 task #1
   (F2 manifest pins).
4. **`10b958c` — `fix(manifest,phase-6): pin Phase-2 LSTM SHA in
   eval_manifest.json (Step-6 F3)`**. Step-8 task #3.
5. **`3773542` — `docs(phase-2): backfill RESULTS.md with F1/F2
   model-selection-criterion narrative`**. Step-8 tasks #4 + #5.
6. **`807a383` — `docs(mentor-review,step-8): cross-cutting
   doc-fix batch (Step-3/4/5/6 deferred items)`**. Step-8 task #6
   (15 files patched: cross-cutting README + 6 per-phase docs +
   2 captions + 1 scoreboard + 5 source-code docstrings/serialiser
   hardenings).
7. **`8d07f26` — `feat(repro): add R1 smoke-reproducibility
   harness + close F1 follow-up`**. Step-8 tasks #7 + #9.
8. **(Pending this commit)** — `docs(mentor-review,step-8): Step 8
   cross-cutting cleanup memo + HANDOFF`. Adds this file +
   `08_HANDOFF.md`.

**Verdict:** **PASS** (no findings, no fixes deferred — Step 8 is
itself the cleanup wave, and every prior-phase deferral has either
shipped or been documented as future-work / Step-9 LaTeX). Test
suite green at 411 / 411; R1 harness verdict `PASS — hash chain
intact; scoreboard schemas valid` (458 OK / 0 FAIL / 2 KNOWN-DIVERGENCE
/ 6 SKIP).

---

## 2. What was reviewed (Step-8 inputs)

### Mentor handoffs read in full

- `docs/mentor_review/00_framing.md` + `00_HANDOFF.md` — protocol invariants.
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 F4 (scoreboard asymmetry).
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2 F1 (LSTM seed/manifest divergence) + F2 (model-selection criterion).
- `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 F1/F2/F3 (asymmetry + env-ctor contract + reward-component count + MTTC clarification).
- `docs/mentor_review/04_detector.md` + `04_HANDOFF.md` — Step-4 F1/F2/F4 (post-3cd2fb9 train counts + scoreboard asymmetry + per-stage-recall column header).
- `docs/mentor_review/05_blue_team.md` + `05_HANDOFF.md` — Step-5 F1/F2/F3/F4/F5/F6.
- `docs/mentor_review/06_benchmark.md` + `06_HANDOFF.md` — Step-6 F1/F2/F3/F4/F5/F6.
- `docs/mentor_review/07_ablation.md` + `07_HANDOFF.md` — Step-7 §5 (Step-8 checklist) + §7 (context recipe) + §8 (open questions).

### Candidate-decision intake (`07_HANDOFF.md §8`)

The eight open candidate decisions were all surfaced at session
start and locked in the mentor-honest pushback discussion:

| Q | Decision | Rationale |
|---|---|---|
| **Q1** Step-2 F1 | (b) doc-only seed-justification | Saturated G3+G4 thresholds; LSTM consumes labels not features; ~30 min retrain ⇒ cascade through Phase-6/7 manifests for re-validation of an already-PASS-strong-margin result |
| **Q2** Step-2 F2 model-selection | balanced-val cross-entropy via early-stop | Implicit in code (`generator_trainer.py`); macro-F1 logged for monitoring only |
| **Q3** doc-fix batching | batch into Step-8 single pile | mentor-recommended throughout the chain |
| **Q4** scoreboard field name | `status` (NOT `verdict`) | matches Phase-6-native `G6_scoreboard.json` |
| **Q6** F12 y-axis re-emit | skip | Step-7 §6.4 + caption rewrites sufficient |
| **Q7** `compromise_rate=1.0` paragraph | author in Step 9 LaTeX | RESULTS.md §6.1 caveat already shipped in Step 7 |
| **Q8** Phase-8 vs Phase-10 routing | (a) Phase 8 was skipped | detached commit chain `8d5dd67`...`a969fd6`+`v0.1.0` is unambiguous evidence; loop continues Step 8 → 9 → 10 |
| **R1 / R2** reproducibility | R1 in Step 8 + R2 in Step 9 | Mentor-honest pushback retired the full-stack re-run in favour of audit-by-hash-chain + thesis appendix |

---

## 3. Findings shipped this session (priority-ordered)

The following are not "findings" in the per-phase sense — they
are the **Step-8 deliverables** organised by which prior-phase
finding they close.

### 3.1 Scoreboard schema unification (Step-8 task #2 / F3)

**Files changed**: `scripts/ablation/close_phase7.py` (rewrite of
`_write_scoreboard` + new `_resolve_status_finding` helper +
`_summary_table` reads `status`), `scripts/blue_team/evaluate_gates.py`
(per-gate `status` enum + nested `gates.G5.5.per_stage` re-keying),
`scripts/detector/close_phase4.py` (NEW, mirrors close_phase7
shape), `docs/results/04_detector/G4_scoreboard.json` (NEW),
`docs/results/05_blue_team/G5_scoreboard.json` (re-emitted v2.0,
`per_algo_summary` byte-identical to v1.0),
`docs/results/07_ablation/G7_scoreboard.json` (re-emitted v2.0
via `/tmp/migrate_g7_scoreboard.py` to preserve hand-added
`note_post_lock_2026-05-06` annotations). **Acceptance**:

```
$ for f in docs/results/0{4,5,7}*/G[457]_scoreboard.json; do
    jq -r '[.. | objects | .status? // empty] | sort | unique' $f
    grep -c '"passes"' $f
  done
```

returns valid enum members on all three; `passes` key count = 0
across all three. Per-gate `status` + `finding_id` cross-link
table:

- **G4.1**=SKIP / **G4.2**=PASS / **G4.3**=PASS / **G4.4**=PASS-WITH-FINDING (D2.1) / **G4.5**=PASS
- **G5.1**=SKIP / **G5.2**=PASS / **G5.3**=PASS / **G5.4**=FAIL-WITH-FINDING (D5.4.1) / **G5.5**=PASS / **G5.6**=SKIP / **G5.7**=PASS
- **G7.1**=PASS / **G7.2**=PASS-WITHOUT-STRETCH (D7.1.1) / **G7.3**=PASS / **G7.4**=FAIL-WITH-FINDING (R7.3) / **G7.5**=PASS / **G7.6**=PASS / **G7.7**=PASS / **G7.8**=PASS / **G7.9**=FAIL-WITH-FINDING (D7.9.1)

### 3.2 Phase-7 manifest SHA pins (Step-8 task #1 / F2)

**Files changed** (4 producers + 4 manifests):
`scripts/ablation/plot_aggressiveness.py`,
`scripts/ablation/plot_pareto.py`,
`scripts/ablation/plot_reward_ablation.py`,
`scripts/ablation/plot_ood_robustness.py` — all four CLI args
extended with `--phase{1,5,6}-*-manifest`; manifest emission
blocks rewritten to include the missing pins. Plus
`docs/results/07_ablation/F{9,10,12,15}_manifest.json` patched
in-place via `/tmp/backfill_phase7_manifest_pins.py`
(idempotent migration). **Acceptance**:

```
$ for m in F9 F10 F12 F15; do
    jq '.inputs.phase5_sweep_manifest != null
        and .inputs.phase6_eval_manifest != null
        and .inputs.phase1_splits_manifest != null' \
      docs/results/07_ablation/${m}_manifest.json
  done
```

returns `true` for all four. Pinned SHAs:
`phase5_sweep_manifest = cc7454320d9acca8...`,
`phase6_eval_manifest = c4a60a8f51d65095...`,
`phase1_splits_manifest = c8574094e7b914fd...`.

### 3.3 Phase-2 LSTM SHA pin in Phase-6 eval_manifest (Step-8 task #3 / Step-6 F3)

**Files changed**: `scripts/benchmark/run_test_eval.py:494` adds
`phase2_lstm` + `phase2_lstm_path` fields to
`eval_manifest::input_hashes`; bumps `schema_version` 1.0 → 1.1.
`docs/results/06_benchmark/RESULTS.md` §8 footnote acknowledges
the post-Step-8 explicit pin and the pre-Step-8 implicit chain.
The on-disk `runs/phase6/eval_manifest.json` is intentionally NOT
mutated (gitignored, hash-chain-immutability per the audit-first
invariant); next `make phase-6` will re-emit with the LSTM pin
and the four figure manifests will atomically re-pin.

### 3.4 Step-2 F1 + F2 — Phase-2 RESULTS.md backfill (Step-8 tasks #4 + #5)

**File created**: `docs/results/02_red_team/RESULTS.md` (~263
lines, mirrors the Phase-3-7 RESULTS skeleton). Records:

- **§5.1 Model-selection criterion**: balanced-val cross-entropy
  via early-stop (`generator_trainer.py:362-369`,
  `early_stopping_patience=8`, `best_observed_val_loss` tracked);
  macro-F1 logged for monitoring only. Rationale: CE is the
  principled selection criterion for a next-token-generative
  model whose downstream env consumes stochastic samples (G3
  KL=0.021 + G4 cosine=0.99999 saturation confirms fidelity).
- **§5.2 Seed justification (`seed=42`)**: full propagation
  chain documented; saturated-gate evidence rules out
  seed-sensitivity at the precision G3/G4 measure. Multi-seed
  sweep flagged as post-thesis future work.
- **§5.3 Splits-manifest SHA divergence forensic**: the recorded
  `82aa1214...` (pre-3cd2fb9) vs. on-disk `1e99d596...`
  (post-3cd2fb9) divergence is documented as a documentation
  drift, not a correctness divergence (the LSTM consumes only
  stage tokens; the leakage-fix corrected per-flow features
  orthogonal to the LSTM's input space; G4=0.99999 saturation
  rules out any model-level divergence).

### 3.5 Cross-cutting doc-fix batch (Step-8 task #6)

15 files patched in commit `807a383`:

- **Cross-cutting**: `docs/results/README.md` — new "Per-phase
  scoreboard / manifest asymmetry" rollup table closing
  Step-1 F4 + Step-2 F4 + Step-3 F1 + Step-4 F1/F2 + Step-5 F2
  in a single audit-trail-completeness narrative.
- **Step-3 F2**: `src/environment/adversarial_env.py` —
  `AdversarialIoTEnv` class-level docstring documents the
  direct-construction contract (production callers must use
  `make_train_env` / `make_eval_env`).
- **Step-3 F3 + Step-5 F6**: `docs/reward-shaping.md` full
  rewrite (retired the v1 `correct_escalation_reward` /
  `patience_bonus` description; canonical six-reward-signals +
  three-asymmetric-guardrails decomposition + explicit
  "MTTC is a metric, not a reward term" paragraph).
  `docs/results/03_env/RESULTS.md` §3 — Step-5 F6 cross-link
  paragraph clarifying the six-vs-nine count divergence.
- **Step-4 F1**: `docs/results/04_detector/RESULTS.md` §3.0 —
  post-3cd2fb9 train counts (281 420 rows; per-stage breakdown).
- **Step-4 F4**: `docs/results/04_detector/RESULTS.md` §3.1
  column headers explicitly read "BENIGN<br/>recall" / etc.
- **Step-5 F3**: `scripts/blue_team/evaluate_gates.py` —
  `_select_best_algo` docstring rewrite (acknowledges the
  triple disagreement; pins the docstring to match the
  `(-mean_reward, -mean_mttc)` sort key).
- **Step-5 F4**: `src/blue_team/env_factory.py` module docstring
  + `make_eval_env` doc — split is caller-supplied via
  `spec.split`, not factory-defaulted.
- **Step-5 F5**: `docs/experiments-mlflow.md` top-of-file scope
  banner — Phase 5 onwards uses JSONL + run-manifest logging
  (PLAN D5.6), not MLflow; document preserved as historical
  reference.
- **Step-6 F1**: `docs/results/06_benchmark/RESULTS.md` G6.1 row
  + §9 footnote; `G6_scoreboard.json::gates.G6.1.value` → 411.
- **Step-6 F2**: `docs/results/06_benchmark/F5_caption.md`
  rewritten with ⓞ oracle marker + 82 %-of-ceiling framing;
  `G6_scoreboard.json::summary.{headline_finding,secondary_finding}`
  + `gates.G6.2.finding_summary` retired the older "rule
  baseline strictly dominates RL" wording.
- **Step-6 F4**: `src/benchmark/eval_runner.py` — `run_policy`'s
  `seed` parameter docstring acknowledges no-op at env layer.
- **Step-6 F5**: `docs/results/06_benchmark/F6_summary.json`
  sanitised in place (NaN → null);
  `scripts/benchmark/plot_stage_action_cm.py` adds
  `_nan_to_none()` pre-pass + `json.dumps(allow_nan=False)`
  enforcement so any future regression fails loudly.
- **Step-6 F6**: `docs/results/06_benchmark/F7_caption.md` —
  "G6.4 FAIL" → "G6.4 PASS-WITH-FINDING per D6.8.1".

### 3.6 F1 follow-up — −43 deleted tests trace (Step-8 task #7)

**Forensic confirmed**:

```
$ git --no-pager log --all --diff-filter=D \
    --pretty=format:'%h %s' --name-only -- 'tests/**'
281860a fix(phase-10,§3.2): delete dead src/benchmarking/ package + tests (D10.2)
tests/test_benchmark_runner.py
tests/test_metrics_collector.py
```

The `git log --all --diff-filter=D` scan returns **exactly** these
two files across the entire repo history. The −43 test deletion
(442 → 411) traces *exclusively* to commit `281860a`
(`tests/test_benchmark_runner.py` 25 tests + `tests/test_metrics_collector.py`
18 tests = 43 tests). The pre-restart `src/benchmarking/` package
(note 'g') is the dead-code source; the live `src/benchmark/`
package and its 28 tests are unaffected. **No orphan-test
regression.**

### 3.7 R1 smoke-reproducibility harness (Step-8 task #9)

**File created**: `scripts/reproducibility_smoke.py` (~370 lines).
Validates the audit-first hash chain end-to-end by checking that
every committed Phase-N manifest's input SHA-256 pins match the
on-disk SHA-256 of those inputs *right now*. Includes a
pre-registered `_KNOWN_DIVERGENCES` table for the documented
Step-1 F4 / Step-2 F1 splits-manifest drift, so the harness
reports those as `KNOWN-DIVERGENCE` (with finding_id
cross-link) rather than `FAIL`.

**Run output (verbatim)**:

```
total OK:               458
total FAIL:             0
total KNOWN-DIVERGENCE: 2  (pre-registered, see _KNOWN_DIVERGENCES)
total SKIP:             6  (gitignored inputs not on disk)

VERDICT: PASS — hash chain intact; scoreboard schemas valid.
```

### 3.8 SKIPPED — Step-8 task #8 (F7(b) F12 y-axis re-emit)

Per Q6 candidate decision: skipped. The Step-7 §6.4 + caption
rewrites already address the misleading-2D-Pareto risk for the
defense.

---

## 4. Itemised doc-fix batch (07_HANDOFF §5 task #6 acceptance)

Every doc-fix listed in the master inventory (Step-3 F1–F3 + Step-4
F1/F2/F4 + Step-5 F1–F6 + Step-6 F1–F5 + Step-6 F6 caption-only) was
either landed in this Step-8 commit pile or was already closed by an
earlier Step-8 phase:

| Finding | Origin | Step-8 closure |
|---|---|---|
| Step-3 F1 (asymmetry) | 03_HANDOFF.md L48 | `docs/results/README.md` rollup (§3.5) |
| Step-3 F2 (env ctor contract) | 03_HANDOFF.md L48 | `adversarial_env.py` class docstring (§3.5) |
| Step-3 F3 (reward-component count + MTTC) | 03_HANDOFF.md L48 | `docs/reward-shaping.md` rewrite (§3.5) |
| Step-3 F4 (MTTC IMPACT-clamp) | 03_HANDOFF.md L48 | deferred Step 9 LaTeX (R2 axis) |
| Step-4 F1 (post-3cd2fb9 counts) | 04_HANDOFF.md L88-119 | `04_detector/RESULTS.md` §3.0 (§3.5) |
| Step-4 F2 (no G4 scoreboard) | 04_HANDOFF.md L114-128 | Phase 1 `364267b` (`G4_scoreboard.json` created) |
| Step-4 F3 (dead-code branch) | 04_HANDOFF.md | deferred (would force model re-emit; not worth the cascade) |
| Step-4 F4 (per-stage-recall header) | 04_HANDOFF.md L131-148 | `04_detector/RESULTS.md` §3.1 (§3.5) |
| Step-4 F5 (R1 thesis cross-link) | 04_HANDOFF.md | deferred Step 9 LaTeX |
| Step-5 F1 (G5.4 cross-link) | 05_HANDOFF.md L120-128 | Phase 1 `364267b` (G5.4 → status:FAIL-WITH-FINDING + finding_id:D5.4.1) |
| Step-5 F2 (implicit hash chain) | 05_HANDOFF.md L132-145 | `docs/results/README.md` rollup footnote ⁵ (§3.5) |
| Step-5 F3 (tie-break docstring) | 05_HANDOFF.md L149-158 | `evaluate_gates.py::_select_best_algo` docstring (§3.5) |
| Step-5 F4 (`make_eval_env` docstring) | 05_HANDOFF.md L162-170 | `env_factory.py` module + function docstring (§3.5) |
| Step-5 F5 (MLflow scope) | 05_HANDOFF.md L173-180 | `docs/experiments-mlflow.md` scope banner (§3.5) |
| Step-5 F6 (Phase-3 reward-component count) | 05_HANDOFF.md L184-194 | `docs/reward-shaping.md` rewrite + `03_env/RESULTS.md §3` cross-link (§3.5) |
| Step-6 F1 (test-count drift) | 06_HANDOFF.md L100-114 | `06_benchmark/RESULTS.md` G6.1 + `G6_scoreboard.json` (§3.5) |
| Step-6 F2 (audit-AF2 reframe) | 06_HANDOFF.md L116-129 | `F5_caption.md` + `G6_scoreboard.json::summary` (§3.5) |
| Step-6 F3 (Phase-2 LSTM SHA pin) | 06_HANDOFF.md L132-148 | Phase 3 `10b958c` |
| Step-6 F4 (run_policy seed semantics) | 06_HANDOFF.md L150-165 | `eval_runner.py:run_policy` docstring (§3.5) |
| Step-6 F5 (NaN → null RFC-7159) | 06_HANDOFF.md L168-175 | `F6_summary.json` sanitised + `plot_stage_action_cm.py` hardened (§3.5) |
| Step-6 F6 (F7 caption verdict drift) | 06_HANDOFF.md L177-181 | `F7_caption.md` (§3.5) |
| Step-6 observation (DQN MANEUVER) | 06_HANDOFF.md L184-190 | not a finding; flagged Step-7 hand-off (already shipped) |
| Step-7 F1/F4/F5/F6/F7a/F8 | shipped in Step-7 commit `11cba37` | ✅ |
| Step-7 F2 (Phase-7 manifest pins) | shipped Step 8 task #1 (§3.2) | ✅ |
| Step-7 F3 (scoreboard schema) | shipped Step 8 task #2 (§3.1) | ✅ |
| Step-7 F6/F13/F14 (post-thesis future work) | 07_ablation.md §7 | retired with Q8=Phase-8-skipped framing |

---

## 5. Tests & checks

- `pytest -q` at HEAD `8d07f26`: **411 passed, 0 failed, 0 skipped**.
- `python -m scripts.reproducibility_smoke`: **VERDICT PASS** (458 OK / 0 FAIL / 2 KNOWN-DIVERGENCE / 6 SKIP).
- `jq '.gates[].status'` across G4 / G5 / G7 scoreboards returns members of the canonical enum (PASS, PASS-WITH-FINDING, PASS-WITHOUT-STRETCH, FAIL-WITH-FINDING, FAIL, SKIP).
- `grep -c '"passes"'` returns 0 across G4 / G5 / G7 scoreboards.
- `jq '.inputs | keys'` across F9 / F10 / F12 / F15 manifests includes all of `phase5_sweep_manifest`, `phase6_eval_manifest`, `phase1_splits_manifest`.

---

## 6. Risks & residual debts

- **R8.1** (low impact): the on-disk `runs/phase6/eval_manifest.json`
  is the pre-Step-8 SHA `c4a60a8f...` and remains the artefact every
  Phase-6 figure manifest pins. The producer-script fix lands in
  code but is not retroactively applied — re-running `make phase-6`
  will produce the new manifest with the LSTM pin and a fresh SHA;
  the four figure manifests will atomically re-pin via
  `make phase-6-figures`. Documented in
  `docs/results/06_benchmark/RESULTS.md §8`. No defense-grade
  risk: the existing pin is byte-perfect on disk.

- **R8.2** (zero-impact): Phase-1 + Phase-2 manifests preserve
  the pre-3cd2fb9 splits-manifest SHA `82aa1214...` as audit-trail
  records of the artefact at lock time. The R1 harness reports
  these as `KNOWN-DIVERGENCE` with the Step-2 F1 finding_id and
  RESULTS.md §5.3 cross-link. No defense-grade risk: documented
  divergence, empirically rules out correctness divergence
  (G4 cosine 0.99999).

- **R8.3** (none, listed for completeness): Step-4 F3 (dead-code
  branch in `train_detector.py:545-552`) was deferred because
  any change to that file would force a Phase-4 re-run with a
  cascade through every downstream phase. Cosmetic; documented;
  no functional impact.

---

## 7. Cross-step rollup — what is now defense-ready

The following Step-1..Step-8 deliverables are now self-contained
and auditable for the defense:

1. **Hash chain end-to-end**: every committed Phase-N manifest's
   input SHA-256 pins are verifiable on-disk via
   `python -m scripts.reproducibility_smoke` (5-second runtime).
   Two pre-registered known-divergences with finding-id
   cross-links + RESULTS.md narrative.

2. **Unified scoreboard schema**: G4/G5/G6/G7 all ship the
   Phase-6-native `status` enum + `finding_id` cross-link. Step-9
   LaTeX automation can read all four scoreboards with one jq
   query.

3. **Phase-2 audit trail**: RESULTS.md backfilled with the
   model-selection-criterion + seed-justification + manifest-SHA
   forensic. Closes the only Phase-without-RESULTS gap.

4. **Reward-shaping documentation**: `docs/reward-shaping.md`
   matches the as-built v2 reward; MTTC explicitly clarified
   as a metric, not a reward term.

5. **Cross-phase hash chain self-containedness**: every
   Phase-7 figure manifest pins phase5+phase6+phase1 explicitly,
   not transitively. Phase-6 eval_manifest pins Phase-2 LSTM
   (post-Step-8 schema v1.1).

6. **Step-9 LaTeX inputs prepared**:
   - 82 %-of-oracle-ceiling reframe propagated through
     `06_benchmark/RESULTS.md`, `F5_caption.md`, and
     `G6_scoreboard.json` summary block.
   - "RL is robust to (not better at) the OOD class" reframe
     (D7.9.1) shipped in Step 7 (commit `11cba37`).
   - `compromise_rate=1.0` thesis-framing paragraph scoped to
     Step 9 LaTeX (RESULTS.md §6.1 caveat already shipped).

---

## 8. Hand-off — what Step 9 owns

`08_HANDOFF.md` carries the Step-9 LaTeX framing checklist. The
high-level scope:

1. **`tex/` rebuild** of all chapters using the as-built RESULTS
   files as canonical sources (not the older PLAN files).
2. **Reproducibility appendix (R2)** — one-page `tex/appendices.tex`
   listing every artefact SHA + git_sha + the exact command to
   reproduce, sourced from the per-phase `manifest.json` files
   and the unified `G[N]_scoreboard.json` records.
3. **Thesis-framing paragraphs** scoped to Step 9 (not Step 8):
   - Q5 — §6.1's "82 % of oracle ceiling" claim becomes the
     canonical thesis result; older "RL beats baselines by 25×"
     framing is retired.
   - Q7 — `compromise_rate=1.0` paragraph (deferred from Step 8
     per Q7 decision).
4. **Step-10 release** owns the post-LaTeX `v1.0.0` tag (Q8=Phase-8-skipped
   framing; F6 MANEUVER coupling + F13/F14 reframed as future-work
   rather than Phase 8).

Step 9 is **read-mostly** — the canonical numerical record across
all phases is now stable and self-contained. The work is to
produce the LaTeX prose that cites the RESULTS files faithfully.
