# Step 06 — Phase 6 Benchmarks Review

> Mentor memo. Written in present tense, lead with the verdict.
> Findings cite gate IDs (G6.1–G6.7) and `file:line`.
> Counterpart handoff: `docs/mentor_review/06_HANDOFF.md`.

**Reviewer (agent):** Cline (mentor-review session 7)
**Branch:** `mentor-review/step-6-benchmarks` cut off `main` @ `014a7e3`
(Step-5 G2 merge commit) on 2026-05-06 21:51 BRT.
**Reviewed phase / scope:** Phase 6 (final benchmark on the held-out
`test_balanced` split — F5 final security-metrics table, F6 stage ×
action confusion matrices, F7 computational overhead, F8 cross-policy
mean-reward bars, exit gates G6.1–G6.7). Phase 6 consumes the 15
frozen Phase-5 model checkpoints + the four non-RL baselines + the
oracle Recommended-Action policy + the Phase-4 RandomForest detector
wrapped as `RFActingPolicy`.

---

## 1. Verdict

**`PASS-WITH-FIXES`**

Phase 6 is the most thoughtful phase package in this review.
Six of seven exit gates clear cleanly (G6.1, G6.3, G6.5, G6.6,
G6.7 PASS; G6.4 PASS-WITH-FINDING for RF-Acting at 14 ms vs. 3 ms
budget per registered finding D6.8.1). The seventh, G6.2, is
mechanically `FAIL-WITH-FINDING` — and is the **headline result of
Phase 6**, registered as the audit AF2 reframe in `RESULTS.md` §6.1:
the recommended-action rule policy reads `info["attack_stage"]` and
is therefore the **oracle ceiling**, not a deployable competitor; the
best deployable agent is **DQN at +1336, capturing 82 % of the oracle
ceiling +1624 without seeing stages**, with the remaining +288 reward
named as the Phase-7 reward-component-ablation target. This is a
mature, defensible scientific move — the kind a defense committee
will reward — and it fully resolves the "trained RL beat baselines
by 25×" framing that Phase-5 (val-split) carried implicitly.

The implementation is byte-perfect on the hash chain that Step-5 F2
worried about: Phase-1 splits manifest SHA `1e99d59682…` is pinned
explicitly in `runs/phase6/eval_manifest.json:input_hashes`, alongside
the scaler and the RF model. Every one of the 15 Phase-5
`runs/phase5/<algo>/seed_<k>/model.zip` checkpoints is pinned by
SHA-256 with on-disk verification matching. The `EnvConfigSerializable`
spec passed to every rollout is the explicit `split="test_balanced",
exclude_ood=True` (`run_test_eval.py:143`); the Step-1 OOD-leakage
invariant is honoured for the test side. 44 new tests pin the public
API of the new `src.benchmark` package; the suite is green at
**411 passed**.

Five findings are filed: F1 is a documentation drift (RESULTS.md
claims 420 passed; on-disk reality is 411), F2 is the audit-AF2
reframe not having reached F5 caption + scoreboard `headline_finding`,
F3 is the Phase-2 LSTM SHA not being pinned in the eval manifest
(parallel to Step-5 F2), F4 is `eval_runner.py:139-144`'s seed
parameter being a documented-but-no-op handoff to the env, and F5 is
NaN-in-JSON for the IMPACT row of `F6_summary.json` (technically
non-RFC-7159). All five batch into Step 8 cross-cutting cleanup;
none are correctness bugs.

---

## 2. What was reviewed

### Frozen audit-trail artefacts (never edited)

- `docs/results/06_benchmark/PLAN.md` — 298 lines; D6.1–D6.10 + the
  two follow-up D-decisions D6.2.1 (G6.2 reframed) and D6.8.1 (G6.4
  RF-Acting disposition).
- `docs/results/06_benchmark/RESULTS.md` — 308 lines; the locked
  scientific record. **§6.1 carries the audit-AF2 oracle-ceiling
  reframe** (the headline contribution) and is the canonical phrasing
  the LaTeX chapter must mirror in Step 9.
- `docs/results/06_benchmark/G6_scoreboard.json` — 98 lines;
  per-gate threshold + value + status + `finding_id` cross-link.
  Scoreboard already implements the schema improvement Step-5 F1
  flagged for cross-cutting clean-up: `status: "FAIL-WITH-FINDING"`
  and `finding_id: "D6.2.1"` make the JSON self-explaining (G6.2
  lines 28–30; G6.4 lines 56–58). This is the verdict-enum +
  finding_ref pattern the review previously asked for.

### Hash-pinned figure outputs

- `F5_table.png` (SHA `8baca1fe…`) + `F5_summary.{json,md,csv}` +
  `F5_caption.md` + `F5_manifest.json` (git_sha `824b825e`).
- `F6_stage_action_cm.png` (SHA `d9f17ae6…`) + `F6_summary.json` +
  `F6_caption.md` + `F6_manifest.json` (git_sha `b63b4d70`).
- `F7_overhead.png` (SHA `e929f4cd…`) + `F7_summary.json` +
  `F7_caption.md` + `F7_manifest.json` (git_sha `dcd8a3b1`).
- `F8_baselines.png` (SHA `7fe11ee9…`) + `F8_summary.json` +
  `F8_caption.md` + `F8_manifest.json` (git_sha `fe105df0`).
- `runs/phase6/eval_manifest.json` (SHA `c4a60a8f51…`) — top-level
  Phase-6 input manifest pinning splits, scaler, RF, all 15
  Phase-5 model.zips per-run; the input artefact every figure
  manifest hashes by reference (D6.9).
- `runs/phase5/sweep_manifest.json` (SHA `cc745432…`) — pinned in
  F7_manifest as the Phase-5 training-time source.

### Code

- `src/benchmark/__init__.py` (53) — package surface.
- `src/benchmark/baseline_policies.py` (311) — `Policy` Protocol,
  `random_policy`, `always_observe`, `always_block`,
  `recommended_action_policy`, `RFActingPolicy`, `SB3PolicyAdapter`,
  the canonical `_RECOMMENDED_BY_STAGE` constant.
- `src/benchmark/eval_runner.py` (348) — `run_policy(...)`, the
  schema-v1.0 EpisodeRecord JSONL emitter + sidecar latency JSONL.
- `src/benchmark/latency.py` (126) — `measure_inference_latency(...)`
  with clock-injection.
- `scripts/benchmark/run_test_eval.py` (535) — Phase-6 sweeper,
  loads all 15 Phase-5 checkpoints + 5 baselines + 1 oracle, writes
  `runs/phase6/eval_manifest.json`. Single-process by design (D6.4).
- `scripts/benchmark/build_summary_table.py` (484) — F5 builder.
- `scripts/benchmark/plot_baselines.py` (~270) — F8 builder.
- `scripts/benchmark/plot_overhead.py` (~290) — F7 builder.
- `scripts/benchmark/plot_stage_action_cm.py` (~310) — F6 builder.

### Tests

- `tests/test_baseline_policies.py` (192 lines, 24 tests).
- `tests/test_benchmark_eval_runner.py` (305 lines, 11 tests).
- `tests/test_benchmark_latency.py` (127 lines, 9 tests).
- Full suite: **`pytest -q` → 411 passed, 0 failed in 71.26 s**
  on `mentor-review/step-6-benchmarks` cut off `main` @ `014a7e3`.

### Cross-references

- `src/blue_team/aggregation.py:176` — `bootstrap_ci(values, *,
  n_resamples=1000, alpha=0.05, seed=0)` — percentile bootstrap,
  reproducible. The CI implementation is shared with Phase 5 (D6
  reuses Phase-5 schema-v1.0 unchanged).
- `src/blue_team/env_factory.py:104,114-115` — `exclude_ood=spec.
  exclude_ood` plumbed end-to-end into the RealizationEngine.
- `src/blue_team/run_config.py:69` — `EnvConfigSerializable.
  exclude_ood: bool = True` (default).
- `src/utils/realization_engine.py:165` — `if exclude_ood:` filter
  actually applied.

---

## 3. Findings (priority-ordered)

### F1 — `[severity: minor]` Test-count drift between `RESULTS.md` and on-disk pytest

`RESULTS.md` §2 (line 61) records the G6.1 verdict as **"420 passed,
0 failed"**, §4 (line 105) summarises **"Total: 376 → **420 tests**
(+44)"**, and §9 (line 308) repeats **"Phase 5 376 → **Phase 6 420**
(+44)"**. `G6_scoreboard.json:14-18` G6.1 also says
`"value": "420 passed"`.

On-disk reality at HEAD `014a7e3` (the Step-5 G2 merge into `main`):
`pytest -q` → **411 passed** in 71.26 s. The 24 + 11 + 9 = 44
benchmark tests are all on disk and collected; the +44 delta over
the historic baseline of 376 is present, but the historic baseline
was not 376 — it was 367 at Phase-5 close (376 was a forward-counting
estimate in PLAN §3.3 "target 376 → 388-392"). The actual baseline
before the 44 benchmark tests was thus 367, not 376, and 367 + 44 =
**411 passed**, which matches the on-disk verification.

The headline gate G6.1 is **still PASS** by its own threshold
(`>= 388 passed`); 411 ≥ 388 by a wide margin. Only the recorded
`value` is wrong by 9 tests.

**Recommended fix (doc-only, Step-8 batch):**
- `RESULTS.md` §2 G6.1 row: `"420 passed, 0 failed"` → `"411 passed,
  0 failed"`.
- `RESULTS.md` §4 line 105: `"Total: 376 → **420 tests** (+44)."` →
  `"Total: 367 → **411 tests** (+44)."`.
- `RESULTS.md` §9: same correction.
- `G6_scoreboard.json` G6.1 `"value"` → `"411 passed"`.

Commit: `docs(phase-6,§2,§4,§9): correct test-count history (411
not 420)`. Disposition: batch into Step 8 with the other Phase-6
doc-fixes.

### F2 — `[severity: minor]` Audit-AF2 oracle-ceiling reframe not propagated to F5 caption + scoreboard `headline_finding`

`RESULTS.md` §6.1 carries the new framing in full ("82 % of oracle
ceiling", "not a deployable defender", "Phase-7 target") and the
F8 caption (`F8_caption.md:5-12,21-27,30-38`) renders it correctly,
including the red dashed reference line at +1624 labelled "oracle
Recommended-Action ceiling". Two downstream artefacts still carry
the older "rule baseline strictly dominates RL" framing:

1. `F5_caption.md:8-10` says *"the trained RL trio is dominated by a
   hand-crafted rule on the held-out split (D6.2.1 finding…)"*. This
   is the older framing; the current framing is "trained RL captures
   82 % of the oracle ceiling without seeing stages".
2. `G6_scoreboard.json:95` `summary.headline_finding` reads
   *"the IoTWarden recommended-action rule baseline (+1624) strictly
   dominates the trained RL trio (DQN +1336 / PPO +1313 / A2C +1297).
   Phase-5's "25x over baseline" claim was a val-split selection-bias
   artefact…"*. Same older framing.

The numbers are unchanged in both — the reframe is a narrative
re-anchoring, not a numerical edit. RESULTS.md §6.1 is the
authoritative source the LaTeX chapter must mirror in Step 9; the
F5 caption and scoreboard `headline_finding` should match it so a
defense reader who only opens those artefacts gets the same story.

**Recommended fix (doc-only, Step-8 batch):**
- `F5_caption.md` lines 8–10: rewrite to "the trained RL trio
  captures 82 % of the oracle Recommended-Action ceiling on the
  held-out split (DQN +1336 / +1624 = 82 %); see RESULTS §6.1
  for the audit-AF2 reframe and Phase-7 hand-off". Also add the
  `ⓞ` oracle marker on the recommended-action row so it's visible
  on the table image.
- `G6_scoreboard.json` `summary.headline_finding` and
  `summary.secondary_finding`: rewrite to mirror RESULTS §6.1's
  oracle-ceiling phrasing.

Commit: `docs(phase-6,§6.1): propagate audit-AF2 oracle-ceiling
reframe to F5 caption + scoreboard headline_finding`. Disposition:
batch into Step 8.

### F3 — `[severity: minor]` Phase-2 LSTM checkpoint not pinned by SHA in `eval_manifest.json`

`runs/phase6/eval_manifest.json:input_hashes` (lines 41-45) pins
three input artefacts:
- `splits_manifest`: `1e99d596…` ✓ (post-`3cd2fb9` Phase-1 splits;
  Step-1 invariant chain).
- `scaler`: `146c8aa7…` ✓ (D6.9).
- `rf_model`: `546a7355…` ✓ (Phase-4 RF detector).

It does **not** pin the Phase-2 LSTM at
`artifacts/generator/phase2/`. Only `args.generator_path` (a string,
"artifacts/generator/phase2") is recorded. On disk that directory
contains four files with SHAs:
- `attack_sequence_generator.pth`: `afd70432…`
- `checkpoint.pth`: `04b08aed…`
- `config.json`: `18f7b84e…`
- `training_config.json`: `c92ab264…`

The Red Team is the upstream of every Phase-6 rollout (it drives the
attack trajectory); without its SHA pinned, a reviewer cannot verify
the rollouts ran against the same generator weights that produced
Phase-5's training trajectories. This is structurally the same
audit-trail gap Step-2 F1 surfaced and Step-5 F2 carried forward.

**Recommended fix (doc + 1-line code, Step-8 batch):**
Extend `run_test_eval.py:_eval_manifest()` (around line 505) to add
two more entries to `input_hashes`:

```python
"input_hashes": {
    "splits_manifest": _sha256(splits_manifest),
    "scaler": _sha256(scaler_path),
    "rf_model": _sha256(rf_path),
    "generator_weights": _sha256(
        Path(args.generator_path) / "attack_sequence_generator.pth"
    ),
    "generator_config": _sha256(
        Path(args.generator_path) / "config.json"
    ),
},
```

Then re-emit `eval_manifest.json` once (no rollout re-run; the
manifest is regenerable from the on-disk JSONLs). The downstream
`F5/F6/F7/F8_manifest.json` pin `eval_manifest.sha256: c4a60a8f5…`,
which would change with this edit — so all four figure manifests
re-emit too. **Note this is the only Phase-6 fix that touches a
hash-pinned artefact.** The figure PNGs themselves do not need to
re-render; only the manifests update.

Commit: `fix(phase-6,§D6.9): pin Phase-2 LSTM checkpoint SHA in
eval_manifest`. Disposition: batch into Step 8 with the unified
hash-chain hardening Step-5 F2 also requested.

### F4 — `[severity: nit]` `eval_runner.run_policy(seed=…)` is documented but does nothing at the env level

`src/benchmark/eval_runner.py:139-144`:

```python
if seed is not None:
    obs = env.reset()  # SB3 DummyVecEnv ignores `seed=` kwarg pre-1.x;
    # fall back to setting the action_space seed via env's RNG
    # by calling reset with a numpy-style call when supported.
else:
    obs = env.reset()
```

Both branches call the identical `env.reset()`. The `seed` parameter
is documented in the docstring (`eval_runner.py:105-107`) as
*"forwarded to env.reset(seed=...) on episode 0 only"*, but the
implementation forwards nothing — the comment acknowledges the gap
("ignores `seed=` kwarg pre-1.x") without fixing it.

Empirical impact on Phase-6 numbers: **none**. The trained RL
adapters use `deterministic=True` so the policy is deterministic
regardless of env RNG state. The random baseline seeds its own
`np.random.default_rng(seed)` in `run_test_eval.py:_roll_random` and
ignores `eval_runner`'s seed. The deterministic baselines have no
RNG. Per-seed reproducibility for Phase 6 is therefore real (you
can re-run today and get the same JSONLs), but it relies on
caller-side seeding and the SB3-env-RNG-not-actually-being-used —
not on the documented `eval_runner` contract.

**Recommended fix (doc-only or 2-line code, Step-8 batch):**
Either (a) drop the `seed` parameter from the public signature and
the docstring, or (b) actually call `env.seed(seed)` /
`env.action_space.seed(seed)` on the underlying VecEnv. Option (a)
is the safer Step-8 fix — option (b) would change the JSONL bytes
on re-run (different env-RNG state could produce different episode
boundaries) and thus invalidate the hash chain. The cleaner move is
to delete the `seed` parameter and update the callers in
`run_test_eval.py` to drop it (they currently pass `seed=seed` /
`seed=0` redundantly).

Commit: `docs(phase-6,eval_runner): clarify run_policy seed
semantics (no-op at env level)`. Disposition: batch into Step 8.

### F5 — `[severity: nit]` `F6_summary.json` IMPACT row uses `NaN` literal (non-RFC-7159 JSON)

`F6_summary.json` lines 60-64, 100-105, 140-146, 180-187, 220-228,
260-269 — every policy's IMPACT row (`matrix[4]`) is filled with
the bare token `NaN`:

```json
[
  NaN,
  NaN,
  NaN,
  NaN,
  NaN
]
```

Bare `NaN` is what Python's `json.dump` emits by default with
`allow_nan=True` (the default), but it is **not valid JSON** under
RFC 7159 / ECMA-404 — strict parsers like `JSON.parse` (browser),
`jq` (without `--ignore-nan` workarounds), and Java's `Jackson`
default reject it. This means the file is consumable by Python
(via `json.load(allow_nan=True)`) but not by every downstream tool
a defense reviewer might use.

The semantic intent is correct (D6.7 — IMPACT row excluded from
proportionality scoring), but the implementation should use either
(a) `null` instead of `NaN`, or (b) drop the IMPACT row entirely
(`matrix` is 4×5 instead of 5×5) and have a separate
`stage_labels_present: ["BENIGN", "RECON", "ACCESS", "MANEUVER"]`
field. Option (a) is the smaller patch.

`build_summary_table.py:182-191` uses `math.nan` for missing-cell
metrics in F5 too, but those cells aren't actually NaN-emitted in
on-disk `F5_summary.json` (every policy has all metrics populated),
so F5 is unaffected. F6 is the only Phase-6 artefact with the issue.

**Recommended fix (one-line code, Step-8 batch):**
In `plot_stage_action_cm.py`'s summary writer, replace `np.nan` with
`None` on the IMPACT row before serialising, or call `json.dump(...,
allow_nan=False)` and convert NaN to `None` upstream.

Commit: `fix(phase-6,plot_stage_action_cm): emit null instead of NaN
in F6_summary.json IMPACT row (RFC-7159)`. Disposition: batch into
Step 8.

---

## 4. Audits performed

### 4.1 Test-split contract (Step-1 invariant)

Phase 6's eval consumes `split="test_balanced"`, `exclude_ood=True`
end-to-end:

- `scripts/benchmark/run_test_eval.py:137-144` — `_eval_env_spec()`
  returns `EnvConfigSerializable(split="test_balanced", exclude_ood=
  True)`. This is the *only* place split is set in Phase 6 (no
  caller can override it via CLI; the CLI accepts `--algos`,
  `--seeds`, `--n-episodes`, splits-manifest path, but **not**
  the split itself — by design, the held-out test split is the
  contract for Phase 6).
- `src/blue_team/env_factory.py:104,114-115` — `make_eval_env`
  passes `exclude_ood=spec.exclude_ood` to `RealizationEngine` and
  logs the values.
- `src/utils/realization_engine.py:165` — `if exclude_ood:` filter
  actually drops the OOD indices.
- `runs/phase6/eval_manifest.json:46-54` — `eval_env` block records
  `split: "test_balanced"`, `exclude_ood: true` as observed values
  for the producing run, providing the audit-trail receipt.

Verdict: **Step-1 invariant honoured**, both at code-level and at
serialisation-level. ✅

### 4.2 Hash chain (G6.7 + cross-phase)

Pinned end-to-end:

| Layer | Path | SHA-256 | Source of truth |
|---|---|---|---|
| Phase-1 splits | `data/processed/ciciot2023/splits/manifest.json` | `1e99d59682…` | `eval_manifest.json:42` |
| Scaler | `data/processed/ciciot2023/scaler.joblib` | `146c8aa7…` | `eval_manifest.json:43` |
| Phase-4 RF | `artifacts/detector/random_forest.joblib` | `546a7355…` | `eval_manifest.json:44` |
| Phase-5 ckpt × 15 | `runs/phase5/<algo>/seed_<k>/model.zip` | varies | per `runs[i].model_sha256` in `eval_manifest.json` |
| Phase-5 sweep manifest | `runs/phase5/sweep_manifest.json` | `cc745432…` | `F7_manifest.json:16` |
| Phase-6 eval manifest | `runs/phase6/eval_manifest.json` | `c4a60a8f51…` | pinned by `F5/F6/F7/F8_manifest.json` |
| F5 outputs | `F5_summary.json` | `9c9ea26f…` | pinned by `F8_manifest.json:13` |

Spot-checked the Phase-5 model.zip SHAs against `eval_manifest.json`'s
recorded values: **15 / 15 byte-perfect match** (e.g.,
`runs/phase5/dqn/seed_0/model.zip` → `e2c11407e0…`, both on disk and
in manifest).

Spot-checked F8 → F5 chain: `F8_manifest.json:13`'s
`f5_summary.sha256: 9c9ea26f259712334fc22000c0639ca28caad0939a8ddde3c35660c0a8229ede`
matches on-disk `shasum -a 256 F5_summary.json` byte-perfect.

The **only** missing link is Phase-2 LSTM (Finding F3 above).
Phase-3 frozen artefacts are not directly hashed because they are
code (committed in git) — the producing git_sha covers them.

Verdict: **Hash chain is byte-perfect for everything currently
pinned**; F3 doc-fix promotes it to gold-standard for the Phase-2
boundary.

### 4.3 Detector-integration question (resolved this step)

Step-5 §8 question 6 asked: *"Phase-6 detector-baseline lane — does
Phase 6 evaluate a 'detector-only' recommended-action policy as one
of its baselines?"*

**Answer:** Yes — but it uses the **Phase-4 RandomForest** at
`artifacts/detector/random_forest.joblib`, **not** the CNN1D
`stage_detector.pt` (Phase-4 G4.x champion). `RFActingPolicy`
(`baseline_policies.py:152-251`) wraps the RF as
`recommended_action_policy(rf.predict(last-step-features))`. The
chain to Phase-4 RF is explicit (`rf_model: 546a7355…` in
`eval_manifest.json`).

The CNN1D `stage_detector.pt: 71e06616…` is **not consumed by Phase
6** at all. Confirmed by:

```bash
$ grep -rn "stage_detector\|StageDetector\|71e06616" \
    src/benchmark/ scripts/benchmark/ tests/test_baseline_*.py \
    tests/test_benchmark_*.py
# (no results)
```

The CNN1D detector is reserved for Phase-9 ablation (per the
detector-augmented observation axis already enumerated in
Phase-7's PLAN — out of Phase-6 scope per D6.5 + the audit-first
contract). **Step-4 open question 4 stays resolved (Phase 5: no);
Step-5 §8 q6 is also resolved (Phase 6: only RF, not CNN1D).**

### 4.4 F5 — final security-metrics table

PLAN §3.1 columns: `mean_reward, mean_mttc, compromise_rate,
mitigated_impact_rate, mean_episode_length, mean_inference_latency_ms,
p95_inference_latency_ms`. On-disk
`F5_summary.json` rows 11-21 (DQN example) deliver every column
+ also `mean_reward_ci_low`, `mean_reward_ci_high`, `n_seeds`,
`n_episodes`, `p50_inference_latency_ms`, `p99_inference_latency_ms`
— a strict superset of PLAN §3.1.

Bootstrap CI is the percentile method on per-seed means
(non-deterministic policies) or per-episode rewards (deterministic
policies), `n_resamples=1000`, `alpha=0.05`, `seed=0` (reproducible).
Implementation: `src/blue_team/aggregation.py:176-213`. Caller:
`build_summary_table.py:206-210`. CIs in F5_summary match RESULTS.md
§1 verbatim (DQN +1336 (1265, 1407), PPO +1313 (1253, 1372), A2C
+1297 (1267, 1337), Recommended +1624 (1572, 1672), RF-Acting +1508
(1455, 1565)). ✅

`compromise_rate = 1.0` for every policy is correct: every Phase-2
trajectory terminates in IMPACT under the upper-triangular generator,
so all policies "reach IMPACT" by the env's definition. The
meaningful security metric is `mitigated_impact_rate` (was the host
isolated *during* compromise?) — Always-BLOCK 1.0, RL 0.15–0.28,
Recommended-Action 0.187. The F5 caption documents this distinction
(`F5_caption.md:11-19`). ✅

### 4.5 F6 — stage × action confusion matrices

`F6_summary.json` axis ordering:
- Rows: `stage_labels = ["BENIGN", "RECON", "ACCESS", "MANEUVER",
  "IMPACT"]` (lines 5-11) ✓.
- Columns: `action_labels = ["OBSERVE", "LOG", "THROTTLE", "BLOCK",
  "ISOLATE"]` (lines 12-18) ✓.
- `recommended_by_stage = {0:0, 1:1, 2:2, 3:3, 4:4}` — identity
  mapping matches Phase-3 frozen contract ✓.
- `g63_excludes_impact: true` (line 27) — D6.7 honoured ✓.
- `g63_threshold: 0.7` (line 26) ✓.

Matrices: per-policy 5×5; non-IMPACT rows row-sum to 1.0 ± 1e-6;
IMPACT row is NaN (Finding F5 above). DQN g63_score 0.7846, PPO
0.7116, A2C 0.7459 — all > 0.70, G6.3 PASS for all three. The
`bin_by_stage` use of `decision_stage` (the stage at action-decision
time, not post-step) is verified at
`tests/test_benchmark_eval_runner.py:248
test_action_counts_by_stage_uses_decision_stage`. ✅

DQN row 3 (MANEUVER) deserves note: 11 % OBSERVE / 5 % LOG / 0.7 %
THROTTLE / 25 % BLOCK / **58 % ISOLATE**. The agent is
*over-isolating* MANEUVER — the same de-escalation-farming pattern
G5.4 surfaced, only here at stage 3 instead of stage 4. Worth
flagging in the Phase-7 reward-component-ablation hand-off.

### 4.6 F7 — computational overhead

`F7_summary.json` contents:
- `platform`: `macOS-26.4.1-arm64-arm-64bit`, `arm`, `arm64`,
  Python 3.9.6 — R6.3 platform fingerprint ✓.
- `g64_thresholds_ms: {rl: 5.0, rf: 3.0, rule: 1.0}` — D6.8 budgets ✓.
- Per-policy `{p50, p95, p99, mean}_ms`, `n_samples`, `budget_ms`,
  `g64_pass`, `policy_class`. All RL: p50 0.07–0.10 ms (≥ 50×
  headroom), `g64_pass: true`. Rule baselines: p50 ≤ 0.001 ms.
  RF-Acting: p50 13.976 ms, `g64_pass: false` ✓ (D6.8.1 finding).
- `phase5_training_seconds_per_algo` and `_hours_per_algo` — DQN
  0.60 h, PPO 0.60 h, A2C 0.61 h.

`plot_overhead.py:217-228` — log-x CDF, "Empirical CDF" axis label,
budget reference lines drawn (search confirmed `axvline` at budget
thresholds), grid + ylim 0–1.02. ✅

Latency capture mechanism: `eval_runner.py:157,159` measures with
`time.perf_counter_ns` inline during the rollout, *not* via the
`measure_inference_latency` micro-benchmark in `latency.py`. The
inline approach measures real env-driven inputs at real env-driven
intervals, which is more realistic than a synthetic obs pool. The
standalone `latency.py` is exposed for future use (and is well-
tested), but it doesn't drive F7. This is a sensible design
choice; just worth noting for the LaTeX chapter so the methodology
description is precise.

F7 caption (`F7_caption.md:14-21`) lists per-policy p50 with budget
verdict including the explicit RF-Acting **"G6.4 FAIL"** wording —
which is harsher than `G6_scoreboard.json` G6.4's
`status: "PASS-WITH-FINDING"`. Minor wording inconsistency; the
scoreboard verdict is the canonical one (the gate **is** PASS-WITH-
FINDING because 7 of 8 policies clear with ≥ 30× headroom; only
RF-Acting individually misses, with finding D6.8.1 documenting the
sklearn-dispatch root cause). Worth noting in Step 8 cleanup.

### 4.7 F8 — RL vs non-RL baselines

`F8_summary.json:73-86` — G6.5: `dqn/ppo/a2c.overlaps_with: []`,
`g65_pass: true` for all three. The CIs do not overlap any non-RL
baseline's CI. ✓.

`plot_baselines.py:170` draws `ax.axvline(rec_floor, color="#dc2626",
linestyle="--")` at +1624 with the oracle annotation in the legend
(per F8 caption). The visual story matches RESULTS §6.1 — RF-Acting
sits closer to the rule ceiling (+1508 vs. +1624, gap +116) than to
the RL cluster (+1297..+1336, gap +172 from RF, +288 to ceiling).
The cross-quadrant trade-off (oracle / RF-Acting / RL / random
reward × inference cost) renders cleanly. ✅

### 4.8 G6 scoreboard schema

`G6_scoreboard.json` already implements the schema improvement
Step-5 F1 + Step-4 G4.4 asked for:
- `gates.G6.2.status = "FAIL-WITH-FINDING"`,
  `finding_id = "D6.2.1"`, `finding_summary` cross-link.
- `gates.G6.4.status = "PASS-WITH-FINDING"`,
  `finding_id = "D6.8.1"`, `finding_summary` cross-link.
- `summary` block with `pass / pass_with_finding / fail_with_finding
  / fail` counts.

This is exactly the verdict-enum + finding_ref pattern Step-5 F1
recommended for cross-cutting cleanup. **Step-5 F1 + Step-4 G4.4
are now half-resolved**: Phase-6 ships the new schema natively;
Phase-4 G4.4 + Phase-5 G5.4 still carry the old `passes: bool`
field with editorial markdown layered on top, and Step 8 should
backfill them to the Phase-6 schema (a one-line JSON regen, no
re-run). Confirm with the candidate.

### 4.9 Per-checkpoint reproducibility

For each (algo, seed):
- Eval env constructed with `seed=seed` at `run_test_eval.py:245`
  → `_build_eval_env(args, seed=seed)` → `make_eval_env(spec=...,
  seed=seed)` → SB3 DummyVecEnv wraps the constructed env.
- SB3 model loaded with `DQN/PPO/A2C.load(model_path, env=env,
  device="cpu")` → wrapped in `SB3PolicyAdapter(model,
  deterministic=True)`.
- Rollout via `run_policy(policy, env, n_episodes=30, seed=seed)`
  — note Finding F4 above: the seed parameter to `run_policy` does
  not actually reach `env.reset(seed=...)`, but determinism still
  holds because the policy is deterministic and the env-side RNG
  is set at env-construction time via the `seed=seed` argument
  threaded through `make_eval_env`.

Empirical sanity: rolling all 15 trained checkpoints + 5 baselines
takes 54.1 s wallclock (`eval_manifest.wallclock_seconds`); the
per-run wallclock varies by 0.001 s but the per-run `eval_jsonl_sha256`
is the same on every re-run (verified by candidate; not re-run by
this audit per the read-only-audit operating rule).

### 4.10 Test-coverage audit

44 new tests cover:
- Every baseline policy: random RNG seeding, constants,
  `recommended_action_policy` info passthrough + KeyError on
  missing key, `RFActingPolicy` slicing (with + without deltas),
  obs-shape validation, out-of-range stage rejection, `SB3PolicyAdapter`
  round-trip + deterministic-flag propagation.
- `run_policy` JSONL round-trip via Phase-5's aggregation reader
  (schema-v1.0 conformance), latency sidecar one-row-per-step,
  decision-stage bookkeeping (the F6 invariant), run_id parser.
- `measure_inference_latency` — n_warmup excluded, deterministic
  clock-injection path, error paths (empty pool, negative warmup,
  zero measure, info_pool length mismatch), info_pool threading.

Verdict: public API of `src.benchmark` is well-pinned. Public API
of the figure-builder scripts is not unit-tested (they are scripts,
not modules), but their outputs are hash-pinned in the figure
manifests, which is the same protection.

---

## 5. Cross-cutting carry-forward

### Open candidate decisions (re-flagged from earlier steps)

1. **Step-2 F1** — Phase-2 LSTM re-run vs. document-only on the
   manifest input-hash divergence. **Now newly relevant:** Step-6
   F3 above asks for the Phase-2 generator weights to be SHA-pinned
   in `eval_manifest.json`. The pin is what ties Phase-6 numbers to
   a specific generator; if option (a) (Step-7 re-run with
   `seed=42` against the post-`3cd2fb9` manifest) is chosen, that
   re-run produces a different `attack_sequence_generator.pth` SHA
   and Phase 6 must re-run too (~10 min CPU). The numerical impact
   is unknown but bounded (the headline 82 %-of-oracle finding is
   a structural property of the reward landscape, not of any
   particular generator weight).

2. **Step-2 F2** — Phase-2 model-selection metric (CE vs. macro-F1).
   No new evidence from Step 6.

3. **Step-3/4/5 doc-fix batching** — Step 3 F1–F3, Step 4 F1–F4,
   Step 5 F1–F6, Step 6 F1–F5 are now all candidates for Step 8
   cross-cutting cleanup. My recommendation remains: **batch all
   into Step 8** rather than landing piecemeal. The atomic Step-8
   commit makes the audit-trail diff readable and avoids
   per-finding hash-chain regenerations except where genuinely
   needed (Step-6 F3 only).

4. **Verdict-enum + finding_ref scoreboard schema** — **half-resolved**
   by Phase 6. `G6_scoreboard.json` ships the new schema natively
   (G6.2 `status: "FAIL-WITH-FINDING"`, `finding_id: "D6.2.1"`).
   Phase-4 G4.4 + Phase-5 G5.4 should be back-filled to the same
   schema in Step 8. The `verdict` enum I previously suggested is
   spelled `status` in the Phase-6 schema; recommend Step 8 use
   `status` everywhere for consistency.

5. **Phase-6 detector-baseline lane (Step-5 §8 q6)** — **resolved
   this step** (§4.3 above). Phase 6 uses the Phase-4 RandomForest
   `random_forest.joblib`, not the CNN1D `stage_detector.pt`.
   Chain to Phase-4 RF is explicit; Phase-4 CNN1D is reserved for
   Phase-9 ablation.

### New raised in Step 6

6. **(F1 above)** — Test-count drift in `RESULTS.md` and
   `G6_scoreboard.json` (420 claimed, 411 on disk). Doc-only fix
   in Step 8.

7. **(F2 above)** — `RESULTS.md` §6.1's audit-AF2 oracle-ceiling
   reframe is the canonical thesis-claim phrasing; F5 caption +
   `G6_scoreboard.json` `headline_finding` carry the older "rule
   dominates RL" framing. Doc-only re-rendering in Step 8 closes
   the gap. The LaTeX `tex/results.tex` rewrite in Step 9 must
   mirror RESULTS §6.1 verbatim.

8. **(F3 above)** — `eval_manifest.json` does not pin the Phase-2
   LSTM SHA. Code-fix in Step 8 (one of the few Phase-6 fixes that
   touches a hash-pinned artefact). Re-emits `eval_manifest.json` →
   regenerates four figure manifests; PNGs unchanged.

9. **(F4 above)** — `run_policy(seed=…)` is documented but no-op at
   env level. Doc-only fix (drop the parameter or its docstring) in
   Step 8.

10. **(F5 above)** — `F6_summary.json` IMPACT row uses `NaN` literal
    (non-RFC-7159 JSON). One-line fix in
    `plot_stage_action_cm.py` to emit `null`; regenerates F6
    manifest + summary; PNG unchanged.

11. **(F7 wording)** — `F7_caption.md:20` says "G6.4 FAIL" for
    RF-Acting but `G6_scoreboard.json` G6.4 reads
    `"PASS-WITH-FINDING"`. Use the scoreboard verdict consistently.
    Doc-only fix in Step 8.

12. **(R6.4-style finding raised by F6 inspection)** — DQN's
    MANEUVER row (stage 3) is **58 % ISOLATE**, the same
    over-aggression pattern G5.4 flagged on IMPACT. The
    Phase-7 reward-component-ablation hand-off should treat MANEUVER
    + IMPACT as a coupled de-escalation-farming axis, not just IMPACT.
    Pure observation; the gate G6.3 still PASSes (proportionality
    band is `|action − rec(stage)| ≤ 1`, and `|4 − 3| = 1` so
    ISOLATE-on-MANEUVER counts as on-band — the band is wider than
    "exact match" by design).

---

## 6. Conclusion and what I recommend

**Verdict: PASS-WITH-FIXES.**

Phase 6 is the strongest phase package in this review. The audit-AF2
oracle-ceiling reframe is the kind of mature scientific move a
defense committee will reward; the gate scoreboard already implements
the Step-5 cross-cutting schema improvement; the hash chain is
byte-perfect for everything currently pinned; the implementation is
clean and well-tested. The five findings filed above are all minor
or nit-level documentation drift, batchable into Step 8.

**No retraining is needed.** The numerical record is correct as
written; only the narrative artefacts (F5 caption, scoreboard
`headline_finding`) and one missing manifest entry (Phase-2 LSTM
SHA) need touch. Step-7's reward-component ablation has a sharply-
defined target — the +288 reward gap to the oracle ceiling — and
the Phase-2 LSTM re-run question (Step-2 F1) gains a small extra
incentive (Step-6 F3 wants a fresh `eval_manifest.json` after the
re-run).

The Step-9 LaTeX chapter must mirror `RESULTS.md` §6.1's "82 % of
oracle ceiling" framing. The older "RL beats baselines by 25×"
phrasing in any draft prose carried over from Phase 5 must be
retired.

I recommend the candidate sign off Step 6 and proceed to Step 7
(Phase 7 ablations: F9 reward-component, F10 aggressiveness, F12
attack sweep, F15 OOD robustness, gate G7).
