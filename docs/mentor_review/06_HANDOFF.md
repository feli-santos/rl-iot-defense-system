# Step `06` — Phase 6 Benchmarks Review — Mentor Review Handoff

**Closed:** `2026-05-06 22:11 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 7)
**Reviewed phase / scope:** Phase 6 (final benchmark on the held-out
`test_balanced` split — F5 final security-metrics table, F6 stage ×
action confusion matrices, F7 computational overhead, F8 cross-policy
mean-reward bars, exit gates G6.1–G6.7). Phase 6 consumes the 15
frozen Phase-5 model checkpoints + the four non-RL baselines + the
oracle Recommended-Action policy + the Phase-4 RandomForest detector
wrapped as `RFActingPolicy`. The audit AF2 oracle-ceiling reframe
recasts Phase-6 as "trained RL captures 82 % of the oracle ceiling
without seeing stages" (DQN +1336 / +1624 = 82 %), with the
remaining +288 reward named as the Phase-7 reward-ablation target.
**Status:** `completed`

---

## 1. What was reviewed

### Frozen audit-trail artefacts (never edited)
- `docs/results/06_benchmark/PLAN.md` (298 lines) — design contract;
  D6.1–D6.10 + the two follow-up D-decisions D6.2.1 (G6.2 reframed)
  and D6.8.1 (G6.4 RF-Acting disposition).
- `docs/results/06_benchmark/RESULTS.md` (308 lines) — locked
  scientific record. **§6.1 carries the audit-AF2 oracle-ceiling
  reframe**: this is the canonical phrasing the LaTeX chapter must
  mirror in Step 9.
- `docs/results/06_benchmark/G6_scoreboard.json` (98 lines) — per-gate
  threshold + value + status + `finding_id` cross-link. **Already
  ships the verdict-enum + finding_ref schema** Step-5 F1 asked for.

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
  Phase-5 model.zips per-run.
- `runs/phase5/sweep_manifest.json` (SHA `cc745432…`) — pinned in
  F7_manifest.

### Code
- `src/benchmark/__init__.py` (53), `baseline_policies.py` (311),
  `eval_runner.py` (348), `latency.py` (126).
- `scripts/benchmark/run_test_eval.py` (535) — sweeper (D6.4 single-
  process). `build_summary_table.py` (484) — F5. `plot_baselines.py`
  (~270) — F8. `plot_overhead.py` (~290) — F7.
  `plot_stage_action_cm.py` (~310) — F6.

### Tests
- `tests/test_baseline_policies.py` (24 tests, 192 lines).
- `tests/test_benchmark_eval_runner.py` (11 tests, 305 lines).
- `tests/test_benchmark_latency.py` (9 tests, 127 lines).
- Full suite: **`pytest -q` → 411 passed, 0 failed in 71.26 s**
  on `mentor-review/step-6-benchmarks` cut off `main` @ `014a7e3`.
  *Note:* `RESULTS.md` records 420; on-disk reality is 411 (Step-6
  Finding F1).

### Docs (mentor-review)
- `README.md`, `00_framing.md` + `00_HANDOFF.md`,
  `01_dataset.md` + `01_HANDOFF.md`,
  `02_red_team.md` + `02_HANDOFF.md` (F1 + F2 still open),
  `03_env.md` + `03_HANDOFF.md`,
  `04_detector.md` + `04_HANDOFF.md`,
  `05_blue_team.md` + `05_HANDOFF.md`,
  `06_benchmark.md` (this step's mentor memo, written this session),
  `06_HANDOFF.md` (this file).
- `HANDOFF_TEMPLATE.md` — template followed for this file.

---

## 2. Verdict

`PASS-WITH-FIXES`

Phase 6 is the most thoughtful phase package in this review.
Six of seven exit gates clear cleanly. G6.2 is mechanically
`FAIL-WITH-FINDING` and is the **headline result of Phase 6** — the
audit AF2 oracle-ceiling reframe in `RESULTS.md` §6.1: trained RL
captures **82 % of the oracle Recommended-Action ceiling** (DQN +1336
/ +1624) without ever seeing the kill-chain stage; the remaining
+288 reward is the Phase-7 reward-component-ablation target. G6.4 is
PASS-WITH-FINDING for RF-Acting at 14 ms vs. 3 ms budget (D6.8.1 —
sklearn dispatch overhead). Hash chain byte-perfect for everything
currently pinned (Phase-1 splits `1e99d59682…`, scaler, RF, all 15
Phase-5 model.zips). Test-split contract honoured at code level
(`run_test_eval.py:143` — `EnvConfigSerializable(split=
"test_balanced", exclude_ood=True)`) and at serialisation level
(`eval_manifest.json:46-54`). Five minor / nit findings filed; all
batch into Step 8 cross-cutting cleanup; none are correctness bugs.

Full memo: `docs/mentor_review/06_benchmark.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: minor]** **F1** — Test-count drift between
   `RESULTS.md` (claims **420 passed**) + `G6_scoreboard.json`
   (claims **420 passed**) and on-disk reality (**411 passed**).
   The +44 benchmark tests are present; the historic baseline was
   367 not 376 (PLAN §3.3 forward-counted to 376). Gate G6.1's
   threshold `>= 388` is met (411 ≥ 388 by wide margin) — only the
   recorded `value` is wrong.
   **Recommended fix (doc-only):** patch `RESULTS.md` §2 / §4 / §9
   and `G6_scoreboard.json` G6.1 `value` to "411 passed". Commit:
   `docs(phase-6,§2,§4,§9): correct test-count history (411 not
   420)`. **Disposition:** batch into Step 8.

2. **[severity: minor]** **F2** — Audit-AF2 oracle-ceiling reframe
   (RESULTS.md §6.1 — "82 % of oracle ceiling, +288 to close")
   not propagated to F5 caption + `G6_scoreboard.json`
   `summary.headline_finding`, which still carry the older "rule
   baseline strictly dominates RL" framing.
   **Recommended fix (doc-only):** rewrite `F5_caption.md:8-10` and
   `G6_scoreboard.json` `summary.{headline_finding,
   secondary_finding}` to mirror RESULTS §6.1; add the `ⓞ` oracle
   marker on the recommended-action row in `F5_table.png` if a
   re-render is cheap (otherwise just the caption). The LaTeX
   `tex/results.tex` rewrite in Step 9 must follow RESULTS §6.1
   verbatim. Commit: `docs(phase-6,§6.1): propagate audit-AF2
   oracle-ceiling reframe to F5 caption + scoreboard
   headline_finding`. **Disposition:** batch into Step 8.

3. **[severity: minor]** **F3** — Phase-2 LSTM checkpoint not pinned
   by SHA in `runs/phase6/eval_manifest.json`. Three input artefacts
   are pinned (`splits_manifest`, `scaler`, `rf_model`); the
   generator at `artifacts/generator/phase2/` is referenced only by
   path string. This is structurally the same audit-trail gap
   Step-2 F1 surfaced and Step-5 F2 carried forward.
   **Recommended fix (1-line code, regenerates 5 manifests):** add
   `generator_weights` and `generator_config` SHA entries to
   `run_test_eval.py:_eval_manifest()`'s `input_hashes` dict (around
   line 505), re-emit `eval_manifest.json` (no rollouts re-run; the
   manifest is regenerable from on-disk JSONLs), then regenerate
   `F5/F6/F7/F8_manifest.json` (PNGs unchanged). Commit:
   `fix(phase-6,§D6.9): pin Phase-2 LSTM checkpoint SHA in
   eval_manifest`. **Disposition:** batch into Step 8 with the
   unified hash-chain hardening Step-5 F2 also requested.

4. **[severity: nit]** **F4** — `src/benchmark/eval_runner.py:139-144`
   `run_policy`'s `seed` parameter is documented as
   *"forwarded to env.reset(seed=...)"* but both branches call
   identical `env.reset()`. Empirical impact on Phase-6 numbers:
   none (RL is `deterministic=True`; random baseline seeds itself
   externally). Reproducibility for Phase 6 is real but relies on
   caller-side seeding, not on the documented contract.
   **Recommended fix (doc-only is safer):** drop the `seed`
   parameter from `run_policy`'s public signature and update the
   `run_test_eval.py` call sites to stop passing it; OR rewrite the
   docstring to say "seed parameter is currently a no-op at the env
   level; rely on caller-side env-construction seeding for
   reproducibility." Commit: `docs(phase-6,eval_runner): clarify
   run_policy seed semantics (no-op at env level)`. **Disposition:**
   batch into Step 8.

5. **[severity: nit]** **F5** — `F6_summary.json` IMPACT row
   (`matrix[4]` for every policy) uses bare `NaN` literal, which
   is not RFC-7159 / ECMA-404 valid JSON (Python's `json.load`
   reads it via `allow_nan=True` default; strict parsers reject it).
   The semantic intent is correct (D6.7 — IMPACT row excluded from
   proportionality scoring); only the serialisation needs fixing.
   **Recommended fix (one-line code, regenerates F6 manifest +
   summary; PNG unchanged):** in `plot_stage_action_cm.py`'s
   summary writer, replace `np.nan` with `None` on the IMPACT row,
   or pass `allow_nan=False` to `json.dump`. Commit:
   `fix(phase-6,plot_stage_action_cm): emit null instead of NaN in
   F6_summary.json IMPACT row (RFC-7159)`. **Disposition:** batch
   into Step 8.

6. **[severity: nit, F7-caption-only]** **F6 (additional, Step-6 §5
   item 11)** — `F7_caption.md:20` says "G6.4 FAIL" for RF-Acting
   but `G6_scoreboard.json` G6.4 status is `"PASS-WITH-FINDING"`.
   Use the scoreboard verdict consistently. Doc-only fix in Step 8.

7. **[observation, not a finding]** DQN's MANEUVER row in F6 shows
   58 % ISOLATE — the same de-escalation-farming pattern G5.4
   flagged on IMPACT, here on stage 3 instead of stage 4. The gate
   G6.3 still PASSes (proportionality band `|action − rec(stage)|
   ≤ 1` is satisfied since `|4 − 3| = 1`). Worth flagging in the
   Phase-7 reward-component ablation hand-off as a *coupled*
   MANEUVER+IMPACT axis rather than IMPACT-only.

Full prose, file:line citations, and recommended commit messages:
`docs/mentor_review/06_benchmark.md` §3 + §5.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/06_benchmark.md` — Step-6 mentor memo (verdict
  PASS-WITH-FIXES + 5 findings + hash-chain reproduction +
  detector-integration audit + Step-1 invariant audit + F5/F6/F7/F8
  realism audits + scoreboard-schema audit + reproducibility audit
  + test-coverage audit + cross-cutting carry-forward).
- `docs/mentor_review/06_HANDOFF.md` — this file.

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: **411 passed in 71.26 s**
on `mentor-review/step-6-benchmarks`.

### Scripts / models
None modified. No re-training, no figure regeneration. **Hash chain
intact** — verified byte-perfect on:
- `eval_manifest.json` SHA `c4a60a8f51…` (matches what
  `F5/F6/F7/F8_manifest.json` claim).
- All 15 Phase-5 `model.zip` SHAs (15/15 byte-perfect match between
  on-disk `shasum -a 256` and `eval_manifest.json:runs[i].
  model_sha256`).
- Phase-1 splits manifest `1e99d59682…` (target met: post-`3cd2fb9`).
- Phase-4 RF `random_forest.joblib` SHA `546a7355…` (pinned).
- F5 → F8 chain (F8_manifest claims `f5_summary.sha256: 9c9ea26f…`
  matching on-disk).
- F5 → F6 → F7 → F8 manifests all pin
  `eval_manifest.sha256: c4a60a8f5…` consistently.

### Git hygiene applied (Phase G1, opening this step)
1. `git checkout main && git pull --ff-only origin main`.
2. `git merge --no-ff mentor-review/step-5-blue-team -F
   /tmp/step5_merge_msg.txt` → merge commit **`014a7e3`** with
   message ref'ing Step-5 verdict + F1–F6 dispositions.
3. `git push origin main` (pushed `014a7e3`).
4. Deleted local + remote `mentor-review/step-5-blue-team`.
5. Cut `mentor-review/step-6-benchmarks` off `main` @ `014a7e3`.
6. Verified policy invariants: `git tag -l` empty, `git branch -a`
   = `main`, `origin/main`, `mentor-review/step-6-benchmarks` only.
7. Ran `pytest -q` → **411 passed in 71.26 s** before any audit
   work.

End state matches policy: one long-lived branch (`main`), zero tags,
current working branch is the per-step topic branch.

### Phase G2 (closing this step) — runs after sign-off
Symmetric to G1. Listed in §6.

---

## 5. Outstanding actions for the next session

The next session executes **Step 7 — Phase 7 Ablations review**
(F9 reward-component, F10 aggressiveness, F12 attack sweep, F15
OOD robustness, gate G7). Phase 7 is where the Phase-6 D6.2.1
"+288 reward gap to oracle ceiling" finding gets its remediation —
the reward-component ablation is the natural follow-up.

### Pre-flight (Phase G1 of Step 7)
- [ ] Verify the candidate has signed off Step 6 either by (a) a
      comment, (b) a merge of `mentor-review/step-6-benchmarks` into
      `main`, or (c) explicit "go" / "Step 7" in chat. If none,
      **stop** and raise.
- [ ] If sign-off given **before** branch merge: execute Phase G2
      ourselves —
  ```
  cat > /tmp/step6_merge_msg.txt <<'MSG' …  (NOT a heredoc; use
  write_to_file)
  git checkout main && git pull --ff-only origin main
  git merge --no-ff mentor-review/step-6-benchmarks -F /tmp/step6_merge_msg.txt
  git push origin main
  git branch -d mentor-review/step-6-benchmarks
  git push origin --delete mentor-review/step-6-benchmarks
  git tag -l   # confirm still empty
  ```
- [ ] Cut `mentor-review/step-7-ablation` off the new `main`.
- [ ] Verify `git tag -l` empty (no tags during the loop, by
      policy).
- [ ] Run `pytest -q` to confirm test count before audit work.
      Expected: 411 passed (Step 6 was strictly read-only audit; no
      tests added or changed). If count differs, **stop** and
      surface.

### Step 7 review checklist (Phase 7 Ablations)
- [ ] Read `docs/results/07_ablation/PLAN.md` in full — frozen audit
      trail. Identify the gates (likely G7.1–G7.k), the figure-IDs
      F9 / F10 / F12 / F15 definitions, and any registered
      D-decisions (D7.x and any post-PLAN D7.x.1 follow-ups, mirror
      of the Phase-6 D6.2.1 / D6.8.1 protocol).
- [ ] Read `docs/results/07_ablation/RESULTS.md` — locked
      scientific record. **Specifically check** whether Phase 7's
      reward-component ablation closes any of the Phase-6 +288
      reward gap to the oracle ceiling; if it does, that's the
      Phase-7 headline.
- [ ] Read `docs/results/07_ablation/G7_scoreboard.json` — if
      present, verify it ships the same `status` enum + `finding_id`
      cross-link schema Phase-6 ships natively.
- [ ] Read `docs/results/07_ablation/manifest.json` (or per-figure
      variants) — verify the input hash chain. **Critical:** Phase
      7 likely re-trains the Phase-5 trio under perturbed reward
      configurations; the manifest should pin the original Phase-5
      model.zips IF used as warm-starts AND the new Phase-7
      checkpoints. The Phase-1 splits manifest SHA `1e99d596…`,
      the Phase-2 LSTM (Step-6 F3 + Step-2 F1 lessons), and the
      `random_forest.joblib` (if F15 OOD eval re-uses it) should all
      be pinned.
- [ ] Read `src/ablation/` (if it exists as a package) and
      `scripts/ablation/` — `run_reward_sweep.py`,
      `run_aggressiveness_sweep.py`, `run_ood_eval.py`,
      `plot_reward_ablation.py`, `plot_aggressiveness.py`,
      `plot_ood_robustness.py`, `plot_pareto.py`, `close_phase7.py`
      (already on disk; visible in the file tree).
- [ ] Read `tests/test_close_phase7_parsers.py` and any
      `tests/test_ablation_*` files.
- [ ] **Realism audit (F9, F10, F12, F15 specifically).**
  - F9 reward-component ablation: which terms swept (de-escalation
    bonus, `penalty_missed_impact`, `reward_proportional`,
    `reward_benign_passive`, `penalty_disproportionate`)? Is the
    sweep grid sensible? Does at least one config close the +288
    gap?
  - F10 aggressiveness sweep: `p_defender_deescalation` axis. How
    many points? Bootstrap CIs?
  - F12 attack sweep: which attack-distribution dimensions varied?
  - F15 OOD robustness: which OOD attack classes evaluated
    (`DDoS-HTTP_Flood`, `Mirai-udpplain`, `VulnerabilityScan`,
    `XSS`)? Pure eval or per-class re-train? Confirms Phase-4 F11
    Vulnerability-Scan recall=0.001 is the "RF can't, but RL
    closes it" claim.
- [ ] **Test-split contract.** Phase 7 ablations must reuse
      `test_balanced` for headline metrics (Phase-6 contract carries
      forward); training perturbations are on the `train` split.
- [ ] **Hash chain.** Verify Phase-7 manifest.json pins:
      Phase-1 splits (`1e99d596…`), Phase-2 LSTM (Step-6 F3 lesson),
      Phase-5 ckpts if used as warm-starts, new Phase-7 ckpts,
      Phase-4 RF if F15 uses it.
- [ ] Re-run `pytest -q` — expect a count that includes
      Phase-7 tests; surface the new total.

### Step 7 outputs (deliverables)
- [ ] Write `docs/mentor_review/07_ablation.md` — full mentor memo,
      lead with verdict (PASS / PASS-WITH-FIXES / FAIL). Cite gate
      IDs (G7.1–G7.k) and file:line. Findings priority-ordered.
- [ ] Write `docs/mentor_review/07_HANDOFF.md` from
      `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for
      **Step 8 (cross-cutting audit + Step 3/4/5/6 doc-fix batch +
      scoreboard-schema backfill for G4.4 + G5.4 + Step-2 F1/F2
      resolution if option a)**.
- [ ] Commit per Conventional Commits
      (`docs(mentor-review,step-7): …`); push to
      `mentor-review/step-7-ablation`.
- [ ] **Pause for candidate sign-off** — do NOT merge to `main`
      without explicit "go" / "Step 8".

### Acceptance criterion for Step 7 PASS
- F9 + F10 + F12 + F15 figures correct (right metric definitions,
  right axis labels, statistical bands labelled, matching PLAN).
- The reward-component ablation (F9) either closes or precisely
  characterises the +288 reward gap Phase-6 left as the Phase-7
  target.
- F15 OOD evaluation either confirms or refutes the thesis claim
  "RL closes the OOD gap that the RF detector exposes on
  VulnerabilityScan" (Phase-4 F11 recall = 0.001).
- Hash chain intact for `docs/results/07_ablation/`. Inputs SHAs
  chain to **post-`3cd2fb9`** Phase-1 splits + Phase-2 LSTM +
  Phase-5 ckpts (warm-starts) + Phase-4 RF (if F15 uses it).
- Test suite green; ablation-scoped tests cover the public API.
- Any fixes filed against documentation (`docs(phase-7,§…)`) unless
  a genuine correctness bug surfaces (then `fix(phase-7,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse --abbrev-ref HEAD     # expect: mentor-review/step-6-benchmarks (this branch)
                                    #   OR main (if Step 6 already merged by candidate)
git --no-pager log --oneline -5     # expect: 06_HANDOFF + 06_benchmark commit on top of 014a7e3
git status                          # expect: clean
git tag -l                          # expect: EMPTY (no tags during the loop, by policy)
git branch -a                       # expect: main, origin/main, current step branch only

pytest -q                           # expect: 411 passed in ~70 s

ls docs/mentor_review/              # expect:
                                    #   README.md, HANDOFF_TEMPLATE.md,
                                    #   00_framing.md, 00_HANDOFF.md,
                                    #   01_dataset.md, 01_HANDOFF.md,
                                    #   02_red_team.md, 02_HANDOFF.md,
                                    #   03_env.md, 03_HANDOFF.md,
                                    #   04_detector.md, 04_HANDOFF.md,
                                    #   05_blue_team.md, 05_HANDOFF.md,
                                    #   06_benchmark.md, 06_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If any expectation fails, **stop** and surface the divergence.
Specifically:
- If `pytest -q` is not 411 passed → Step 6 was strictly read-only
  audit + memo, so any test count change is unexpected.
- If a tag exists → policy violation; cut it before continuing.
- If `mentor-review/step-5-blue-team` still exists locally or
  remotely → Phase G2 of Step 5 didn't fully complete; re-do the
  deletion.

If sign-off has been received but the branch hasn't been merged yet,
execute Phase G2 (use `write_to_file`, NOT a heredoc, for the merge
message — heredocs in `execute_command` mangle in this terminal):

```bash
# Write /tmp/step6_merge_msg.txt via write_to_file with content:
#
# Merge mentor-review/step-6-benchmarks into main
#
# Step 6 (Phase 6 RL benchmarks audit — F5 final security metrics,
# F6 stage×action confusion matrices, F7 computational overhead, F8
# cross-policy reward bars, gates G6.1-G6.7, hash chain, test-split
# contract, detector-integration question) closed at PASS-WITH-FIXES.
#
# Memo: docs/mentor_review/06_benchmark.md
# Handoff: docs/mentor_review/06_HANDOFF.md
#
# Five minor / nit findings filed; F1 (test-count drift 420 vs 411
# on disk), F2 (audit-AF2 oracle-ceiling reframe not propagated to F5
# caption + scoreboard headline_finding), F3 (Phase-2 LSTM SHA not
# pinned in eval_manifest — parallel to Step-5 F2), F4 (run_policy
# seed parameter no-op at env level), F5 (F6_summary.json NaN literal
# non-RFC-7159) — all batched into Step 8.
#
# Six of seven exit gates PASS / PASS-WITH-FINDING; G6.2 mechanically
# FAIL-WITH-FINDING is the headline result of Phase 6 (D6.2.1 / audit
# AF2: trained RL captures 82 % of oracle Recommended-Action ceiling
# on test_balanced; +288 reward gap is the Phase-7 reward-ablation
# target). G6.4 PASS-WITH-FINDING for RF-Acting at 14 ms vs 3 ms
# budget (D6.8.1 — sklearn dispatch overhead). Hash chain byte-
# perfect for everything currently pinned (Phase-1 splits 1e99d596…,
# scaler, RF, all 15 Phase-5 model.zips). Step-1 invariant honoured
# at code (run_test_eval.py:143) and serialisation
# (eval_manifest.json:46-54) levels. Phase-4 CNN1D detector NOT
# consumed by Phase 6 (Step-5 §8 q6 resolved); Phase 6 uses the
# Phase-4 RandomForest detector wrapped as RFActingPolicy (D6.5).
# G6_scoreboard.json natively ships the verdict-enum + finding_ref
# schema Step-5 F1 + Step-4 G4.4 asked for; Step 8 should backfill
# G4.4 + G5.4 to the same schema. Full suite green at 411 passed.
#
git checkout main && git pull --ff-only origin main
git merge --no-ff mentor-review/step-6-benchmarks -F /tmp/step6_merge_msg.txt
git push origin main
git branch -d mentor-review/step-6-benchmarks
git push origin --delete mentor-review/step-6-benchmarks
git checkout -b mentor-review/step-7-ablation
git tag -l            # confirm still empty
git branch -a         # expect: main, origin/main, mentor-review/step-7-ablation
```

> Use `write_to_file` to create `/tmp/step6_merge_msg.txt`, NOT a
> shell heredoc — heredocs in `execute_command` mangle in this
> terminal (per Step-3 / Step-4 / Step-5 handoff git-policy lesson).

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions.
2. `docs/mentor_review/00_framing.md` — locked thesis claims P1/P2/P3
   and R1/R2; IoTWarden's role (inspiration only); chapter outline.
3. `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff
   (still in force).
4. `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1
   dataset audit; Findings 1–6 doc-fixes shipped + Finding 7
   deferred; the post-`3cd2fb9` splits manifest (SHA `1e99d596…`)
   is the canonical Phase-1 output.
5. `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
   red-team audit; Findings F1 (Phase-2 manifest input-hash
   divergence) and F2 (model-selection metric) still **open and
   need candidate decision**.
6. `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 env
   audit; F1–F3 doc-fixes batched to Step 8.
7. `docs/mentor_review/04_detector.md` + `04_HANDOFF.md` — Step-4
   stage-detector audit; F1–F4 doc-fixes batched to Step 8.
8. `docs/mentor_review/05_blue_team.md` + `05_HANDOFF.md` — Step-5
   blue-team audit; F1–F6 batched to Step 8.
9. `docs/mentor_review/06_benchmark.md` — Step-6 mentor memo (full
   prose; cite by Finding number F1–F5).
10. `docs/mentor_review/06_HANDOFF.md` (this file) — the resume
    point.
11. `docs/results/07_ablation/PLAN.md` — Phase-7 plan (frozen; **do
    not edit**).
12. `docs/results/07_ablation/RESULTS.md` — Phase-7 scientific
    record (locked).
13. `docs/results/07_ablation/G7_scoreboard.json` (or per-gate
    variants) — numerical gate verdicts. Verify it uses the same
    `status` + `finding_id` schema Phase-6 ships natively.
14. `docs/results/07_ablation/manifest.json` (or per-figure
    variants) — Phase-7 hash chain. **Verify input SHAs chain to
    post-`3cd2fb9` Phase-1 splits AND to Phase-5 ckpts (if used as
    warm-starts) AND to Phase-2 LSTM (Step-6 F3 lesson) AND to
    Phase-4 RF (if F15 uses it).**
15. `scripts/ablation/run_reward_sweep.py`,
    `run_aggressiveness_sweep.py`, `run_ood_eval.py` — Phase-7
    sweepers.
16. `scripts/ablation/plot_reward_ablation.py`,
    `plot_aggressiveness.py`, `plot_ood_robustness.py`,
    `plot_pareto.py` — F9 / F10 / F12 / F15 figure builders.
17. `scripts/ablation/close_phase7.py` — Phase-7 closer.
18. `tests/test_close_phase7_parsers.py` and any other
    `test_ablation_*.py` files — Phase-7 test coverage.

Skim these for reference (do not read in full):

- `docs/results/06_benchmark/RESULTS.md` §6.1 + §7 — the Phase-7
  hand-offs are enumerated there.
- `docs/results/06_benchmark/G6_scoreboard.json` — for the schema
  reference Phase-7 should mirror.
- `docs/architecture.md`
- `docs/thesis_results_map.md`
- root `README.md`

Then visually inspect Phase-7 figures:

```bash
ls docs/results/07_ablation/
# open the PNGs the directory contains — F9, F10, F12, F15.
```

---

## 8. Open questions for the user

Re-flagged from earlier steps + raised this step:

1. **[carry from Step 2 / Step 3 / Step 4 / Step 5]** **Step-2 F1 —
   Phase-2 manifest input-hash divergence.** Still pending. Confirm
   option (a) Step-7 re-run with `seed=42` against the post-`3cd2fb9`
   manifest (recommended), or option (b) document-only in a
   backfilled Phase-2 RESULTS.md? *Step-6 newly-relevant takeaway:*
   if option (a) is chosen, Phase 6 must re-run too because
   `eval_manifest.json` will pin a new
   `attack_sequence_generator.pth` SHA (Step-6 F3). The numerical
   impact on Phase-6's headline 82 %-of-oracle finding is unknown
   but bounded — the +288 gap is a structural property of the reward
   landscape, not of any particular generator weight.

2. **[carry from Step 2]** **Step-2 F2 — model-selection metric.**
   Was balanced-val cross-entropy or macro-F1 the intended Phase-2
   model-selection criterion? No new evidence from Step 6.

3. **[carry from Steps 3 / 4 / 5 + raised in Step 6]** **Step-3
   F1–F3 + Step-4 F1/F2/F3/F4 + Step-5 F1/F2/F3/F4/F5/F6 + Step-6
   F1/F2/F3/F4/F5 batching into Step 8 cross-cutting cleanup.** All
   minor / nit doc-fixes plus the one code-fix (Step-6 F3:
   `eval_manifest` Phase-2 SHA pin). Confirm batched (recommended)
   over piecemeal landing.

4. **[half-resolved by Step 6 + carry-forward]** **Verdict-enum +
   `finding_ref` scoreboard schema.** Phase-6 `G6_scoreboard.json`
   ships the new schema natively (`status: "FAIL-WITH-FINDING"`,
   `finding_id: "D6.2.1"`). Phase-4 G4.4 + Phase-5 G5.4 still carry
   the older `passes: bool` field with editorial markdown layered on
   top. Confirm Step 8 backfills G4.4 + G5.4 to the Phase-6 schema
   (the field name is `status` — recommend Step 8 use `status`
   everywhere, NOT `verdict`, for consistency).

5. **Resolved this step (Step-5 §8 q6)** — Phase-6 detector-baseline
   lane. Answer: Phase 6 uses the Phase-4 RandomForest detector
   `random_forest.joblib`, **not** the CNN1D `stage_detector.pt`.
   Chain to Phase-4 RF is explicit (`rf_model: 546a7355…` in
   `eval_manifest.json:input_hashes`). The CNN1D detector is
   reserved for Phase-9 ablation.

6. **[New, raised in Step 6]** **Step-9 LaTeX chapter framing.**
   `RESULTS.md` §6.1's audit-AF2 "82 % of oracle ceiling" framing
   is the canonical thesis claim Phase 6 supports. Confirm Step 9's
   `tex/results.tex` rewrite mirrors RESULTS §6.1 verbatim and
   retires any older "RL beats baselines by 25×" prose carried over
   from Phase-5 drafts.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, no model, no hash-pinned
  figure, no test was touched. Pytest count unchanged at 411.
- **Risk noticed (carry-forward to Step 7):** Step-2 Finding 1's
  manifest input-hash divergence (Phase-2 LSTM was demonstrably
  trained on the pre-`3cd2fb9` leaky splits prior). If option (a)
  re-run, Phase 6's eval_manifest must regenerate (Step-6 F3 makes
  this trivial — the manifest is regenerable from JSONLs in seconds).
- **Risk noticed (Step-6 F3):** Phase-6 hash chain to upstream
  Phase-2 LSTM is implicit. Per Step-2 F1 / Step-5 F2, this is the
  same audit-trail invariant at a different boundary. Step-8 doc-fix
  + 1-line code-fix (`run_test_eval.py:_eval_manifest()` adds two
  `input_hashes` entries) closes it.
- **Risk noticed (Step-6 F2):** F5 caption + scoreboard
  `headline_finding` carry the older "rule dominates RL" framing;
  RESULTS.md §6.1 is the new canonical phrasing. If Step 9 LaTeX is
  written from F5 caption rather than from RESULTS.md §6.1, the
  thesis chapter will inherit the old framing. Step 8 must close
  this gap before Step 9 starts.
- **Risk noticed (Step-7 territory):** RESULTS.md §6.1's +288 gap
  becomes Phase-7's reward-component-ablation target. The Phase-7
  reward-component ablation is the natural follow-up; PLAN §7
  already lists the candidate axes (de-escalation bonus,
  `penalty_missed_impact`, `reward_proportional`,
  `reward_benign_passive`, `penalty_disproportionate`). The
  Phase-6 F6 inspection also flags MANEUVER (stage 3) as a
  coupled de-escalation-farming axis; the Phase-7 sweep should
  treat MANEUVER+IMPACT as a coupled remediation target, not
  IMPACT-only.
- **Risk noticed (carry-forward to Step 8):** **all six phases**
  now exhibit per-phase scoreboard or hash-chain asymmetry findings
  (Step-1 F4, Step-2 F4, Step-3 F1, Step-4 F2 + G4.4, Step-5 F1 +
  F2, Step-6 F2 + F3). Step 8 must consolidate. The good news:
  Phase 6 *natively* ships the unified `status` enum + `finding_id`
  cross-link schema; the cross-cutting fix is to backfill Phase 4 +
  Phase 5 scoreboards to the same shape, plus the Phase-2-LSTM-SHA
  one-line code-fix.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or
  out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next
  agent and `07_ablation.md` is opened.

Per the operating rule *"One step per session. Do not start Step 7
until the candidate signs off Step 6."* — Step 7 may not begin
without an explicit "go" / "Step 7" / merge of this branch.
