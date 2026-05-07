# Step `07` — `Phase 7 Ablations` — Mentor Review Handoff

**Closed:** `2026-05-06 22:55 -03:00`
**Author (agent):** Cline (mentor agent), on behalf of Prof. Dr. Denis Fantinato
**Reviewed phase / scope:** `Phase 7 — Ablations + OOD-class Robustness`
**Status:** `completed (PASS-WITH-FIXES, awaiting candidate sign-off)`

---

## 1. What was reviewed

### Artifacts
- `docs/results/07_ablation/PLAN.md` — frozen plan (read-only); gates G7.1–G7.9, R7.x risks, D7.x decisions
- `docs/results/07_ablation/RESULTS.md` — locked scientific record (now extended with §6.1 caveat, §6.3 caveat, §6.4 sharpening, §7 MANEUVER bullet, §8.1 eval contract, §9 footnote)
- `docs/results/07_ablation/G7_scoreboard.json` — gate scoreboard (now annotated with `note_post_lock_2026-05-06` on G7.1 + G7.6)
- `docs/results/07_ablation/F9_summary.json` (12 reward cells), `F10_summary.json` (6 p-values × PPO + rule), `F12_summary.json` (32 candidates), `F15_summary.json` (4 OOD × 8 policies)
- `docs/results/07_ablation/F9_manifest.json`, `F10_manifest.json`, `F12_manifest.json`, `F15_manifest.json` — hash-chain inputs
- `docs/results/07_ablation/F9_caption.md`, `F10_caption.md`, `F12_caption.md`, `F15_caption.md` — thesis-paper captions (F9, F10, F12 rewritten with disambiguation; F15 left as-is)
- `docs/results/07_ablation/F9_reward_ablation.png`, `F10_aggressiveness.png`, `F12_pareto.png`, `F15_ood_robustness.png` — figure assets

### Code
- `scripts/ablation/run_reward_sweep.py` — F9 driver; verified eval split = `test_balanced`, `exclude_ood=True` at lines 299, 301
- `scripts/ablation/run_aggressiveness_sweep.py` — F10 driver; same eval contract
- `scripts/ablation/run_ood_eval.py` — F15 driver with hybrid OOD realiser (commit `87b80dc`)
- `scripts/ablation/plot_reward_ablation.py` — F9 plotter + `_evaluate_g72` two-strand evaluator
- `scripts/ablation/plot_aggressiveness.py` — F10 plotter + G7.3 evaluator
- `scripts/ablation/plot_pareto.py` — F12 plotter + G7.4 evaluator
- `scripts/ablation/plot_ood_robustness.py` — F15 plotter + G7.8 + G7.9 evaluators
- `scripts/ablation/close_phase7.py` — closer; `_write_scoreboard` at lines 355–374
- `tests/test_close_phase7_parsers.py` (12 tests) — pytest-summary parser + G7.2 two-strand logic
- `tests/test_phase31_impact_terminal.py` (8 tests) — `impact_is_terminal` codepath
- `tests/test_train_agent_reward_overrides.py` (14 tests) — CLI override plumbing
- `src/environment/adversarial_env.py` — `impact_is_terminal: bool = True` field
- `src/blue_team/run_config.py` — `EnvConfigSerializable` extended 7→18 fields
- `src/blue_team/env_factory.py` — `_build_env_config` reward-field forwarding

### Docs
- `docs/mentor_review/06_HANDOFF.md` — Step-7 entry point; §5 checklist + §7 context-loading recipe + §8 open questions
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — section structure for this file
- `docs/results/06_benchmark/RESULTS.md` (§6.1) — Phase-6 oracle ceiling +1624 anchor

---

## 2. Verdict

**`PASS-WITH-FIXES`.**

Phase 7 is the strongest scientific deliverable since Phase 1. F9
partially closes the +288 oracle-ceiling gap that Phase 6 left
(PPO `impact_is_terminal=False` reaches +1542, Δ_DQN +205.6,
71 % of the gap, with `mitigated_impact_rate` 0.153→0.900 = 5.9×).
F15 honestly activates the pre-registered D7.9.1 narrowing of the
OOD claim ("RL is *robust to* `VulnerabilityScan`, not *better
at* it"; DQN +1313 vs RF-Acting +1611, Δ=−298, CIs disjoint). F10
delivers monotone PPO sensitivity to attacker aggressiveness
(G7.3 PASS). F12 fails its frontier threshold *as pre-registered*
(R7.3, G7.4 FAIL-WITH-FINDING) — and the actual situation is
sharper than R7.3 suggested (the y-axis is identically zero on
`test_balanced`; F7 in the memo). Test-split contract verified
end-to-end in code; hash chain intact for the data-flow path
(F9 + F15) but two manifests under-pin upstream SHAs (F2, batched
to Step 8). Scoreboard schema uses `passes:bool` instead of
Phase-6's native `status:enum + finding_id` (F3, also Step 8).
Six doc-fixes shipped this session (F1, F4, F5, F6, F7a, F8 +
two thesis-framing caveats); two findings batched to Step 8 (F2,
F3); two deferred to future work / Step 9 LaTeX (F6 MANEUVER
coupling, F9 `compromise_rate` thesis paragraph).

---

## 3. Findings (priority-ordered)

Full memo at `docs/mentor_review/07_ablation.md` §3. Concise
summary here:

1. **[major]** F1 — `RESULTS.md §9` and `G7_scoreboard.json#G7.1`
   record `454 passed`, but HEAD `pytest --collect-only -q`
   reports `411 collected`. Forensic: Phase-10 commit `281860a`
   deleted 43 dead-code tests after Phase-7 lock. Doc-only.
   **Shipped this session** as RESULTS §9 footnote +
   scoreboard `note_post_lock_2026-05-06` on G7.1 + G7.6.

2. **[major]** F2 — `F10_manifest.json` and `F12_manifest.json`
   under-pin upstream SHAs (no `phase6_eval_manifest` SHA on
   F10; no `phase5_sweep_manifest` SHA on either). Hash chain
   on disk is intact (verified `cc7454…/c4a60a…` etc.); the
   gap is in the *recorded* chain. Mirror of Phase-6 F3.
   **Batched to Step 8** as a reproducibility-hygiene wave with
   the Phase-2-LSTM-SHA-pin code-fix.

3. **[major]** F3 — `G7_scoreboard.json` uses `passes:bool` +
   free-text `interpretation` field; Phase 6 ships native
   `status:enum + finding_id`. Cross-cutting; same backfill
   target as Step-4 G4.4 + Step-5 G5.4.
   **Batched to Step 8** as the verdict-enum unification wave.
   Use `status` (Phase-6 native), NOT `verdict`.

4. **[major]** F4 — F9 baseline `mitigated_impact_rate = 0.273`
   is easy to misread vs the §6.1 baseline 0.153 (the 5.9×
   ratio uses 0.153 as the denominator). **Shipped this
   session** as F9 caption clarification + RESULTS §1
   parenthetical.

5. **[minor]** F5 — Test-split contract on
   `split="test_balanced", exclude_ood=True` is satisfied in
   code but not stated in RESULTS §8. **Shipped this session**
   as new RESULTS §8.1 "Eval contract" subsection.

6. **[major; defer]** F6 — F9 does not address the MANEUVER
   (kill-chain stage 3) de-escalation-farming pattern that
   Phase-6 F6 inspection flagged for DQN at 58 % ISOLATE; only
   IMPACT-stage semantics are flipped. Phase 8 / future-work
   territory. **Documented this session** in RESULTS §7 as a
   Phase-7-surfaced-but-not-addressed item.

7. **[major]** F7 — F12 `security_gain` is identically 0.0
   across all 32 candidates because `compromise_rate = 1.0` for
   every Phase-7 cell and every Phase-6 anchor on
   `test_balanced`. The "Pareto frontier" is degenerate (1-D
   scatter on availability_cost), not just "approximately
   linear". **Shipped this session (option a)** as F12 caption
   rewrite + RESULTS §6.4 sharpening paragraph. **Step-8
   candidate decision (option b):** re-emit F12 with
   `mitigated_impact_rate` y-axis if the candidate wants the
   figure to land in the thesis as a 2-D plot.

8. **[minor]** F8 — F10 PPO at `p=1.0` reaches +2047, exceeding
   the Phase-6 oracle ceiling +1624 because the high-`p` cells
   operate in a strictly easier MDP (defender always
   de-escalates). Not a comparison; needs explicit disclaimer.
   **Shipped this session** as F10 caption + RESULTS §6.3
   caveat.

9. **[nit]** F9 — `compromise_rate = 1.0` is honest but
   uncomfortable; the F9 +1542 / mit-rate=0.900 win lives in
   the post-IMPACT mitigation regime, not pre-IMPACT
   prevention. **Documented this session** as RESULTS §6.1
   caveat; full thesis-framing paragraph deferred to Step 9
   (LaTeX framing).

---

## 4. Actions taken in this session

- **Files added:**
  - `docs/mentor_review/07_ablation.md` (full mentor memo, ~480 lines)
  - `docs/mentor_review/07_HANDOFF.md` (this file)

- **Files edited:**
  - `docs/results/07_ablation/RESULTS.md` — added §1 baseline-disambiguation parenthetical (F4); §6.1 `compromise_rate` caveat (F9-paragraph); §6.3 F10-MDP caveat (F8); §6.4 F12 degeneracy sharpening (F7a); §7 MANEUVER-coupling bullet (F6); §8.1 new "Eval contract" subsection (F5); §9 post-locking footnote (F1)
  - `docs/results/07_ablation/G7_scoreboard.json` — added `note_post_lock_2026-05-06` to G7.1 + G7.6 (F1)
  - `docs/results/07_ablation/F9_caption.md` — rewritten with `n_episodes` disclosure + 5.9× baseline disambiguation (F4)
  - `docs/results/07_ablation/F10_caption.md` — rewritten with monotonicity-vs-absolute-level caveat (F8)
  - `docs/results/07_ablation/F12_caption.md` — rewritten with degenerate-y-axis disclosure (F7a)

- **Files deleted:** none

- **Tests added / changed:** none (audit-only, per Step 7 mandate)

- **Scripts added / refactored:** none (Step-7 finding F2 + F3 batched to Step 8)

- **Results re-run, if any:** none (no model retraining; no plot
  regeneration). All artefacts under `docs/results/07_ablation/`
  retained their original git_sha / SHA-256 references; hash
  chain on disk byte-perfect against recorded SHAs.

- **Git phases:** G2 of Step 6 executed at session start (merge
  `mentor-review/step-6-benchmarks` → `main` as `1d78fec`,
  pushed, branch deleted local + remote). G1 of Step 7 cut
  `mentor-review/step-7-ablation` off `main` at `1d78fec`.

---

## 5. Outstanding actions for the next session

These belong to Step 8 (cross-cutting cleanup wave). Each item
is checkable + concrete.

- [ ] **F1 follow-up (optional):** investigate whether the −43
      tests reflect any *unintended* loss beyond the 43 dead
      `src/benchmarking/` tests — acceptance criterion: write
      a one-line summary in `08_*.md` confirming all 43 deleted
      tests trace to commit `281860a`, no orphan-test
      regression.
- [ ] **F2 fix:** patch `scripts/ablation/plot_aggressiveness.py`
      and `scripts/ablation/plot_pareto.py` to embed
      `phase5_sweep_manifest` SHA in `F10_manifest.json` and
      both `phase5_sweep_manifest` + `phase6_eval_manifest`
      SHAs in `F12_manifest.json`. Optionally add an explicit
      `phase1_splits_manifest` pin to all four. Acceptance:
      `inputs.keys()` of every Phase-7 manifest contains
      `phase5_sweep_manifest` and `phase6_eval_manifest` (where
      applicable).
- [ ] **F3 fix:** rewrite `scripts/ablation/close_phase7.py:_write_scoreboard`
      to emit the Phase-6-native schema: replace `passes:bool`
      with `status: "PASS"|"PASS-WITH-FINDING"|"PASS-WITHOUT-STRETCH"|"FAIL-WITH-FINDING"|"FAIL"`,
      add `finding_id` field referencing `R7.3`/`D7.9.1`/etc.
      Re-emit `G7_scoreboard.json`. Backfill same on `G4_scoreboard.json` +
      `G5_scoreboard.json`. Acceptance: `jq '.gates[].status'` on
      all 3 scoreboards returns valid enum members; no `passes`
      key remains. Use **`status`** (NOT `verdict`).
- [ ] **Step-3/4/5/6/7 doc-fix batch:** consolidate all
      previously-deferred doc-fixes from
      `03_HANDOFF.md`/`04_HANDOFF.md`/`05_HANDOFF.md`/`06_HANDOFF.md`/`07_HANDOFF.md`
      §5 lists into a single Step-8 commit pile. Acceptance:
      `08_HANDOFF.md` §4 itemises every doc-fix landed.
- [ ] **Step-2 F1 decision-execution (option a or b):** if
      candidate chose (a) Phase-2 LSTM re-run with `seed=42` —
      execute the re-run, regenerate Phase-2 RESULTS.md, re-emit
      Phase-6 `eval_manifest.json` to pin the new
      `attack_sequence_generator.pth` SHA. If (b) — backfill a
      seed-justification paragraph in Phase-2 RESULTS.md.
      Acceptance: `08_*.md` records the choice + executes it.
- [ ] **Step-2 F2 decision-execution:** record candidate's
      choice on Phase-2 model-selection criterion (balanced-val
      cross-entropy vs macro-F1) in Phase-2 RESULTS.md.
- [ ] **Step-6 F3 fix:** emit Phase-2 LSTM SHA pin in Phase-6
      `eval_manifest.json` (trivial code-fix; manifest is
      regenerable in seconds from on-disk JSONLs).
- [ ] **F7(b) candidate decision:** if the candidate wants F12
      to land in the thesis as a 2-D figure, re-emit it with
      `mitigated_impact_rate` y-axis (~30 lines in
      `plot_pareto.py`). Otherwise the §6.4 + caption rewrites
      shipped this session are sufficient.

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse HEAD                 # expect: tip of mentor-review/step-7-ablation
                                   #         (the docs(mentor-review,step-7) commit)
git --no-pager log --oneline -5    # expect: docs(mentor-review,step-7) on top of
                                   #         1d78fec Merge mentor-review/step-6-benchmarks
                                   #         on top of 55f8f6d Phase 6 mentor commit
                                   #         on top of 014a7e3 Phase 5 merge
git status                         # expect: clean
git tag -l                         # expect: EMPTY (mentor-loop policy; v0.1.0 in
                                   #         later detached commits but NOT yet on main
                                   #         or merged at HEAD-of-step-7)
pytest -q                          # expect: 411 passed (see RESULTS §9 footnote)
ls docs/mentor_review/             # expect: 07_ablation.md + 07_HANDOFF.md present
```

If any of those expectations fail, **stop** and surface the
divergence before continuing.

**Phase-G2 of Step 7** is owed at the next session start: merge
`mentor-review/step-7-ablation` → `main` with `--no-ff -F /tmp/merge-step-7.txt`,
push `main`, delete the branch local + remote, then cut
`mentor-review/step-8-cleanup` off the new `main`.

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any
work in Step 8:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` … `06_HANDOFF.md` — the
   six predecessor steps in order; skim first read, then
   re-read each `_HANDOFF.md` §5 list (those are the items
   Step 8 owns)
3. `docs/mentor_review/07_ablation.md` — Step 7 full memo
   (§3 Findings = the priority-ordered Step-8 input)
4. `docs/mentor_review/07_HANDOFF.md` (this file) — the resume
   point
5. `docs/results/07_ablation/RESULTS.md` — Phase-7 scientific
   record (now extended with §6.1/§6.3/§6.4/§7/§8.1/§9
   caveats and footnotes)
6. `docs/results/07_ablation/PLAN.md` — frozen plan (read-only)
7. `docs/results/07_ablation/G7_scoreboard.json` — current
   schema (will be re-emitted in Step 8 F3 fix)
8. `docs/results/06_benchmark/G6_scoreboard.json` — target
   schema for Step 8 F3 backfill (Phase-6-native `status`
   enum + `finding_id`)
9. `docs/results/04_detector/G4_scoreboard.json` and
   `docs/results/05_blue_team/G5_scoreboard.json` — also
   targets for Step 8 F3 backfill

Skim these for reference if needed (do not read in full):

- `docs/thesis_results_map.md`
- `docs/architecture.md`
- root `README.md`
- `scripts/ablation/close_phase7.py:_write_scoreboard` (lines
  355–374) — the function that needs to be rewritten for F3

---

## 8. Open questions for the user

These are eight cross-step decisions still owed by the candidate.
None block Step-7 sign-off; all surface in Step 8 unless resolved
sooner.

1. **Step-2 F1 path** — Phase-2 LSTM re-run with `seed=42`
   against the post-`3cd2fb9` manifest (option a) or document-
   only in a backfilled Phase-2 RESULTS.md (option b)? *Note:
   option (a) forces a Phase-6 `eval_manifest.json` re-emit to
   pin the new `attack_sequence_generator.pth` SHA — Step-6 F3
   makes this trivial.*
2. **Step-2 F2** — was balanced-val cross-entropy or macro-F1
   the intended Phase-2 model-selection criterion? Need a
   one-paragraph documentation choice.
3. **Step-3 / 4 / 5 / 6 / 7 doc-fix batching into Step 8** —
   confirm the cross-phase batch over piecemeal commits.
4. **Verdict-enum scoreboard schema backfill (F3)** — for G4.4
   + G5.4 + G7.x, match Phase-6-native `status` enum +
   `finding_id`. Recommend **`status`** (NOT `verdict`) to
   avoid a schema split.
5. **Step-9 LaTeX framing** — RESULTS.md §6.1's "82 % of
   oracle ceiling" + §6.2's "robust to (not better at) the OOD
   class" must be the canonical thesis claims; older "RL beats
   baselines by 25×" must be retired. Plus the §6.1 caveat
   paragraph on `compromise_rate=1.0` (F9 of memo) needs full
   thesis-framing.
6. **NEW (F7(b))** — F12 y-axis remediation. Doc-fix only
   (option a, **shipped this session**) or re-emit with
   `mitigated_impact_rate` y-axis (option b, requires explicit
   re-run opt-in)? Mentor-recommendation in `07_ablation.md`
   §3 F7 is **option a is sufficient unless the figure is to
   land in the thesis as a 2-D plot.**
7. **NEW (F9-paragraph)** — `compromise_rate = 1.0` thesis-
   framing paragraph: author in Step 9 (LaTeX) or fold into
   RESULTS.md §6.1 in Step 8?
8. **Phase-8 vs Phase-10 routing.** Commit `8d5dd67`
   (`docs(handoff): rewrite for Phase-7 closeout — D2
   (Phase 8 vs 10) decision required`) and the chain
   `f1a68f3`, `fa1a791`, `281860a`, `8c6e665`, `0a1352d`,
   `2deda39`, `a969fd6` (with `v0.1.0` tag) suggest the
   candidate already executed Phase 10 in parallel and
   **skipped Phase 8**. Is this confirmed? If so, F6 (MANEUVER
   coupling) and F13/F14 deferrals from Phase-7 RESULTS §7
   should be reframed as **future-work / post-thesis** rather
   than Phase 8. The mentor-review loop continues straight
   from Step 8 (cleanup) → Step 9 (LaTeX) → Step 10
   (release-tag, possibly `v1.0.0`).

---

## 9. Risks introduced or noticed

- **Risk: post-locking RESULTS drift on test-count claim** —
  likelihood: ALREADY ACTUALISED (the `454` claim was already
  stale at session start); impact: low (committee will
  ask, but the §9 footnote shipped this session pre-empts
  the question); mitigation: §9 footnote + scoreboard `note`
  shipped.

- **Risk: F12 figure as 2-D plot misleads the defense
  committee** — likelihood: medium (committee may read F12
  before reading §6.4); impact: medium (could trigger a
  "this is not a Pareto plot" challenge); mitigation: F12
  caption rewrite + RESULTS §6.4 sharpening paragraph
  shipped this session; F7(b) decision pending if 2-D
  remediation is desired.

- **Risk: F10 absolute-level mis-comparison against §6.1
  ceiling** — likelihood: medium-low (candidate is unlikely
  to make this slip in defense, but a committee member
  reading §6.3 in isolation might); impact: medium;
  mitigation: F10 caption + RESULTS §6.3 caveat shipped.

- **Risk: scoreboard schema split between G4.4/G5.4/G7.x
  (`passes`) and G6.x (`status`) silently complicates Step-9
  LaTeX automation that reads scoreboards** — likelihood:
  medium; impact: low (mechanical script change); mitigation:
  Step 8 F3 backfill — *recommend Step 8 lands the unified
  schema before Step 9 starts*.

- **Risk: Phase-7-skipped-Phase-8 is implicit** — if Q8
  resolves "Phase 8 was skipped", the future-work scoping
  in RESULTS §7 needs a one-line update from "Phase 8
  owns…" to "future work / post-thesis owns…" to avoid the
  defense committee asking "where is Phase 8 then?";
  Step-9 LaTeX framing job.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment,
  or out-of-band confirmation), **or**
- the "Outstanding actions" list above is empty.

When the candidate types "go" / "Step 8" / merges this branch:

1. Phase G2 of Step 7: merge `mentor-review/step-7-ablation` →
   `main` `--no-ff -F /tmp/merge-step-7.txt`, push, delete
   branch local + remote.
2. Phase G1 of Step 8: cut `mentor-review/step-8-cleanup` off
   the new `main`.
3. Begin Step 8: cross-cutting cleanup wave per §5 above.
