# Step `05` — Phase 5 Blue Team RL Training Review — Mentor Review Handoff

**Closed:** `2026-05-06 ~14:30 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 6)
**Reviewed phase / scope:** Phase 5 (PPO / DQN / A2C × 5 seeds against
the Phase-2 LSTM Red Team on the Phase-3 environment, Phase-1 train
split with `exclude_ood=True`; thesis figures F3 + F4; appendix
table T1; exit gates G5.1–G5.7; reward-function fidelity vs Phase-3
frozen contract; detector-integration question carried from Step 4)
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts (frozen audit trail; never edited)
- `docs/results/05_blue_team/PLAN.md` (511 lines) — design contract;
  D5.1–D5.11 locked + dated D-decisions D5.3.1, D5.4.1, D5.10.1
  (the probe-driven gate revisions).
- `docs/results/05_blue_team/RESULTS.md` (302 lines) — locked
  scientific record; §2 final scoreboard; §3 headline numbers; §4
  four findings (Finding 2 = de-escalation farming = G5.4
  PASS-with-finding); §5 lessons learned.
- `docs/results/05_blue_team/G5_scoreboard.json` (260 lines) —
  mechanical gate verdicts. Note: `G5.4.passes = false` is the
  honest mechanical reading; RESULTS.md upgrades to PASS-with-
  finding by editorial judgement.
- `docs/results/05_blue_team/F3_manifest.json` — F3 hash chain
  (30 input JSONLs + 2 outputs; producing git_sha `03353d54068f`).
- `docs/results/05_blue_team/F4_manifest.json` — F4 hash chain
  (same 30 inputs + 2 outputs; producing git_sha
  `03353d54068f-dirty`).
- `docs/results/05_blue_team/F3_summary.json` — per-algo per-seed
  numerical truth (train + eval windows); SHA `229814e8…`.
- `docs/results/05_blue_team/F4_summary.json` — marginal action
  share per bin + per-stage at three checkpoints + `g5_5_per_stage`;
  SHA `5ab4e6cf…`.
- `docs/results/05_blue_team/F3_learning_curves.png` (SHA
  `d03fcd9d…`).
- `docs/results/05_blue_team/F4_action_distribution.png` (SHA
  `424c4dc0…`).
- `docs/results/05_blue_team/T1_hparams.json` + `T1_hparams.md` —
  per-algo hyperparameter table.
- `docs/results/05_blue_team/F3_caption.md` + `F4_caption.md`.
- `runs/phase5/sweep_manifest.json` — top-level sweep record.
- `runs/phase5/<algo>/seed_<k>/run_manifest.json` (15 files) —
  per-run frozen config + post-run telemetry; spot-checked PPO
  seed 0.

### Code
- `src/blue_team/__init__.py`, `run_config.py` (204), `env_factory.py`
  (184), `aggregation.py` (424), `callbacks.py` (507) — Phase-5
  package.
- `src/algorithms/adversarial_algorithm.py` (337) — SB3 wrapper.
- `scripts/blue_team/train_agent.py` (460) — single-run entrypoint.
- `scripts/blue_team/run_phase5.py` (171) — sweep driver (D5.6
  subprocess fan-out).
- `scripts/blue_team/evaluate_gates.py` (238) — G5.x evaluator.
- `scripts/blue_team/plot_learning_curves.py` (316) — F3.
- `scripts/blue_team/plot_action_dist.py` (351) — F4.
- `scripts/blue_team/dump_hparams.py` (122) — T1.

### Tests
- `tests/test_blue_team_aggregation.py` (24 tests, 311 lines).
- `tests/test_blue_team_callbacks.py` (259 lines).
- `tests/test_blue_team_env_factory.py` (236 lines).
- `tests/test_blue_team_run_config.py` (93 lines).
- `tests/test_blue_team_train_agent.py` (164 lines).
- `tests/test_train_agent_reward_overrides.py` (315 lines).
- Full suite: **`pytest -q` → 411 passed in 64.71 s** on
  `mentor-review/step-5-blue-team` (cut off `main` @ `81804cc` =
  Step-4 merge `81804cc`).

### Docs
- `docs/mentor_review/README.md` — directory conventions.
- `docs/mentor_review/00_framing.md` + `00_HANDOFF.md` — locked
  thesis claims P1/P2/P3 + R1/R2.
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1.
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
  Findings F1, F2 still open.
- `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 F2
  monkey-patch contract still in force.
- `docs/mentor_review/04_detector.md` + `04_HANDOFF.md` — Step-4;
  open question 4 (detector-Phase-5 integration) is **resolved
  this step**: Phase 5 does NOT consume `stage_detector.pt`.
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — template for this file.
- `docs/mentor_review/05_blue_team.md` — Step-5 mentor memo (written
  this session).

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-5 RL training package is the cleanest Phase-package this
review has seen. Six of seven exit gates PASS mechanically (G5.2
+1350.7 reward, G5.3 19.24 MTTC, G5.5 max share 0.45 ≪ 0.70, G5.7
manifests present; G5.1 + G5.6 implicitly via 411-pytest). G5.4 is
mechanically FAIL (mitigated-impact rate 0.263 < 0.50) and editorially
PASS-WITH-FINDING per RESULTS.md §4 Finding 2 — the agent learned to
farm de-escalation bonuses and accept the IMPACT loss, which is the
headline thesis result on reward hacking. Step-1 invariant honoured
both by code and by serialisation. The Phase-4 detector is **NOT**
in the agent's observation pipeline (D5.2 design intent) — Step-4
open question 4 resolved. F3/F4/T1 numerically and structurally
correct. Hash chain *internal* to Phase 5 byte-perfect; chain *to
upstream Phase-1 splits* is implicit (Finding F2). Six minor / nit
findings, all batchable into Step 8.

Full memo: `docs/mentor_review/05_blue_team.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: minor]** **F1** — G5.4 mechanical FAIL
   (`G5_scoreboard.json:207-213`) ↔ narrative PASS-WITH-FINDING
   (`RESULTS.md §2 + §4 Finding 2`) is not cross-linked inside the
   JSON. A defense reader who only opens the scoreboard will see
   `passes: false` and miss the editorial layer.
   **Recommended fix:** add a `verdict` and `finding_ref` field to
   the gate dict so the JSON is self-explaining. Commit:
   `docs(phase-5,§2): cross-link G5.4 scoreboard PASS-with-finding to
   RESULTS.md §4 Finding 2`. **Disposition:** batch into Step 8.

2. **[severity: minor]** **F2** — Hash chain back to the
   post-`3cd2fb9` Phase-1 splits manifest (SHA `1e99d596…`) is
   *implicit* (`run_manifest.json::paths` records the path string
   only; `F3_manifest.json` and `F4_manifest.json` pin only
   output-side JSONLs; no top-level Phase-5 `manifest.json` exists).
   Phase 5 demonstrably ran on the post-fix manifest (file-mtime
   correlation), but a defense reviewer cannot verify it without
   reproducing that correlation. Compare to Phase-4 which pins six
   input SHAs explicitly.
   **Recommended fix (option a, recommended):** add a top-level
   `docs/results/05_blue_team/manifest.json` pinning the splits
   manifest, the Phase-2 LSTM checkpoint, the dataset
   `features.npy`/`labels.npy`, the producing git_sha, and the
   eight Phase-5 outputs. Commit: `docs(phase-5,§hash-chain): add
   top-level manifest.json pinning Phase-1 + Phase-2 input SHAs`.
   **Disposition:** batch into Step 8.

3. **[severity: nit]** **F3** —
   `scripts/blue_team/evaluate_gates.py:81-89` `_select_best_algo`
   tie-break: docstring claims "lowest std", code uses `(-mean_mttc)`,
   PLAN §8 D5.11 says "lower variance". Triple disagreement; never
   fires in practice (mean gaps are large) but visible to any
   reviewer who reads the code. **Recommended fix (doc-only):**
   amend PLAN §8 D5.11 + docstring to acknowledge MTTC tie-break.
   Commit: `docs(phase-5,§D5.11): align tie-break docstring + PLAN
   with code (highest MTTC)`. **Disposition:** batch into Step 8.

4. **[severity: nit]** **F4** — `src/blue_team/env_factory.py:10-13`
   docstring claims `make_eval_env` defaults to `val_balanced`, but
   the function imposes no default — the split lives at the caller
   layer. **Recommended fix:** docstring rewrite OR keyword-default
   on the function signature (the docstring rewrite is the cleaner,
   stable-SHA option). Commit: `docs(phase-5,env_factory): clarify
   make_eval_env split is caller-supplied`. **Disposition:** batch
   into Step 8.

5. **[severity: nit]** **F5** — MLflow setup exists in
   `docs/experiments-mlflow.md` but Phase 5 doesn't use it; the
   Step-4 handoff §5 incorrectly forecast Phase 5 as "the first
   phase with MLflow runs". Phase-5 D5.6 is intentionally
   JSONL-based. **Recommended fix:** doc-fix in
   `docs/experiments-mlflow.md` clarifying scope, or delete the
   doc. Commit: `docs(experiments-mlflow): clarify scope; Phase 5
   uses JSONL + run_manifest`. **Disposition:** batch into Step 8.

6. **[severity: nit, Phase-3 doc only]** **F6** — Phase-3
   RESULTS.md §3 "six-term reward" mismatches Phase-5 wiring count
   (`env_factory.py:53-73` plumbs nine reward fields plus
   `action_cost_scale`). The six logical terms collapse the three
   sub-modulators (`penalty_overreact_benign / penalty_block_benign /
   penalty_block_recon`) but the Phase-3 doc doesn't say so.
   **Recommended fix:** Phase-3 doc-fix in RESULTS.md §3. Commit:
   `docs(phase-3,§3): clarify reward decomposition (six terms +
   three modulators)`. **Disposition:** batch into Step 8 with the
   Step-3 doc batch.

Full prose, file:line citations, and recommended commit messages:
`docs/mentor_review/05_blue_team.md` §3.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/05_blue_team.md` — Step-5 mentor memo (verdict
  PASS-WITH-FIXES + 6 findings + hash-chain reproduction +
  detector-integration audit + Step-1 invariant audit + reward-
  function audit + hyperparameter audit + F3/F4/T1 realism audit +
  reproducibility audit + test-coverage audit + carry-forward table).
- `docs/mentor_review/05_HANDOFF.md` — this file.

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: **411 passed in 64.71 s**
on `mentor-review/step-5-blue-team`.

### Scripts / models
None modified. No re-training, no figure regeneration. **Hash chain
intact** — verified byte-perfect for the four hash-pinned outputs in
`docs/results/05_blue_team/` (F3 PNG + summary, F4 PNG + summary)
and three random spot checks of input JSONLs in
`runs/phase5/{ppo,dqn,a2c}/seed_*/`.

### Git hygiene applied (Phase G1, opening this step)
1. `git checkout main && git pull --ff-only origin main`.
2. `git merge --no-ff mentor-review/step-4-detector -F /tmp/step4_merge_msg.txt`
   → merge commit **`81804cc`** with message ref'ing Step-4 verdict +
   F1/F2/F3/F4/F5 dispositions.
3. `git push origin main` (pushed `81804cc`).
4. Deleted local + remote `mentor-review/step-4-detector`.
5. Cut `mentor-review/step-5-blue-team` off `main` @ `81804cc`.
6. Verified policy invariants: `git tag -l` empty,
   `git branch -a` = `main`, `origin/main`,
   `mentor-review/step-5-blue-team` only.
7. Ran `pytest -q` → **411 passed in 63.37 s** before any audit work.

End state matches policy: one long-lived branch (`main`), zero tags,
current working branch is the per-step topic branch.

### Phase G2 (closing this step) — runs after sign-off
Symmetric to G1. Listed in §6.

---

## 5. Outstanding actions for the next session

The next session executes **Step 6 — Phase 6 Benchmarks review**
(F5, F6, F7, F8, G6). Phase 6 consumes the 15 Phase-5 model
checkpoints to produce the final benchmark table, stage × action
confusion matrices, computation-overhead plots, and (if applicable)
a comparison against the IoTWarden recommended-action oracle.

### Pre-flight (Phase G1 of Step 6)
- [ ] Verify the candidate has signed off Step 5 either by (a) a
      comment, (b) a merge of `mentor-review/step-5-blue-team` into
      `main`, or (c) explicit "go" / "Step 6" in chat. If none,
      **stop** and raise.
- [ ] If sign-off given **before** branch merge: execute Phase G2
      ourselves —
  ```
  git checkout main && git pull --ff-only origin main
  git merge --no-ff mentor-review/step-5-blue-team -F /tmp/step5_merge_msg.txt
  git push origin main
  git branch -d mentor-review/step-5-blue-team
  git push origin --delete mentor-review/step-5-blue-team
  git tag -l   # confirm still empty
  ```
- [ ] Cut `mentor-review/step-6-benchmarks` off the new `main`.
- [ ] If any Step-5 *fix* commits were applied (F1/F2/F3/F4/F5/F6
      doc-fixes), pull them onto `main` first so Step 6 starts from
      corrected state. (Recommendation per §3: batch all into
      Step 8 — so Step 6 starts from the unmodified Step-5 state.)
- [ ] Run `pytest -q` to confirm 411 passed before audit work. If
      count differs, **stop** and surface.
- [ ] Verify `git tag -l` is empty (no tags during the loop, by
      policy).

### Step 6 review checklist (Phase 6 Benchmarks)
- [ ] Read `docs/results/06_benchmark/PLAN.md` in full — frozen audit
      trail. Note the gates (likely G6.1–G6.k) and the figure-ID
      definitions for F5, F6, F7, and F8 (final benchmark, stage ×
      action confusion matrices, computation overhead, possibly an
      oracle-vs-trained comparison).
- [ ] Read `docs/results/06_benchmark/RESULTS.md` — locked scientific
      record.
- [ ] Read `docs/results/06_benchmark/manifest.json` — verify hash
      chain via `shasum -a 256`. **Critical:** confirm input SHAs
      chain to the Phase-5 model checkpoints
      (`runs/phase5/<algo>/seed_<k>/model.zip`) AND to the
      post-`3cd2fb9` Phase-1 splits manifest (Step-2 F1 + Step-5 F2
      lessons) AND to the Phase-4 `stage_detector.pt: 71e06616…` if
      Phase 6 includes the detector-augmented evaluation lane.
      *(Reminder from Step 5: Phase-5 hash chain to upstream is
      implicit; Phase-6 should be explicit per Phase-4's pattern.)*
- [ ] Read `docs/results/06_benchmark/G6_scoreboard.json` if present;
      if absent, file as a finding consistent with Step-5 F1's
      scoreboard self-explaining-ness roll-up.
- [ ] Read `src/benchmark/baseline_policies.py` —
      `RandomPolicy`, `RecommendedActionPolicy` (the IoTWarden
      oracle), any other baselines.
- [ ] Read `src/benchmark/eval_runner.py` — gold-standard evaluation
      loop. Verify it consumes `split="test_balanced"` (not
      `val_balanced`, which is Phase-5's eval split — Phase 6 must
      use a *fresh* split that the agent never saw at training time).
- [ ] Read `src/benchmark/latency.py` — F7 computation-overhead
      measurement.
- [ ] Read `scripts/benchmark/run_test_eval.py` — evaluation
      entrypoint; verify it loads model checkpoints from
      `runs/phase5/<algo>/seed_<k>/model.zip` and that the eval env
      is constructed with `split="test_balanced"`,
      `exclude_ood=True`. *(F0a/F0b/F11 / Step-1 invariant.)*
- [ ] Read `scripts/benchmark/build_summary_table.py` — F5 table.
- [ ] Read `scripts/benchmark/plot_baselines.py` — likely the
      reward-comparison plot.
- [ ] Read `scripts/benchmark/plot_overhead.py` — F7.
- [ ] Read `scripts/benchmark/plot_stage_action_cm.py` — F6.
- [ ] Read `tests/test_baseline_policies.py`,
      `test_benchmark_eval_runner.py`, `test_benchmark_latency.py`.
- [ ] **Realism audit (F5, F6, F7, F8 specifically).**
  - F5 final benchmark table: agents × baselines × metrics. Verify
    metric set matches PLAN. Verify the 5-seed aggregation is
    bootstrapped correctly.
  - F6 stage × action confusion matrices: 5×5 (or 5×N) per algo.
    Confirm stage ordering `[BENIGN, RECON, ACCESS, MANEUVER,
    IMPACT]` × action ordering `[OBSERVE, LOG, THROTTLE, BLOCK,
    ISOLATE]`.
  - F7 computation overhead: latency per decision (median, p95,
    p99) on the IoT-target hardware profile.
  - F8 — verify what F8 is in PLAN (may be the oracle-vs-RL
    headline comparison or a security-metrics decomposition).
- [ ] **Test-split contract.** Phase 6 *must* use `test_balanced`,
      not `val_balanced` or `train`. Cite file:line.
- [ ] **Detector-integration question (Phase 6).** Step 5 confirmed
      Phase 5 does NOT use the detector. Phase 6 may use it as an
      evaluation-time *baseline* (e.g., "detector-only" recommended-
      action policy) or as part of a co-design. Verify which lane
      Phase 6 occupies and that the chain to `stage_detector.pt:
      71e06616…` is explicit if used.
- [ ] **Hash chain.** Verify Phase-6 `manifest.json` pins:
      Phase-1 splits (`1e99d596…`), Phase-2 LSTM
      (`artifacts/generator/phase2/attack_sequence_generator.pth`
      hash from Phase-2 manifest), all 15 Phase-5 model checkpoints
      (`runs/phase5/<algo>/seed_<k>/model.zip`), and (if used)
      Phase-4 detector (`71e06616…`).
- [ ] Re-run `pytest -q` — expect 411 passed (Step 6 is read-only
      audit; no model re-training).

### Step 6 outputs (deliverables)
- [ ] Write `docs/mentor_review/06_benchmark.md` — full mentor memo,
      lead with verdict (PASS / PASS-WITH-FIXES / FAIL). Cite gate
      IDs (G6.1–G6.k) and file:line. Findings priority-ordered by
      severity.
- [ ] Write `docs/mentor_review/06_HANDOFF.md` from
      `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for
      **Step 7 (Phase 7 Ablations: F9, F10, F12, F15, G7 + the
      Step-2 F1 / Step-2 F2 re-run if option a)**.
- [ ] Commit per Conventional Commits
      (`docs(mentor-review,step-6): …`); push to
      `mentor-review/step-6-benchmarks`.
- [ ] **Pause for candidate sign-off** — do NOT merge to `main`
      without explicit "go" / "Step 7".

### Acceptance criterion for Step 6 PASS
- F5 + F6 + F7 + F8 figures correct (right test split, right axis
  ordering, right metric definitions).
- Phase-6 evaluation consumes `split="test_balanced"`,
  `exclude_ood=True` (Step-1 invariant honoured for the test split;
  cite file:line in `eval_runner.py` / `run_test_eval.py`).
- Hash chain intact for `docs/results/06_benchmark/`. Inputs SHAs
  chain to **post-`3cd2fb9`** Phase-1 splits + Phase-5 model
  checkpoints + (if used) Phase-4 `stage_detector.pt: 71e06616…`.
- Test suite green (411 passed); benchmark-scoped tests cover the
  public API.
- Any fixes filed against documentation (`docs(phase-6,§…)`) unless
  a genuine correctness bug surfaces (then `fix(phase-6,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse --abbrev-ref HEAD     # expect: mentor-review/step-5-blue-team (this branch)
                                    #   OR main (if Step 5 already merged by candidate)
git --no-pager log --oneline -5     # expect: 05_HANDOFF + 05_blue_team commit on top of 81804cc
git status                          # expect: clean
git tag -l                          # expect: EMPTY (no tags during the loop, by policy)
git branch -a                       # expect: main, origin/main, current step branch only

pytest -q                           # expect: 411 passed in ~65 s

ls docs/mentor_review/              # expect:
                                    #   README.md, HANDOFF_TEMPLATE.md,
                                    #   00_framing.md, 00_HANDOFF.md,
                                    #   01_dataset.md, 01_HANDOFF.md,
                                    #   02_red_team.md, 02_HANDOFF.md,
                                    #   03_env.md, 03_HANDOFF.md,
                                    #   04_detector.md, 04_HANDOFF.md,
                                    #   05_blue_team.md, 05_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If any expectation fails, **stop** and surface the divergence.
Specifically:
- If `pytest -q` is not 411 passed → Step 5 was strictly read-only
  audit + memo, so any test count change is unexpected.
- If a tag exists → policy violation; cut it before continuing.
- If `mentor-review/step-4-detector` still exists locally or
  remotely → Phase G2 of Step 4 didn't fully complete; re-do the
  deletion.

If sign-off has been received but the branch hasn't been merged yet,
execute Phase G2:

```bash
cat > /tmp/step5_merge_msg.txt <<'MSG'
Merge mentor-review/step-5-blue-team into main

Step 5 (Phase 5 Blue Team RL training audit — F3 learning curves,
F4 action distribution, T1 hyperparameters, gates G5.1-G5.7, hash
chain, reward-function fidelity, detector-integration question)
closed at PASS-WITH-FIXES.

Memo: docs/mentor_review/05_blue_team.md
Handoff: docs/mentor_review/05_HANDOFF.md

Six minor / nit findings filed; F1 (G5.4 mechanical FAIL ↔ narrative
PASS-with-finding not cross-linked in scoreboard JSON), F2 (hash
chain to post-3cd2fb9 Phase-1 splits is implicit, not explicit), F3
(_select_best_algo tie-break disagrees with PLAN D5.11 and own
docstring), F4 (make_eval_env docstring claims an undelivered
default), F5 (MLflow not used despite experiments-mlflow.md setup),
F6 (Phase-3 RESULTS.md "six-term reward" mismatches Phase-5 wiring
count of nine fields) — all batched into Step 8.

Six of seven exit gates PASS mechanically (G5.2 +1350.7 reward,
G5.3 19.24 MTTC, G5.5 max per-stage share 0.45, G5.7 manifests
present; G5.1 + G5.6 implicitly via 411-pytest); G5.4 is
mechanically FAIL (mitigated-impact 0.263 < 0.50) and editorially
PASS-WITH-FINDING per RESULTS.md §4 Finding 2 (de-escalation
farming = headline thesis result on reward hacking). Step-1
invariant honoured by code (train_agent.py:182,196,187,201) and by
serialisation (run_manifest.json). Phase-4 detector NOT consumed
by Phase 5 (D5.2 design intent) — Step-4 open question 4 resolved.
Hash chain internal to Phase 5 byte-perfect; chain to upstream
Phase-1 is implicit (Finding F2). Full suite green at 411 passed.
MSG
git checkout main && git pull --ff-only origin main
git merge --no-ff mentor-review/step-5-blue-team -F /tmp/step5_merge_msg.txt
git push origin main
git branch -d mentor-review/step-5-blue-team
git push origin --delete mentor-review/step-5-blue-team
git checkout -b mentor-review/step-6-benchmarks
git tag -l            # confirm still empty
git branch -a         # expect: main, origin/main, mentor-review/step-6-benchmarks
```

> Use `write_to_file` to create `/tmp/step5_merge_msg.txt`, NOT a
> shell heredoc — heredocs in `execute_command` mangle in this
> terminal (per Step-3 / Step-4 handoff §6 git-policy lesson).

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
   need candidate decision**; F8 (transition_mask) resolved benign.
6. `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 env
   audit; F1–F3 doc-fixes batched to Step 8; F4 deferred to Step 9
   LaTeX; F5 (Step-2 F8 carry-forward) resolved benign.
7. `docs/mentor_review/04_detector.md` + `04_HANDOFF.md` — Step-4
   stage-detector audit; F1–F4 doc-fixes batched to Step 8; F5
   deferred to Step 9 LaTeX. **Open question 4 (Phase-5 detector
   integration) — resolved this step.**
8. `docs/mentor_review/05_blue_team.md` — Step-5 mentor memo (full
   prose; cite by Finding number F1–F6).
9. `docs/mentor_review/05_HANDOFF.md` (this file) — the resume point.
10. `docs/results/06_benchmark/PLAN.md` — Phase-6 plan (frozen; **do
    not edit**).
11. `docs/results/06_benchmark/RESULTS.md` — Phase-6 scientific
    record (locked).
12. `docs/results/06_benchmark/G6_scoreboard.json` (or per-gate
    variants) — numerical gate verdicts. If absent, treat as
    Phase-3/Phase-4-style asymmetry roll-up.
13. `docs/results/06_benchmark/manifest.json` — Phase-6 hash chain.
    **Verify input SHAs chain back to post-`3cd2fb9` Phase-1 splits
    AND to all 15 Phase-5 model checkpoints AND (if used) to
    Phase-4 `stage_detector.pt: 71e06616…`.**
14. `src/benchmark/baseline_policies.py` — Random + Recommended
    (oracle) baselines.
15. `src/benchmark/eval_runner.py` — evaluation loop; verify
    `split="test_balanced"` (Step-1 invariant for the test side).
16. `src/benchmark/latency.py` — F7 computation overhead.
17. `scripts/benchmark/run_test_eval.py` — Phase-6 entrypoint;
    verify it loads `runs/phase5/<algo>/seed_<k>/model.zip` and
    consumes `test_balanced`.
18. `scripts/benchmark/{build_summary_table,plot_baselines,plot_overhead,plot_stage_action_cm}.py` — figures.
19. `tests/{test_baseline_policies,test_benchmark_eval_runner,test_benchmark_latency}.py` — Phase-6 test coverage.

Skim these for reference (do not read in full):

- `docs/benchmarking-results.md`
- `docs/architecture.md`
- `docs/thesis_results_map.md`
- root `README.md`

Then visually inspect Phase-6 figures:

```bash
ls docs/results/06_benchmark/
# open the PNGs the directory contains — F5, F6, F7, F8.
```

---

## 8. Open questions for the user

Re-flagged from earlier steps + raised this step:

1. **[carry from Step 2 / Step 3 / Step 4]** **Step-2 F1 — Phase-2
   manifest input-hash divergence.** Still pending. Confirm option
   (a) Step-7 re-run with `seed=42` against the post-`3cd2fb9`
   manifest (recommended), or option (b) document-only in a
   backfilled Phase-2 RESULTS.md? *Step-5 takeaway: Phase 5's
   per-figure manifests pin output JSONLs but not upstream Phase-1
   splits — Finding F2 — which is structurally the same audit-trail
   gap Step-2 F1 surfaced. The Step-7 re-run scope grows by one
   small artefact (a new top-level Phase-5 `manifest.json`) if
   option (a) is chosen.*

2. **[carry from Step 2]** **Step-2 F2 — model-selection metric.**
   Was balanced-val cross-entropy or macro-F1 the intended Phase-2
   model-selection criterion? Phase 4 explicitly uses val-macro-F1
   (`stage_detector.py:202-211`); Phase 5 doesn't have a model-
   selection axis (PPO/DQN/A2C run for a fixed timestep budget,
   each saves its final model). If Phase 2 was meant to use
   macro-F1 → `fix(phase-2,trainer)` + Step-7 re-run; if balanced-
   val CE → doc-fix in Phase-2 RESULTS.md.

3. **[carry from Step 4 + Step 5]** **Step-3 F1–F3 + Step-4
   F1/F2/F3/F4 + Step-5 F1/F2/F3/F4/F5/F6 batching into Step 8.**
   All minor / nit doc-fixes. Confirm they are batched into Step 8
   cross-cutting cleanup (recommended) rather than landed
   piecemeal? My recommendation: **batch**. Confirm.

4. **Resolved this step** — Step-4 open question 4 (Phase-5
   detector-checkpoint integration). Answer: **NOT integrated**, by
   D5.2 design intent (PLAN §A3 + §8 D5.2 verbatim). The Phase-4
   detector is reserved for Phase-6 evaluation-time baselines and
   Phase-7 ablations (specifically the "detector-augmented
   observation" axis if it is included in the Phase-7 sweep).

5. **[New, raised in Step 5]** **G5.4 mechanical-vs-narrative
   verdict format.** Two phases now (Phase 4 G4.4, Phase 5 G5.4)
   have produced PASS-WITH-FINDING gates where the mechanical JSON
   reads `passes: false` and the narrative reads "PASS-with-finding".
   Should the scoreboard JSON schema gain a `verdict` enum
   (`pass|fail|pass_with_finding`) plus a `finding_ref` cross-link
   so the JSON is self-explaining? Recommendation: **yes**,
   land in Step 8 with a one-line JSON regen for both phases (no
   model retraining). Confirm.

6. **[New, raised in Step 5]** **Phase-6 detector-baseline lane.**
   Does Phase 6 evaluate a "detector-only" recommended-action policy
   (i.e., "use `stage_detector.pt`'s argmax as the recommended
   action and play that") as one of its baselines? If yes, the
   chain to Phase-4 `stage_detector.pt: 71e06616…` is required at
   Phase 6. The Step-6 audit will confirm.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, no model, no hash-pinned
  figure, no test was touched. Pytest count unchanged at 411.
- **Risk noticed (carry-forward to Step 7):** Step-2 Finding 1's
  manifest input-hash divergence (Phase-2 LSTM was demonstrably
  trained on the pre-`3cd2fb9` leaky splits prior). Recommended fix:
  Step-7 re-run.
- **Risk noticed (Step-2 F2):** model-selection metric ambiguity in
  Phase 2 (still open). Phase 4 is consistent with macro-F1 selection.
  Phase 5 has no model-selection ambiguity (fixed total_timesteps,
  saves final model).
- **Risk noticed (Step-5 F2):** Phase-5 hash-chain to upstream
  Phase-1 is implicit. Per the Step-1 / Step-2 / Phase-4 lessons,
  this is the same audit-trail invariant Step-2 F1 surfaced —
  just at a different boundary. Step-8 doc-fix (a top-level
  `docs/results/05_blue_team/manifest.json` pinning input SHAs) is
  the cleanest closure.
- **Risk noticed (carry-forward to Step 8):** five phases now exhibit
  per-phase scoreboard or hash-chain asymmetry findings (Step-1 F4,
  Step-2 F4, Step-3 F1, Step-4 F2, Step-5 F2). Step 8 must
  consolidate. The good news: Phase 5 *does* ship a
  `G5_scoreboard.json` (cf. Phase 3 / Phase 4 absences), so the
  scoreboard half of the asymmetry is closed; the hash-chain half
  needs the unified-input-SHA pattern that Phase 4 already
  pioneered.
- **Risk noticed (Step-7 territory):** RESULTS.md §4 Finding 2
  identifies the headline thesis result — reward hacking via
  de-escalation farming. The Phase-7 reward-component ablation is
  the natural follow-up; PLAN §3.2 already lists the candidate
  axes. If Step-2 F1 chooses option (a), Step-7 also includes the
  Phase-2 LSTM re-run — the work scope is well-defined.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or
  out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next
  agent and `06_benchmark.md` is opened.

Per the operating rule *"One step per session. Do not start Step 6
until the candidate signs off Step 5."* — Step 6 may not begin
without an explicit "go" / "Step 6" / merge of this branch.
