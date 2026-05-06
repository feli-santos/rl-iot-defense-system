# Step `04` — Phase 4 Stage Detector Review — Mentor Review Handoff

**Closed:** `2026-05-06 ~14:00 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 5)
**Reviewed phase / scope:** Phase 4 (Stage Detector + supervised
baselines — StageDetector MLP, RandomForest, 1-D CNN; thesis figure
F11; exit gates G4.1–G4.5; OOD-class behaviour; train/val/test
isolation; hash-chain integrity)
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts (frozen audit trail; never edited)
- `docs/results/04_detector/PLAN.md` (340 lines) — design contract;
  D1, D2, D2.1, D3 locked; §A4 imbalance audit; §3.3 gates G4.1–G4.5
  with thresholds.
- `docs/results/04_detector/RESULTS.md` (177 lines) — locked scientific
  record; §2 final scoreboard; §3 headline numbers; §4 three findings;
  §5 the unplanned `3cd2fb9` Phase-1 leakage discovery.
- `docs/results/04_detector/F11_summary.json` (SHA `955f99ff…`) —
  per-model, per-split, per-stage, per-OOD numerical truth + gates dict.
- `docs/results/04_detector/manifest.json` — git SHA
  **`3cd2fb90ac7a`**; six input hashes (chain to post-`3cd2fb9`
  Phase-1 splits manifest, byte-perfect); six output hashes.
- `docs/results/04_detector/F11_per_stage_recall.png`
  (SHA `b6bd1871…`) — 1775×694 RGBA at 150 dpi.
- `docs/results/04_detector/F11_caption.md` (SHA `4250dfb8…`).

### Code
- `src/detector/__init__.py` (45 lines) — public API surface.
- `src/detector/stage_detector.py` (351 lines) — production MLP head;
  StageDetectorConfig + _MLP + StageDetector class (fit/predict/
  predict_proba/save/from_checkpoint).
- `src/detector/cnn1d.py` (331 lines) — Tharewal-style 1-D CNN
  baseline.
- `src/detector/random_forest.py` (111 lines) — RF baseline wrapper.
- `src/detector/evaluation.py` (221 lines) — sklearn-compatible
  metrics; STAGE_NAMES = `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]`
  (line 23).
- `scripts/detector/train_detector.py` (616 lines) — Phase-4
  entrypoint; `_verify_disjoint` runtime defence at lines 111-123.

### Tests
- `tests/test_detector.py` (331 lines, **23 tests**, all PASS in 7.75 s
  on the Step-4 branch). 4 `Test*` classes covering evaluation
  (sklearn parity), StageDetector (separable-cluster, save/load,
  latency budget), RandomForest, CNN1D.
- Full suite: **`pytest -q` → 411 passed in 79.81 s** on
  `mentor-review/step-4-detector` (cut off `main` @ `193ded3` =
  Step-3 merge).

### Docs
- `docs/mentor_review/README.md` — directory conventions.
- `docs/mentor_review/00_framing.md` — locked thesis claims P1/P2/P3
  and R1/R2; IoTWarden as inspiration only.
- `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff.
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 audit;
  post-`3cd2fb9` splits manifest is canonical.
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
  Findings 1 (manifest input-hash divergence; honoured cleanly here)
  and 2 (model-selection metric; still open) and 8
  (transition_mask carry-forward; resolved benign).
- `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 F1
  (no Phase-3 manifest/scoreboard) recurs structurally here as
  Step-4 F2.
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — template for this file.
- `docs/mentor_review/04_detector.md` — Step-4 mentor memo (written
  this session).

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-4 detector stack is faithfully implemented and mechanically
verified. **All five exit gates G4.1–G4.5 PASS** (G4.4 as
*PASS-with-finding* by design, per PLAN §8.D2 revised in step 4.5):
G4.2 = 0.7855 macro-F1 (≥ 0.75); G4.3 = 0.539 worst per-stage recall
at RECON (≥ 0.50); G4.4 = OOD asymmetry recorded as thesis result;
G4.5 = 0.039 ms median per-sample latency (≤ 1 ms). Hash chain
byte-perfect end-to-end against the **post-`3cd2fb9`** Phase-1 splits
manifest — the Step-2 F1 lesson is honoured cleanly. The Step-1
invariant (train consumes only OOD-disjoint rows) is honoured both by
construction and by runtime `_verify_disjoint` assertion. Five minor
doc/cosmetic findings, all batchable into Step 8.

Full memo: `docs/mentor_review/04_detector.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: minor]** **F1** — `docs/results/04_detector/PLAN.md`
   §A4 lines 99-107 cite *pre-`3cd2fb9`* train counts (309 566 total;
   RECON 26 967, ACCESS 23 198, MANEUVER 33 947, IMPACT 127 209). The
   actual post-fix train index file holds 281 420 rows
   (RECON 27 038, ACCESS 23 173, MANEUVER 33 939, IMPACT 127 270).
   The 28 146 delta is exactly 70% of the four OOD classes' total
   (40 209) — i.e., the OOD rows the bug had folded into train. PLAN
   is frozen audit trail; do not edit. **Recommended fix:** one-paragraph
   "as-built post-`3cd2fb9` train counts" subsection in
   `docs/results/04_detector/RESULTS.md` OR a cross-cutting note in
   `docs/results/README.md` rolled together with Step-1 F4 / Step-2 F4
   / Step-3 F1 / Step-4 F2. Commit:
   `docs(phase-4,§A4-as-built): note post-3cd2fb9 train counts vs PLAN
   §A4`. **Disposition:** batch into Step 8.

2. **[severity: minor]** **F2** — `docs/results/04_detector/` ships
   `manifest.json` (with `gates_status` mirror) and a `gates` dict
   inside `F11_summary.json`, but no top-level `G4_scoreboard.json`.
   This is the same asymmetry across Phases 1/2/3/4 (Step-1 F4,
   Step-2 F4, Step-3 F1). Numerical truth is intact; the asymmetry is
   purely cosmetic. **Recommended fix:** roll into the unified
   cross-cutting paragraph in `docs/results/README.md`. Commit:
   `docs(audit-trail,readme): document per-phase scoreboard
   asymmetry`. **Disposition:** batch into Step 8.

3. **[severity: minor]** **F3** — Dead-code branch
   `rf.run_info.__dict__ if False else {…}` at
   `scripts/detector/train_detector.py` lines 545-552. Vestigial
   refactor artefact; unreachable branch. **Recommended fix:**
   simplify to the dict literal. Caveat: any change to
   `train_detector.py` triggers a re-run if we want hash-chain
   recompute, which is Step-7 territory. **Disposition:** defer to
   Step 7 (Phase-7 re-run) or Step 8 cleanup, whichever happens
   first.

4. **[severity: nit]** **F4** —
   `docs/results/04_detector/RESULTS.md` lines 30-34 show a "headline
   numbers" table whose cells (0.819, 0.539, 0.801, 0.770, 0.998 for
   StageDetector) are **per-stage recall**, not per-class F1. Column
   header lacks a metric label and the leftmost "Macro-F1" column
   invites the reader to assume per-class F1. **Recommended fix:**
   add an explicit "Per-stage recall on test_balanced" sub-header.
   Commit: `docs(phase-4,§3.1): clarify per-stage cells are recall,
   not F1`. **Disposition:** batch into Step 8.

5. **[severity: nit, defense narrative]** **F5** —
   `docs/mentor_review/00_framing.md` §3 R1 frames OOD behaviour at
   the *RL* policy level; RESULTS.md §4 Finding 3 reports the
   *detector*'s OOD asymmetry. The two senses of "OOD performance"
   are not cross-linked; the defense committee may conflate them.
   **Recommended fix:** one sentence in §4.4 (Stage Detection) and
   §9.3 (Robustness) of the LaTeX rebuild explicitly linking R1 to
   G4.4. **Disposition:** Step 9 LaTeX rebuild.

Full prose, file:line citations, and recommended commit messages:
`docs/mentor_review/04_detector.md` §3.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/04_detector.md` — Step-4 mentor memo (verdict
  PASS-WITH-FIXES + 5 findings + hash-chain reproduction + gate
  reproduction + hyperparameter audit + isolation audit + F11 realism
  audit + test-coverage audit + carry-forward table + open
  candidate-decisions section).
- `docs/mentor_review/04_HANDOFF.md` — this file.

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: **411 passed**.
Detector-scoped: **23 passed in 7.75 s**.

### Scripts / models
None modified. No re-training, no figure regeneration. **Hash chain
intact** (verified byte-perfect against `manifest.json` and against
the post-`3cd2fb9` Phase-1 splits manifest).

### Git hygiene applied (Phase G1, opening this step)
1. `git checkout main && git pull --ff-only origin main`.
2. `git merge --no-ff mentor-review/step-3-env -F /tmp/step3_merge_msg.txt`
   → merge commit **`193ded3`** with message ref'ing Step-3 verdict +
   F1/F2/F3/F4/F5 dispositions.
3. `git push origin main` (pushed `193ded3`).
4. Deleted local + remote `mentor-review/step-3-env`.
5. Cut `mentor-review/step-4-detector` off `main` @ `193ded3`.
6. Verified policy invariants: `git tag -l` empty, `git branch -a` =
   `main`, `origin/main`, `mentor-review/step-4-detector` only.
7. Ran `pytest -q` → **411 passed in 79.81 s** before any audit work.

End state matches policy: one long-lived branch (`main`), zero tags,
current working branch is the per-step topic branch.

### Phase G2 (closing this step) — runs after sign-off
Symmetric to G1. Listed in §6.

---

## 5. Outstanding actions for the next session

The next session executes **Step 5 — Phase 5 Blue Team RL training
review** (F3, F4, T1, G5). Phase 5 is the heart of the thesis: PPO
agent trained on the Phase-3 environment using the Phase-4 detector
checkpoint.

### Pre-flight (Phase G1 of Step 5)
- [ ] Verify the candidate has signed off Step 4 either by (a) a
      comment, (b) a merge of `mentor-review/step-4-detector` into
      `main`, or (c) explicit "go" / "Step 5" in chat. If none,
      **stop** and raise.
- [ ] If sign-off given **before** branch merge: execute Phase G2
      ourselves —
  ```
  git checkout main && git pull --ff-only origin main
  git merge --no-ff mentor-review/step-4-detector -F /tmp/step4_merge_msg.txt
  git push origin main
  git branch -d mentor-review/step-4-detector
  git push origin --delete mentor-review/step-4-detector
  git tag -l   # confirm still empty
  ```
- [ ] Cut `mentor-review/step-5-blue-team` off the new `main`.
- [ ] If any Step-4 *fix* commits were applied (F1/F2/F4 doc-fixes),
      pull them onto `main` first so Step 5 starts from corrected
      state.
- [ ] Run `pytest -q` to confirm 411 passed before audit work. If
      count differs, **stop** and surface.
- [ ] Verify `git tag -l` is empty (no tags during the loop, by
      policy).

### Step 5 review checklist (Phase 5 Blue Team RL training)
- [ ] Read `docs/results/05_blue_team/PLAN.md` in full — frozen audit
      trail. Note the gates (likely G5.1–G5.k) and the figure-ID
      definitions for F3, F4 (and possibly T1).
- [ ] Read `docs/results/05_blue_team/RESULTS.md` — locked scientific
      record. Reward curves, KL trajectories, action distributions,
      final-policy gates.
- [ ] Read `docs/results/05_blue_team/manifest.json` — verify hash
      chain via `shasum -a 256`. **Critical:** confirm input SHAs
      chain to the **post-`3cd2fb9`** Phase-1 splits manifest AND to
      the Phase-4 `stage_detector.pt` (SHA `71e06616…`). If Phase 5's
      manifest pins the *pre*-fix splits or a stale detector, that's a
      Phase-2-F1-style divergence.
- [ ] Read `docs/results/05_blue_team/G5_scoreboard.json` if present;
      if absent, file as a finding consistent with Step-4 F2's
      scoreboard-asymmetry roll-up.
- [ ] Read `src/blue_team/run_config.py` — `BlueTeamConfig` /
      `EnvConfigSerializable` defaults; PPO hyperparameters
      (learning rate, n_steps, batch size, n_epochs, gamma, gae_lambda,
      clip_range, ent_coef, vf_coef, target_kl).
- [ ] Read `src/blue_team/env_factory.py` — `make_train_env` /
      `make_eval_env`; confirm Step-3 F2 doc-only disposition still
      applies (env-ctor non-split-aware default, monkey-patched by
      factory).
- [ ] Read `src/blue_team/aggregation.py` — multi-environment
      reward/metric aggregation.
- [ ] Read `src/blue_team/callbacks.py` — early-stop, MLflow, KL
      divergence, action-distribution logging.
- [ ] Read `src/algorithms/adversarial_algorithm.py` — SB3 PPO wrapper.
- [ ] Read `scripts/blue_team/train_agent.py` — Phase-5 entrypoint;
      confirm it consumes `split="train"`, `exclude_ood=True` via
      `RealizationEngine.from_split_manifest` (Step-1 invariant,
      Step-3 F2 contract).
- [ ] Read `scripts/blue_team/run_phase5.py` — outer Phase-5
      orchestration (sweeps?).
- [ ] Read `scripts/blue_team/evaluate_gates.py` — gate-check script.
- [ ] Read `scripts/blue_team/plot_action_dist.py` — F3/F4 figure
      production.
- [ ] Read `scripts/blue_team/plot_learning_curves.py`.
- [ ] Read `scripts/blue_team/dump_hparams.py`.
- [ ] Read `tests/test_blue_team_*.py` (5 test files: aggregation,
      callbacks, env_factory, run_config, train_agent_reward_overrides)
      and `tests/test_train_agent_reward_overrides.py`.
- [ ] **Realism audit (F3, F4 specifically).** F3 is "training reward
      curves" (likely train + eval, mean over n seeds, with shaded
      band). F4 is "policy action distribution by stage" (5×5 or 5×N
      heatmap). Verify:
  - learning curves use the right metric (likely cumulative reward,
    moving average); seed count and band semantics (95% CI?
    min-max?) are stated;
  - F4 ordering is `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]` × the
    action-set ordering from `src/utils/label_mapper.py` (likely
    OBSERVE, LOG, THROTTLE, ISOLATE, RESET); verify against
    `00_framing.md` §2 IoTWarden recommended-action mapping.
- [ ] **Train-env vs eval-env contract.** Re-verify the candidate
      uses `make_train_env` for training (with `split="train"`,
      `exclude_ood=True`) and `make_eval_env` for evaluation (with
      `split="val_balanced"` or `"test_balanced"`, also
      `exclude_ood=True`). Cite file:line.
- [ ] **Detector usage.** Confirm whether the agent's observation
      includes the Phase-4 `stage_detector.pt` outputs (Phase-4
      RESULTS.md §6 envisions this as an option for ablation in Phase
      5). If yes: cite the integration point and verify the
      checkpoint SHA matches Phase-4's `71e06616…`. If no: confirm
      that's an intentional design choice and that Phase-4 detector
      is reused only at evaluation (Phase 6/7).
- [ ] **Reward function.** Re-verify the six-term reward
      (proportionality + mitigation + disproportionate-penalty +
      defender-de-escalation-bonus + IMPACT-terminal-bonus +
      step-penalty per Step-3 F3 / RESULTS.md §3) is consumed
      verbatim by Phase 5. If Phase 5 overrides any term default,
      cite the override + the rationale.
- [ ] **Reproducibility.** Confirm Phase-5 training is deterministic
      given `--seed`. Cite the seed propagation chain (numpy, torch,
      SB3, env, RNG for action sampling).
- [ ] **Hyperparameter audit.** PPO defaults: cite the lr, n_steps,
      batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef,
      vf_coef, target_kl, total_timesteps. Cross-check against PLAN.md.
- [ ] **MLflow.** Phase 5 is the first phase with MLflow runs (per
      `docs/experiments-mlflow.md`). Verify the run IDs cited in
      `manifest.json` correspond to actually-existing local MLflow
      directories OR document that MLflow is local-only and
      manifest.json is the canonical record.
- [ ] **T1 (table).** PLAN.md likely defines T1 as the per-seed
      learning summary table (final reward, KL, % of episodes with
      IMPACT-terminal, etc.). Verify the numbers in T1 match
      `F3_summary.json` / `F4_summary.json` byte-for-byte.
- [ ] Re-run `pytest -q` — expect 411 passed (Step 5 is read-only
      audit; no model re-training).

### Step 5 outputs (deliverables)
- [ ] Write `docs/mentor_review/05_blue_team.md` — full mentor memo,
      lead with verdict (PASS / PASS-WITH-FIXES / FAIL). Cite gate
      IDs (G5.1–G5.k) and file:line. Findings priority-ordered by
      severity.
- [ ] Write `docs/mentor_review/05_HANDOFF.md` from
      `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for
      **Step 6 (Phase 6 Benchmarks: F5, F6, F7, F8, G6)**.
- [ ] Commit per Conventional Commits
      (`docs(mentor-review,step-5): …`); push to
      `mentor-review/step-5-blue-team`.
- [ ] **Pause for candidate sign-off** — do NOT merge to `main`
      without explicit "go" / "Step 6".

### Acceptance criterion for Step 5 PASS
- F3 (learning curves) and F4 (action distribution) are correct: right
  splits, right action/stage axis ordering, statistical bands clearly
  labelled.
- Phase-5 training consumes `split="train"`, `exclude_ood=True`
  (Step-1 invariant honoured; cite file:line in
  `make_train_env` / `train_agent.py`).
- Hash chain intact for `docs/results/05_blue_team/`. Input SHAs chain
  to **post-`3cd2fb9`** Phase-1 splits AND to Phase-4
  `stage_detector.pt: 71e06616…` if the agent observation pipeline
  consumes it.
- Reward function in code matches RESULTS.md §3 of Phase 3
  (six terms; MTTC is a *metric*, not a reward term — Step-3 F3
  contract honoured).
- Test suite green (411 passed); blue-team-scoped tests cover the
  public API.
- Any fixes filed against documentation (`docs(phase-5,§…)`) unless
  a genuine correctness bug surfaces (then `fix(phase-5,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse --abbrev-ref HEAD     # expect: mentor-review/step-4-detector (this branch)
                                    #   OR main (if Step 4 already merged by candidate)
git --no-pager log --oneline -5     # expect: 04_HANDOFF + 04_detector commit on top of 193ded3
git status                          # expect: clean
git tag -l                          # expect: EMPTY (no tags during the loop, by policy)
git branch -a                       # expect: main, origin/main, current step branch only

pytest -q                           # expect: 411 passed in ~80 s

ls docs/mentor_review/              # expect:
                                    #   README.md, HANDOFF_TEMPLATE.md,
                                    #   00_framing.md, 00_HANDOFF.md,
                                    #   01_dataset.md, 01_HANDOFF.md,
                                    #   02_red_team.md, 02_HANDOFF.md,
                                    #   03_env.md, 03_HANDOFF.md,
                                    #   04_detector.md, 04_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If any expectation fails, **stop** and surface the divergence.
Specifically:
- If `pytest -q` is not 411 passed → Step 4 was strictly read-only
  audit + memo, so any test count change is unexpected.
- If a tag exists → policy violation; cut it before continuing.
- If `mentor-review/step-3-env` still exists locally or remotely →
  Phase G2 of Step 3 didn't fully complete; re-do the deletion.

If sign-off has been received but the branch hasn't been merged yet,
execute Phase G2:

```bash
cat > /tmp/step4_merge_msg.txt <<'MSG'
Merge mentor-review/step-4-detector into main

Step 4 (Phase 4 Stage Detector audit — F11, gates G4.1-G4.5,
hash chain, OOD-class behaviour) closed at PASS-WITH-FIXES.

Memo: docs/mentor_review/04_detector.md
Handoff: docs/mentor_review/04_HANDOFF.md

Five minor findings filed; F1 (PLAN §A4 train counts are pre-3cd2fb9),
F2 (no G4_scoreboard.json — recurrence of Step-3 F1 asymmetry), F3
(dead-code branch in train_detector.py), F4 (RESULTS.md §3.1
recall-vs-F1 column-header ambiguity) batched into Step 8
cross-cutting cleanup. F5 (RL-OOD vs detector-OOD framing not
cross-linked) deferred to Step 9 LaTeX.

All 5 exit gates G4.1-G4.5 PASS (G4.4 PASS-with-finding by design).
Hash chain byte-perfect against post-3cd2fb9 Phase-1 splits manifest
(Step-2 F1 lesson honoured cleanly). Step-1 invariant honoured by
construction + runtime _verify_disjoint assertion. Full suite green
at 411 passed.
MSG
git checkout main && git pull --ff-only origin main
git merge --no-ff mentor-review/step-4-detector -F /tmp/step4_merge_msg.txt
git push origin main
git branch -d mentor-review/step-4-detector
git push origin --delete mentor-review/step-4-detector
git checkout -b mentor-review/step-5-blue-team
git tag -l            # confirm still empty
git branch -a         # expect: main, origin/main, mentor-review/step-5-blue-team
```

> Use `write_to_file` to create `/tmp/step4_merge_msg.txt`, NOT a
> shell heredoc — heredocs in `execute_command` mangle in this
> terminal (per Step-3 handoff §6 git-policy lesson).

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` — locked thesis claims P1/P2/P3
   and R1/R2; IoTWarden's role (inspiration only); chapter outline
3. `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still
   in force)
4. `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1
   dataset audit; cite Findings 1–6 doc-fixes shipped + Finding 7
   deferred to Step 8; the post-`3cd2fb9` splits manifest is the
   canonical Phase-1 output and every downstream phase must chain
   back to it
5. `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
   red-team audit; Finding 1 (Phase-2 manifest input-hash divergence)
   and Finding 2 (model-selection metric) are still **open and need
   candidate decision**; Finding 8 (transition_mask) resolved benign
6. `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 env
   audit; F1–F3 doc-fixes batched to Step 8; F4 deferred to Step 9
   LaTeX; F5 (Step-2 F8 carry-forward) resolved benign
7. `docs/mentor_review/04_detector.md` — Step-4 mentor memo (full
   prose; cite by Finding number F1–F5)
8. `docs/mentor_review/04_HANDOFF.md` (this file) — the resume point
9. `docs/results/05_blue_team/PLAN.md` — Phase-5 plan (frozen; **do
   not edit**)
10. `docs/results/05_blue_team/RESULTS.md` — Phase-5 scientific record
    (locked)
11. `docs/results/05_blue_team/G5_scoreboard.json` (or per-gate
    variants) — numerical gate verdicts. If absent, treat as Phase-3
    /Phase-4-style asymmetry (Step-3 F1 / Step-4 F2 roll-up).
12. `docs/results/05_blue_team/manifest.json` — Phase-5 hash chain.
    **Verify input SHAs chain back to post-`3cd2fb9` Phase-1 splits
    manifest AND to Phase-4 `stage_detector.pt` SHA `71e06616…`** if
    the agent observation includes detector outputs.
13. `src/blue_team/run_config.py` — `BlueTeamConfig` /
    `EnvConfigSerializable`; PPO hyperparameters
14. `src/blue_team/env_factory.py` — `make_train_env` /
    `make_eval_env` (Step-3 F2 contract: factory monkey-patches a
    split-aware engine post-construction)
15. `src/blue_team/aggregation.py` — multi-env reward/metric
    aggregation
16. `src/blue_team/callbacks.py` — training callbacks
17. `src/algorithms/adversarial_algorithm.py` — SB3 PPO wrapper
18. `scripts/blue_team/train_agent.py` — Phase-5 CLI (verify
    `split="train"`, `exclude_ood=True` consumption — Step-1
    invariant)
19. `scripts/blue_team/run_phase5.py` — outer Phase-5 orchestration
20. `scripts/blue_team/evaluate_gates.py` — gate-check script
21. `scripts/blue_team/plot_action_dist.py` — F4 figure
22. `scripts/blue_team/plot_learning_curves.py` — F3 figure
23. `tests/test_blue_team_*.py` (5 files) +
    `tests/test_train_agent_reward_overrides.py` — Phase-5 test
    coverage

Skim these for reference (do not read in full):

- `docs/reward-shaping.md` (cross-check vs Phase-3 RESULTS.md §3 six
  terms)
- `docs/rl-training.md`
- `docs/experiments-mlflow.md`
- `docs/architecture.md`
- `docs/thesis_results_map.md` (per-figure thesis chapter mapping)
- root `README.md`

Then visually inspect Phase-5 figures:

```bash
ls docs/results/05_blue_team/
# open <any PNGs the directory contains> — F3, F4, possibly T1
```

---

## 8. Open questions for the user

Re-flagged from earlier steps + raised this step:

1. **[carry from Step 2 / Step 3]** **Step-2 F1 — Phase-2 manifest
   input-hash divergence.** Still pending. Confirm option (a) Step-7
   re-run with `seed=42` against the post-`3cd2fb9` manifest
   (recommended), or option (b) document-only in a backfilled Phase-2
   RESULTS.md? Step-4 takeaway: the fix protocol *works* — Phase 4's
   manifest chains to the post-fix splits byte-perfectly. Phase 2 is
   the outlier because the LSTM trainer was not re-run after
   `3cd2fb9`. The Step-7 re-run scope crystallises once this is
   answered.
2. **[carry from Step 2]** **Step-2 F2 — model-selection metric.**
   Was balanced-val cross-entropy or macro-F1 the intended Phase-2
   model-selection criterion? If balanced-val CE → doc-fix only; if
   macro-F1 → `fix(phase-2,trainer): …` + Step-7 re-run. *Phase 4's
   StageDetector explicitly selects on val-macro-F1
   (`stage_detector.py:202-211`), so Phase 4 is consistent with one
   half of the Step-2 F2 question — but the candidate still owes a
   confirmation that this is the design intent across both Phase 2
   and Phase 4.*
3. **[Step 4]** **Step-3 F1–F3 + Step-4 F1/F2/F3/F4 batching.** All
   minor doc-fixes. Confirm they are batched into Step 8
   cross-cutting cleanup (recommended) rather than landed
   piecemeal? My recommendation: **batch**. Confirm.
4. **[Step 4]** **Phase-5 detector-checkpoint integration.** Does the
   Phase-5 RL agent's observation pipeline consume the Phase-4
   `stage_detector.pt` outputs? If yes, the Step-5 audit must verify
   the checkpoint SHA matches `71e06616…`. If no, the Phase-4
   detector is reused only at Phase-6/7 evaluation time, which is
   fine but should be documented. Please confirm.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, no model, no hash-pinned
  figure, no test was touched. Pytest count unchanged at 411.
- **Risk noticed (carry-forward to Step 7):** Step-2 Finding 1's
  manifest input-hash divergence (Phase-2 LSTM was demonstrably
  trained on the pre-`3cd2fb9` leaky splits prior). Recommended fix:
  Step-7 re-run.
- **Risk noticed (Step-2 F2):** model-selection metric ambiguity in
  Phase 2 (still open). Phase 4 is consistent with macro-F1 selection
  — if Phase 2 was meant to do the same, that's a `fix(phase-2,…)` at
  Step 7. If Phase 2 was meant to use balanced-val CE, that's a doc
  divergence to fix in RESULTS.md.
- **Risk noticed (Step-4 F1):** PLAN.md §A4 contains pre-`3cd2fb9`
  numbers. The defense committee may notice the discrepancy if they
  cross-check with `F11_summary.json`. Mitigation:
  RESULTS.md/`docs/results/README.md` cross-cutting note in Step 8.
- **Risk noticed (carry-forward to Step 8):** four phases now exhibit
  per-phase scoreboard-asymmetry findings (Step-1 F4, Step-2 F4,
  Step-3 F1, Step-4 F2). Step 8 must consolidate.
- **Risk noticed (carry-forward to Step 9 LaTeX rebuild):** detector-
  OOD vs RL-OOD framing (Step-4 F5) needs explicit cross-link in §4.4
  + §9.3.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or
  out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next
  agent and `05_blue_team.md` is opened.

Per the operating rule *"One step per session. Do not start Step 5
until the candidate signs off Step 4."* — Step 5 may not begin
without an explicit "go" / "Step 5" / merge of this branch.
