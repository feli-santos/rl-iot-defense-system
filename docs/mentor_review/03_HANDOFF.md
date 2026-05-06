# Step `03` — Phase 3 Environment Review — Mentor Review Handoff

**Closed:** `2026-05-06 ~03:45 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 4)
**Reviewed phase / scope:** Phase 3 (Adversarial RL environment v2 — MDP, reward, exit gates G3.1–G3.7, `impact_is_terminal` lever, OOD-leakage boundary, transition_mask carry-forward)
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts
- `docs/results/03_env/PLAN.md` — frozen audit trail; bugs B1–B6, exit gates G3.1–G3.7, sequencing, risks R1–R3.
- `docs/results/03_env/RESULTS.md` — locked as-built record; bug-fix table, lifecycle pseudocode, six-term reward formula, default constants table, per-gate scoreboard, Iteration 1–3 lessons.
- `docs/results/03_env/manifest.json` — **does not exist** (Step-3 Finding F1).
- `docs/results/03_env/G3_scoreboard.json` — **does not exist** (Step-3 Finding F1).

### Code
- `src/environment/adversarial_env.py` (716 lines) — the env. Re-read in full.
- `src/blue_team/env_factory.py` (184 lines) — split-aware factories `make_train_env`/`make_eval_env`; the `RealizationEngine.from_split_manifest(...)` call site at line 100; the monkey-patch at line 107 that gives the env its split-aware engine.
- `src/blue_team/run_config.py` (lines 50–96) — `EnvConfigSerializable` dataclass; defaults `impact_is_terminal=True`, `exclude_ood=True`, `p_defender_deescalation=0.6`, `split="train"` / `"val_balanced"`.
- `src/utils/realization_engine.py` (re-verified Step-1 invariant) — `from_split_manifest` factory, `exclude_ood=True` default at line 110, OOD-stripping at line 165.
- `src/algorithms/adversarial_algorithm.py` — selective; no impact on Phase-3 MDP semantics.

### Tests
- Full suite: **`pytest -q` → 411 passed in 68.10 s** on `mentor-review/step-3-env` (off `main` @ `d4acfca`).
- Phase-3-scoped: `tests/test_phase3_env_gates.py` (13 tests, G3.1–G3.7) + `tests/test_phase31_impact_terminal.py` (6 tests, both `True`/`False` branches of the lever) + `tests/test_realization_engine_split_aware.py` (9 tests, OOD-exclusion + split isolation) → **30/30 passed in 4.09 s**.
- Generic env: `tests/test_adversarial_env.py` (29 tests), `tests/test_realization_engine.py` (19 tests). Inside the 411.
- Lever forwarding: `tests/test_train_agent_reward_overrides.py`, `tests/test_blue_team_env_factory.py` confirm the F9/F10 sweep substrate is byte-correct. Inside the 411.

### Docs
- `docs/mentor_review/README.md` — directory conventions
- `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2); IoTWarden recommended-action mapping (used as design choice)
- `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still in force)
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 dataset audit
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2 red-team audit; Finding 8 carry-forward resolved here
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — template for this handoff
- `docs/mentor_review/03_env.md` — Step-3 mentor memo, written this session.

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-3 v2 environment is faithfully implemented and mechanically verified. **All 7 exit gates G3.1–G3.7 PASS** on current `main` (30/30 Phase-3-scoped tests; 411/411 full suite). The MDP — state space, action space, transition function, IMPACT-clamp, defender-driven de-escalation, terminal-condition logic — matches `PLAN.md` and `RESULTS.md` line-by-line. The reward function (six implemented terms + inline IMPACT terminal accounting) matches `RESULTS.md` §3 and the IoTWarden-aligned recommended-action mapping locked in `00_framing.md` §2. The two Phase-7 levers — `impact_is_terminal` (P3 structural lever) and `p_defender_deescalation` (F10 aggressiveness sweep) — are exposed cleanly and tested in both branches.

Three minor findings: **F1** (no `manifest.json`/`G3_scoreboard.json` for Phase 3 — by design per PLAN §3.3, but the asymmetry should be documented once in `docs/results/README.md`); **F2** (env's `__init__` builds a non-split-aware engine; production split-awareness is monkey-patched by the factory — works in production but a latent footgun for direct callers); **F3** (PLAN-vs-RESULTS reward-component list mismatch; the task prompt's "MTTC as a reward component" framing is wrong — MTTC is a metric, not a reward term). Plus one nit (**F4**, MTTC IMPACT-clamp bias deferred to Step 9 LaTeX) and the Step-2 carry-forward (**F5**) resolved benign for Phase 3 (zero `set_transition_mask` calls outside `tests/test_transition_mask.py`).

Full memo: `docs/mentor_review/03_env.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: minor]** **F1** — `docs/results/03_env/` contains only `PLAN.md` and `RESULTS.md`; no `manifest.json` and no `G3_scoreboard.json`. The G3 verdicts live as a Markdown table in `RESULTS.md` §4. **By design** (PLAN §3.3: Phase 3 is infrastructure-only, no thesis figures), but the asymmetry breaks the pattern other phases use. Recommended fix: one-paragraph note in `docs/results/README.md` clarifying the asymmetry across Phases 1, 2, and 3 (rolls up Step-1 F4 and Step-2 F4). Commit: `docs(phase-3,§audit-trail): …`.

2. **[severity: minor]** **F2** — `src/environment/adversarial_env.py:305` constructs `RealizationEngine(dataset_path)` non-split-aware. Production path is correct because `src/blue_team/env_factory.py:107` monkey-patches the engine post-construction (`env._realization_engine = engine`), but a direct caller of `AdversarialIoTEnv(...)` silently bypasses the OOD-exclusion + train-only restriction. Recommended fix (preferred): class-level docstring on `AdversarialIoTEnv` documenting the contract that direct construction is for synthetic-data tests only and production must use `make_train_env`/`make_eval_env`. Optional deeper fix: add an `engine: Optional[RealizationEngine] = None` injection seam to `__init__` and have the factory pass the engine in (eliminates the `# type: ignore[attr-defined]`). Commit: `docs(phase-3,§env-init): …`.

3. **[severity: minor]** **F3** — Reward-component count divergence: PLAN.md §3.1 step 2 lists *four* terms; RESULTS.md §3 documents *six* (matches code at `adversarial_env.py:659-716`); the Step-3 task prompt says *"the four documented components — proportionality reward, mitigation, disproportionate-penalty, MTTC"* — which incorrectly lists MTTC as a reward term (MTTC is a per-episode metric exposed in `info["mttc_steps"]`, not part of the reward at all). PLAN.md is frozen and not edited; RESULTS.md is correct. Recommended fix: one sentence in `docs/reward-shaping.md` listing the six implemented terms with their as-built defaults, and an explicit clarification that MTTC is a metric. Carry the same correction into Step 9 LaTeX rebuild §3.4. Commit: `docs(phase-3,§reward): …`.

4. **[severity: nit]** **F4** — RESULTS.md §7 risk R2 acknowledges the IMPACT-clamp creates a hard left wall on MTTC at `min_episode_length=20`. If the thesis ever quotes mean MTTC, that footnote needs to propagate. Defer to Step 9 LaTeX rebuild; no Phase-3 commit needed.

5. **[carry-forward, resolved benign]** **F5** — Step-2 Finding 8 (transition_mask vs episode_generator IMPACT-state divergence) does **not** become a Step-3 correctness finding because Phase 3 never calls `set_transition_mask`. Grep confirmed: zero references in `src/environment/`, `src/blue_team/`, `src/algorithms/`, or `scripts/`. Remains a Step-8 cross-cutting cleanup task.

Full prose, file:line citations, and recommended commit messages: `docs/mentor_review/03_env.md` §7.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/03_env.md` — Step-3 mentor memo (verdict PASS-WITH-FIXES + 5 findings + MDP-correctness table + reward-formula audit + G3 reproduction table + transition_mask carry-forward resolution).
- `docs/mentor_review/03_HANDOFF.md` — this file.

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: **411 passed**. Scoped slice (G3 gates + impact_is_terminal lever + split-aware engine): **30 passed in 4.09 s**.

### Scripts / models
None modified. No re-training, no figure regeneration. **Hash chain (where it exists) intact.** The Phase-3 hash-chain *gap* (no `manifest.json` for Phase 3) is **F1**, not a regression — by design per PLAN §3.3.

### Git hygiene applied (Phase G1, opening this step)
1. `git checkout main && git pull --ff-only origin main`.
2. `git merge --no-ff mentor-review/step-2-red-team -F /tmp/step2_merge_msg.txt` → merge commit **`d4acfca`** with message ref'ing Step-2 verdict + Findings 1 and 8 carry-forward. Used `--no-ff` to preserve the Step-2 commit `2e10725` as a discrete atom in `main`'s history.
3. `git push origin main` (pushed `d4acfca`).
4. Deleted local + remote `mentor-review/step-2-red-team`.
5. Cut `mentor-review/step-3-env` off `main` @ `d4acfca`.
6. Verified policy invariants: `git tag -l` empty, `git branch -a` = `main`, `origin/main`, `mentor-review/step-3-env` only.
7. Ran `pytest -q` → 411 passed before any audit work.

End state matches policy: one long-lived branch (`main`), zero tags, current working branch is the per-step topic branch.

### Phase G2 (closing this step) — runs after sign-off
Symmetric to G1. Listed in §6.

---

## 5. Outstanding actions for the next session

The next session executes **Step 4 — Phase 4 Stage Detector review** (F11, realism). The Phase-4 deliverable is the multi-class supervised stage detector (likely a 1-D CNN; possibly a Random Forest baseline) producing the F11 confusion matrix and per-class precision/recall/F1.

### Pre-flight (Phase G1 of Step 4)
- [ ] Verify the candidate has signed off Step 3 either by (a) a comment, (b) a merge of `mentor-review/step-3-env` into `main`, or (c) explicit "go" / "Step 4" in chat. If none, **stop** and raise.
- [ ] If sign-off given **before** branch merge: execute Phase G2 ourselves —
  ```
  git checkout main && git pull --ff-only origin main
  git merge --no-ff mentor-review/step-3-env -F /tmp/step3_merge_msg.txt
  git push origin main
  git branch -d mentor-review/step-3-env
  git push origin --delete mentor-review/step-3-env
  git tag -l   # confirm still empty
  ```
- [ ] Cut `mentor-review/step-4-detector` off the new `main`.
- [ ] If any Step-3 *fix* commits were applied (F1/F2/F3 doc-fixes), pull them onto `main` first so Step 4 starts from corrected state.
- [ ] Run `pytest -q` to confirm 411 passed before audit work. If count differs, **stop** and surface.
- [ ] Verify `git tag -l` is empty (no tags during the loop, by policy).

### Step 4 review checklist (Phase 4 Stage Detector — F11, realism)
- [ ] Read `docs/results/04_detector/PLAN.md` in full — frozen audit trail. Note the gates (likely G4.1, G4.2, …) and the F11 confusion-matrix definition.
- [ ] Read `docs/results/04_detector/RESULTS.md` — locked scientific record. Per-class precision/recall/F1, F11 image, any caveats.
- [ ] Read `docs/results/04_detector/G4_scoreboard.json` (or per-gate variants) — numerical truth. **If absent**, file a finding consistent with Step-3 F1's resolution (asymmetry note in `docs/results/README.md`).
- [ ] Read `docs/results/04_detector/manifest.json` — verify hash chain via `shasum -a 256`. Confirm input SHAs chain back to the **post-3cd2fb9** Phase-1 splits manifest. (Step-2 Finding 1 was about Phase-2's input-SHA divergence; verify Phase 4 doesn't have the same issue.)
- [ ] Read `src/detector/stage_detector.py` — the public API.
- [ ] Read `src/detector/cnn1d.py` — 1-D CNN architecture, training loop, hyperparameters.
- [ ] Read `src/detector/random_forest.py` — baseline detector.
- [ ] Read `src/detector/evaluation.py` — confusion-matrix and metric computation.
- [ ] Read `scripts/detector/train_detector.py` — Phase-4 CLI, splits consumption (must be `split="train"`, `exclude_ood=True`).
- [ ] Read `tests/test_detector.py` — test coverage.
- [ ] **Realism audit (F11 specifically).** F11 is the kill-chain confusion matrix on the held-out `val_balanced` (or `test_balanced`) split. Verify:
  - the matrix is computed on the *correct* split (no train rows; OOD-class rows excluded by default; if included, that's a separate evaluation cell — confirm it's not the headline F11);
  - row/column ordering matches the kill-chain stages [BENIGN, RECON, ACCESS, MANEUVER, IMPACT];
  - per-class recall ≥ whatever PLAN.md threshold the gate sets (likely ≥0.50 or ≥0.70 per class);
  - the figure is a thesis-clean PNG (publication-grade resolution, axis labels in English, colourbar visible).
- [ ] **OOD-class behaviour audit.** Confirm whether the detector was *trained* with `exclude_ood=True` (Step-1 invariant) and whether F11 is *evaluated* with the OOD class either masked out (for the headline matrix) or shown as a separate row/column for transparency. The candidate's R1 thesis claim ("RL is robust to but not better at OOD") was framed at the *RL* level; the detector's OOD behaviour is a separate question.
- [ ] **Train/val/test isolation.** Re-verify the detector's data-loading code consumes `split="train"`, evaluates on `split="val_balanced"` (or `test_balanced` for the F11 published version). Cite file:line.
- [ ] **Class imbalance handling.** CICIoT2023 is heavily skewed toward DDoS. The `train` split is built balanced (Step-1 G1.5/G1.6 PASS). Confirm the detector code does not reintroduce imbalance via additional sampling.
- [ ] **Hyperparameters.** Cite the architecture (1-D CNN layer counts, kernel sizes, dropout, optimiser, learning rate, batch size, epochs, early-stopping criterion if any). Cross-check vs PLAN.md.
- [ ] Re-run `pytest -q` — expect 411 passed (Step 4 is read-only audit).

### Step 4 outputs (deliverables)
- [ ] Write `docs/mentor_review/04_detector.md` — full mentor memo, lead with verdict (PASS / PASS-WITH-FIXES / FAIL). Cite gate IDs (G4.1, G4.2, …) and file:line. Findings priority-ordered by severity.
- [ ] Write `docs/mentor_review/04_HANDOFF.md` from `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for **Step 5 (Phase 5 Blue Team RL training: F3, F4, T1, G5)**.
- [ ] Commit per Conventional Commits (`docs(mentor-review,step-4): …`); push to `mentor-review/step-4-detector`.
- [ ] **Pause for candidate sign-off** — do NOT merge to `main` without explicit "go" / "Step 5".

### Acceptance criterion for Step 4 PASS
- F11 (confusion matrix) is correct: right split, right axis ordering, per-class metrics meet PLAN.md gate thresholds.
- Detector training consumes `split="train"`, `exclude_ood=True` (Step-1 invariant honoured).
- Hash chain intact for `docs/results/04_detector/`.
- Test suite green (411 passed); detector-scoped tests cover the public API.
- Any fixes filed against documentation (`docs(phase-4,§…)`) unless a genuine correctness bug surfaces (then `fix(phase-4,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse --abbrev-ref HEAD     # expect: mentor-review/step-3-env (this branch)
                                    #   OR main (if Step 3 already merged by candidate)
git --no-pager log --oneline -5     # expect: 03_HANDOFF + 03_env commits on top of d4acfca
git status                          # expect: clean
git tag -l                          # expect: EMPTY (no tags during the loop, by policy)
git branch -a                       # expect: main, origin/main, current step branch only

pytest -q                           # expect: 411 passed in ~66 s

ls docs/mentor_review/              # expect:
                                    #   README.md, HANDOFF_TEMPLATE.md,
                                    #   00_framing.md, 00_HANDOFF.md,
                                    #   01_dataset.md, 01_HANDOFF.md,
                                    #   02_red_team.md, 02_HANDOFF.md,
                                    #   03_env.md, 03_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If any expectation fails, **stop** and surface the divergence. Specifically:
- If `pytest -q` is not 411 passed → Step 3 was strictly read-only audit + memo, so any test count change is unexpected.
- If a tag exists → policy violation; cut it before continuing.
- If `mentor-review/step-2-red-team` still exists locally or remotely → Phase G2 of Step 2 didn't fully complete; re-do the deletion.

If sign-off has been received but the branch hasn't been merged yet, execute Phase G2:

```bash
cat > /tmp/step3_merge_msg.txt <<'MSG'
Merge mentor-review/step-3-env into main

Step 3 (Phase 3 Environment audit — MDP, reward, gates G3.1–G3.7) closed at
PASS-WITH-FIXES.

Memo: docs/mentor_review/03_env.md
Handoff: docs/mentor_review/03_HANDOFF.md

Five findings filed; F1–F3 are minor doc-fixes (manifest/scoreboard
asymmetry, env-ctor split-awareness contract, reward-component
documentation). F4 deferred to Step 9 LaTeX rebuild (MTTC IMPACT-clamp
bias). F5 (Step-2 carry-forward, transition_mask) resolved benign:
Phase 3 never calls set_transition_mask; remains a Step-8 cross-cutting
cleanup.

All 7 exit gates G3.1–G3.7 PASS; full suite green at 411 passed.
MSG
git checkout main && git pull --ff-only origin main
git merge --no-ff mentor-review/step-3-env -F /tmp/step3_merge_msg.txt
git push origin main
git branch -d mentor-review/step-3-env
git push origin --delete mentor-review/step-3-env
git checkout -b mentor-review/step-4-detector
git tag -l            # confirm still empty
git branch -a         # expect: main, origin/main, mentor-review/step-4-detector
```

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2), chapter outline, IoTWarden's role (inspiration only)
3. `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still in force)
4. `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 dataset audit (cite findings by number; Findings 1–6 doc-fixes shipped, Finding 7 deferred to Step 8)
5. `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2 red-team audit (Finding 1 = Phase-2 manifest input-hash divergence, Step-7 re-run scheduled; Finding 8 = transition_mask carry-forward, **resolved benign by Step 3** — see `03_env.md` §6)
6. `docs/mentor_review/03_env.md` — Step-3 mentor memo (full prose; cite by Finding number F1–F5)
7. `docs/mentor_review/03_HANDOFF.md` (this file) — the resume point
8. `docs/results/04_detector/PLAN.md` — Phase-4 plan (frozen; **do not edit**)
9. `docs/results/04_detector/RESULTS.md` — Phase-4 scientific record (locked)
10. `docs/results/04_detector/G4_scoreboard.json` (or per-gate variants) — numerical gate verdicts. If absent, treat as Phase-3-style asymmetry (Step-3 F1).
11. `docs/results/04_detector/manifest.json` — Phase-4 hash chain. Verify input SHAs chain back to **post-3cd2fb9** Phase-1 outputs (Step-2 Finding 1's lesson).
12. `src/detector/stage_detector.py` — detector public API
13. `src/detector/cnn1d.py` — 1-D CNN architecture
14. `src/detector/random_forest.py` — baseline detector
15. `src/detector/evaluation.py` — F11 confusion matrix + metrics
16. `scripts/detector/train_detector.py` — Phase-4 CLI (verify `split="train"`, `exclude_ood=True` consumption — Step-1 invariant)
17. `tests/test_detector.py` — Phase-4 test coverage

Skim these for reference (do not read in full):

- `docs/dataset_card.md` (Step-1 verified post-fix)
- `docs/data-pipeline.md` (Step-1 verified)
- `docs/kill-chain-mapping.md` (Step-1 Finding 3 fix shipped: per-stage rationale paragraphs added — see commit `4b81b0b`)
- `docs/thesis_results_map.md` (per-figure thesis chapter mapping)
- `docs/architecture.md`
- root `README.md`

Then visually inspect Phase-4 figures:

```bash
ls docs/results/04_detector/
# open <any PNGs the directory contains> — F11 confusion matrix, possibly F11b per-class metrics
```

---

## 8. Open questions for the user

1. **Step-3 Findings F1–F3 (doc-fixes).** Land them now as part of Step 3, or batch into Step 8 (cross-cutting audit)? My recommendation: **batch into Step 8**, because F1 rolls up two earlier asymmetry findings (Step-1 F4, Step-2 F4) and they share a single resolution paragraph in `docs/results/README.md`. Confirm.
2. **Step-3 Finding F2 (env-ctor split-awareness contract).** Doc-fix only (preferred), or doc + 5-line code-fix to add the `engine` injection seam? My recommendation: **doc-only at this stage**; the code-fix is a candidate for Step 7 (Phase-7 re-run territory) where touching `adversarial_env.py` doesn't risk breaking a frozen artefact. Confirm.
3. **Step-2 Finding 1 (Phase-2 manifest input-hash divergence).** Still pending. Confirm option (a) Step-7 re-run with `seed=42` vs (b) document-only in Phase-2 RESULTS.md backfill. The Phase-2 handoff §8 question 1 left this open. Phase 4 review is a good moment to lock the answer because Step 7's re-run scope crystallises here.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, no model, no hash-pinned figure, no test was touched. Pytest count unchanged at 411.
- **Risk noticed (carry-forward to Step 7):** Step-2 Finding 1's manifest input-hash divergence (Phase-2 LSTM was demonstrably trained on the pre-`3cd2fb9` leaky splits prior). Recommended fix: Step-7 re-run.
- **Risk noticed (Step-3 F2):** the env-constructor non-split-aware default. Production safe (factory monkey-patch); future test or script that bypasses the factory could leak val/test/OOD. Mitigation: docstring fix per F2.
- **Risk noticed (carry-forward to Step 8):** the transition_mask vs episode_generator IMPACT-state divergence (Step-2 F8, resolved benign for Phase 3 by Step 3). Phase 3 doesn't exercise it; Step 8 should reconcile.
- **Risk noticed (carry-forward to Step 9 LaTeX rebuild):** the MTTC IMPACT-clamp bias (Step-3 F4). The thesis prose in §3.4 needs a footnote on the lifecycle-floor structural cap.
- **Risk noticed (carry-forward to Step 9 LaTeX rebuild):** the reward-component documentation needs to track RESULTS.md's six-term version, not PLAN.md's four-term sketch nor the task prompt's incorrect MTTC-as-reward framing (Step-3 F3).

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next agent and `04_detector.md` is opened.

Per the operating rule *"One step per session. Do not start Step 4 until the candidate signs off Step 3."* — Step 4 may not begin without an explicit "go" / "Step 4" / merge of this branch.
