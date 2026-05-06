# Step `02` — Phase 2 Red Team Review — Mentor Review Handoff

**Closed:** `2026-05-06 ~02:40 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 3)
**Reviewed phase / scope:** Phase 2 (Red Team — LSTM episode generator, F1 learning curves, F2 transition-matrix comparison)
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts
- `docs/results/02_red_team/PLAN.md` — frozen audit trail; gates G1–G5 defined.
- `docs/results/02_red_team/manifest.json` — Phase-2 hash chain (`figure_id="F1+F2"`).
- `docs/results/02_red_team/F1_summary.json` — best epoch, gate values, transition matrices, per-stage F1.
- `docs/results/02_red_team/F1_learning_curves.png` (1902×716 px, 8-bit RGBA, 80 KB) + `.caption.md`.
- `docs/results/02_red_team/F2_transition_matrix_comparison.png` (2020×678 px, 8-bit RGBA, 81 KB) + `.caption.md`.
- `data/processed/ciciot2023/splits/manifest.json` — current (post-`3cd2fb9`) Phase-1 manifest, used for the LSTM-prior delta analysis.

### Code
- `scripts/red_team/train_lstm.py` (520 lines) — Phase-2 CLI; loads train-split prior, synthesizes 50 000 episodes, trains the LSTM, evaluates, emits F1+F2 + summary + manifest.
- `src/generator/episode_generator.py` (539 lines) — `EpisodeGenerator` Markov sampler + stateless helpers (`episodes_to_numpy`, `stage_distribution_from_split_manifest`).
- `src/generator/attack_sequence_generator.py` (lines 1–120 read; 5×5 LSTM next-token predictor).
- `src/generator/transition_mask.py` (226 lines) — *not exercised* by Phase 2.
- `src/training/generator_trainer.py` (1041 lines) — training loop, balanced-val split, early stopping, MLflow.

### Tests
- Full `pytest -q` → **411 passed** in 66.7 s.
- Phase-2 scoped: `tests/test_red_team_helpers.py` (88 lines), `test_episode_generator.py` (611), `test_attack_sequence_generator.py` (334), `test_generator_trainer.py` (572), `test_transition_mask.py` (206) — all passing inside the 411-test suite.
- Quantitative side-check: reverse-engineered the LSTM training prior from `F1_summary.json::stage_frequency_train_prior` and computed cosine similarity vs both candidate manifests (pre-/post-`3cd2fb9`). Result: pre-fix cosine = **0.9999999999940188**, post-fix cosine = **0.9980** — confirms Finding 1.

### Docs
- `docs/mentor_review/README.md` — directory conventions
- `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2)
- `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still in force)
- `docs/mentor_review/01_dataset.md` — Step-1 mentor memo (read in full)
- `docs/mentor_review/01_HANDOFF.md` — Step-1 handoff (read in full)
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — handoff format (this file applies it)
- `docs/mentor_review/02_red_team.md` — Step-2 mentor memo, written this session.

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-2 LSTM is genuinely fit-for-purpose: the 5-token language model has learned the synthetic Kill-Chain Markov grammar (max per-cell deviation 0.012 across 25 cells; cosine of LSTM rollouts vs ground-truth rollouts = 1.0000), all four pre-registered gates G1–G4 PASS with strong margins, and `pytest -q` is green at **411 passed**. Three doc/manifest defects must land before binding: **(F1, major)** the Phase-2 manifest's `inputs["splits/manifest.json"]` SHA points to the pre-`3cd2fb9` (leaky) Phase-1 splits manifest — re-run at Step 7 will fix this with negligible expected change to gate values; **(F2, minor)** PLAN.md / 00 + 01 handoff wording says model selection is by macro-F1, but the code (`use_macro_f1_stopping=False`) selects by balanced-val cross-entropy; **(F4, minor)** Phase 2 has no `RESULTS.md` (same asymmetry already flagged for Phase 1). Plus three nits (F3, F5, F6) and one carry-forward (transition_mask vs episode_generator absorbing-state divergence; only relevant if Phase 3+ uses the mask). Full memo: `docs/mentor_review/02_red_team.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: major]** `docs/results/02_red_team/manifest.json:6` declares `splits/manifest.json` SHA `82aa1214…` (pre-`3cd2fb9` leaky manifest). On-disk SHA is `1e99d596…` (post-`3cd2fb9`). The LSTM was demonstrably trained on the pre-fix prior — confirmed by cosine 0.99999 vs the pre-fix train-distribution estimate (vs 0.998 post-fix). Per-stage frequency delta (max 2.3 pp) propagates only into the 5 BENIGN-row cells of the transition matrix; gates G1–G4 are internal to the generator and unaffected, but the audit-trail invariant is broken. Two acceptable fixes (memo §3 Finding 1): **(a)** re-run at Step 7 with seed=42 against the post-fix manifest (preferred); **(b)** document-only in `RESULTS.md`.

2. **[severity: minor]** Model-selection metric mismatch. PLAN.md §3.2, 00_HANDOFF.md §5, 01_HANDOFF.md §5 step-2 acceptance, and the Step-2 prompt all say "balanced-val macro-F1 is the model-selection metric." Code (`scripts/red_team/train_lstm.py:337-355` + `src/training/generator_trainer.py:445-452`) selects by minimum balanced-val cross-entropy; macro-F1 is reported as a secondary diagnostic. The saved checkpoint corresponds to epoch 1 (`val_loss=0.854`), which is **not** the macro-F1 maximum. Doc fix recommended: replace the wording in future memos and (where authored) in `RESULTS.md` / `F1_caption.md`. PLAN.md is frozen; do not edit.

3. **[severity: minor]** Stage-1 (RECON) F1 = 0 on the holdout (no RECON predictions ever produced). Caption already half-acknowledges this. Add one sentence sharpening the explanation ("IMPACT absorbing dominates ambiguous histories; downstream consumption — Phase 5 — uses the LSTM only as an episode sampler, not as a per-token classifier").

4. **[severity: minor]** Phase 2 has no `RESULTS.md` (same asymmetry as Step-1 Finding 4). Recommend the same option (b) — one paragraph in `docs/results/README.md` documenting the asymmetry once.

5. **[severity: minor]** `tex/figs/lstm_*.png` carries three legacy qualification-era figures that don't match Phase-2-v2 framing. Step 9 (LaTeX rebuild) owns this — flagged as carry-forward.

6. **[severity: nit]** `scripts/red_team/train_lstm.py:305-306` seeds numpy but not `torch.manual_seed`. One-line fix; defer to Step 7 if Finding 1 (a) re-run is scheduled, else apply now.

7. **[severity: nit]** `manifest.json` does not pin producing-script SHA inline (only commit-level `git_sha`). Same as Step-1 Finding 7. Defer to Step 8.

8. **[carry-forward, not a Step-2 finding]** `src/generator/transition_mask.py:79-80` allows `IMPACT→BENIGN` while `src/generator/episode_generator.py:269-271` hard-codes IMPACT absorbing. Phase 2 does not exercise the mask. Step 3 should verify whether the RL env (Phase 3) sets `use_transition_mask=True`; if yes, this becomes a Step-3 finding; if no, Step-8 cross-cutting.

Full prose, file:line citations, and recommended commit messages: `docs/mentor_review/02_red_team.md` §3.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/02_red_team.md` — Step-2 mentor memo (verdict PASS-WITH-FIXES + 7 findings + invariant table + LSTM convergence narrative).
- `docs/mentor_review/02_HANDOFF.md` — this file.

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: **411 passed**.

### Scripts / models
None modified. No re-training, no figure regeneration. **Output-side hash chain intact.** Input-side splits-manifest divergence (Finding 1) is documented but not yet re-resolved.

### Git hygiene applied (Phase G1, opening this step)
1. `git checkout main && git pull --ff-only origin main`
2. `git merge --no-ff mentor-review/step-1-dataset` → merge commit `90e5195`. Used `--no-ff` over squash to preserve the 5 Step-1 Conventional-Commits-scoped doc-fix commits as individual atoms in `main`'s history.
3. `git push origin main`.
4. Deleted local + remote `mentor-review/step-1-dataset`.
5. Cut `mentor-review/step-2-red-team` off `main`.
6. Verified policy invariants: `git tag -l` empty, `git branch -a` = `main`, `origin/main`, `mentor-review/step-2-red-team` only.
7. Ran `pytest -q` to confirm 411 passed before any audit work.

One-time pager fix: `git config --global core.pager cat` plus `pager.{tag,branch,log}=cat`, after a `less`-pager hijack interrupted an earlier shell turn. This is now permanent on the candidate's machine.

End state matches policy: one long-lived branch (`main`), zero tags, current working branch is the per-step topic branch.

### Phase G2 (closing this step) — runs after sign-off
Symmetric to G1. Listed in §6.

---

## 5. Outstanding actions for the next session

The next session executes **Step 3 — Phase 3 Environment review** (MDP correctness, reward shape, env gates G3.1–G3.7).

### Pre-flight (Phase G1 of Step 3)
- [ ] Verify the candidate has signed off Step 2 either by (a) a comment, (b) a merge of `mentor-review/step-2-red-team` into `main`, or (c) explicit "go" / "Step 3" in chat. If none, **stop** and raise.
- [ ] If sign-off given **before** branch merge: execute Phase G2 ourselves —
  ```
  git checkout main && git pull --ff-only origin main
  git merge --no-ff mentor-review/step-2-red-team \
    -m "Merge mentor-review/step-2-red-team into main\n\nStep 2 (Phase 2 Red Team audit) closed at PASS-WITH-FIXES."
  git push origin main
  git branch -d mentor-review/step-2-red-team
  git push origin --delete mentor-review/step-2-red-team
  git tag -l   # confirm still empty
  ```
- [ ] Cut `mentor-review/step-3-env` off the new `main`.
- [ ] If any Step-2 *fix* commits were applied (e.g. `fix(phase-2,seed): also pin torch.manual_seed`), pull them onto `main` first so Step 3 starts from corrected state.
- [ ] Run `pytest -q` to confirm 411 passed before audit work. If count differs, **stop** and surface.
- [ ] Verify `git tag -l` is empty (no tags during the loop, by policy).

### Step 3 review checklist (Phase 3 Environment — MDP, reward, gates G3.1–G3.7)
- [ ] Read `docs/results/03_env/PLAN.md` in full — frozen audit trail. Understand the MDP: state space, action space, transition function, reward.
- [ ] Read `docs/results/03_env/RESULTS.md` — locked scientific record. Note all gate verdicts.
- [ ] Read `docs/results/03_env/G3_scoreboard.json` (or per-gate scoreboard files) — numerical truth, immutable.
- [ ] Read `docs/results/03_env/manifest.json` — verify hash chain: input SHAs chain back to Phase-1 outputs and Phase-2 LSTM checkpoint; output figure / summary SHAs match on-disk via `shasum -a 256`.
- [ ] Read `src/environment/adversarial_env.py` — Gym-style env, `observation_space`, `action_space`, `step()`, `reset()`, MDP correctness.
- [ ] Read `src/utils/realization_engine.py` (already audited in Step 1) — verify that the env constructs it via `from_split_manifest(..., exclude_ood=True)` so OOD classes never appear in training.
- [ ] Read `src/algorithms/adversarial_algorithm.py` — high-level orchestration, if relevant for env semantics.
- [ ] Read `tests/test_adversarial_env.py`, `tests/test_phase3_env_gates.py`, `tests/test_phase31_impact_terminal.py`, `tests/test_realization_engine.py` — full coverage of MDP invariants.
- [ ] **Reward function audit:** locate the reward calculation (likely in `adversarial_env.py::step()` or a helper). Verify the four components from PLAN.md (proportionality reward, mitigation, disproportionate-penalty, MTTC) are implemented as documented.
- [ ] Verify proportionality semantics: stage-action pairings (BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK, IMPACT→ISOLATE) match the IoTWarden-inspired mapping in `00_framing.md` §2.
- [ ] Verify `impact_is_terminal=True` is the default (P3 evidence) and that the alternative `impact_is_terminal=False` (Phase 7's structural lever) is exposed cleanly.
- [ ] Verify defender de-escalation probability `p_defender_de-escalation` is exposed and that Phase-7 sweep `F10` consumes it.
- [ ] **OOD-leakage check (carry-forward from Step 1).** Does the env ever load val/test/OOD rows during training-time `step()` calls? It must consume only the `train` split (with `exclude_ood=True`).
- [ ] **Carry-forward from Step 2 Finding 8.** Does the env (or any consumer of `AttackSequenceGenerator` in Phase 3) call `set_transition_mask()`? If yes, the `transition_mask.py:79-80` IMPACT→BENIGN allowance vs the absorbing-state hard-code in `episode_generator.py:269-271` becomes a Step-3 correctness finding.
- [ ] Re-run `pytest -q` — expect 411 passed (Step 3 is read-only).

### Step 3 outputs (deliverables)
- [ ] Write `docs/mentor_review/03_env.md` — full mentor memo, lead with verdict (PASS / PASS-WITH-FIXES / FAIL). Cite gate IDs (G3.1, G3.2, …) and file:line. Findings priority-ordered.
- [ ] Write `docs/mentor_review/03_HANDOFF.md` from `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for **Step 4 (Phase 4 Stage Detector review: F11, realism)**.
- [ ] Commit per Conventional Commits (`docs(mentor-review,step-3): …`); push to `mentor-review/step-3-env`.
- [ ] **Pause for candidate sign-off** — do NOT merge to `main` without explicit "go" / "Step 4".

### Acceptance criterion for Step 3 PASS
- MDP semantics correct: state space, action space, transition function, terminal-condition logic match PLAN.md and RESULTS.md.
- Reward function implementation matches the documented four-component formulation.
- All Phase-3 exit gates (G3.1–G3.7) PASS in `G3_scoreboard.json` and the verdicts are reproducible from the env tests.
- No OOD-leakage at the Phase-3 boundary (env consumes train-only, OOD-excluded).
- Hash chain intact for `docs/results/03_env/`.
- Any fixes filed against documentation (`docs(phase-3,§…)`) unless a genuine correctness bug surfaces (then `fix(phase-3,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse --abbrev-ref HEAD     # expect: mentor-review/step-2-red-team (this branch)
                                    #   OR main (if Step 2 already merged by candidate)
git --no-pager log --oneline -5     # expect: 02_HANDOFF + 02_red_team commits on top of 90e5195
git status                          # expect: clean
git tag -l                          # expect: EMPTY (no tags during the loop, by policy)
git branch -a                       # expect: main, origin/main, current step branch only

pytest -q                           # expect: 411 passed in ~66 s

ls docs/mentor_review/              # expect:
                                    #   README.md, HANDOFF_TEMPLATE.md,
                                    #   00_framing.md, 00_HANDOFF.md,
                                    #   01_dataset.md, 01_HANDOFF.md,
                                    #   02_red_team.md, 02_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If any expectation fails, **stop** and surface the divergence. Specifically:
- If `pytest -q` is not 411 passed → Step 2 was strictly read-only audit + memo, so any test count change is unexpected.
- If a tag exists → policy violation; cut it before continuing.
- If `mentor-review/step-1-dataset` still exists locally or remotely → Phase G1 of Step 2 didn't fully complete; re-do the deletion.

If sign-off has been received but the branch hasn't been merged yet, execute Phase G2:

```bash
git checkout main && git pull --ff-only origin main
git merge --no-ff mentor-review/step-2-red-team \
  -m "Merge mentor-review/step-2-red-team into main

Step 2 (Phase 2 Red Team audit) closed at PASS-WITH-FIXES."
git push origin main
git branch -d mentor-review/step-2-red-team
git push origin --delete mentor-review/step-2-red-team
git checkout -b mentor-review/step-3-env
git tag -l            # confirm still empty
git branch -a         # expect: main, origin/main, mentor-review/step-3-env
```

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2), chapter outline, IoTWarden's role (inspiration only)
3. `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still in force)
4. `docs/mentor_review/01_dataset.md` — Step-1 mentor memo (cite by Finding number; doc-fixes 1–6 applied; Finding 7 deferred to Step 8)
5. `docs/mentor_review/01_HANDOFF.md` — Step-1 handoff
6. `docs/mentor_review/02_red_team.md` — Step-2 mentor memo (full prose; cite by Finding number)
7. `docs/mentor_review/02_HANDOFF.md` (this file) — the resume point
8. `docs/results/03_env/PLAN.md` — Phase-3 plan (frozen; **do not edit**)
9. `docs/results/03_env/RESULTS.md` — Phase-3 scientific record (locked)
10. `docs/results/03_env/G3_scoreboard.json` (or per-gate variants) — numerical gate verdicts
11. `docs/results/03_env/manifest.json` — Phase-3 hash chain
12. `src/environment/adversarial_env.py` — the env under review
13. `src/utils/realization_engine.py` — split-aware feature sampler (already audited in Step 1; verify Phase 3 consumes via `from_split_manifest(..., exclude_ood=True)`)
14. `src/algorithms/adversarial_algorithm.py` — orchestration (read selectively if needed for env semantics)
15. `tests/test_adversarial_env.py`, `tests/test_phase3_env_gates.py`, `tests/test_phase31_impact_terminal.py`, `tests/test_realization_engine.py`, `tests/test_realization_engine_split_aware.py` — Phase-3 test coverage

Skim these for reference (do not read in full):

- `docs/environment.md` — Phase-3 user-facing doc
- `docs/reward-shaping.md` — reward design narrative
- `docs/dataset_card.md` (Step-1 verified post-fix; trust the code over the card if anything looks off)
- `docs/data-pipeline.md` (correct on scaler — `StandardScaler`)
- `docs/kill-chain-mapping.md` (Step-1 Finding 3 fix shipped: per-stage rationale paragraphs added — see commit `4b81b0b`)
- `docs/thesis_results_map.md` (per-figure thesis chapter mapping)
- `docs/architecture.md`
- root `README.md`

Then, if relevant, visually inspect the Phase-3 figures (which I have not yet enumerated):

```bash
ls docs/results/03_env/
# open <any PNGs the directory contains>
```

---

## 8. Open questions for the user

1. **Step-2 Finding 1 (manifest input-hash divergence).** Option (a) re-run F1+F2 at Step 7 with the post-fix manifest, or option (b) document-only in `RESULTS.md`. My recommendation is (a) at Step 7. Confirm.
2. **Step-2 Finding 2 (model-selection metric).** Confirm balanced-val cross-entropy (the code's actual behaviour) is the intended criterion. If the original intent was macro-F1 + recall gates, that's a `fix(phase-2,trainer)` commit and a Step-7 re-run.
3. **Step-2 Finding 4 (RESULTS.md asymmetry).** Same as Step-1 Finding 4: option (a) backfill or option (b) document-once. My recommendation: (b).
4. **Step-3 OOD-leakage carry-forward.** The Step-3 review will verify the env consumes only the post-fix `train` split with OOD excluded. If the env mistakenly loads val/test/OOD during training, that's a Phase-3 blocker comparable to the Phase-1 leakage that `3cd2fb9` fixed. Confirm this is the right level of scrutiny.
5. **Step-3 transition-mask carry-forward.** Step 3 should verify whether `AdversarialEnv` (or any Phase-3 consumer of the LSTM) calls `set_transition_mask()`. If yes, the `transition_mask.py:79-80` ↔ `episode_generator.py:269-271` divergence (mask permits IMPACT→BENIGN; episode generator hard-codes absorbing) becomes a Step-3 finding.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, model, hash-pinned figure, or test was touched. Pytest count unchanged at 411.
- **Risk noticed (carry-forward to Step 7):** the Phase-2 manifest's `inputs["splits/manifest.json"]` SHA is the pre-`3cd2fb9` (leaky) manifest's SHA. The transmitted leakage to the LSTM is bounded (≤ 2.3 pp per stage on 5/25 transition-matrix cells) and the gates G1–G4 are unaffected, but the audit-trail invariant is broken. **Mitigation:** Step 7 re-run with the corrected manifest.
- **Risk noticed (carry-forward to Step 3):** the `transition_mask.py:79-80` ↔ `episode_generator.py:269-271` IMPACT-state divergence. Phase 2 doesn't exercise it; Phase 3+ may. **Mitigation:** check on Step 3 entry.
- **Risk noticed (carry-forward to Step 8 cross-cutting):** the wording "balanced-val macro-F1 is the model-selection metric" propagated through 00_HANDOFF and 01_HANDOFF and the Step-2 prompt without ever being verified against the code. Step 8 should sanity-check every "model-selection metric" / "early-stopping criterion" claim across all phases against the actual training scripts.
- **Risk noticed (carry-forward to Step 9 LaTeX rebuild):** the `tex/figs/lstm_*.png` legacy figures from the qualification draft must be replaced with `F1_learning_curves.png` and `F2_transition_matrix_comparison.png`. The qualification-era confusion-matrix figure has no equivalent under Phase-2-v2 framing (Step-2 Finding 5).

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next agent and `03_env.md` is opened.

Per the operating rule *"One step per session. Do not start Step 3 until the candidate signs off Step 2."* — Step 3 may not begin without an explicit "go" / "Step 3" / merge of this branch.
