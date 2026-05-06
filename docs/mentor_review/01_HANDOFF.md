# Step `01` — Phase 0–1 Dataset Review — Mentor Review Handoff

**Closed:** `2026-05-06 ~01:30 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 2)
**Reviewed phase / scope:** Phase 0–1 (dataset, splits, kill-chain mapping, anti-leakage protocol, F0 figures)
**Status:** `completed`

---

## 1. What was reviewed

### Artefacts
- `docs/results/01_dataset/F0_class_distribution.png` — F0a, 2063×863, 160 DPI, 8-bit RGBA, 161 KB
- `docs/results/01_dataset/F0_stage_distribution.png` — F0b, 1343×780, 160 DPI, 8-bit RGBA, 73 KB
- `docs/results/01_dataset/F0_class_distribution.caption.md` — F0a caption + reading guide
- `docs/results/01_dataset/F0_stage_distribution.caption.md` — F0b caption + reading guide
- `docs/results/01_dataset/F0_summary.json` — numerical aggregate (n_classes=34, n_stages=5, n_total=442 237)
- `docs/results/01_dataset/manifest.json` — F0 hash chain (verified intact)
- `docs/dataset_card.md` — Hugging-Face-style dataset card (229 lines)
- `docs/data-pipeline.md` — anti-leakage and processing protocol (91 lines)
- `docs/kill-chain-mapping.md` — per-class to per-stage table (69 lines)

### Code
- `scripts/data/build_split_indices.py` — split builder, OOD pre-extraction, hash-manifest emitter (448 lines)
- `src/utils/label_mapper.py` — `KillChainStage` IntEnum + `AbstractStateLabelMapper`
- `src/utils/realization_engine.py` — `from_split_manifest` factory + `allowed_indices` enforcement
- `src/utils/dataset_processor.py` — scaler / feature-selection sections (selective read)

### Tests
- Full `pytest -q` → **411 passed** in 64.3 s
- Targeted `tests/test_build_split_indices.py` + `tests/test_realization_engine_split_aware.py` + `tests/test_label_mapper.py` → 73 passed in 2.4 s
- Disjointness regression confirmed at `tests/test_build_split_indices.py:181-198` (the Phase-4 fix `3cd2fb9` invariants)

### Docs
- `docs/mentor_review/README.md` — directory conventions
- `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2), chapter outline
- `docs/mentor_review/00_HANDOFF.md` — Step-0c handoff (read in full)
- `docs/mentor_review/HANDOFF_TEMPLATE.md` — handoff format (this file applies it)
- `docs/thesis_results_map.md` — F0a/F0b → Ch. 4 §4.1 mapping

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-1 substrate is sound: 442 237-row processed snapshot, OOD removed before train/val/test split (commit `3cd2fb9`), disjointness asserted by regression test, F0 hash chain intact, 411/411 tests passing. Three documentation defects must land before binding: `dataset_card.md` names the wrong scaler (`MinMax` vs the code's `Standard`); the same card's split table and OOD note describe the pre-`3cd2fb9` reality; `kill-chain-mapping.md` carries the per-class table but no per-class rationale for committee scrutiny. Plus a missing `PLAN.md`/`RESULTS.md` for Phase 1 (asymmetry vs every other phase). Full memo: `docs/mentor_review/01_dataset.md`.

---

## 3. Findings (priority-ordered)

1. **[severity: major]** `docs/dataset_card.md:115-118` and `:99` say `MinMaxScaler` / "MinMax scaling"; code (`src/utils/dataset_processor.py:25,79,232,288,877`) and `docs/data-pipeline.md:19,66` say `StandardScaler`. Doc fix: `docs(phase-1,§4): correct scaler name in dataset_card.md`.

2. **[severity: major]** `docs/dataset_card.md:128-138` shows pre-`3cd2fb9` split sizes (train 309 566, val 44 224, test 88 447, sum=all=442 237) and `:163-168` claims "OOD indices overlap with train/val/test by construction". Both stale. Post-fix sizes are train ≈281 420, val ≈40 202, test ≈80 414, ood=40 209, with disjointness from OOD asserted at `tests/test_build_split_indices.py:181-198`. Doc fix: `docs(phase-1,§5): align dataset_card.md with post-3cd2fb9 implementation`.

3. **[severity: major]** `docs/kill-chain-mapping.md` is a bare assignment table with no per-class rationale. Step-1 acceptance criterion explicitly demands "defensible rationale that survives committee scrutiny." Per-class contestable cases enumerated in `01_dataset.md` §3 Finding 3 (MITM-ArpSpoofing, Mirai-* variants, DictionaryBruteForce, Backdoor_Malware). No re-mapping requested — just the prose. Doc fix: `docs(phase-1,§3): add per-stage rationale paragraphs`.

4. **[severity: minor]** `docs/results/01_dataset/` has no `PLAN.md` or `RESULTS.md`. Phases 02–10 all do. Recommend option (b) from memo: one paragraph in `docs/results/README.md` documenting the asymmetry once, citing `dataset_card.md` and `data-pipeline.md` as the substantive substitutes.

5. **[severity: minor]** F0 figure ID inconsistency: `manifest.json::figure_id="F0"`, F0a caption titled "Figure F0", F0b caption titled "Figure F0b", `thesis_results_map.md` uses "F0a"/"F0b". Standardize to F0a/F0b in caption titles and split `manifest.json` into two entries.

6. **[severity: nit]** F0a caption: "Five small classes" then lists seven.

7. **[severity: nit]** `manifest.json` does not pin script SHA inline (only `git_sha` of producing commit). Defer to Step 8 cross-cutting audit if other phases follow a different convention.

Full prose, file:line citations, and recommended commit messages: `docs/mentor_review/01_dataset.md` §3.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/01_dataset.md` — Step-1 mentor memo (verdict PASS-WITH-FIXES + 7 findings + invariant table)
- `docs/mentor_review/01_HANDOFF.md` — this file

### Files edited
None.

### Files deleted
None.

### Tests
None added or changed. Full suite re-run: 411 passed.

### Scripts / models
None modified. No re-training, no figure regeneration. Hash chain intact.

### Git hygiene applied (one-time, this session)
Per the candidate's session-opening directive *"main is the only branch that should exist after a branch merge; remove any tag right now; re-tag at the end of development":*

- Created local-only rescue tag `rescue/pre-cleanup-2026-05-06`, deleted at end of cleanup.
- Fast-forwarded `main` from `ecdbe69` → `a969fd6` (= former `v0.1.0`) → `26f753e` (= former tip of `mentor-review/step-0c-framing`). No merge commits, no force-pushes, full linear history preserved.
- Deleted **local** branches: `feature/reward-shaping`, `feature/lstm-training-upgrades`, `mentor-review/step-0c-framing`.
- Deleted **local** tags: `v0.1.0`, `pre-mentor-restart`, and the rescue tag.
- Pushed updated `main` to `origin/main` (now at `26f753e`).
- Deleted **remote** refs: `feature/reward-shaping`, `v0.1.0`, `pre-mentor-restart`. (`feature/lstm-training-upgrades` had no remote ref; the local stale cache was pruned via `git remote prune origin`.)
- Cut `mentor-review/step-1-dataset` off the new `main` for this session's work.

End state matches the directive: `main` (only long-lived branch on origin), zero tags, current working branch is the per-step topic branch.

---

## 5. Outstanding actions for the next session

The next session executes **Step 2 — Phase 2 Red Team review** (F1 LSTM learning curves, F2 transition matrix comparison, LSTM convergence quality).

### Pre-flight
- [ ] Verify `git rev-parse --abbrev-ref HEAD` returns `mentor-review/step-1-dataset` (or whatever branch this handoff was committed on); if `main`, the candidate likely already merged Step 1 — cut a fresh `mentor-review/step-2-red-team` off `main`.
- [ ] Verify the candidate has signed off Step 1 either by (a) a commit explicitly accepting the verdict, (b) a comment in this thread, or (c) a merge of `mentor-review/step-1-dataset` into `main`. If none, **stop** and raise the question.
- [ ] If any Step-1 *fix* commits were applied (e.g. `docs(phase-1,§3): add kill-chain rationale`), pull them onto `main` before cutting the Step-2 branch so Step 2 starts from the corrected docs.

### Step 2 review checklist (Phase 2 Red Team — F1, F2)
- [ ] Read `docs/results/02_red_team/PLAN.md` in full — understand the pre-registered Phase-2 plan, hypotheses, gates.
- [ ] Read `docs/results/02_red_team/manifest.json` — verify hash chain (input data hashes match Phase-1 outputs; output figure hashes match on-disk PNGs via `shasum -a 256`).
- [ ] Visually inspect `docs/results/02_red_team/F1_learning_curves.png` and `F2_transition_matrix_comparison.png` (open in Quick Look or describe via PIL). Are axis labels, legends, fonts, palettes publication-clean?
- [ ] Read `docs/results/02_red_team/F1_learning_curves.caption.md` — does it match what's plotted? Is the loss/accuracy claim defensible?
- [ ] Read `docs/results/02_red_team/F2_transition_matrix_comparison.caption.md` — same. Does the comparison narrative (LSTM-learned vs ground-truth Markov) hold up?
- [ ] Read `docs/results/02_red_team/F1_summary.json` — numerical record of training metrics. Look for: best-epoch, final macro-F1, per-stage F1, train/val gap (overfit signal).
- [ ] Read `scripts/red_team/train_lstm.py` — confirm: train-only fit, balanced-val early stopping (`docs/mentor_review/00_HANDOFF.md` mentioned macro-F1 early stop), seed pinning, MLflow run id captured in manifest.
- [ ] Read `src/generator/episode_generator.py` and `src/generator/attack_sequence_generator.py` — does the LSTM consume realistic episode structure? Does it respect the kill-chain transition mask (`src/generator/transition_mask.py`)?
- [ ] Read `tests/test_red_team_helpers.py`, `tests/test_episode_generator.py`, `tests/test_attack_sequence_generator.py`, `tests/test_generator_trainer.py` — coverage of training loop, episode integrity, mask correctness.
- [ ] Re-run `pytest -q` — expect 411 passed (Step 2 is read-only).
- [ ] Sanity-check that the LSTM is trained on the **in-distribution** train pool only — i.e. `RealizationEngine.from_split_manifest(... split="train", exclude_ood=True)` (or equivalent) is what `train_lstm.py` calls. **This is the OOD-leakage check at the Phase-2 boundary.**
- [ ] Check whether the `_run_phase7_background.sh` / `_finalize_phase7_background.sh` shell scripts reference Phase-2 artefacts in any way (carry-forward from Step-0c §6 finding).

### Step 2 outputs (deliverables)
- [ ] Write `docs/mentor_review/02_red_team.md` — full mentor memo, verdict (PASS / PASS-WITH-FIXES / FAIL), findings priority-ordered, citations to F1, F2, file:line. Lead with the verdict.
- [ ] Write `docs/mentor_review/02_HANDOFF.md` from `HANDOFF_TEMPLATE.md` — outstanding-actions checklist for **Step 3 (Phase 3 Environment review: MDP correctness, reward, gates)**.
- [ ] Commit per Conventional Commits (`docs(mentor-review,step-2): …`); do **not** merge to `main` without candidate sign-off.

### Acceptance criterion for Step 2 PASS
- LSTM converged honestly (no clear overfit; balanced-val macro-F1 is the model-selection metric and matches the saved best-epoch claim).
- F2 transition matrix comparison demonstrates the LSTM has actually learned attack-sequence structure (not just emitted the marginal distribution).
- F1 + F2 PNGs are publication-clean.
- LSTM training did not see OOD-class rows or test/val rows.
- Any fixes filed against documentation (`docs(phase-2,§…)`) unless a genuine correctness defect surfaces (in which case `fix(phase-2,§…)`).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment (StandardScaler, not MinMax — see Step-1 Finding 1)
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse HEAD                 # expect: <commit SHA at handoff>; today's tip is on
                                    #         mentor-review/step-1-dataset, commit pending
git --no-pager log --oneline -5    # expect Step-1 commits on top of 26f753e
git status                         # expect clean
git tag -l                         # expect EMPTY (no tags during the loop, by policy)
git branch -a                      # expect: main, origin/main, current step branch only

source .venv/bin/activate
pytest -q                          # expect: 411 passed

ls docs/mentor_review/             # expect: README.md, HANDOFF_TEMPLATE.md,
                                    #         00_framing.md, 00_HANDOFF.md,
                                    #         01_dataset.md, 01_HANDOFF.md
                                    # (this file is the highest <NN>_HANDOFF.md)
```

If `pytest -q` is not 411 passed, **stop** and surface the divergence — Step 1 was strictly read-only audit + memo authoring, no code or test changes.

If a tag exists, **stop**: per policy, no tags are created during the mentor-review loop. A single release tag (likely `v1.0.0`) is cut at the end of Step 10.

If `main` has fewer than the expected commits, the candidate may not yet have merged Step 1 — that is fine; just confirm the working branch is the right one.

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` — locked thesis claims (P1–P3, R1–R2), chapter outline, IoTWarden's role (inspiration only)
3. `docs/mentor_review/00_HANDOFF.md` — Step-0c framing handoff (still in force)
4. `docs/mentor_review/01_dataset.md` — Step-1 mentor memo (full findings; cite by Finding number)
5. `docs/mentor_review/01_HANDOFF.md` (this file) — the resume point
6. `docs/results/02_red_team/PLAN.md` — Phase-2 plan (frozen audit trail; **do not edit**)
7. `docs/results/02_red_team/manifest.json` — Phase-2 hash chain
8. `docs/results/02_red_team/F1_learning_curves.caption.md` and `F2_transition_matrix_comparison.caption.md` — what F1 and F2 claim to show
9. `docs/results/02_red_team/F1_summary.json` — numerical record (per-stage F1, best epoch, train/val gap)
10. `scripts/red_team/train_lstm.py` — Red Team training entry point
11. `src/generator/episode_generator.py` and `src/generator/transition_mask.py` — episode structure + kill-chain mask
12. `src/training/generator_trainer.py` — actual training loop
13. `tests/test_red_team_helpers.py`, `tests/test_episode_generator.py`, `tests/test_attack_sequence_generator.py`, `tests/test_generator_trainer.py`, `tests/test_transition_mask.py` — Phase-2 test coverage

Skim these for reference (do not read in full):

- `docs/dataset_card.md` (Step-1 verified; note Findings 1, 2 — the dataset card scaler name and split table are stale; trust the code over the card)
- `docs/data-pipeline.md` (correct on scaler — `StandardScaler`)
- `docs/kill-chain-mapping.md` (Step-1 Finding 3 — table is right, prose is missing)
- `docs/thesis_results_map.md` (per-figure thesis chapter mapping)
- `docs/architecture.md`
- root `README.md`

Then visually open the F1, F2 PNGs:

```bash
open docs/results/02_red_team/F1_learning_curves.png
open docs/results/02_red_team/F2_transition_matrix_comparison.png
```

---

## 8. Open questions for the user

1. **Step-1 Finding 1 (scaler).** Is `StandardScaler` (the code) the intended choice? The dataset card will be corrected to match. If `MinMaxScaler` was the actual intent and the code drifted, that's a **correctness** finding requiring re-processing — please flag explicitly.
2. **Step-1 Finding 4 (Phase-1 audit-trail asymmetry).** Option (a) retroactive `PLAN.md`+`RESULTS.md` or option (b) document-the-asymmetry-once in `docs/results/README.md`. My recommendation: (b).
3. **Step-1 Finding 3 (kill-chain rationale).** Should the candidate (you / Felipe) draft the per-class rationale, or do you want me to draft it as a `docs(phase-1,§3): …` commit on `mentor-review/step-1-dataset` *before* moving to Step 2? This is the single highest-leverage fix in the dissertation per my read.
4. **Step-2 OOD-leakage check.** The Step-1 memo confirms the OOD-removal-before-split invariant at the Phase-1 boundary. The Step-2 review will check that `train_lstm.py` consumes the in-distribution-only `train` split (no `XSS`, `VulnerabilityScan`, `Mirai-udpplain`, `DDoS-HTTP_Flood` in Red Team training). Confirm this is the right check for Step 2's acceptance.

---

## 9. Risks introduced or noticed

- **None introduced this session.** No code, no model artefact, no hash-pinned figure was touched. Pytest count unchanged at 411.
- **Risk of confusion (mitigated):** `dataset_card.md` Findings 1+2 mean the dataset card actively misrepresents the actual data pipeline. Until the doc fix lands, future agents *must* trust the code (`src/utils/dataset_processor.py`) and `docs/data-pipeline.md` over the card. This handoff `§3 Findings` and `§7 Context-loading` flag the disagreement explicitly.
- **Risk noticed (carry forward to Step 2):** The Step-2 LSTM training script must consume `RealizationEngine.from_split_manifest(..., exclude_ood=True)` or equivalent at the train pool. If it does not, Phase-2 results have OOD leakage and we have a **blocker** for the dissertation. Step 2 §5 checklist includes this verification.
- **Risk noticed (carry forward to Step 8 cross-cutting):** The "five small classes" → seven names typo in F0a caption is a hint that captions are hand-authored rather than generated. Step 8 should sanity-check every figure caption against its `summary.json`.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the next agent and `02_red_team.md` is opened.

Per the operating rule *"One step per session. Do not start Step 2 until the candidate signs off Step 1."* — Step 2 may not begin without an explicit "go" / "Step 2" / merge of this branch.
