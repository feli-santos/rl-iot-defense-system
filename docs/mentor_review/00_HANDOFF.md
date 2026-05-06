# Step `00` — Framing & Scope — Mentor Review Handoff

**Closed:** `2026-05-05 ~01:00 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 1)
**Reviewed phase / scope:** framing, audience lock-in, IoTWarden role, chapter outline
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts (read in full)
- `README.md` — root project overview (still pre-mentor-review framing)
- `docs/HANDOFF.md` — Phase-7 → Phase-10 historical handoff
- `docs/README.md` — docs nav index
- `docs/decisions.md` — design-decision ledger
- `docs/architecture.md` — architecture overview
- `docs/data-pipeline.md`, `docs/dataset_card.md`, `docs/generator.md`,
  `docs/rl-training.md`, `docs/benchmarking-results.md`,
  `docs/experiments-mlflow.md`, `docs/metrics-glossary.md`,
  `docs/environment.md`, `docs/configuration.md`,
  `docs/reproducibility.md`, `docs/walkthrough.md`,
  `docs/kill-chain-mapping.md`, `docs/reward-shaping.md` — full
  technical doc set
- `docs/thesis_results_map.md` — F0–F15 + T1 mapping
- `docs/papers/IoTWarden- A Deep Reinforcement Learning Based Real-time Defense System...pdf` — inspiring paper
- `docs/results/06_benchmark/RESULTS.md` — Phase-6 chapter (the audit-AF2 reframe doc)
- `docs/results/06_benchmark/F8_caption.md` and `F7_caption.md`
- `docs/results/05_blue_team/RESULTS.md`, `F3_caption.md`, `F4_caption.md`
- `docs/results/07_ablation/RESULTS.md`, `F10_caption.md`
- `CHANGELOG.md` (top entries Phase 6 / 7 / 10)
- `tex/thesis.tex`, `tex/thesis.cls`, `tex/{introduction,background,methodology,results,conclusions,appendices}.tex` — qualification draft

### Code (architecture-level, full read)
- `src/environment/adversarial_env.py`
- `src/algorithms/adversarial_algorithm.py`
- `src/blue_team/run_config.py`, `env_factory.py`, `aggregation.py`
- `src/benchmark/baseline_policies.py`, `eval_runner.py`
- `main.py`
- partial: `scripts/blue_team/train_agent.py`,
  `scripts/blue_team/run_phase5.py`,
  `scripts/benchmark/run_test_eval.py`

### Artifacts on disk (inventoried, not opened)
- All `docs/results/<NN>_<name>/` directories — confirmed every
  Tier-1 + Tier-2 figure (F0a, F0b, F1, F2, F3, F4, F5, F6, F7,
  F8, F9, F10, F11, F12, F15, T1) is present with manifests,
  captions, and summary JSONs. F13 / F14 reframed as future work.

---

## 2. Verdict

`PASS-WITH-FIXES`

The project is in remarkably strong shape for an MSc defense. All
required figures exist on disk with hash-pinned manifests; the
seven-phase audit-first workflow has produced a clean and unusually
honest scientific record. The main hygiene work remaining is
**framing** — earlier docs leaned on IoTWarden faithfulness in ways
that no longer match the candidate's intended thesis story. Step 0c
applied an aggressive but surgical reframe pass: forward-facing docs
softened, frozen audit trail (PLAN.md / scoreboards) preserved, no
numerical results touched. Doc tree is now consistent and ready for
the per-phase mentor-review walkthrough that begins in Step 1.

---

## 3. Findings (priority-ordered)

1. **[severity: minor]** *Ten "IoTWarden-aligned" annotations on
   forward-facing docs misrepresented the thesis contract.*
   The docs claimed several figures (F3, F4, F7, F10) "reproduced"
   or "were aligned with" specific IoTWarden figures. Per the
   candidate's decision, the thesis stops short of head-to-head
   comparison; the figures are direct CICIoT2023 results with
   IoTWarden as inspiration only. **Action taken:** softened in
   `README.md`, `docs/thesis_results_map.md`, the five caption
   files (`F3`, `F4`, `F7`, `F8`, `F10`), and the two RESULTS files
   (`05_blue_team`, `07_ablation`). PLAN.md / scoreboard files
   intentionally untouched (frozen audit trail).

2. **[severity: minor]** *F8 caption inconsistently called the
   oracle Recommended-Action policy "the IoTWarden hand-crafted
   rule baseline (floor)" while elsewhere correctly calling it
   the oracle ceiling.*
   This was already inconsistent within `06_benchmark/RESULTS.md`
   §6.1 (audit AF2 reframed it as a ceiling). **Action taken:** F8
   caption now consistent with the audit-AF2 framing.

3. **[severity: minor]** *`docs/README.md` referenced `src/benchmarking/*`
   (the deleted pre-restart package) as a source of truth.*
   The current canonical package is `src/benchmark/` (no `g`).
   **Action taken:** corrected the reference and added the missing
   `src/blue_team/` and `src/detector/` entries.

4. **[severity: minor]** *`docs/HANDOFF.md` is the canonical "next
   agent" prompt but has been superseded by Phase-10 closure and
   the mentor-review workflow.*
   The file's content (especially the audit-fix narrative around
   commits `7537493` and `396f827`) is still cited from
   `07_ablation/RESULTS.md` §5, so deleting it would break
   references. **Action taken:** added a STATUS banner at the top
   marking the file as superseded historical record and
   redirecting readers to `docs/mentor_review/`.

5. **[severity: nit]** *No mentor-review directory existed.*
   **Action taken:** created `docs/mentor_review/` with `README.md`
   (directory purpose + naming conventions), `HANDOFF_TEMPLATE.md`
   (reusable pattern), `00_framing.md` (locked thesis claims and
   chapter outline), and this file (`00_HANDOFF.md`).

6. **[severity: nit, deferred]** *Phase-7 has three one-off shell
   scripts (`_run_phase7_background.sh`,
   `_finalize_phase7_background.sh`,
   `scripts/ablation/close_phase7.py` driven via shell) that should
   be Make targets.*
   Out of scope for Step 0c; deferred to Step 7 as a refactor.

---

## 4. Actions taken in this session

### Files added
- `docs/mentor_review/README.md`
- `docs/mentor_review/HANDOFF_TEMPLATE.md`
- `docs/mentor_review/00_framing.md` — **the canonical thesis-framing memo**
- `docs/mentor_review/00_HANDOFF.md` — this file

### Files edited
- `README.md` — TL;DR softened (no longer "extends IoTWarden");
  claim P2 no longer claims "qualitatively reproduces IoTWarden Fig. 6";
  *"Inspiring paper"* section renamed to *"Inspiring work"* and
  reworded; operating-principles section now points to
  `docs/mentor_review/`.
- `docs/README.md` — banner added redirecting to `mentor_review/`;
  fixed stale `src/benchmarking/*` reference; added missing
  `src/blue_team/`, `src/detector/` entries.
- `docs/HANDOFF.md` — STATUS banner at top declaring the file
  superseded historical record; rest of content preserved intact.
- `docs/thesis_results_map.md` — restructured. Tier columns and
  *"Aligned with IoTWarden Fig. X"* annotations dropped; replaced
  with *Thesis chapter* / *Thesis section* columns. F13 / F14
  relabelled as future-work.
- `docs/results/05_blue_team/F3_caption.md` — IoTWarden alignment
  note removed; "IoTWarden recommended policy" → "oracle
  recommended-action policy".
- `docs/results/05_blue_team/F4_caption.md` — same pattern.
- `docs/results/06_benchmark/F7_caption.md` — IoTWarden alignment
  note removed.
- `docs/results/06_benchmark/F8_caption.md` — corrected the
  oracle-ceiling framing; consistent with audit-AF2 in RESULTS.md
  §6.1.
- `docs/results/07_ablation/F10_caption.md` — IoTWarden alignment
  note removed.
- `docs/results/05_blue_team/RESULTS.md` — six prose softening
  edits; **no numerical results changed**.
- `docs/results/07_ablation/RESULTS.md` — five prose softening
  edits including §6.3 retitled *"Sensitivity to attacker
  aggressiveness (G7.3 PASS)"*; **no numerical results, no gate
  verdicts, no manifest hashes touched**.
- `CHANGELOG.md` — appended a top entry documenting the framing
  pass.

### Files NOT touched (intentional)
- All `docs/results/<phase>/PLAN.md` files — frozen audit trail.
- All `G<N>_scoreboard.json` files — numerical truth, immutable.
- All figure PNGs and `manifest.json` files — hash-chain pinned.
- `docs/results/00_phase0_diagnosis.md` — historical pre-restart audit.
- `docs/decisions.md` — already free of IoTWarden-faithfulness language.
- `tex/*.tex`, `tex/*.bib`, `tex/*.cls` — Step 9 LaTeX rebuild owns these.

### Tests / scripts / models
Read-only review; no source-code modules, scripts, or tests were
modified. `pytest -q` count unchanged at 411.

### Results re-runs
None. No model trained, no plot regenerated, no JSON or PNG
overwritten.

---

## 5. Outstanding actions for the next session

The next session executes **Step 1 — Phase 0–1 Dataset review**.

Concrete checklist for the next agent:

- [ ] Open and visually inspect `docs/results/01_dataset/F0_class_distribution.png`
      and `F0_stage_distribution.png` — do they look publication-ready?
      Are axis labels, font sizes, legends correct?
- [ ] Read `docs/results/01_dataset/F0_summary.json` and
      `docs/results/01_dataset/manifest.json` — are the numerical
      summaries honest (no surprising imbalances)? Is the manifest
      hash chain intact (does it reference the splits manifest by
      SHA)?
- [ ] Read `docs/dataset_card.md` end-to-end — does the per-class
      mapping table cover every CICIoT2023 class? Is the OOD-class
      reservation justified?
- [ ] Read `docs/data-pipeline.md` — anti-leakage protocol clear?
      `StandardScaler.fit` on train only? `feature_selection`
      protocol defensible?
- [ ] Read `docs/kill-chain-mapping.md` — every CICIoT2023 attack
      class assigned to exactly one of the 5 stages? Rationale per
      assignment? Any contentious assignments that a committee
      member might challenge?
- [ ] Read `scripts/data/build_split_indices.py` — disjointness
      assertions present (the Phase-4 fix `3cd2fb9`)? OOD removal
      happens *before* train/val/test split? Are the
      regression tests in `tests/test_build_split_indices.py` actually
      testing the hash-chain invariants?
- [ ] Read `src/utils/label_mapper.py` — is `KillChainStage` a frozen
      `IntEnum`? Are the per-class assignments centralized?
- [ ] Read `src/utils/realization_engine.py` — is the
      `from_split_manifest` factory correctly excluding OOD by default?
      Is `allowed_indices` properly enforced?
- [ ] Write `docs/mentor_review/01_dataset.md` — full mentor memo
      with verdict (PASS / PASS-WITH-FIXES / FAIL), the findings
      list, the actions taken or recommended.
- [ ] Write `docs/mentor_review/01_HANDOFF.md` from
      `HANDOFF_TEMPLATE.md` — resume point for Step 2 (Phase 2 Red
      team review: F1, F2).

Acceptance criterion for Step 1: the splits are honest, the kill-
chain mapping is defensible to a committee, the F0 plots are
publication-clean and labelled correctly, and any fixes needed are
filed against the dataset documentation (not the code, unless a
genuine correctness bug surfaces).

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git status                         # may include the doc-cleanup edits if not yet committed
pytest -q                          # expect: 411 passed (Step 0c-exec is doc-only, no code changes)
ls docs/mentor_review/             # expect: README.md, HANDOFF_TEMPLATE.md,
                                   #         00_framing.md, 00_HANDOFF.md
                                   # (this file is the highest <NN>_HANDOFF.md)
```

If `pytest -q` is not 411 passed, **stop** and surface the
divergence — Step 0c-exec was strictly documentation-only, so any
test count change is unexpected.

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/00_framing.md` — locked thesis claims, chapter
   outline, IoTWarden's role (inspiration only)
3. `docs/mentor_review/00_HANDOFF.md` (this file) — the resume point
4. `docs/results/01_dataset/F0_summary.json` — numerical record of F0
5. `docs/results/01_dataset/manifest.json` — F0 hash chain
6. `docs/dataset_card.md` — Hugging-Face-style dataset card
7. `docs/data-pipeline.md` — anti-leakage and processing protocol
8. `docs/kill-chain-mapping.md` — per-class to per-stage mapping
9. `scripts/data/build_split_indices.py` — split builder with
   disjointness invariants
10. `src/utils/label_mapper.py` — `KillChainStage` IntEnum
11. `src/utils/realization_engine.py` — split-aware feature sampling
12. `tests/test_build_split_indices.py` — disjointness regression
    tests

Skim these for reference (do not read in full):

- `README.md`
- `docs/thesis_results_map.md`
- `docs/architecture.md`
- `docs/results/01_dataset/PLAN.md`
- `docs/results/00_phase0_diagnosis.md`

Then visually open the F0 PNGs:

```bash
open docs/results/01_dataset/F0_class_distribution.png
open docs/results/01_dataset/F0_stage_distribution.png
```

---

## 8. Open questions for the user

n/a — the framing decisions are locked in
`docs/mentor_review/00_framing.md`. New questions, if any, should
be raised in `01_dataset.md` for the candidate to answer at Step 1
sign-off.

---

## 9. Risks introduced or noticed

- **None introduced.** No code, model, or hash-pinned artefact was
  touched.
- **Noticed (carry forward):**
  - The `docs/results/01_dataset/PLAN.md` was not opened during
    Step 0; it should be read in Step 1 as part of the Phase-0/1
    review.
  - The `docs/results/03_env/PLAN.md` text we glimpsed contains
    "We use IoTWarden's recommended-action mapping" prose. This is
    *correct* (we genuinely adopted that mapping) and is preserved
    in the frozen audit trail. The Step-9 LaTeX rebuild should
    quote this attribution faithfully in Chapter 3.
  - Phase-7 has the `_run_phase7_background.sh` /
    `_finalize_phase7_background.sh` / `close_phase7.py` shell
    pipeline that's a candidate for refactor into Make targets.
    Step 7 owns this.

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment,
  or out-of-band confirmation), **or**
- the "Outstanding actions" list in §5 has been started by the
  next agent and `01_dataset.md` is opened.
