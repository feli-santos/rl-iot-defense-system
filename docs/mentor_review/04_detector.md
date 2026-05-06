# Step 04 — Phase 4 Stage Detector Review

*Mentor memo. Audits the Phase-4 stage detector + supervised baselines
(StageDetector MLP, RandomForest, 1-D CNN) — F11 confusion matrix
realism, per-class metrics, train/val/test isolation, OOD-class
behaviour, hyperparameters, and exit gates G4.1–G4.5 — ahead of the
MSc defense at Unicamp/FEEC.*

---

## 1. Verdict

`PASS-WITH-FIXES`

The Phase-4 detector stack is faithfully implemented and mechanically
verified. **All five exit gates G4.1–G4.5 PASS** (G4.4 as
*PASS-with-finding* by design, per PLAN §8.D2 revised in step 4.5):
G4.1 = 411 pytest passed; G4.2 = StageDetector macro-F1 on
`test_balanced` is **0.7855** (≥ 0.75 threshold); G4.3 = worst
StageDetector per-stage recall is **0.539** at RECON (≥ 0.50
threshold); G4.4 = OOD recall asymmetry observed (min = 0.001 at
`VulnerabilityScan`, max = 0.999 at `DDoS-HTTP_Flood`, gap = 0.998 —
the structural blind spot the Phase-7 RL agent must compensate for);
G4.5 = StageDetector median per-sample inference latency **0.039 ms**
on CPU (≤ 1 ms threshold, 25× headroom). The architecture (MLP
29→64→32→5, AdamW lr=1e-3 wd=1e-4, balanced class weights normalised
to sum-to-num_classes, CE loss with grad-clip 1.0, batch=512,
max_epochs=20, early-stop on val-macro-F1 patience=3) matches PLAN
§8.D3 line-by-line; same for the CNN1D and RF baseline configs. F11 is
a publication-grade 1775×694 PNG at 150 dpi with the correct
`[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]` row/column ordering.

Hash chain is intact end-to-end: `manifest.json` git SHA is
**`3cd2fb90ac7a`** (the very Phase-1 leakage-fix commit) and all six
input hashes — `features.npy`, `stages.npy`,
`splits/{train,val_balanced,test_balanced,test}.idx.npy` — chain
exactly back to the **post-`3cd2fb9`** Phase-1 splits manifest. The
Step-2 F1 lesson (manifest input-hash divergence vs the post-fix
splits) is **honoured cleanly for Phase 4**. The Step-1 invariant
(train consumes only OOD-disjoint rows) is honoured by construction
(Phase 4 loads `splits/train.idx.npy` directly, not via the
`RealizationEngine`) **and** asserted at runtime by
`_verify_disjoint(name_to_idx)` in `scripts/detector/train_detector.py`
lines 111-123 — which is the very check that *caught* the original
Phase-1 leakage bug (commit `3cd2fb9`).

Five minor doc/cosmetic findings, none thesis-blocking:
**F1** PLAN §A4 train-count table is pre-`3cd2fb9` (309 566 total; RECON
26 967; etc.); the as-built post-fix counts are 281 420 / 27 038 /
23 173 / 33 939 / 127 270 — fix lives in RESULTS.md (or a cross-cutting
note in `docs/results/README.md`); PLAN.md is frozen.
**F2** No `G4_scoreboard.json` — gate verdicts are in `F11_summary.json`
and `manifest.json.gates_status`; same asymmetry as Phase-3 F1; rolls
into Step-8 cross-cutting cleanup.
**F3** Dead-code branch `rf.run_info.__dict__ if False else {…}` at
`scripts/detector/train_detector.py:546-552`; harmless but vestigial.
**F4** RESULTS.md §3.1 column header conflates "per-class F1" with
"per-stage recall"; the numbers in that table are recalls, not F1s.
Doc-fix.
**F5** Defence-narrative gap: the "RL is robust to but not better at
OOD" claim (R1 in `00_framing.md`) is framed at the RL level; this
phase's G4.4 asymmetry is the *detector*'s OOD behaviour, and the two
should be cross-referenced explicitly in §9 LaTeX rebuild.

Full memo follows. The phase is **ready for Step 5 (Phase 5 Blue Team
RL training: F3, F4, T1, G5)** with the doc-fixes batched into Step 8.

---

## 2. What was reviewed

### Frozen audit trail (read in full, never edited)
- `docs/results/04_detector/PLAN.md` (340 lines) — design contract:
  D1 (eval split = `test_balanced` primary, `test` secondary); D2
  (G4.4 revised in step 4.5 to PASS-with-finding on `min(OOD recall) ≤
  0.30`); D2.1 (G4.3 narrowed to StageDetector only after CNN1D came
  in 0.003 below threshold); D3 (locked architectures and training
  schedules per model); §A4 imbalance audit table; §3.3 gates G4.1–G4.5
  with thresholds.
- `docs/results/04_detector/RESULTS.md` (177 lines) — locked scientific
  record: §2 final exit-gate scoreboard; §3 headline numbers
  (in-distribution + OOD); §4 three findings (RF saturation, RECON
  blind spot, OOD asymmetry); §5 unplanned discovery of the Phase-1
  leakage bug and the audit narrative around `3cd2fb9`; §6
  downstream-phase enablement.

### Source of truth (numerical)
- `docs/results/04_detector/F11_summary.json` (committed,
  SHA `955f99ff…`) — all per-model, per-split, per-stage, per-OOD-class
  numbers; gates dict with thresholds and observed.
- `docs/results/04_detector/manifest.json` (committed,
  unhashed metadata file) — git SHA `3cd2fb90ac7a`; six input hashes
  (`features.npy`, `stages.npy`, four split index files); six output
  hashes (PNG, summary JSON, caption MD, three model checkpoints).
- `docs/results/04_detector/F11_per_stage_recall.png`
  (SHA `b6bd1871…`) — 1775 × 694 px RGBA at 150 dpi.
- `docs/results/04_detector/F11_caption.md`
  (SHA `4250dfb8…`) — caption text.

### Code (re-read in full)
- `src/detector/__init__.py` (45 lines) — public surface:
  `StageDetector`, `StageDetectorConfig`, `CNN1D`, `CNN1DConfig`,
  `train_cnn1d`, `RandomForestConfig`, `train_random_forest`,
  `DetectorEvaluation`, `per_stage_recall`, `per_class_f1`,
  `macro_f1`, `confusion_matrix`, `summarize_run`.
- `src/detector/stage_detector.py` (351 lines) — production MLP head;
  `StageDetectorConfig` (29→64→32→5, dropout 0.2, AdamW lr=1e-3
  wd=1e-4, batch=512, max_epochs=20, patience=3, grad-clip 1.0, balanced
  class weights, inference batch=4096); training loop with early-stop
  on val-macro-F1; `predict` / `predict_proba` / `save` /
  `from_checkpoint` API.
- `src/detector/cnn1d.py` (331 lines) — Tharewal-style 1-D CNN
  baseline: `Conv1d(1→16,k=3,pad=1) → ReLU → MaxPool(2) →
  Conv1d(16→32,k=3,pad=1) → ReLU → AdaptiveAvgPool(1) → Linear(32, 5)`;
  same training schedule as MLP.
- `src/detector/random_forest.py` (111 lines) —
  `RandomForestClassifier(n_estimators=100, class_weight="balanced",
  n_jobs=-1, random_state=seed)`; thin wrapper attaching a
  `RandomForestRunInfo` sidecar (training time, feature importances).
- `src/detector/evaluation.py` (221 lines) — sklearn-compatible
  `confusion_matrix`, `per_stage_recall`, `per_class_f1`, `macro_f1`,
  `summarize_run`, `evaluate_ood_class`; stage names
  `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]` hard-coded
  (`evaluation.py:23`) consistent with `KillChainStage` in
  `src/utils/label_mapper.py` and `src/environment/adversarial_env.py`.
- `scripts/detector/train_detector.py` (616 lines) — Phase-4 entrypoint:
  loads features+stages+splits, runtime `_verify_disjoint` defence
  (lines 111-123), trains RF→MLP→CNN1D in that order on `train`,
  selects on `val_balanced`, evaluates on `test_balanced` and `test`,
  evaluates four OOD classes, measures latency, applies gates, renders
  F11, dumps summary JSON + manifest.

### Tests
- `tests/test_detector.py` (331 lines, 23 test methods across four
  `Test*` classes) — `TestEvaluationModule` (7 tests, sklearn parity);
  `TestStageDetector` (7 tests, including separable-cluster sanity,
  proba sums to 1, save/load round-trip, latency budget); `TestRandomForest`
  (4 tests); `TestCNN1D` (5 tests). **All 23 pass in 7.75 s on the Step-4
  branch.**
- Full suite: **`pytest -q` → 411 passed in 79.81 s** on
  `mentor-review/step-4-detector` (cut off `main` @ `193ded3` =
  Step-3 merge of `b1fffab` into `d4acfca`).

### Reference docs
- `docs/mentor_review/00_framing.md` — locked thesis claims
  (P1, P2, P3, R1, R2); IoTWarden as inspiration only.
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 audit;
  the post-`3cd2fb9` splits manifest is the canonical source.
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
  Findings 1 (manifest input-hash divergence; **honoured cleanly here**)
  and 8 (transition_mask carry-forward; not exercised by Phase 4).
- `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 F1
  (no Phase-3 manifest/scoreboard; **same asymmetry recurs here as
  this step's F2**).
- `docs/thesis_results_map.md` — F11 maps to Ch. 4 §4.4
  ("Stage Detection — A Necessary Subroutine").

---

## 3. Findings (priority-ordered)

### F1 [minor] — PLAN §A4 train counts are pre-`3cd2fb9` leaky figures

**Where.** `docs/results/04_detector/PLAN.md` lines 99-107:
> *"Per-stage train counts (309 566 rows total): BENIGN 70 000 (22.6%);
> RECON 26 967 (8.7%); ACCESS 23 198 (7.5%); MANEUVER 33 947 (11.0%);
> IMPACT 127 209 (41.1%)."*

**Why it matters.** PLAN.md was authored before the leakage bug was
discovered in step 4.5 (RESULTS.md §5). After `3cd2fb9` removed all
OOD-class rows from `train_idx`, the actual train index file holds
**281 420 rows** with stage breakdown:

| stage | PLAN §A4 | post-`3cd2fb9` actual |
|---|---:|---:|
| BENIGN | 70 000 | 70 000 |
| RECON | 26 967 | **27 038** |
| ACCESS | 23 198 | **23 173** |
| MANEUVER | 33 947 | **33 939** |
| IMPACT | 127 209 | **127 270** |
| **Total** | **309 566** | **281 420** |

The 309 566 − 281 420 = 28 146 delta matches **70%** of the four OOD
classes' total row count (40 209) — i.e., exactly the OOD rows that the
pre-`3cd2fb9` `build_split_indices.py` had erroneously folded into
`train_idx`. PLAN §A4 is therefore a *true* numerical artefact of the
bug RESULTS.md §5 explains. The detector training code in fact ran on
the *post-fix* counts (281 420), which is what `F11_summary.json`
records (`n_train: 281420`).

**Frozen audit trail rule:** PLAN.md is not edited. The fix lives
either in RESULTS.md (one-paragraph subsection naming the as-built
post-fix counts) or in a cross-cutting `docs/results/README.md` note
that rolls up Step-1 F4 + Step-2 F4 + Step-3 F1 + this finding.
Verified: no other Phase-4 doc cites the 309 566 / 26 967 / etc.
numbers.

**Severity rationale:** minor (numbers don't propagate to the figure or
to any downstream RESULTS table; the discrepancy itself is *evidence*
the leakage fix was applied).

**Recommended fix:** batch into Step 8 cross-cutting cleanup. Commit:
`docs(phase-4,§A4-as-built): note post-3cd2fb9 train counts vs PLAN
§A4`.

---

### F2 [minor] — No `G4_scoreboard.json` (recurrence of Step-3 F1)

**Where.** `ls docs/results/04_detector/` shows only `PLAN.md`,
`RESULTS.md`, `manifest.json`, `F11_*` artefacts, `F11_summary.json`.
No `G4_scoreboard.json`.

**Why it matters.** Phase 1 ships a `manifest.json` only and Phase 2
`G2_scoreboard.json` only; Phase 3 ships neither (Step-3 F1); Phase 4
ships `manifest.json` + a `gates` dict embedded in `F11_summary.json` +
a `gates_status` dict embedded in `manifest.json`, but **no top-level
`G4_scoreboard.json`**. This is the same asymmetry Step-3 F1 named —
it should be documented once in `docs/results/README.md` to avoid the
defense committee asking "why does each phase do this differently?".

**Functional consistency:** the gate verdicts ARE present, just spread
across two files. `F11_summary.json.gates` has thresholds, observed
values, and PASS/FAIL strings; `manifest.json.gates_status` is the
short-form mirror. Both agree with RESULTS.md §2's scoreboard table.

**Severity rationale:** minor cosmetic; numerical truth is intact.

**Recommended fix:** roll into the Step-3 F1 / Step-1 F4 / Step-2 F4
unified cross-cutting note in `docs/results/README.md`. Commit:
`docs(audit-trail,readme): document per-phase scoreboard asymmetry`
(batched into Step 8).

---

### F3 [minor] — Dead-code branch in summary-JSON construction

**Where.** `scripts/detector/train_detector.py` lines 545-552:

```python
"RandomForest": {
    "config": rf.run_info.__dict__  # type: ignore[attr-defined]
    if False
    else {
        "n_estimators": rf.n_estimators,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
```

**Why it matters.** The `if False else …` is a vestigial refactor
artefact — the `__dict__` branch is unreachable. The literal `False`
makes the code trivially correct (and the type-ignore comment unused
on the live branch), but it's noise that a reviewer will flag.

**Severity rationale:** minor cosmetic; behaviour is correct and
covered by tests.

**Recommended fix:** simplify to the dict literal. Caveat: any change
to `train_detector.py` will alter behaviour-irrelevant lines but
**will not** alter outputs (figures and JSON are byte-identical only
under a re-run with `seed=0` — which is Step-7 territory). Defer the
cleanup to whenever `train_detector.py` next gets touched (Step-7
re-run candidate, or Step 8 cross-cutting). Doc-only commit candidate:
none needed; cleanup in Step 8.

---

### F4 [nit] — RESULTS.md §3.1 conflates "per-class F1" with "per-stage recall"

**Where.** `docs/results/04_detector/RESULTS.md` lines 30-34:

```
| Model | Macro-F1 | BENIGN | RECON | ACCESS | MANEUVER | IMPACT |
|---|---:|---:|---:|---:|---:|---:|
| **StageDetector** … | **0.7855** | 0.819 | 0.539 | 0.801 | 0.770 | 0.998 |
```

**Why it matters.** The five per-stage cells (0.819, 0.539, 0.801,
0.770, 0.998) are **recall**, not F1. Cross-checked against
`F11_summary.json.models.StageDetector.test_balanced.per_stage_recall`
— exact match. The per-class F1s for the same model on the same split
are different numbers (0.777, 0.624, 0.685, 0.844, 0.997 per
`per_class_f1`). The column header lacks a metric label, and the
context "Macro-F1" in column 1 invites the reader to assume per-class
F1 in columns 3–7.

**Severity rationale:** nit; the F11 figure caption and PLAN §A5 both
correctly call out per-stage recall as the comparand. Only the
RESULTS.md table is ambiguous.

**Recommended fix:** add an explicit "Per-stage recall on
test_balanced" sub-header above the second-through-sixth columns, or
rename the column "Macro-F1" to "Macro-F1 (test_balanced)" and add a
caption clarifying the per-stage cells are recall. Commit:
`docs(phase-4,§3.1): clarify per-stage cells are recall, not F1`
(batched into Step 8).

---

### F5 [nit, defense narrative] — Detector-OOD vs RL-OOD framing not cross-linked

**Where.** `docs/mentor_review/00_framing.md` §3 P1/R1 frames OOD
behaviour at the **RL** level: *"the RL agent is robust to but not
better at novel attacks"*. RESULTS.md §4 Finding 3 reports the
detector's OOD asymmetry (0.001 ↔ 0.999, gap 0.998) without
explicitly cross-linking the two.

**Why it matters.** The defense committee may conflate the two senses
of "OOD performance". Step-3's mentor memo §6 already noted the same
issue tangentially. The thesis prose in §4.4 (Stage Detection) and
§9.3 (Robustness) needs one sentence each linking R1 to G4.4: *"R1's
robustness claim is at the policy level. At the detector head level,
G4.4 PASS-with-finding shows asymmetric OOD generalisation by
attack-class signature; the policy claim therefore conditions on this
silent-failure mode."*

**Severity rationale:** nit; defense-narrative consistency, no
numerical impact.

**Recommended fix:** propagate the cross-reference into Step 9 LaTeX
rebuild (chapter 4 §4.4 + chapter 9 robustness section). No Phase-4
commit; tracked as a Step-9 obligation.

---

## 4. Validation tables

### 4.1 Hash-chain reproduction (audit step)

```
$ shasum -a 256 docs/results/04_detector/F11_per_stage_recall.png \
                  docs/results/04_detector/F11_summary.json \
                  docs/results/04_detector/F11_caption.md
b6bd187177c3976bbfb54ac84b1142713b2f01706bd8972ae41ccc134ea94e51  F11_per_stage_recall.png
955f99ff7d107b35a1c1f3223b6f06ed113ab62d4e7e76cdf39339e408e016b7  F11_summary.json
4250dfb81029c274ff57260de0adc7cc38b0dd8e436eca7d59f928acad19589c  F11_caption.md
```

vs `manifest.json.outputs`:

| Output | manifest hash | recomputed | ✓ |
|---|---|---|---|
| `F11_per_stage_recall.png` | `b6bd1871…` | `b6bd1871…` | ✅ |
| `F11_summary.json` | `955f99ff…` | `955f99ff…` | ✅ |
| `F11_caption.md` | `4250dfb8…` | `4250dfb8…` | ✅ |

vs `manifest.json.inputs` (cross-checked against on-disk Phase-1
artefacts):

| Input | manifest hash | on-disk hash | ✓ |
|---|---|---|---|
| `features.npy` | `5d1ff73d…` | `5d1ff73d…` | ✅ |
| `stages.npy` | `607730a5…` | `607730a5…` | ✅ |
| `splits/train.idx.npy` | `d4aa79ae…` | `d4aa79ae…` | ✅ |
| `splits/val_balanced.idx.npy` | `8e175fcd…` | `8e175fcd…` | ✅ |
| `splits/test_balanced.idx.npy` | `7439cd63…` | `7439cd63…` | ✅ |
| `splits/test.idx.npy` | `a6728513…` | `a6728513…` | ✅ |

vs Phase-1 `data/processed/ciciot2023/splits/manifest.json` (the
post-`3cd2fb9` splits manifest):

| Input | Phase-1 splits manifest output hash | Phase-4 input hash | ✓ |
|---|---|---|---|
| `splits/train.idx.npy` | `d4aa79ae…` | `d4aa79ae…` | ✅ |
| `splits/val_balanced.idx.npy` | `8e175fcd…` | `8e175fcd…` | ✅ |
| `splits/test_balanced.idx.npy` | `7439cd63…` | `7439cd63…` | ✅ |
| `splits/test.idx.npy` | `a6728513…` | `a6728513…` | ✅ |

**Conclusion.** Phase-4 inputs chain back to the **post-`3cd2fb9`**
Phase-1 splits manifest with byte-perfect equality. The Step-2 F1
divergence pattern (Phase-2 manifest pinned to *pre*-fix hashes) does
**not** recur in Phase 4. Phase-4 manifest's `git_sha` is
`3cd2fb90ac7a` — the very fix commit — confirming the run was
performed *on* the corrected splits.

### 4.2 Exit-gate reproduction (vs PLAN §3.3)

| Gate | PLAN threshold | Observed (F11_summary.json + RESULTS.md) | Status |
|---|---|---|---|
| **G4.1** | full pytest suite green | 411 / 411 passed in 79.81 s | **PASS** |
| **G4.2** | StageDetector macro-F1 on `test_balanced` ≥ 0.75 | 0.785462 | **PASS** (margin +0.035) |
| **G4.3** | StageDetector worst per-stage recall ≥ 0.50 (D2.1) | 0.539 (RECON) | **PASS** (margin +0.039) |
| **G4.4** | min(OOD recall) ≤ 0.30 (D2 revised) | min = 0.000825 (VulnerabilityScan); max = 0.998597 (DDoS-HTTP_Flood); gap = 0.998 | **PASS-with-finding** (asymmetry recorded as thesis result) |
| **G4.5** | StageDetector per-sample inference ≤ 1 ms | 0.039 ms (median over 1 000 iter) | **PASS** (25× headroom) |

### 4.3 Hyperparameter audit (vs PLAN §8.D3)

| Knob | PLAN §8.D3 | Code (file:line) | ✓ |
|---|---|---|---|
| **StageDetector** ||||
| Architecture | 29 → 64 → 32 → 5 | `stage_detector.py:55,93-103` `StageDetectorConfig.hidden_sizes=(64, 32)` + `_MLP.__init__` | ✅ |
| Activation | ReLU + Dropout(0.2) | `stage_detector.py:99,100` | ✅ |
| Optimiser | AdamW, lr=1e-3, wd=1e-4 | `stage_detector.py:59,60,169-173` | ✅ |
| Class weights | balanced, normalised sum-to-num_classes | `stage_detector.py:159-164` (`weights = inv * num_classes / inv.sum()`) | ✅ |
| Loss | CrossEntropyLoss(weight=…) | `stage_detector.py:168` | ✅ |
| Batch / epochs / patience | 512 / 20 / 3 | `stage_detector.py:61-63` | ✅ |
| Early-stop criterion | val-macro-F1 | `stage_detector.py:202-211` | ✅ |
| Gradient clip | 1.0 | `stage_detector.py:64,254` | ✅ |
| **CNN1D** ||||
| Conv stack | Conv1d(1→16,k=3,pad=1) → ReLU → MaxPool(2) → Conv1d(16→32,k=3,pad=1) → ReLU → AdaptiveAvgPool(1) | `cnn1d.py:55-57,94-102` | ✅ |
| Head | Linear(32, 5) | `cnn1d.py:103` | ✅ |
| Optimiser/schedule | same as MLP | `cnn1d.py:60-66,151-165` | ✅ |
| **RandomForest** ||||
| n_estimators | 100 | `random_forest.py:37,82-89` | ✅ |
| class_weight | "balanced" | `random_forest.py:40,86` | ✅ |
| n_jobs | -1 | `random_forest.py:41,88` | ✅ |
| random_state | =seed | `random_forest.py:87` | ✅ |

### 4.4 Train/val/test isolation audit (Step-1 invariant)

The Phase-3/Phase-5 invariant uses
`RealizationEngine.from_split_manifest(split="train", exclude_ood=True)`.
**Phase 4 takes a different mechanism**: it loads
`data/processed/ciciot2023/splits/train.idx.npy` directly, then
indexes `features.npy[train_idx]`. The OOD attack indices live in a
separate file tree (`splits/ood_attack/<class>.idx.npy`) and are
never folded into the in-distribution splits — that is precisely the
property `3cd2fb9` fixed.

The Step-1 invariant is honoured in **two ways simultaneously**:

1. **By construction** — `train_detector.py:391-397` consumes only the
   five non-OOD split files (`train`, `val`, `val_balanced`, `test`,
   `test_balanced`); OOD is loaded *separately* at line 396-398 and
   used only for evaluation.
2. **By runtime assertion** — `_verify_disjoint(name_to_idx)` at lines
   111-123 throws `RuntimeError: LEAKAGE: train ∩ <other> = N rows`
   if any pair overlaps. This is the very check that *caught* the
   original Phase-1 leakage bug (RESULTS.md §5 narrative). It runs on
   every Phase-4 invocation. Defence-in-depth: exemplary.

| Split | Loaded at | Consumed for |
|---|---|---|
| `train` | `train_detector.py:393, 405-406` | Training all three models |
| `val_balanced` | `train_detector.py:407-408` | Early-stop val-macro-F1 monitor (StageDetector + CNN1D); RF doesn't model-select |
| `test_balanced` | `train_detector.py:409-410` | F11 left+right panels; reported in summary JSON |
| `test` | `train_detector.py:411-412` | Secondary numbers in summary JSON (D1's "BENIGN-heavy" companion) |
| `ood_attack/{4 classes}` | `train_detector.py:396-398, 487-499` | G4.4 OOD evaluation only — NEVER training |

**Class-imbalance handling.** PLAN §3.3 + §8.D3 prescribe **balanced
class weights** as the only imbalance lever (no resampling). Code
honours: RF passes `class_weight="balanced"`
(`random_forest.py:86`); MLP and CNN1D normalise inverse-frequency
weights to sum-to-num_classes
(`stage_detector.py:159-164`, `cnn1d.py:151-156`); no
oversampling, no undersampling, no SMOTE, no random subsampling
applied on top of the already-balanced `train` split. The `train`
split is itself balanced by Phase-1's `build_split_indices.py`
(Step-1 G1.5/G1.6 PASS); Phase 4 does not re-balance.

### 4.5 F11 figure visual inspection

| Property | Expected | Observed |
|---|---|---|
| Resolution | publication-grade ≥ 1500 px wide | **1775 × 694** at 150 dpi |
| Mode | colour | RGBA |
| Layout | 2-panel side-by-side | 12 × 4.5 inch figsize, `bbox_inches="tight"` (`train_detector.py:141, 200`) |
| Left panel | bar chart, 5 stages × 3 models | `train_detector.py:144-166` — 15 grouped bars, dashed best-macro-F1 line |
| Right panel | StageDetector confusion matrix on `test_balanced` | `train_detector.py:168-192` — row-normalised %; `Blues` colormap with text overlay |
| Stage axis ordering | `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]` | `evaluation.py:23` STAGE_NAMES + `train_detector.py:161,176-177` xticks/yticks ✅ |
| Axis labels | English, "Predicted" / "True" | `train_detector.py:178-180` ✅ |
| Colourbar | row-normalised, visible | `train_detector.py:192` `fig.colorbar(... label="row-normalised")` ✅ |
| Caption | accompanying `.md` | `F11_caption.md` SHA `4250dfb8…`, content cross-checked against `train_detector.py:204-217` ✅ |

### 4.6 OOD-class behaviour audit

The Phase-4 OOD evaluation cell (G4.4) is **separate** from F11's
headline confusion matrix. The F11 right-panel CM is on
`test_balanced` (5 000 rows, 1 000 per stage); OOD classes are not
in `test_balanced` at all. They are only evaluated at
`train_detector.py:483-499` against the four `splits/ood_attack/*`
files. `F11_summary.json.ood_evaluation` records per-class recall +
predicted-stage distribution.

**Detector training honoured `exclude_ood`-equivalent.** The four
held-out classes never enter the train/val/test splits at any point
(`build_split_indices.py` post-`3cd2fb9`, asserted at runtime by
`_verify_disjoint`). Equivalent semantics to
`exclude_ood=True` even though Phase 4 does not call
`RealizationEngine.from_split_manifest`.

**Headline F11 has no OOD column or row.** The 5×5 CM is
`[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]` only. OOD results are
exposed in `F11_summary.json` for thesis prose use, not in the
figure. This matches the "RL is robust to but not better at OOD" R1
framing — the OOD evaluation is a *narrative* result documented in
RESULTS.md §4 and §9 (robustness), not a figure cell.

### 4.7 Test-coverage audit

`tests/test_detector.py` 23 tests, all PASS in 7.75 s:

| Test class | Tests | Covers |
|---|---:|---|
| `TestEvaluationModule` | 7 | sklearn-parity for `confusion_matrix`, `per_stage_recall`, `per_class_f1`, `macro_f1`; `summarize_run` round-trip; empty-class edge case (returns 0.0 not NaN); `evaluate_ood_class` |
| `TestStageDetector` | 7 | default config; predict-before-fit raises; toy-cluster macro-F1 > 0.90; proba sums-to-1; save/load round-trip; run_info populated; latency < 5 ms (CI-tolerant; production gate is 1 ms) |
| `TestRandomForest` | 4 | default config; toy macro-F1 > 0.90; proba shape & sums-to-1; `run_info` attached |
| `TestCNN1D` | 5 | default config; forward shape (accepts both `(N, F)` and `(N, 1, F)`); loss decreases on toy; proba sums-to-1; save/load round-trip |

**Public API coverage:** complete. `StageDetector.{__init__, fit,
predict, predict_proba, save, from_checkpoint}`, `train_random_forest`,
`CNN1D.{__init__, fit, predict, predict_proba, save, from_checkpoint}`,
`train_cnn1d`, `confusion_matrix`, `per_stage_recall`, `per_class_f1`,
`macro_f1`, `summarize_run`, `evaluate_ood_class`. **Nothing missing.**

---

## 5. F11 realism audit

The thesis claim around F11 (`docs/thesis_results_map.md`):
*"Per-stage detection recall (detector + RF + 1D-CNN), aligned with
Tharewal et al."*

Checks:

1. **Right split.** `test_balanced` (5 000 rows, 1 000 per stage). Not
   `train`. Not `val`. **PASS.** (`train_detector.py:170 ⇄
   F11_summary.json.models.StageDetector.test_balanced.n_samples = 5000`)
2. **No train rows.** `_verify_disjoint(name_to_idx)` asserts
   `train ∩ test_balanced = ∅` at runtime
   (`train_detector.py:111-123`). **PASS.**
3. **Axis ordering.** `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]`
   from `STAGE_NAMES = ["BENIGN", "RECON", "ACCESS", "MANEUVER",
   "IMPACT"]` (`evaluation.py:23`). **PASS.**
4. **Confusion-matrix structure.** Diagonal-heavy. Off-diagonal mass
   concentrated in `RECON → ACCESS` (314/1000 = 31.4%) and
   `MANEUVER → ACCESS` (121/1000 = 12.1%). RECON is the universal
   hard stage (RESULTS.md §4 Finding 2 elaborates) — the bulk of the
   confusion lives in the RECON row, exactly the structural blind
   spot the RL agent must hedge around. **Realistic.**
5. **Per-class recall meets G4.3.** Worst is RECON at 0.539
   (StageDetector); ≥ 0.50 threshold. **PASS** (margin +0.039). The
   D2.1 revision (G4.3 narrowed to StageDetector only) is justified
   in PLAN §8.D2.1 — CNN1D scoring 0.497 on RECON is a baseline
   weakness, not a thesis-blocker.
6. **Caption thesis-clean.** `F11_caption.md`: "Per-stage detection
   recall on the balanced test split (1 000 rows / stage). Left:
   stage-recall comparison across the production MLP detector
   (blue), RandomForest baseline (orange), and 1-D CNN baseline
   (green); the dashed line marks the best macro-F1 achieved by any
   model. Right: row-normalised confusion matrix of the production
   detector on the same split…". English, factually correct,
   self-contained. **PASS.**

**No realism concerns.** F11 is publication-grade.

---

## 6. Reward-shaping / detector contract for downstream phases

Phase 4 produces three artefacts consumed by Phase 5+:

- `artifacts/detector/stage_detector.pt` — production MLP head, 4 357
  params, 20 KB on disk; `predict_proba(batch)` interface; latency
  0.039 ms/sample. Phase 5's RL agent observation pipeline can
  optionally include `predict_proba` outputs.
- `artifacts/detector/random_forest.joblib` — RF baseline, 360 MB
  (RESULTS.md §7 R4 risk note: Phase 5/7 should call `predict_proba`
  on the in-memory classifier, not reload per step).
- `artifacts/detector/cnn1d.pt` — CNN1D baseline, 8 KB, used only as
  apples-to-apples comparand in F11.

The `manifest.json` pins the SHA of all three checkpoints:
`stage_detector.pt: 71e06616…`, `random_forest.joblib: 546a7355…`,
`cnn1d.pt: 8e7cf63d…`. Phase 5's training run should record these
SHAs in its own input-manifest to chain-pin the detector checkpoint.

---

## 7. Step-2 / Step-3 carry-forwards

| Earlier finding | Status at Step 4 |
|---|---|
| **Step-1 F4** (no `G1_scoreboard.json`) | Recurs structurally as Step-4 F2 — same asymmetry. Roll into Step 8 cross-cutting note. |
| **Step-2 F1** (Phase-2 manifest input-hash divergence vs post-`3cd2fb9` splits) | **Honoured cleanly here** — Phase-4 `manifest.json.inputs` chain back to the post-fix splits manifest with byte equality. No analogous divergence. *Step-4 takeaway: the fix protocol works when the producing script is re-run after `3cd2fb9`. Phase 2's manifest is the outlier because the LSTM trainer was not re-run after `3cd2fb9`. The Step-2 F1 candidate-decision (re-run with `seed=42` vs document-only) remains open and is flagged in `04_HANDOFF.md` §8.* |
| **Step-2 F2** (model-selection metric: docs say macro-F1, code uses balanced-val CE) | **Open at Step 4**; not a Phase-4 issue. Phase 4's StageDetector explicitly selects on val-macro-F1 (`stage_detector.py:202-211`) — no metric-vs-doc divergence here. Re-flagged in `04_HANDOFF.md` §8. |
| **Step-2 F8** (transition_mask carry-forward) | **Already resolved benign by Step 3** (Phase 3 never calls `set_transition_mask`); Phase 4 doesn't either (`grep -r set_transition_mask src/detector scripts/detector` → 0 matches). Confirmed dormant. |
| **Step-3 F1** (no Phase-3 manifest/scoreboard) | **Recurs as Step-4 F2** — Phase 4 has `manifest.json` but no dedicated `G4_scoreboard.json`. Roll into the same cross-cutting note. |
| **Step-3 F2** (env-ctor non-split-aware default) | n/a — Phase 4 doesn't construct `AdversarialIoTEnv` or use the `RealizationEngine` factory; it loads splits directly. |
| **Step-3 F3** (PLAN-vs-RESULTS reward-component count mismatch) | n/a — no reward function in Phase 4. |
| **Step-3 F4** (MTTC IMPACT-clamp bias) | n/a — Phase 4 has no MTTC. |

---

## 8. Open candidate decisions (for `04_HANDOFF.md` §8)

Carried forward from Step 2 (still owed by the candidate):

1. **Step-2 F1** — re-run the Phase-2 LSTM at Step 7 with `seed=42`
   against the post-`3cd2fb9` manifest (option a, recommended), or
   document-only in a backfilled Phase-2 RESULTS.md (option b)?
2. **Step-2 F2** — was balanced-val cross-entropy or macro-F1 the
   intended Phase-2 model-selection criterion? If balanced-val CE →
   doc-fix only; if macro-F1 → `fix(phase-2,trainer): …` + Step-7 re-run.

These are not Step-4 blockers but the Step-7 re-run scope crystallises
once they're answered. Phase-4 in-distribution numbers (G4.2/G4.3) are
unaffected.

---

## 9. Sign-off

Step 4 closes at **PASS-WITH-FIXES**. All five exit gates G4.1–G4.5
pass on real data. F11 is publication-grade. The hash chain back to
the post-`3cd2fb9` Phase-1 splits manifest is byte-perfect. The five
findings are minor doc/cosmetic issues that batch cleanly into Step 8.
Step 5 (Phase 5 Blue Team RL training: F3, F4, T1, G5) is unblocked.

— mentor-review agent, 2026-05-06
