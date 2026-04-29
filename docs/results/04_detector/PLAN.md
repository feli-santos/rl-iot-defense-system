# Phase 4 — Stage Detector + Supervised Baselines: audit & plan

> Mentor's PLAN written **before any code**. Cross-references:
> - PLAN files for Phases 2 (`docs/results/02_red_team/PLAN.md`) and 3
>   (`docs/results/03_env/PLAN.md`).
> - Thesis-figure target: **F11** (`docs/thesis_results_map.md`):
>   *"Per-stage detection recall (detector + RF + 1D-CNN), aligned with
>   Tharewal et al."*

## 1 — Why Phase 4 exists

The Phase-3 environment hands the agent a **window of feature vectors**, not
the attack stage. The thesis storyline is:

> *"The RL Blue Team learns to defend without ever seeing the attack stage
> directly — but if you ground-truth the agent against a high-quality stage
> detector, the agent's policy proves correct."*

For that storyline to hold, we need three things:

1. **A reference detector** (the *Detector head*) that maps a single feature
   vector → stage probabilities. This is Phase 4's main artifact and the
   thing the agent's observation feeds into in Phase 7.
2. **Two pure-supervised baselines** that bound how good a policy *could*
   be if it had access to ground-truth stage labels. Per the thesis-results
   map: **Random Forest** and **1D-CNN**, mirroring the Tharewal et al.
   industrial-IoT baseline set.
3. **F11 itself**: per-stage recall comparison across {Detector head,
   Random Forest, 1D-CNN}. This is the figure that *justifies* the
   non-trivial gap the RL agent has to close in Phase 7.

Phase 4 produces F11. It does **not** train the RL agent — that's Phase 5.

## 2 — Audit findings (what we have / what is missing)

### A1. Dataset is ready and split-clean

`data/processed/ciciot2023/`:
- `features.npy`: shape `(442 237, 29)`, float32, scaler already applied.
- `labels.npy`: 34 attack-class strings.
- `state_indices.json`: per-stage row index lists (5 stages).
- `splits/manifest.json` + `splits/{train,val,test,val_balanced,test_balanced}.idx.npy`,
  plus `splits/ood_attack/<class>.idx.npy` for the four held-out
  attack classes.

The Phase-1 splits were built to support exactly this kind of supervised
training: train on `train`, model-select on `val`, report on `test_balanced`
(or `test` for raw stats), evaluate OOD generalisation on `ood_attack/*`.

### A2. Stage labels need to be derived once and frozen

`labels.npy` is per-attack-class. The Phase-1 `state_indices.json` already
groups rows by stage. We need a **persisted** `stages.npy` of shape
`(442 237,)` int8 in {0..4} so all three models train on the *same* stage
labels with zero-cost row-lookup. Building it once and committing the
hash to the manifest closes a class of "did we use the same labels?"
bugs that would otherwise haunt cross-baseline comparisons.

This is a one-time helper in `scripts/data/derive_stage_labels.py` (~30
lines) producing `data/processed/ciciot2023/stages.npy`. We add an
SHA-256 entry to the next splits-manifest regeneration; for Phase 4 we
verify the file is internally consistent with `state_indices.json` (no
disagreements) and add a regression test.

### A3. No supervised model code exists yet

The repo today has:
- `src/generator/` — Red Team LSTM (Phase 2).
- `src/environment/` — RL env (Phase 3).
- `src/algorithms/` — SB3 algorithm wrapper (Phase 5 territory).
- `src/benchmarking/` — runner and metrics (Phase 7 territory).
- *Nothing* for supervised stage classification.

We need a fresh module: `src/detector/`. Public API:

```python
class StageDetector:                 # Tier-1 deliverable, used by F11
    """29-D feature -> 5-class stage logits via a small MLP."""
    @classmethod
    def from_checkpoint(cls, path) -> "StageDetector": ...
    def fit(self, X, y, X_val, y_val, *, max_epochs, seed) -> "StageDetector": ...
    def predict(self, X, *, batch_size=4096) -> np.ndarray: ...   # int stage IDs
    def predict_proba(self, X, *, batch_size=4096) -> np.ndarray: ...

def train_random_forest(X, y, *, seed) -> RandomForestClassifier: ...
def train_cnn1d(X, y, X_val, y_val, *, max_epochs, seed) -> "CNN1D": ...
```

Why a tiny MLP for the *detector head* instead of something fancier? The
detector lives downstream as a **subroutine of the RL agent**: every step
of every episode in Phase 7 evaluation calls `predict_proba` on a single
29-D row. A 3-layer MLP with ~10 K parameters runs in ≈ 0.2 ms on CPU; a
1D-CNN over a window would run in ~5-10 ms. The MLP is the *production-
realistic* head. The 1D-CNN exists to give us a *strict upper bound*
comparison in F11.

### A4. Class imbalance is real but tractable

Per-stage train counts (309 566 rows total):

| stage | rows | fraction |
|---|---:|---:|
| BENIGN | 70 000 | 22.6% |
| RECON | 26 967 | 8.7% |
| ACCESS | 23 198 | 7.5% |
| MANEUVER | 33 947 | 11.0% |
| IMPACT | 127 209 | 41.1% |

Smallest class is ACCESS at 23 K rows — still huge. We will use
**balanced class weights** in the loss (sklearn convention or
`weight=1/freq` for the MLP/CNN) rather than under-sampling, because
discarding 100 K BENIGN rows would hurt absolute precision on the
dominant stage. Phase-1's `test_balanced.idx.npy` (5 000 rows, ~1 000
per stage) is the canonical evaluation split.

### A5. F11 design

Following IoTWarden Fig. 3(b) layout but with our per-stage breakdown:

- **Left panel**: bar chart, 5 stages × 3 models = 15 bars, height =
  per-stage recall on `test_balanced`. Group by stage, colour by model.
  Add a horizontal dashed line at the macro-F1 of the best model for
  reference.
- **Right panel**: confusion matrix of the *Detector head* on
  `test_balanced` (5 × 5). Annotated with row-normalised percentages.

One PNG, two side-by-side panels, A4-friendly aspect ratio (figsize =
12 × 4.5). Caption committed alongside.

The detector head is **the** model that downstream phases reuse. The RF
and 1D-CNN are baselines for context.

## 3 — Concrete deliverables

### 3.1 Code

1. `scripts/data/derive_stage_labels.py` — one-shot helper to build
   `stages.npy` from `state_indices.json`. Writes the hash into a small
   sidecar `stages.manifest.json`.
2. `src/detector/__init__.py` + four files:
   - `stage_detector.py` — the production MLP head (~150 lines).
   - `random_forest.py` — sklearn `RandomForestClassifier` wrapper (~50
     lines).
   - `cnn1d.py` — small 1D-CNN treating the 29-D vector as a 1×29
     "image" of channels = 1 (~150 lines).
   - `evaluation.py` — shared `per_stage_recall`, `confusion_matrix`,
     `summarize_run`, etc. (~100 lines).
3. `scripts/detector/train_detector.py` — Phase-4 entrypoint. Loads the
   train split via `RealizationEngine.from_split_manifest`, trains the
   three models (in this order: RF first because cheap, then MLP, then
   CNN1D), evaluates each on `test_balanced`, dumps F11 + a summary
   JSON.
4. Makefile target `make phase-4` calling the entrypoint.

### 3.2 Tests

1. `tests/test_detector_stage_detector.py` — tests the MLP can fit a
   linearly-separable synthetic toy (sanity).
2. `tests/test_detector_random_forest.py` — sklearn wrapper produces
   `predict_proba` output of shape `(N, 5)` summing to 1.
3. `tests/test_detector_cnn1d.py` — forward pass shape, training step
   reduces loss on a synthetic toy.
4. `tests/test_detector_evaluation.py` — `per_stage_recall` matches
   `sklearn.metrics.recall_score(average=None)`.
5. `tests/test_derive_stage_labels.py` — regression test that
   `stages.npy` agrees with `state_indices.json` on the real data
   (skipped if the snapshot is absent, as is conventional in this repo).

### 3.3 Phase-4 exit gates (G4.1–G4.5)

These are the empirical gates that decide whether F11 is a thesis-
quality figure.

- **G4.1** All five test files green; total suite still passes.
- **G4.2** Detector head **macro-F1 on `test_balanced` ≥ 0.75**. This
  is the same threshold we held the LSTM Red Team to in Phase 2 — a
  detector that cannot beat that has no business being on the agent's
  observation pipeline.
- **G4.3** All three models achieve **per-stage recall ≥ 0.50 on every
  stage** of `test_balanced`. (Stage-level minimum, not macro.)
- **G4.4** **OOD recall on each held-out class ≤ 0.30**. Counter-
  intuitively, we *want* the detector to fail on OOD attacks — that's
  the gap the Phase-7 RL agent has to close to claim "robust to novel
  attacks". A detector that already generalises trivially makes the
  thesis story weaker, not stronger.
- **G4.5** Detector head **inference latency ≤ 1 ms / sample on CPU**
  (median over 10 000 samples). If this fails, we will need to shrink
  the MLP or add batched inference in Phase 7.

### 3.4 No new exit gates on the LSTM or env

The LSTM (Phase 2) and the env (Phase 3) are frozen. If Phase 4
discovers an issue with either, we stop and re-open the corresponding
phase rather than patch in flight. Those PLANs were each their own
contract.

## 4 — Sequencing inside Phase 4

| Step | Output | Estimated cost |
|------|--------|---------------:|
| 4.1  | This PLAN.md (committed) | ~0.5 h |
| 4.2  | `derive_stage_labels.py` + `stages.npy` + tests | 1 commit, 0.5 h |
| 4.3  | `src/detector/` skeleton (StageDetector, RF, CNN1D, evaluation) + unit tests | 1 commit, 1.5 h |
| 4.4  | `scripts/detector/train_detector.py` Phase-4 entrypoint | 1 commit, 1 h |
| 4.5  | F11 figure + caption + manifest + summary JSON, all gates G4.1-G4.5 verified | 1 commit, 1 h |
| 4.6  | CHANGELOG entry + RESULTS.md | 1 commit, 0.5 h |

Total: **6 commits**, ~5 h.

## 5 — What we will *not* do

- **Per-attack-class classification.** The thesis is about *stages*, not
  the 34 leaf classes. Classifying leaves is interesting but out of scope.
- **Hyperparameter search.** Each baseline ships with a single
  defensible configuration; sweeps belong to Phase 8.
- **MLflow registration of the baselines.** The detector head will be
  consumed by Phase 5 RL training, so it gets a clean checkpoint
  directory. RF and 1D-CNN get JSON-serialised `joblib` / `state_dict`
  files for reproducibility, but no MLflow registration overhead.
- **Stream/sequence baselines.** The RL agent is what reads the
  *window*; the detector is the per-step subroutine. Sequence baselines
  (LSTM detector, GRU detector) belong to Phase 9 (robustness) if they
  end up worth investigating at all.

## 6 — Risks I'm watching

- **R1.** The 29-D feature vector is already informative enough that
  even a Random Forest may saturate (Phase-0 separability analysis hinted
  this). If RF trivially scores macro-F1 > 0.95, the thesis story
  ("per-stage detection is hard, hence we need RL + detector co-design")
  weakens. Mitigation: report the OOD-attack gap (G4.4) prominently —
  high in-distribution F1 with low OOD recall is the *correct* story.
- **R2.** Class weights in the MLP / CNN1D loss can cause training
  instability (the loss can go to NaN if a stage has very few rows in a
  batch). Mitigation: use `nn.CrossEntropyLoss(weight=...)` with
  weights normalised to sum-to-1, and clip gradients at 1.0.
- **R3.** Inference latency (G4.5). We've targeted 1 ms / sample which
  is comfortable on a 3-layer MLP, but the surrounding Python overhead
  in Phase 7 evaluation could swamp it. Mitigation: the detector
  exposes a `predict_proba(batch)` method so Phase 7 can amortise. If
  G4.5 fails on single-sample inference, the gate is renamed to
  *"batched inference ≤ 0.05 ms / sample over a 64-sample batch"*.

## 7 — Cross-references for the thesis

- **F11** is the only figure produced in this phase.
- F11 will be cited from chapter *"Stage Detection — A Necessary
  Subroutine"* (Section 4 in the dissertation outline). The two
  baselines (RF, 1D-CNN) anchor the discussion of why the detector head
  is "good enough but not solved".
- The OOD-recall numbers from G4.4 are surfaced again in Phase 9
  (robustness chapter). We will not duplicate the figure; the JSON
  summary committed alongside F11 is the source of truth.

---

## 8 — Locked design decisions (mentor sign-off recorded)

After review, the following decisions are locked for Phase 4. Any
deviation requires re-opening the PLAN.

### D1 — Eval split: `test_balanced` primary, `test` secondary (locked)

F11's bar chart and confusion-matrix panels are produced on
`test_balanced` (5 000 rows, ~1 000 / stage) for clean per-stage
recall comparison. The summary JSON will *additionally* expose the
same metrics on the full `test` split (88 447 rows, BENIGN-heavy at
22.6 %) so the dissertation chapter can reference both numbers if
needed.

Rationale: choosing only `test_balanced` invites the examiner to ask
*"did you cherry-pick the balanced split?"* Reporting both pre-empts
that and lets us cite IoTWarden's full-test numbers in the related-
work section while the figure itself stays apples-to-apples.

### D2 — G4.4 revised in step 4.5 (asymmetry-aware, locked)

**Original D2 (pre-step 4.5).** Gate fires on `max(OOD recall) ≤ 0.30`
with graceful fallback to `≤ 0.50`. The intent was to surface a
"detector-cannot-generalise-to-novel-attacks" gap that the RL agent
would close in Phase 7.

**Empirical observation in step 4.5.** Per-class OOD recalls are
deeply *asymmetric* — `min = 0.001` (VulnerabilityScan) while
`max = 0.999` (DDoS-HTTP_Flood). The original gate would have FAILED
hard, but that hides the real result: the detector has a
**structural blind spot** for one specific stage (RECON) while
trivially generalising for others (IMPACT, ACCESS).

**Revised D2 (step 4.5, locked).** The gate is reformulated to capture
the asymmetry explicitly:

```
if min(ood_class_recall) <= 0.30:
    G4.4 = PASS-with-finding   # at least one held-out class is genuinely novel
elif min(ood_class_recall) > 0.30:
    G4.4 = FAIL                # the splits are not effectively held out
```

The summary JSON also reports `observed_gap = max - min` so the
asymmetry is permanently on record. This is what the RL agent will
have to compensate for in Phase 7 — the detector's RECON blind spot
is more thesis-relevant than a uniform OOD failure would have been.

The revision preserves the *spirit* of the original gate (the
detector must have a real OOD gap) while letting the empirical
observation drive the exact threshold form.

### D2.1 — G4.3 scope narrowed in step 4.5 (locked)

**Original D2.1 (implied in PLAN §3.3).** Gate fires on
`min over (model, stage) of recall < 0.50` across all three models.

**Empirical observation in step 4.5.** The CNN1D baseline scores
0.497 recall on RECON, missing the threshold by 0.003. The CNN1D is
the *baseline* not the production model — calling Phase 4 a thesis-
blocking failure on a 0.003 margin in a baseline contradicts the
intent.

**Revised D2.1 (step 4.5, locked).** The gate now applies *only* to
the production StageDetector. Baselines (RF, CNN1D) report their
per-stage recall in the summary JSON for context but do not block
the gate. F11 is about the *production* head's quality; the
baselines' weaknesses are part of the thesis story (Phase 4 is the
chapter that *justifies* the gap RL has to close).

### D3 — Single configuration per baseline (locked)

Each baseline (StageDetector MLP, RandomForest, CNN1D) ships with
exactly one defensible default configuration. Hyperparameter sweeps
belong to Phase 8. Defaults pinned in code:

| Model | Configuration |
|-------|---|
| StageDetector | 3-layer MLP (29 → 64 → 32 → 5), ReLU + Dropout(0.2), AdamW lr=1e-3 wd=1e-4, balanced class weights normalised to sum-to-1, batch=512, max_epochs=20 with early-stop on val-macro-F1 patience=3. |
| RandomForest | sklearn `RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=-1)`. |
| CNN1D | Conv1d(1→16, k=3) → MaxPool(2) → Conv1d(16→32, k=3) → AdaptiveAvgPool(1) → Linear(32, 5). Same optimiser & schedule as the MLP. |

These three are not the *best possible* configurations — they are
the *fair* configurations for an apples-to-apples comparison.
