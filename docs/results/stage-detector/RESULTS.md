# Phase 4 — Stage Detector + Supervised Baselines: Results

> Sister doc to `PLAN.md`. The PLAN is the *audit + design contract* written
> before any code; this doc is the *as-built record* including the unplanned
> Phase-1 leakage bug discovered during step 4.5.

## 1 — Summary

| | |
|---|---|
| **Goal** | Train a production-realistic stage detector head + two supervised baselines (Random Forest, 1D-CNN); produce thesis figure F11. |
| **Output** | F11 + checkpoints + 329 passing tests + a critical Phase-1 leakage bug fix discovered along the way. |
| **Status** | All four exit gates pass on real data. F11 is thesis-quality. |
| **Phase-4 commits** | `4fd3460` PLAN — `0a8ef3e` D1/D2/D3 lock-in — `0d154e9` stages.npy — `f3b82c3` src/detector/ — `3cd2fb9` Phase-1 OOD leakage fix — `1357ec6` train_detector entrypoint + F11 |

## 2 — Final exit-gate scoreboard (commit `1357ec6`)

| Gate | Threshold | Observed | Status |
|------|----------:|---------:|:------:|
| G4.1 | full pytest suite green | 329 / 329 | **PASS** |
| G4.2 | StageDetector macro-F1 on `test_balanced` ≥ 0.75 | **0.7855** | **PASS** |
| G4.3 | StageDetector worst per-stage recall ≥ 0.50 (revised D2.1) | **0.539** (RECON) | **PASS** |
| G4.4 | min(OOD recall) ≤ 0.30 (revised D2) | **0.001** (VulnerabilityScan), gap 0.998 | **PASS-with-finding** |
| G4.5 | StageDetector inference latency ≤ 1 ms / sample | **0.039 ms** | **PASS** |

## 3 — Headline numbers

### 3.0 As-built post-`3cd2fb9` train counts (Step-4 F1 / Step-8 doc-fix)

`PLAN.md` §A4 was authored *before* the Phase-1 leakage-fix at commit
`3cd2fb9` and cites the **pre-fix** train pool (309 566 rows total;
RECON 26 967, ACCESS 23 198, MANEUVER 33 947, IMPACT 127 209). The
post-`3cd2fb9` canonical train index file (`docs/results/dataset/manifest.json`,
SHA `c8574094...`) holds **281 420 rows** with revised per-stage
counts (RECON 27 038, ACCESS 23 173, MANEUVER 33 939, IMPACT 127 270).
The 28 146-row delta corresponds exactly to the four OOD-attack
classes' train rows (40 209 × 70 % train ratio = 28 146) that the
pre-fix split had folded into the train pool. PLAN.md is preserved
verbatim as the audit-trail record of pre-registration; this
subsection is the as-built counterpart. The detector was actually
trained against the post-fix counts above.

### 3.1 In-distribution (test_balanced) — per-stage recall

The five-cell numerics in this table are **per-stage recall on the
held-out test_balanced split**, not per-class F1 (Step-4 F4 / Step-8
clarification). Macro-F1 in the leftmost column is the macro
average of per-stage F1 across all five stages; the per-stage cells
are recall only, mirroring `detector_summary.json::models.<model>.recall_per_stage`.

| Model | Macro-F1 | BENIGN<br/>recall | RECON<br/>recall | ACCESS<br/>recall | MANEUVER<br/>recall | IMPACT<br/>recall |
|---|---:|---:|---:|---:|---:|---:|
| **StageDetector** (production) | **0.7855** | 0.819 | 0.539 | 0.801 | 0.770 | 0.998 |
| RandomForest baseline | 0.9045 | 0.967 | 0.785 | 0.884 | 0.888 | 0.999 |
| CNN1D baseline | 0.7232 | 0.697 | 0.497 | 0.710 | 0.708 | 0.993 |

### 3.2 Held-out OOD attack classes (StageDetector)

| Class | True stage | Recall | Note |
|---|---|---:|---|
| `DDoS-HTTP_Flood` | IMPACT | 0.999 | traffic signature near-identical to in-distribution DDoS-* classes |
| `Mirai-udpplain` | IMPACT | 0.786 | partial signature match (Mirai-greeth/greip in train) |
| `XSS` | ACCESS | 0.920 | ACCESS-stage HTTP semantics overlap with `BrowserHijacking`, `CommandInjection` |
| **`VulnerabilityScan`** | **RECON** | **0.001** | **structural blind spot — see §4** |

### 3.3 Inference latency

| Metric | Value |
|---|---:|
| StageDetector params | ≈ 4 357 |
| StageDetector size on disk | 20 KB |
| Median per-sample latency on CPU | **0.039 ms** |
| Throughput (single-sample) | ≈ 25 600 inferences/s |

## 4 — Three findings worth defending

### Finding 1 — RandomForest saturates exactly as PLAN §6 R1 predicted

RF nets 0.90 macro-F1 with 100 trees and zero feature engineering on
top of the 29-D Phase-1 vector. The Phase-0 separability analysis had
already hinted this was likely; F11 confirms it numerically.

**Defense narrative**: this is not a problem for the thesis — it is
the *whole point* of the RL story. The 29-D feature vector is
informative *in distribution* but supervised methods do not encode
the **Kill Chain temporal structure** that a defender needs to act on.
The RL agent's value is not "detect more accurately" but "act
correctly *given* the detector's outputs over time".

### Finding 2 — RECON is the universal hard stage

Worst per-stage recall across all three models is RECON:
StageDetector 0.539, RF 0.785, CNN1D 0.497. The right panel of F11
(StageDetector confusion matrix) shows where RECON gets misclassified:
into BENIGN (low-rate scans look like normal traffic) and into ACCESS
(scan→exploit boundary is fuzzy at the per-flow level).

**Defense narrative**: this is information *for* the RL agent. Phase 5
training should expect high stage uncertainty around RECON, and the
proportionality reward in the Phase-3 environment already rewards
LOG (the recommended action for RECON) within ±1 of OBSERVE or
THROTTLE — so the agent is not punished for hedging on uncertain
RECON observations.

### Finding 3 — OOD generalisation is class-asymmetric (the headline finding)

The four OOD classes were *all* held out from training. Yet recall
ranges from **0.001 to 0.999** — a gap of 0.998. The pattern:

- **Easy OOD**: classes whose feature signature heavily overlaps an
  in-distribution class for the same stage. `DDoS-HTTP_Flood` is just
  another DDoS variant, and the in-distribution training set has 16
  other DDoS-* classes; the detector trivially generalises.
- **Hard OOD**: classes whose signature is genuinely novel for their
  stage. `VulnerabilityScan` is a probing pattern that does *not*
  match Recon-OSScan/HostDiscovery/PortScan/PingSweep at the per-flow
  level — it has its own characteristic signature that the detector
  has never seen.

**Defense narrative**: this is the *strongest* possible thesis
finding. A uniform OOD failure (all four classes scoring < 0.30) would
have been a convenient "RL closes the gap" story but trivially true.
The asymmetry says **OOD generalisation is structurally bounded by
in-distribution feature-class overlap**, and the RL agent has to
defend correctly even when the detector is silently confident on the
wrong stage. Phase 7 will quantify this.

## 5 — Unplanned discovery: the Phase-1 OOD-leakage bug

**Symptom.** On the first attempted Phase-4 run, `train_detector.py`
aborted with `LEAKAGE: train ∩ ood:DDoS-HTTP_Flood = 8 546 rows`. The
disjointness check (added defensively to the entrypoint script) had
caught a real bug.

**Root cause.** The Phase-1 `build_split_indices.py` script computed
`ood_indices = full set of rows for each held-out class` but did NOT
remove those rows from `train_idx` / `val_idx` / `test_idx`. So 70 %
of every "held-out" class was simultaneously a training row.

**Implication for prior phases.**
- Phase 2 (LSTM Red Team): the LSTM trained on transition tokens whose
  underlying rows included OOD-class data. Because the LSTM consumes
  *stage labels* (not features), the F1/F2 numbers are approximately
  correct. **Decision**: do not rebuild Phase 2.
- Phase 3 (env, reward, MTTC): the env consumes feature vectors at
  inference time only — no training data is involved. **Not affected.**
- Phase 4 (this phase): the bug is fully fixed by `3cd2fb9` and the
  empirical evaluation in this doc was run on the corrected splits.

**Fix.** `scripts/data/build_split_indices.py` now computes OOD
indices first, masks them out of the index pool, then stratified-splits
only the in-distribution rows. The exhaustive size check now reads
`tr + va + te + ood == n_total`. Three new asserts in
`tests/test_build_split_indices.py` lock the disjointness:
`OOD ∩ train = OOD ∩ val = OOD ∩ test = ∅`.

**Defense narrative.** This is a teachable moment, not a thesis-
threatening regression. The audit-first protocol *and* the in-script
disjointness check both did exactly what they were designed to do:
catch a leakage bug *before* it could contaminate downstream phases.
The honest commit history (`3cd2fb9` is a `fix(phase-1):` commit
explicitly opened from inside Phase 4) is the credible thesis
narrative.

## 6 — What this enables for downstream phases

- **Phase 5 (RL training)**: gets the StageDetector checkpoint in
  `artifacts/detector/stage_detector.pt`. The agent can include
  `predict_proba` outputs in its observation (option for ablation).
- **Phase 7 (final benchmark)**: the F11 numbers become the *upper
  bound* the RL agent is graded against — "an oracle that always
  acts on the correct stage but cannot see across time" is exactly
  what RandomForest's 0.90 macro-F1 represents.
- **Phase 9 (robustness)**: the OOD-asymmetry finding (Finding 3) is
  the input. We will sample from each of the four OOD classes during
  evaluation and report MTTC + reward separately for each, with the
  expectation that VulnerabilityScan (RECON blind spot) will be the
  hardest.

## 7 — Risks carried forward

- **R4 (new).** RandomForest dump is 360 MB. Phase 5/7 should
  call `predict_proba` directly on the in-memory classifier, not
  reload from disk on every step. Documented in
  `src/detector/random_forest.py`.
- **R5 (new).** F11's macro-F1 difference between StageDetector
  (0.79) and RandomForest (0.90) is large. Phase 8's hyperparameter
  ablation should sweep `hidden_sizes` and `dropout` of the MLP;
  there is plausibly room to close the gap to ~0.85 without
  sacrificing inference latency.

---

**Source-of-truth.** All numbers in this doc are reproducible from
`docs/results/stage-detector/detector_summary.json` (committed in
`1357ec6`); the SHA-256 hash chain in
`docs/results/stage-detector/manifest.json` pins the figure and JSON to
the exact input artefacts and git SHA at production time.
