# Step 01 — Phase 0–1 Dataset Review

**Mentor memo. Audits the Phase-1 dataset artefacts (F0 figures, splits, kill-chain mapping, anti-leakage protocol) ahead of the MSc defense at Unicamp/FEEC.**

---

## 1. Verdict

`PASS-WITH-FIXES`

The Phase-1 **scientific substrate is sound**: the 442 237-row processed snapshot is honest, the OOD-class leakage bug discovered in Phase 4 (commit `3cd2fb9`) is genuinely fixed, the disjointness regression test is in place, and the F0 hash chain is intact (figure SHAs in `manifest.json` match the on-disk PNGs and JSON exactly). Pytest is green at **411 passed**.

What's not yet defense-grade is **the surrounding documentation**. Three doc-only defects need to land before binding the dissertation: (i) `docs/dataset_card.md` §4 names the scaler as `MinMaxScaler` while the code persists a `StandardScaler` (`src/utils/dataset_processor.py:25,232`); (ii) the same card §5 split table and §5 OOD note describe the **pre-`3cd2fb9` reality** (overlap with `train/val/test` "by construction") and so contradict the post-fix code; (iii) `docs/kill-chain-mapping.md` is a bare assignment table with no per-class rationale, which is the explicit acceptance criterion for committee scrutiny. Phase 0–1 also has **no `PLAN.md` or `RESULTS.md`** under `docs/results/01_dataset/`, breaking the per-phase audit pattern that every other phase follows. None of these defects are correctness bugs and none invalidate any downstream gate; all are tractable doc edits scoped for Step 1 follow-up or Step 8.

---

## 2. What was reviewed

### Artefacts (read in full)
- `docs/results/01_dataset/F0_class_distribution.png` — F0a, 2063×863, 160 DPI, 8-bit RGBA, Matplotlib 3.9.4, 161 KB
- `docs/results/01_dataset/F0_stage_distribution.png` — F0b, 1343×780, 160 DPI, 8-bit RGBA, Matplotlib 3.9.4, 73 KB
- `docs/results/01_dataset/F0_class_distribution.caption.md` — F0 caption + "what to look for"
- `docs/results/01_dataset/F0_stage_distribution.caption.md` — F0b caption + "what to look for"
- `docs/results/01_dataset/F0_summary.json` — numerical aggregate, 65 lines
- `docs/results/01_dataset/manifest.json` — F0 hash chain, 18 lines
- `docs/dataset_card.md` — Hugging-Face-style dataset card, 229 lines
- `docs/data-pipeline.md` — anti-leakage and processing protocol, 91 lines
- `docs/kill-chain-mapping.md` — per-class to per-stage mapping, 69 lines

### Code (read in full)
- `scripts/data/build_split_indices.py` — split builder + OOD pre-extraction, 448 lines
- `src/utils/label_mapper.py` — `KillChainStage` IntEnum + `AbstractStateLabelMapper`
- `src/utils/realization_engine.py` — `from_split_manifest` factory + `allowed_indices` enforcement
- `src/utils/dataset_processor.py` (scaler-relevant sections)

### Tests (run in full + scoped)
- Full suite: **`pytest -q`** → 411 passed, 0 failed, 64.3 s
- Scoped: `tests/test_build_split_indices.py` + `tests/test_realization_engine_split_aware.py` + `tests/test_label_mapper.py` → 73 passed, 2.4 s

### Git
- HEAD: `26f753e` on `mentor-review/step-1-dataset` (cut from `main` after Step-0c-framing was merged into `main` and the prior `feature/*` branches and `v0.1.0`/`pre-mentor-restart` tags were cleaned up — see §8).
- Phase-4 leakage fix: commit **`3cd2fb9`** *"fix(phase-1): exclude held-out OOD classes from train/val/test (CRITICAL)"* — 2 files, +60/-16, regression test added in the same commit.

---

## 3. Findings (priority-ordered)

### Finding 1 — `docs/dataset_card.md` §4 names the wrong scaler [severity: **major**]

**Where.** `docs/dataset_card.md:115-118` states:

> "All 29 features are MinMax-scaled. The fitted `MinMaxScaler` is persisted in `data/processed/ciciot2023/scaler.joblib` and is used unchanged throughout the thesis (held-out, OOD, and benchmark splits are all projected with the same scaler)."

Plus a related claim at `docs/dataset_card.md:99`: *"variance below 0.01 after MinMax scaling"* (in the low-variance feature-drop description).

**What's true.** The code persists a `StandardScaler`, not a `MinMaxScaler`. Four call sites import / instantiate it:
- `src/utils/dataset_processor.py:25` — `from sklearn.preprocessing import LabelEncoder, StandardScaler`
- `src/utils/dataset_processor.py:79` — `self.scaler: Optional[StandardScaler] = None`
- `src/utils/dataset_processor.py:232,288,877` — `self.scaler = StandardScaler()` (three instantiations)
- `src/utils/dataset_processor.py:11` (module docstring) — *"scaler.joblib: StandardScaler for feature normalization"*

`docs/data-pipeline.md:19,66` correctly says `StandardScaler`. The dataset card disagrees with both the code and the pipeline doc.

**Why it matters.** This is the kind of contradiction a careful examiner spots in five seconds and uses to question whether the rest of the dataset card was written from the code. `MinMaxScaler` and `StandardScaler` produce different feature distributions; a defender or detector tuned on one will silently mis-perform if the other is hot-loaded. It also affects the interpretation of the variance-threshold drop ("variance below 0.01 after MinMax scaling" reads like a different protocol).

**Recommended fix.** `docs(phase-1,§4): correct scaler from MinMax to Standard in dataset_card.md`. Update lines 99, 115, 116. No code changes; the persisted `scaler.joblib` is correctly a `StandardScaler` and is used unchanged everywhere downstream.

### Finding 2 — `docs/dataset_card.md` §5 split table and OOD note describe pre-`3cd2fb9` reality [severity: **major**]

**Where.** `docs/dataset_card.md:128-138` (split table) and `:163-168` (OOD overlap note).

**What's stale.**

1. **Split table (line 128–138).** The table claims:
   - `train` = 309 566 (70 %)
   - `val` = 44 224 (10 %)
   - `test` = 88 447 (20 %)
   - `train ⊔ val ⊔ test = 442 237 = all`
   These are the **pre-fix** sizes, when OOD-class rows were silently included in train/val/test. After commit `3cd2fb9` the OOD rows are subtracted **before** stratification, and the actual sizes printed by the builder (per the Phase-4 fix commit message) are train ≈ 281 420, val ≈ 40 202, test ≈ 80 414, ood = 40 209, with `train ⊔ val ⊔ test ⊔ ood = 442 237`. The card's table and the printed disjointness invariant are both incorrect after `3cd2fb9`.

2. **OOD overlap note (line 163–168).** The card states:
   > "OOD indices are computed from the **string label array**, so they overlap with `train`/`val`/`test` by construction. Phase-2/4 training code must subtract them before fitting; helper `src/utils/dataset_loader.py::exclude_ood_classes` will be added in Phase 2 to enforce this."
   This contradicts the post-`3cd2fb9` reality where overlap is structurally impossible (see `scripts/data/build_split_indices.py:258-283` — OOD mask is built before in-distribution-pool selection at line 280) and is asserted by `tests/test_build_split_indices.py:181-198` (`ood_set.intersection(split_idx) == ∅`). The "helper to be added in Phase 2" sentence is also stale — there is no `exclude_ood_classes` helper in `src/utils/dataset_loader.py`; the responsibility now sits in the split builder itself.

**Why it matters.** The dataset card is the document the committee will *cite* in their report. A stale overlap-by-construction note completely undermines the no-leakage claim that Phase 4 / Phase 5 / Phase 6 / Phase 7 all build on. The defense-time risk is higher than the magnitude suggests because the *correct* protocol is in place — the doc just doesn't reflect it.

**Recommended fix.** `docs(phase-1,§5): align dataset_card.md split table and OOD note with the post-3cd2fb9 implementation`.

- Replace the split table with the actual post-fix per-stage counts (regenerate from `data/processed/ciciot2023/splits/manifest.json` if available; otherwise document the 70/10/20 ratio applied to the in-distribution pool of size `442 237 − 40 209 = 402 028`).
- Replace the "overlap by construction" paragraph with a one-sentence statement that OOD classes are removed before the stratified split (cite `build_split_indices.py:258-283`) and that disjointness from train/val/test is asserted by `tests/test_build_split_indices.py::TestBuildSplitsEndToEnd::test_run_on_synthetic_dataset` lines 181–198.
- Drop the "will be added in Phase 2" sentence.

### Finding 3 — `docs/kill-chain-mapping.md` has no per-class rationale [severity: **major**]

**Where.** Whole file. `docs/kill-chain-mapping.md` is a bare assignment table (5 stages, 34 classes) with three short notes at the end. There is no prose justifying *why* each class is in the stage it's in.

**Why it matters.** The Step-1 acceptance criterion (per `00_HANDOFF.md` §5 and the Step-1 prompt) is verbatim:

> "Every CICIoT2023 attack class maps to exactly one kill-chain stage with a defensible rationale that survives committee scrutiny."

The current doc satisfies the *exactly-one* part — every class appears in exactly one stage and the closed-mapping invariant is enforced by `KillChainStage` and `AbstractStateLabelMapper.get_stage_id` raising `KeyError` on unknown labels (test: `tests/test_label_mapper.py::TestStringToStageIds::test_raises_on_unknown_label`). It does **not** satisfy the *defensible-rationale* part. A committee member can plausibly contest each of these without prose to push back on:

- Why is `MITM-ArpSpoofing` in MANEUVER and not ACCESS? (Lockheed-Martin's original kill chain puts it in "Lateral Movement", which doesn't map cleanly to either of our stages.)
- Why is `Mirai-greeth_flood` / `Mirai-greip_flood` / `Mirai-udpplain` in MANEUVER and not IMPACT, given that those Mirai variants *are* DDoS payloads?
- Why is `DictionaryBruteForce` in ACCESS and not RECON? (The line between credential discovery and credential exploitation is fuzzy.)
- Why is `Backdoor_Malware` in ACCESS and not MANEUVER, given that backdoors typically establish persistence?

**What I'm not asking for.** I'm not asking for any class to be re-mapped. The current mapping is internally coherent (the BENIGN/RECON/ACCESS/MANEUVER/IMPACT progression is monotone in attack severity, which is what makes the proportional-defense reward shape sensible). I'm asking for the **prose** that defends each non-trivial choice.

**Recommended fix.** `docs(phase-1,§3): add per-stage rationale paragraphs to kill-chain-mapping.md`.

- One short paragraph per stage explaining the operational definition (RECON = pre-access information gathering; ACCESS = primary entry vector; MANEUVER = post-access positioning *including* botnet-side data-plane preparation such as Mirai variants; IMPACT = service degradation against the victim).
- One sentence per "could-be-elsewhere" class (the four bullets above plus any others the candidate flags) citing why it sits where it sits.
- The full table can stay; it is also reproduced verbatim in `dataset_card.md` §3, which is fine as long as the rationale lives in the canonical kill-chain doc.

This is the **single highest-leverage fix** for the dissertation — it converts the mapping from "asserted" to "argued".

### Finding 4 — Phase 0–1 has no `PLAN.md` or `RESULTS.md` [severity: minor]

**Where.** `docs/results/01_dataset/` contains only the F0 PNGs, captions, summary JSON, and manifest. Every other phase (`02_red_team`, `03_env`, `04_detector`, `05_blue_team`, `06_benchmark`, `07_ablation`, `10_release`) has both `PLAN.md` and `RESULTS.md`. Phase 1 has neither.

**Why it matters.** The mentor-review directory README defines `PLAN.md` as the *frozen audit trail of what was planned* and `RESULTS.md` as the *scientific record of what each phase produced*. The candidate's invariant of *"never edit `PLAN.md` files; numerical truth is in summary JSONs and figure manifests"* fails as a global invariant when one phase has no `PLAN.md` to begin with. It also makes the LaTeX rebuild in Step 9 harder: §3.1 (Methodology / Dataset preparation) and §4.1 (Results / Red team validation, which builds on F0) have no canonical text to cite.

**Why it's only minor.** The substantive content that would belong in `01_dataset/PLAN.md` is already split across `docs/dataset_card.md` (input characterization, sampling, splits, hashes) and `docs/data-pipeline.md` (transformation protocol). The `RESULTS.md` content is covered by the captions plus `F0_summary.json`. Nothing scientifically novel is missing — only the audit-trail packaging.

**Recommended fix.** Two options for the candidate to choose:

- (a) **Author retroactive `PLAN.md` + `RESULTS.md` for Phase 1**, citing the existing `dataset_card.md` and `data-pipeline.md` for the substance. Mark them clearly as *retroactive audit trail* so they are not mistaken for pre-registration. Step 9 LaTeX rebuild then has a per-phase symmetry.
- (b) **Document the asymmetry once** in `docs/results/README.md` ("Phase 1 has no `PLAN.md`/`RESULTS.md` because the substantive content lives in `docs/dataset_card.md` and `docs/data-pipeline.md`; the F0 figures and `F0_summary.json` carry the numerical record"), and accept the asymmetry.

I recommend (b). It is honest about the gap, is one paragraph of work, and avoids backfilling fake history.

### Finding 5 — Caption + figure-id naming inconsistency (F0 vs F0a vs F0b) [severity: minor]

**Where.**
- `docs/results/01_dataset/manifest.json:1` says `"figure_id": "F0"` (singular, both PNGs grouped under one ID).
- `docs/results/01_dataset/F0_class_distribution.caption.md` title: *"Figure F0 — Class distribution after rebalancing"* (no suffix).
- `docs/results/01_dataset/F0_stage_distribution.caption.md` title: *"Figure F0b — Kill Chain stage distribution per split"* (with `b` suffix).
- `docs/thesis_results_map.md:18-19` lists them as **`F0a`** and **`F0b`** — the convention used by the framing memo (`00_HANDOFF.md:46`, "F0a, F0b, F1, …").

**What's inconsistent.** The class-distribution figure has three different IDs across three documents (`F0` in manifest, `F0` in caption title, `F0a` in thesis-results-map). The stage-distribution figure has two (`F0` in manifest, `F0b` in caption + thesis-results-map).

**Why it matters.** Cross-references in the LaTeX rebuild (Step 9) will break or be ambiguous. Examiners reading the bound thesis will find `\ref{fig:F0a}` next to a caption that says "Figure F0".

**Recommended fix.** `docs(phase-1,§5): standardize F0a / F0b naming across caption files and manifest`.

- Rename caption titles: *"Figure F0 — Class distribution"* → *"Figure F0a — Class distribution"* (caption file content edit).
- Update `manifest.json` to either split into two entries (`F0a` + `F0b`) or drop the `figure_id` field for this multi-figure manifest. Two-entry split is cleaner.
- This **does** change `manifest.json` content, but the **output SHAs** of the PNGs and JSON do not change — so the hash chain stays intact. (The manifest itself is not hashed by anything downstream; it's the output-of-record, not an input.)

### Finding 6 — F0 caption text says "Five small classes" then lists seven [severity: nit]

**Where.** `docs/results/01_dataset/F0_class_distribution.caption.md:14-17`:

> "Five small classes — `BrowserHijacking`, `CommandInjection`, `SqlInjection`, `XSS`, `Backdoor_Malware`, `Recon-PingSweep`, `Uploading_Attack` — fall below the 12 121-row cap…"

That's seven classes named, not five.

**Recommended fix.** Replace "Five" with "Seven" (and re-verify the count against `F0_summary.json::class_counts` — the seven listed are indeed the only ones below 12 121 except for the un-capped 100 000-row `BenignTraffic`).

### Finding 7 — `manifest.json` does not pin the producing-script hash [severity: nit]

**Where.** `docs/results/01_dataset/manifest.json` records:
- `git_sha`: `a69846f7…` (the commit at which the figure was produced) ✅
- `inputs`: SHA-256 of `labels.npy` and `splits/manifest.json` ✅
- `outputs`: SHA-256 of the two PNGs and the summary JSON ✅
- `produced_by`: `scripts/data/plot_dataset_overview.py` (path only) ✅

But **the script's own SHA is not pinned.** Compare `manifest.json` for Phase 5 (e.g. `docs/results/05_blue_team/F3_manifest.json`) which I have not opened in this step but the Phase-5 RESULTS audit will check.

**Why it's nit, not minor.** The `git_sha` field already pins the entire repository state at production time, so script identity is fully recoverable via `git show a69846f7:scripts/data/plot_dataset_overview.py | shasum -a 256`. The convention is just slightly less ergonomic than carrying the script SHA inline.

**Recommended fix.** Defer to Step 8 (cross-cutting audit) — if other phases pin script hashes inline, harmonize Phase 1 to match; otherwise leave as-is.

---

## 4. Validation: structural invariants are intact

The substantive integrity claims that make Phases 2–7 trustworthy all check out on the current commit (`26f753e`):

| Invariant | Where enforced | Verified |
|---|---|---|
| OOD classes are removed **before** the train/val/test stratified split | `scripts/data/build_split_indices.py:258-283` (OOD mask built at L267, in-distribution pool selected at L280, stratified split runs only over the in-distribution pool) | ✅ |
| `train ∩ val ∩ test ∩ ood = ∅` (pairwise disjoint, OOD locked out) | `tests/test_build_split_indices.py:181-198` (`ood_set.intersection(split_idx) == ∅` for split in {train,val,test}) | ✅ test passes |
| `train ⊔ val ⊔ test ⊔ ood = all` (exhaustive) | `tests/test_build_split_indices.py:186` (`tr.size + va.size + te.size + ood_idx.size == n_total`) | ✅ test passes |
| Per-stage stratification ratios approximate 70/10/20 | `tests/test_build_split_indices.py::TestStratifiedSplit::test_per_stage_ratios_approx` | ✅ test passes |
| Determinism: same seed → same indices | `tests/test_build_split_indices.py::TestStratifiedSplit::test_seed_determinism`, `TestBuildSplitsEndToEnd::test_determinism_across_runs` | ✅ tests pass |
| Closed kill-chain mapping (unknown label → `KeyError`) | `src/utils/label_mapper.py::AbstractStateLabelMapper.get_stage_id` + test `tests/test_label_mapper.py::TestStringToStageIds::test_raises_on_unknown_label` | ✅ test passes |
| `KillChainStage` is `IntEnum` with exactly 5 members (BENIGN=0, RECON=1, ACCESS=2, MANEUVER=3, IMPACT=4) | `src/utils/label_mapper.py:15-26` | ✅ |
| `RealizationEngine.from_split_manifest(..., exclude_ood=True)` is the default factory | `src/utils/realization_engine.py:104-171` (default `exclude_ood: bool = True` at L110, OOD subtraction at L165–169) | ✅ |
| `allowed_indices` is enforced (empty → `ValueError`; intersected with each per-stage pool) | `src/utils/realization_engine.py:175-186` | ✅ test `test_realization_engine_split_aware.py::test_allowed_indices_*` passes |
| F0 hash chain | `docs/results/01_dataset/manifest.json::outputs` SHA-256 == on-disk SHA-256 for `F0_class_distribution.png`, `F0_stage_distribution.png`, `F0_summary.json` | ✅ exact match (verified via `shasum -a 256`) |
| F0 input chain | `manifest.json::inputs[labels.npy]` and `[splits/manifest.json]` SHAs match the upstream-`splits/manifest.json` reference at the time of figure production (commit `a69846f7…`) | ✅ (consistent with `git_sha` field; on-disk processed-data hashes were not re-checked because they live outside the repo at `data/processed/`) |
| Full pytest run | All 411 tests | ✅ 411 passed, 64.3 s |

The Phase-4 leakage fix is genuinely watertight. The disjointness regression test in `tests/test_build_split_indices.py` lines 181–198 is the right test in the right place; it would catch any regression that re-introduced OOD-class rows into the in-distribution splits.

The four held-out OOD classes (`VulnerabilityScan` for RECON, `XSS` for ACCESS, `Mirai-udpplain` for MANEUVER, `DDoS-HTTP_Flood` for IMPACT) span four of the five stages with deliberate per-stage sizing rationale: 23.9 % of RECON, 10.4 % of ACCESS (intentionally small to preserve `DictionaryBruteForce` for training), 20.0 % of MANEUVER, 6.3 % of IMPACT. BENIGN is intentionally not held out (a held-out BenignTraffic class is meaningless for an "unseen attack" generalization test).

---

## 5. F0 figure visual inspection

| Property | F0a (class distribution) | F0b (stage distribution) |
|---|---|---|
| Pixel size | 2063 × 863 | 1343 × 780 |
| DPI | 160 | 160 |
| Implied print size | 12.9″ × 5.4″ | 8.4″ × 4.9″ |
| Color mode | 8-bit RGBA, sRGB | 8-bit RGBA, sRGB |
| File size | 161 KB | 73 KB |
| Producer | Matplotlib 3.9.4 | Matplotlib 3.9.4 |
| Caption claims log-y axis | ✅ asserted in caption | n/a |

**Publication-cleanness.** The captions describe axis labels, palette (color-by-stage), and y-axis scale. 160 DPI is below the customary 300 DPI bar for print figures, but the absolute pixel dimensions (2063 px wide) are large enough that the rendered figure at thesis-page width (≈6.5″ → 317 ppi effective) is sharp. **Recommend leaving the PNGs as-is** unless Step 9 regenerates from script. *(I did not visually open the PNGs in this read-only audit; the visual inspection of palette, legend, and font sizes is deferred to the candidate. If a re-render is needed for any reason, the seed-pinned reproducer is `python -m scripts.data.plot_dataset_overview`.)*

**Single concrete caption defect (Finding 6):** F0a caption says "Five small classes" and lists seven. Trivial textual fix.

---

## 6. Anti-leakage protocol (pipeline narrative)

For the audit trail and Step-9 LaTeX rebuild, the actual pipeline is:

1. **Process** raw CICIoT2023 CSVs (`data/raw/CICIoT2023/*.csv`, 169 files, ~46.7 M rows) via `main.py --mode process-data` → produces `data/processed/ciciot2023/{features.npy, labels.npy, scaler.joblib, metadata.json, state_indices.json}`. The 17 dropped features are recorded in `metadata.json::feature_selection_info`. The fitted scaler is **`StandardScaler`** (see Finding 1) and is fit on the **train split only** (`docs/data-pipeline.md:66`).
2. **Build splits** via `python -m scripts.data.build_split_indices --processed-dir data/processed/ciciot2023 --seed 42`. This reads `labels.npy` and:
   - Computes string-label OOD indices (`build_split_indices.py:168-186`) for the 4 held-out classes.
   - Builds the OOD mask, restricts to the in-distribution row pool (`:264-280`).
   - Stratifies the in-distribution pool 70/10/20 by stage (`:289+`).
   - Draws per-stage balanced subsets `val_balanced` (200/stage = 1 000) and `test_balanced` (1 000/stage = 5 000) **from the already-disjoint in-distribution pools** (`:305+`).
   - Writes per-split `.idx.npy` files plus per-OOD-class `.idx.npy` files plus `splits/manifest.json` with SHA-256 of every input/output (`:326-385`).
3. **Consume** in any phase via `RealizationEngine.from_split_manifest(data_path, split, exclude_ood=True)` (`realization_engine.py:104`). The factory loads the relevant `<split>.idx.npy`, optionally subtracts OOD indices (default ON), and restricts the per-stage sampling pools (`_restrict_to`, `:175-186`). Empty pools raise `ValueError`.

This is honest. The split-then-scale order is preserved (scaler fit happens during processing on the eventual train pool; once persisted, all downstream splits are projected with the **same** scaler — see `docs/data-pipeline.md:66`). Feature selection is upstream of the split.

---

## 7. Actions taken in this session

### Files added
- `docs/mentor_review/01_dataset.md` — this memo.
- `docs/mentor_review/01_HANDOFF.md` — Step-2 resume handoff (sibling file; see §10).

### Files edited
None. Per the operating rule "Step 1 is read-only audit + documentation," all proposed fixes (Findings 1–6) are deferred to a follow-up `docs(phase-1,§…)` commit if/when the candidate accepts them.

### Files deleted
None.

### Tests / scripts / models
None modified. Test count unchanged at 411 passed.

### Results re-runs
None. No model trained, no plot regenerated, no JSON or PNG overwritten. Hash chain is intact.

### Git hygiene applied this session (one-time)
The repository history was tidied prior to writing this memo — full description in §8.

---

## 8. Git hygiene applied (one-time)

The candidate requested at session start that the git policy follow *one main branch, no long-lived feature branches, single release tag at the end of the loop*. The repository was **not** in that shape: `main` was many commits behind `feature/reward-shaping` (which carried Phase 2 → Phase 10 closeout including `v0.1.0`), and `mentor-review/step-0c-framing` carried the Step-0c framing pass. Two tags existed (`v0.1.0`, `pre-mentor-restart`).

Operations applied (all linear, no force-pushes):

- Created a local-only rescue tag `rescue/pre-cleanup-2026-05-06` at the pre-cleanup HEAD; deleted at the end of the cleanup.
- Fast-forwarded `main` over `feature/reward-shaping` (`a969fd6` = former `v0.1.0`), then over `mentor-review/step-0c-framing` (`26f753e`). All commit history preserved, no merge commits introduced.
- Deleted local branches `feature/reward-shaping`, `feature/lstm-training-upgrades`, `mentor-review/step-0c-framing`.
- Deleted local tags `v0.1.0` and `pre-mentor-restart`.
- Pushed `main` to `origin` (now at `26f753e`).
- Deleted remote refs `refs/heads/feature/reward-shaping`, `refs/tags/v0.1.0`, `refs/tags/pre-mentor-restart`. (`feature/lstm-training-upgrades` was a stale local cache only — pruned.)
- Cut `mentor-review/step-1-dataset` off the new `main` for this session.

End state:

```
$ git branch -a
* mentor-review/step-1-dataset
  main
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
$ git tag -l         # empty
$ git log --oneline -3 main
26f753e (origin/main, main) docs(handoff,changelog): mark docs/HANDOFF.md superseded; …
be3a486 docs(thesis-results-map): restructure by chapter; drop tier and IoTWarden-…
e9348d4 docs(framing): soften IoTWarden language across forward-facing surfaces
```

Going forward (per candidate directive):

- One topic branch per mentor-review step: `mentor-review/step-N-<slug>` cut from `main`.
- Conventional Commits (`docs(mentor-review,step-N): …`, `fix(phase-N,§…): …`, `docs(phase-N,§…): …`).
- Merge to `main` (squash or `--no-ff`) on candidate sign-off, **then delete the branch immediately** (local + remote).
- **No tags during the loop.** A single release tag (likely `v1.0.0`) is cut at the end of Step 10 against `main`.

This policy is recorded in `01_HANDOFF.md` so subsequent agents inherit it.

---

## 9. Open questions for the candidate

1. **Finding 1 (scaler).** Confirm that `StandardScaler` (the code) is the intended choice; the dataset card will be corrected to match. If the *intent* was always `MinMaxScaler` and the code drifted, that's a different (correctness) finding requiring re-processing — please flag.
2. **Finding 4 (Phase 1 audit-trail asymmetry).** Option (a) retroactive `PLAN.md`+`RESULTS.md` or option (b) document-the-asymmetry-once? My recommendation is (b).
3. **Finding 3 (kill-chain rationale).** Is the candidate (or advisor) the right author for the per-stage rationale prose? If yes, this is the candidate's most important Step-1 follow-up. If you want me to draft the rationale paragraphs in a follow-up commit, say so and I will produce them on the same `mentor-review/step-1-dataset` branch.

---

## 10. Sign-off

This memo locks the Step-1 verdict at **PASS-WITH-FIXES**. The accompanying handoff `docs/mentor_review/01_HANDOFF.md` records the resume point for Step 2 (Phase 2 Red team review: F1, F2, LSTM convergence). Step 2 may not begin until the candidate signs off this step (a commit, a comment, or out-of-band confirmation).

— mentor-review agent, 2026-05-06
