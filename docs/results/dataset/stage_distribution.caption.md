# Figure F0b — Kill Chain stage distribution per split

**LaTeX caption (proposed).**
*Per-stage row counts for the train (70 %), val (10 %), and test (20 %)
splits produced by `scripts/data/build_split_indices.py` with seed 42,
after reserving the ten held-out out-of-distribution attack classes.
Hatching distinguishes splits, color encodes the Kill Chain stage. The
splits are stratified by stage, so per-stage ratios are exact (70/10/20)
up to integer rounding.*

**What to look for.**
1. The 7-1-2 ratio is preserved exactly within every stage — confirming
   the stratified split implementation.
2. Stages 1 (RECON) and 2 (ACCESS) are the rarest in the test split
   (~5 k rows each); per-stage F1 on those classes therefore has
   higher variance and motivates the **balanced** held-out splits
   (`val_balanced`, `test_balanced`) used for Stage Detector model selection.
3. Stage 4 (IMPACT) dominates each split — 29 091 rows in test alone — so
   any model that only optimizes overall accuracy will still look good
   on this distribution. This is exactly the failure mode quantified
   during initial dataset diagnostics.

**How it was generated.** `python -m scripts.data.plot_dataset_overview`,
reading `data/processed/ciciot2023/splits/{train,val,test}.idx.npy`.
