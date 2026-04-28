# Figure F0 — Class distribution after rebalancing

**LaTeX caption (proposed).**
*Distribution of the 34 CICIoT2023 attack classes in the processed
snapshot after `smart_balanced` resampling (n = 442 237). Bars are colored
by their Kill Chain stage. The y-axis is logarithmic. Classes with fewer
than 5 000 rows are annotated with their absolute count.*

**What to look for.**
1. The 27 most-frequent attack classes are capped at exactly 12 121 rows
   each (top of the per-attack bars), confirming the per-class cap derived
   from the smallest "frequent" class.
2. The IMPACT-stage bars dominate the chart (sixteen DDoS/DoS variants),
   which explains why a stage-conditioned classifier still has a 5.25:1
   imbalance even after rebalancing.
3. Five small classes — `BrowserHijacking`, `CommandInjection`,
   `SqlInjection`, `XSS`, `Backdoor_Malware`, `Recon-PingSweep`,
   `Uploading_Attack` — fall below the 12 121-row cap and are the
   bottleneck for ACCESS- and RECON-stage statistics.

**How it was generated.** `python -m scripts.data.plot_dataset_overview`,
reading `data/processed/ciciot2023/labels.npy`. Reproducible with seed 42.
