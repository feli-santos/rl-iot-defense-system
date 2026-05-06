# Dataset Card — `ciciot2023` (processed snapshot v1)

> Hugging-Face-style dataset card for the immutable processed snapshot used
> by every result in this thesis. This card is the authoritative description
> for examiners; if it disagrees with the source code, the card is wrong and
> must be regenerated.

## At a glance

| Field                        | Value                                                          |
|------------------------------|----------------------------------------------------------------|
| Source                       | CICIoT2023 (Canadian Institute for Cybersecurity, 2023)        |
| Snapshot                     | `data/processed/ciciot2023/`                                   |
| Snapshot version             | `splits/manifest.json`, `version = 1`                          |
| Number of samples            | **442 237** rows                                               |
| Feature dimension            | **29** (post feature-selection, see §4)                        |
| Number of classes (labels)   | **34** original CICIoT2023 attack labels                       |
| Number of stages (Kill Chain)| **5** (BENIGN, RECON, ACCESS, MANEUVER, IMPACT)                |
| Master split seed            | **42**                                                         |
| Train / val / test ratios    | **0.7 / 0.1 / 0.2** (stratified by stage)                      |
| Balanced eval splits         | val 200/stage (1 000 rows), test 1 000/stage (5 000 rows)      |
| OOD-attack splits (held-out) | 4 classes (3 × 12 121 + XSS 3 846 = 40 209 rows), one per attack stage |

The processed snapshot is the result of `main.py --mode process-data`
(commit `pre-mentor-restart`) applied to the **full 169-file CICIoT2023
release** (≈46.7 M raw rows). Splits and the hash manifest are produced by
`python -m scripts.data.build_split_indices` from the same processed
snapshot.

## 1 — Provenance

- **Original dataset.** *CICIoT2023: A Real-Time Dataset and Benchmark for
  Large-Scale Attacks in IoT Environment*, Neto, Dadkhah, Ferreira, Zohourian,
  Lu, Ghorbani — University of New Brunswick, 2023.
- **Raw archive.** 169 CSV files, 47 features each, **46 686 579** flow
  records labelled across 34 attack classes plus `BenignTraffic`.
- **Local raw path.** `data/raw/CICIoT2023/*.csv` (not committed).
- **Processed pipeline.** `src/utils/dataset_processor.py` — described in §4.
- **Pipeline configuration snapshot.** `config.yml`, sections
  `dataset` and `feature_selection` — relevant fields are reproduced in
  `data/processed/ciciot2023/metadata.json`.

## 2 — Why a *processed* snapshot, not the raw CSVs

Training the LSTM Red Team and the RL Blue Team requires three properties
that the raw release does not have on its own:

1. **Bounded class imbalance.** Raw CICIoT2023 has a 1 059:1 ratio between
   the largest class (`DDoS-ICMP_Flood`, 7.2 M) and the smallest
   (`Uploading_Attack`, 1 252). Without resampling, every model collapses
   to predicting the majority class — exactly the failure mode diagnosed
   in `docs/results/00_phase0_diagnosis.md` §2.1.
2. **Bounded compute footprint.** 47 M float32 rows × 47 features ≈ 8.7 GB,
   too large to keep in memory on a research workstation.
3. **Deterministic feature space.** Two CSVs in the release ship with
   slightly different column orderings; the processor canonicalizes them
   and freezes a single 29-D feature vector everywhere downstream.

Sampling strategy (recorded in `metadata.json → sampling_info`):

- **Mode.** `smart_balanced`.
- **BENIGN cap.** 100 000 rows (sampled from `BenignTraffic` only).
- **Attack budget.** 400 000 rows split evenly across the 27 attack classes
  with ≥ 12 121 rows in the raw release (capped per-class at 12 121 to match
  the smallest "frequent" class). Classes below that floor (e.g.
  `Uploading_Attack`, 1 252) are kept in full.
- **Result.** 442 237 rows total, 34 unique attack labels, stage imbalance
  ratio of **5.25** (vs 1 059 in the raw release).

## 3 — Kill Chain mapping

The thesis works at the abstract Kill Chain level rather than the 34-class
fine-grained level. The mapping is maintained in
`src/utils/label_mapper.py::AbstractStateLabelMapper` and is reproduced
below for archival completeness. **This mapping is closed**: any new
CICIoT2023 label encountered at processing time will raise an explicit
`KeyError` (see test `TestStringToStageIds::test_raises_on_unknown_label`).

| Stage | ID | Name      | CICIoT2023 labels included                                                                                                                                                                                                                                                                                |
|-------|----|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | B  | BENIGN    | `BenignTraffic`                                                                                                                                                                                                                                                                                          |
| 1     | R  | RECON     | `Recon-PortScan`, `Recon-OSScan`, `Recon-HostDiscovery`, `Recon-PingSweep`, `VulnerabilityScan`                                                                                                                                                                                                          |
| 2     | A  | ACCESS    | `SqlInjection`, `CommandInjection`, `XSS`, `Backdoor_Malware`, `BrowserHijacking`, `Uploading_Attack`, `DictionaryBruteForce`                                                                                                                                                                            |
| 3     | M  | MANEUVER  | `MITM-ArpSpoofing`, `DNS_Spoofing`, `Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain`                                                                                                                                                                                                          |
| 4     | I  | IMPACT    | All 12 `DDoS-*` variants + 4 `DoS-*` variants (full list: `DDoS-ICMP_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-PSHACK_Flood`, `DDoS-SYN_Flood`, `DDoS-RSTFINFlood`, `DDoS-SynonymousIP_Flood`, `DDoS-ICMP_Fragmentation`, `DDoS-UDP_Fragmentation`, `DDoS-ACK_Fragmentation`, `DDoS-HTTP_Flood`, `DDoS-SlowLoris`, `DoS-UDP_Flood`, `DoS-TCP_Flood`, `DoS-SYN_Flood`, `DoS-HTTP_Flood`) |

The class `Uploading_Attack` is mapped to ACCESS but only contributes 1 252
rows in the snapshot (smallest class). Statistical conclusions about ACCESS
are dominated by `DictionaryBruteForce`.

## 4 — Feature engineering

Of the 46 numerical features in the raw CICIoT2023 release, 17 are dropped
during processing — recorded in `metadata.json → feature_selection_info`:

- **Zero-variance dropped (4)**: `DHCP`, `IRC`, `SMTP`, `Telnet` — protocols
  that never occur in the snapshot.
- **Low-variance dropped (7)**: `ARP`, `DNS`, `IPv`, `LLC`, `SSH`,
  `cwr_flag_number`, `ece_flag_number` — variance below 0.01 on the raw
  per-feature scale (pre-StandardScaler).
- **High-correlation dropped (6)**: `Magnitue` (sic), `Number`, `Radius`,
  `Srate`, `Std`, `Weight` — Pearson > 0.95 with another retained feature.

The remaining **29 features** are listed in `metadata.json → feature_columns`
in canonical order and split across:

- **Flow timing**: `flow_duration`, `Duration`, `Rate`, `Drate`, `IAT`.
- **Header / size**: `Header_Length`, `Tot sum`, `Min`, `Max`, `AVG`, `Tot size`.
- **TCP flags**: `fin_flag_number`, `syn_flag_number`, `rst_flag_number`,
  `psh_flag_number`, `ack_flag_number`, `ack_count`, `syn_count`,
  `fin_count`, `urg_count`, `rst_count`.
- **Protocol indicators**: `Protocol Type`, `HTTP`, `HTTPS`, `TCP`, `UDP`, `ICMP`.
- **Distribution moments**: `Covariance`, `Variance`.

All 29 features are zero-mean / unit-variance scaled with
`sklearn.preprocessing.StandardScaler`. The scaler is fit on the **train
split only** (see `docs/data-pipeline.md` §Anti-leakage protocol) and
persisted to `data/processed/ciciot2023/scaler.joblib`. The same fitted
scaler is used unchanged on the val, test, balanced-eval, and OOD splits
throughout the thesis — code reference: `src/utils/dataset_processor.py`
(`StandardScaler` instantiated at lines 232, 288, 877; persisted via the
processor's `save_artifacts` path).

A raw (un-scaled) copy of the same 29 features is preserved in
`features_raw.npy` for ablation studies that require interpretable units.

## 5 — Splits (immutable, seed-pinned)

The split builder is `scripts/data/build_split_indices.py`. With seed 42
and the default ratios it produces:

| Split           | Rows    | Stage 0 | 1     | 2     | 3     | 4       |
|-----------------|--------:|--------:|------:|------:|------:|--------:|
| **all**         | 442 237 | 100 000 | 50 746| 36 950| 60 605| 193 936 |
| **train** (70 %)| 309 566 |  70 000 | 35 522| 25 865| 42 424| 135 755 |
| **val** (10 %)  |  44 224 |  10 000 |  5 075|  3 695|  6 060|  19 394 |
| **test** (20 %) |  88 447 |  20 000 | 10 149|  7 390| 12 121|  38 787 |
| **val_balanced**|   1 000 |     200 |    200|    200|    200|     200 |
| **test_balanced**|  5 000 |   1 000 |  1 000|  1 000|  1 000|   1 000 |

**Disjoint and exhaustive.** `train ⊔ val ⊔ test` = `all`; verified by
`tests/test_build_split_indices.py::TestStratifiedSplit`.

**Balanced subsets ⊆ pools.** `val_balanced ⊆ val`,
`test_balanced ⊆ test`. The balanced splits are designed for the LSTM
held-out evaluation (so per-stage F1 is meaningful) and the RL benchmark's
"per-stage decision" matrices.

### OOD-attack splits

Four CICIoT2023 classes are reserved as OOD held-outs — the model never
sees these labels during *any* training phase, but they are evaluated in
Phase 7 to test generalization:

| Class                 | Stage    | Rows   | % of stage |
|-----------------------|----------|-------:|-----------:|
| `VulnerabilityScan`   | RECON    | 12 121 |    23.9 %  |
| `XSS`                 | ACCESS   |  3 846 |    10.4 %  |
| `Mirai-udpplain`      | MANEUVER | 12 121 |    20.0 %  |
| `DDoS-HTTP_Flood`     | IMPACT   | 12 121 |     6.3 %  |

> The ACCESS choice is deliberately small. `DictionaryBruteForce` (12 121,
> 32.8 % of ACCESS rows) was rejected because removing it would starve the
> ACCESS classifier of training data; `XSS` is large enough (3 846 rows
> ≈ Mirai-class size) for a meaningful held-out experiment yet costs only
> ~10 % of stage data.

> **Important.** OOD indices are computed from the **string label array**,
> so they overlap with `train`/`val`/`test` by construction. Phase-2/4
> training code must subtract them before fitting; helper
> `src/utils/dataset_loader.py::exclude_ood_classes` will be added in
> Phase 2 to enforce this.

## 6 — Hash manifest

Every model run records the SHA-256 of the data it consumed. The current
manifest hashes (`splits/manifest.json`):

```
features.npy        5d1ff73d8cc1dc3706db4ca0381e9b88c542ec40a5ae5d6e5fd784e4d166dcc7
labels.npy          fb6bbbdc3b5a35ba201e6d087fc135ac94325c4867ec5d40383d8f82ebb3f743
metadata.json       a9e68b68d82902ac293258444a0acbcdf819393fcc2973b68ab4df664924a99c
scaler.joblib       146c8aa762e0bed97da0d5f9714515213f4b210a82737178aaaddcd1a6705dbf
state_indices.json  83f28824e28b66dd033f3490800715de117b39b6428401e83858a34bbb079b86
```

If any of these change, downstream MLflow runs become invalid — the
benchmarking layer (Phase 7) will refuse to compare runs with mismatched
data hashes.

## 7 — Limitations & caveats (declared up-front)

1. **Synthetic balance.** The 442 k-row snapshot is an *artificially
   rebalanced* view; the natural CICIoT2023 distribution is much more
   skewed. Results in this thesis explicitly do not generalize to a
   real-world packet stream where IMPACT-class DDoS dominates.
2. **Flow-level granularity.** CICIoT2023 features are aggregated over
   flows (≈100 packets), not per-packet. Sub-flow attack progression
   (e.g. multiple stages within a single flow) is invisible to the
   classifier.
3. **No temporal ordering across rows.** Rows are shuffled during
   processing. Sequence-level training (Phase 2) constructs synthetic
   episodes via `EpisodeGenerator`; we do *not* claim to have learned
   from real attack timelines.
4. **`Uploading_Attack` underrepresented.** 1 252 rows is too few for
   reliable per-class statistics; we report ACCESS-stage metrics aggregated.
5. **OOD ≠ unseen.** OOD-attack classes are different *labels* from the
   train classes, but they may share a Kill Chain stage. The OOD-attack
   evaluation tests stage-generalization-given-novel-class, not novel-stage
   detection.

## 8 — How to reproduce this snapshot from scratch

```bash
# 1. Place the raw CICIoT2023 CSVs at data/raw/CICIoT2023/.
# 2. Process them (slow: ~15 min on a workstation):
python main.py --mode process-data \
    --config config.yml \
    --data-path data/processed/ciciot2023

# 3. Build the split index files and hash manifest (fast: <2 s):
python -m scripts.data.build_split_indices \
    --processed-dir data/processed/ciciot2023 \
    --seed 42

# 4. Verify hashes against this card:
sha256sum data/processed/ciciot2023/{features,labels}.npy \
    data/processed/ciciot2023/metadata.json
```

If the hashes diverge, the snapshot is no longer the v1 of this card.
Bump `splits/manifest.json::version`, regenerate this card, and document
the change in `CHANGELOG.md`.
