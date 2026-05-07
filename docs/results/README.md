# `docs/results/` — Canonical Thesis Figures

This directory is the **single source of truth** for figures, tables, and
narrative summaries that will appear in the thesis. Anything outside this
directory is considered exploratory.

## Layout

```
docs/results/
├── README.md                 # this file
├── 00_phase0_diagnosis.md    # baseline diagnosis of pre-restart results
├── 01_dataset/               # Phase 1 — dataset card + EDA-derived plots
├── 02_red_team/              # Phase 2 — LSTM Red Team curves and analysis
├── 03_environment/           # Phase 3 — env sanity checks, reward shape
├── 04_detector/              # Phase 4 — supervised detector + RF / 1D-CNN
├── 05_blue_team/             # Phase 5 — DQN/PPO/A2C learning curves
├── 06_benchmark/             # Phase 6 — RL algorithm benchmark (F5/F6/F7/F8)
├── 07_ablation/              # Phase 7 — sensitivity and ablation (F9/F10/F12/F15)
├── 08_robustness/            # Phase 8 — perturbation / drift studies (F13/F14)  [reserved]
├── 10_release/               # Phase 10 — open-source hygiene + v0.1.0 release tag
└── thesis_figures.md         # Index: figure id ↔ filename ↔ caption
```

## Phase-1 has no `PLAN.md` / `RESULTS.md` (intentional)

`docs/results/01_dataset/` carries only the F0 figures, captions, the F0
summary JSON, and the figure manifest. Unlike Phases 02–10 it does **not**
contain a `PLAN.md` or `RESULTS.md`. This is intentional, not an oversight,
and is documented here so that future agents do not "fix" the asymmetry by
backfilling fake history.

The substantive content that would belong in those two files lives in two
canonical documents that pre-date the per-phase audit-trail convention:

- `docs/dataset_card.md` — the Hugging-Face-style dataset card. Covers
  provenance, sampling strategy, kill-chain projection (cross-referenced
  to `docs/kill-chain-mapping.md`), feature engineering, splits (post
  Phase-4 OOD-leakage fix, commit `3cd2fb9`), hash manifest, declared
  limitations, and reproduction recipe. This is the document the
  defense committee will read.
- `docs/data-pipeline.md` — the anti-leakage and processing protocol:
  split-then-scale order, `StandardScaler` fit on train only, feature
  selection rules.

The numerical record for Phase 1 is the F0 figures plus
`docs/results/01_dataset/F0_summary.json` and `manifest.json`; the
narrative is in the two documents above. Step 9 (LaTeX rebuild) draws
its §3.1 (Methodology — Dataset preparation) and §4.1 introductory
paragraph from `dataset_card.md` directly, and the Phase-2 results
chapter inherits the splits from there.

If a future revision *does* want a `PLAN.md` and `RESULTS.md` for
Phase 1, they should be authored as a single `docs(phase-1):` commit
with both files explicitly marked as **retroactive audit trail**, not
mistaken for pre-registration. Source-of-truth status would then move
from `dataset_card.md` to the new `RESULTS.md`, and the dataset card
would point upward.

## Per-phase scoreboard / manifest asymmetry (audit-trail rollup)

> Cross-cutting note added in Step 8 (cross-cutting cleanup wave) to
> document a benign asymmetry across the per-phase audit trails — see
> mentor-review findings Step-1 F4, Step-2 F4, Step-3 F1, Step-4 F2.

The audit-trail artefacts under `docs/results/<phase>/` are not uniform
across phases — by design. The pattern was adopted incrementally:
Phase 4 was the first phase to ship a top-level `manifest.json` with
explicit input-SHA pinning + a `gates_status` mirror; Phase 6 was the
first to ship a top-level `G<N>_scoreboard.json` JSON record next to
`RESULTS.md`. Earlier phases either pre-date one or both conventions
or scoped their numerical record into a different artefact:

| Phase | `PLAN.md` | `RESULTS.md` | `manifest.json` | `G<N>_scoreboard.json` | Numerical record lives in |
|---|:---:|:---:|:---:|:---:|---|
| **1 — Dataset** | – | – | ✅ | – | `F0_summary.json` + `manifest.json` (+ `dataset_card.md`, `data-pipeline.md`) |
| **2 — Red Team (LSTM)** | ✅ | ✅ ¹ | ✅ | – ² | `F1_summary.json::gates_values` |
| **3 — Environment** | ✅ | ✅ | – ³ | – ³ | `RESULTS.md` §4 (Phase-3 is infrastructure-only; no figures) |
| **4 — Stage detector** | ✅ | ✅ | ✅ | ✅ ⁴ | `F11_summary.json::gates` (+ Step-8-derived `G4_scoreboard.json`) |
| **5 — Blue Team (RL)** | ✅ | ✅ | – ⁵ | ✅ | `G5_scoreboard.json` (Phase-6-native schema since Step 8) |
| **6 — Benchmarks** | ✅ | ✅ | – ⁶ | ✅ | `G6_scoreboard.json` (canonical schema) |
| **7 — Ablations** | ✅ | ✅ | – ⁷ | ✅ | `G7_scoreboard.json` (Phase-6-native schema since Step 8) |

¹ Phase-2 RESULTS.md was authored in Step 8 (cross-cutting cleanup
  wave) to backfill the RESULTS-pattern that Phases 3–7 each followed
  at production time. The Phase-2 *numerical record* lives in
  `F1_summary.json::gates_values` and is byte-perfect from commit
  `283ca29e`; RESULTS.md is the documentation companion.

² No `G2_scoreboard.json` because the gate verdicts live in
  `F1_summary.json::gates_passed` + `gates_thresholds` +
  `gates_values`. The asymmetry is cosmetic; the numerical record is
  intact.

³ No `manifest.json` and no `G3_scoreboard.json` for Phase 3 by
  design — PLAN §3.3 scopes Phase 3 as infrastructure-only with no
  thesis figures, only env-correctness gates evaluated as a
  Markdown table in `RESULTS.md` §4. The producing scripts
  (`tests/test_phase3_env_gates.py`, `tests/test_adversarial_env.py`)
  are the audit-trail backstop.

⁴ Step-4 originally shipped `manifest.json::gates_status` (mixed-
  case) and `F11_summary.json::gates`; Step 8 added the dedicated
  `G4_scoreboard.json` derived artefact to align with the Phase-6
  schema (5 gates: G4.1 SKIP / G4.2 PASS / G4.3 PASS / G4.4
  PASS-WITH-FINDING [D2.1] / G4.5 PASS). Producer:
  `scripts/detector/close_phase4.py`.

⁵ Phase 5 emits per-figure manifests (`F3_manifest.json`,
  `F4_manifest.json`, `T1_hparams.json`) plus
  `runs/phase5/<algo>/seed_<k>/run_manifest.json` records on the
  per-run side. The hash chain back to the post-`3cd2fb9` Phase-1
  splits manifest is *implicit* (path strings in run-manifests, not
  SHAs in a top-level Phase-5 manifest). Step-5 F2 surfaced this;
  the Step-8 batch documents the asymmetry rather than introducing
  a new top-level manifest (no scientific change; Phase-5 numbers
  are unchanged byte-for-byte).

⁶ Phase 6 emits per-figure manifests (`F5/F6/F7/F8_manifest.json`)
  + the run-side `runs/phase6/eval_manifest.json` (gitignored).
  Each per-figure manifest pins the run-side `eval_manifest.json`
  by SHA; the chain is explicit at the figure level.

⁷ Phase 7 emits per-figure manifests
  (`F9/F10/F12/F15_manifest.json`); since Step-8 task #1 these all
  pin `phase5_sweep_manifest`, `phase6_eval_manifest`, and
  `phase1_splits_manifest` SHAs explicitly so the chain is
  self-contained at the figure level (no transitive lookups).

The takeaway for a defense reviewer: the *content* of each phase's
audit trail is intact — every gate has a recorded threshold and an
observed value, every figure ships an SHA chain back to the inputs
that produced it, and the cross-phase chain back to the post-`3cd2fb9`
Phase-1 splits manifest (`c8574094...`) is verifiable end-to-end. The
asymmetry is in *which file* carries the records for a given phase,
not in whether the records exist. The Step-1/2/3/4/5 findings
flagging the asymmetry have all been resolved either by adding the
missing artefact (Step 8 added `G4_scoreboard.json` and Phase-2
RESULTS.md) or by documenting the by-design exception (Phase 1
intentional per `dataset_card.md` precedent; Phase 3 intentional
per PLAN §3.3 infrastructure-only scope).

## Per-figure conventions

Every figure committed under `docs/results/<phase>/` MUST be paired with:

- A PNG (300 DPI minimum, transparent background OK).
- A `.caption.md` file containing the LaTeX-ready caption and a 2–4 sentence
  interpretation.
- A `manifest.json` in the phase directory with at least:

  ```json
  {
    "figure_id": "F3",
    "title": "RL learning curves",
    "produced_by": "scripts/plot_learning_curves.py",
    "mlflow_run_ids": ["..."],
    "data_hash": "<sha256 of underlying CSV/JSON>",
    "git_sha": "<sha at generation time>"
  }
  ```

## Regenerating

All figures are regenerated (deterministically, given fixed seeds) by:

```
make reproduce-thesis
```

A figure that cannot be regenerated by `make reproduce-thesis` does NOT belong
in this directory.

## Authoring discipline

1. **No exploratory plots.** Move them to `notebooks/` first.
2. **Captions are written in `caption.md` first**, before the plot is finalized.
   This forces clarity about what the figure is supposed to show.
3. **Every plot includes a "what to look for" box** in its caption — for the
   thesis defense, examiners will be looking exactly there.

See `docs/thesis_results_map.md` for the full list of planned figures.
