# Step 02 — Phase 2 Red Team Review

**Mentor memo. Audits the Phase-2 LSTM Red Team (F1 learning curves, F2 transition-matrix comparison) ahead of the MSc defense at Unicamp/FEEC.**

---

## 1. Verdict

`PASS-WITH-FIXES`

The Phase-2 **scientific substrate is sound**: the LSTM is a 5-token Kill-Chain language model trained end-to-end on synthetic episodes (no real-flow features, no row indices), all four pre-registered exit gates (G1 i.i.d. gap 0.035 ≤ 0.25; G2 token-accuracy 0.977 ≥ 0.55; G3 KL 0.021 ≤ 0.05; G4 cosine 0.99999 ≥ 0.90) clear with strong margins, F2 demonstrates the LSTM has actually learned the Markov structure (max per-cell deviation ≈ 0.012 across 25 cells), and `pytest -q` is green at **411 passed**. The architecture matches PLAN.md §1.1: embedding → 1×LSTM(32) → linear(5), trained with cross-entropy on next-token prediction over windows of length 5 sliding over 50 000 synthetic episodes.

What's not yet defense-grade is **input-side reproducibility**. Three issues need to land before binding the dissertation: **(F1, major)** `manifest.json::inputs["splits/manifest.json"]` records SHA `82aa1214…` — the **pre-`3cd2fb9` (leaky) Phase-1 splits manifest** — while the on-disk file is the **post-`3cd2fb9`** corrected manifest (SHA `1e99d596…`). I confirmed this by reverse-engineering the LSTM's training prior from `F1_summary.json::stage_frequency_train_prior`: cosine similarity = **0.9999999999940188** to the pre-fix train-distribution estimate, vs **0.9980** to the post-fix distribution. This is a hash-chain break at the Phase-1→Phase-2 boundary that doesn't invalidate Phase 2 (the LSTM consumes only stage-frequency *counts*, never per-row data, so the leakage transmitted to the model is bounded by 2.3 percentage points on the 5-cell BENIGN-row sub-block of the transition matrix — and the gate values, all internal to the generator, are unaffected). But the audit-trail invariant *"every figure pinned to its inputs by SHA, hashes match on disk"* is broken until either the manifest is regenerated (Step 7) or the divergence is documented in-place. **(F2, minor)** PLAN.md §3.2 + 00_HANDOFF.md §5 claim "balanced-val macro-F1 is the model-selection metric"; in fact `train_lstm.py` ships with `use_macro_f1_stopping=False`, so the saved checkpoint is the **balanced-val cross-entropy** minimum (epoch 1, val_loss 0.854). The actual `val_macro_f1_max` was 0.444, achieved at a later epoch — the script throws away that checkpoint. **(F3, minor)** `tex/figs/lstm_*.png` carries three legacy figures that the LaTeX rebuild (Step 9) must replace with `F1` / `F2`. None of these are correctness bugs.

---

## 2. What was reviewed

### Artefacts (read in full)
- `docs/results/02_red_team/PLAN.md` — frozen audit trail, exit gates G1–G5 defined.
- `docs/results/02_red_team/manifest.json` — Phase-2 hash chain (4 fields: inputs, outputs, git_sha, all_gates_passed=true).
- `docs/results/02_red_team/F1_summary.json` — best epoch, gate values, transition matrices, per-stage F1.
- `docs/results/02_red_team/F1_learning_curves.png` (1902×716, 8-bit RGBA, 80 KB) + `.caption.md`.
- `docs/results/02_red_team/F2_transition_matrix_comparison.png` (2020×678, 8-bit RGBA, 81 KB) + `.caption.md`.

### Code (read in full)
- `scripts/red_team/train_lstm.py` (520 lines) — Phase-2 CLI: prior loader, episode synthesis, training, evaluation, F1+F2 plotting, manifest emission.
- `src/generator/episode_generator.py` (539 lines) — `EpisodeGenerator` Markov sampler + stateless helpers (`episodes_to_numpy`, `stage_distribution_from_split_manifest`).
- `src/generator/attack_sequence_generator.py` (head, lines 1–120) — `AttackSequenceGenerator` (Embedding → LSTM → Linear) and config defaults.
- `src/generator/transition_mask.py` (226 lines) — `TransitionMask`. **Not used by Phase 2** (`use_transition_mask=False` is the default and is never overridden in `train_lstm.py`).
- `src/training/generator_trainer.py` (1041 lines) — training loop, balanced-validation split, early stopping, MLflow logging, model persistence.

### Tests (run + scoped)
- Full suite: `pytest -q` → **411 passed**, 0 failed, 66.7 s.
- Scoped Phase-2 coverage: `tests/test_red_team_helpers.py` (88 lines, 8 tests on `episodes_to_*` and `stage_distribution_from_split_manifest`), `tests/test_episode_generator.py` (611 lines), `tests/test_attack_sequence_generator.py` (334 lines), `tests/test_generator_trainer.py` (572 lines), `tests/test_transition_mask.py` (206 lines) — **all passing**.

### Git
- Phase-2 producing commit: `283ca29` *"refactor(phase-2): stateless episode helpers + train-split prior loader"* (2026-04-28).
- Followed in immediate sequence by `e15be4d` *"feat(phase-2): scripts/red_team/train_lstm.py + Makefile target"* and `88ad3d7` *"docs(phase-2): F1 + F2 figures with all four exit gates passing"*.
- Phase-1 OOD-leakage fix `3cd2fb9` *"fix(phase-1): exclude held-out OOD classes from train/val/test (CRITICAL)"* (2026-04-29) **landed AFTER `283ca29`** but is **not** an ancestor of it. Verified via `git merge-base --is-ancestor 3cd2fb9 283ca29` → returns false. This is the temporal root cause of Finding 1.

---

## 3. Findings (priority-ordered)

### Finding 1 — `manifest.json::inputs` records the pre-`3cd2fb9` (leaky) splits manifest [severity: **major**]

**Where.**
- `docs/results/02_red_team/manifest.json:6` declares `"data/processed/ciciot2023/splits/manifest.json"` SHA = `82aa12149d2e0ee5a2424a7da44719df885ac18495590344e6d393e22d72b5c5`.
- On-disk `data/processed/ciciot2023/splits/manifest.json` SHA = `1e99d596826d054e337a8a84e060b1e9d7c15b44a1cbbda425b6bbdd311e0e0d` (manifest `generated_at` = 2026-04-29T16:39:36 UTC).
- The Phase-2 producing commit (`283ca29`) precedes the Phase-1 OOD-leakage fix (`3cd2fb9`) in the linear history.

**What's true (the substantive evidence).** I reverse-engineered which splits-manifest the LSTM actually trained on. `train_lstm.py:402-403` records `train_freq /= train_freq.sum()` into `summary["stage_frequency_train_prior"]` — i.e. the raw frequencies of the pre-Laplace stage counts the model received. Comparing to the two candidate manifests:

| | BENIGN | RECON | ACCESS | MANEUVER | IMPACT |
|---|---:|---:|---:|---:|---:|
| LSTM training prior (from `F1_summary.json`) | 0.2261 | 0.1147 | 0.0836 | 0.1370 | 0.4385 |
| Post-`3cd2fb9` train (current `splits/manifest.json`) | 0.2487 | 0.0961 | 0.0823 | 0.1206 | 0.4522 |
| Pre-`3cd2fb9` train estimate (70 % of `all`-stage counts) | 0.2261 | 0.1147 | 0.0836 | 0.1370 | 0.4385 |

`cosine(LSTM_prior, post-fix freq) = 0.998038`. `cosine(LSTM_prior, pre-fix freq) = 0.99999999999`. The pre-fix match is exact to numerical precision. **The LSTM was trained on the leaky-Phase-1 splits manifest** that included OOD-class rows in the train pool (specifically `VulnerabilityScan` ⊂ RECON-stage train pool, `Mirai-udpplain` ⊂ MANEUVER-stage train pool, plus the smaller `XSS` and `DDoS-HTTP_Flood` contributions).

**Why it matters — and why it's *not* a Phase-2 correctness blocker.** The LSTM consumes the splits manifest exactly **once**, in `train_lstm.py:302`:

```python
train_prior = stage_distribution_from_split_manifest(splits_manifest, "train")
```

`stage_distribution_from_split_manifest` reads the integer stage *counts* and returns them — never any per-row features, never any indices, never any class labels. Those counts then shape only:
- 5 cells of the 25-cell transition matrix (`trans[0, j] = 0.6 * stage_dist[j]` for j∈{1..4}, the BENIGN→j transitions; `episode_generator.py:248-250`),
- the initial-state distribution for the 20 % of episodes that don't start with BENIGN (`episode_generator.py:307-316`).

Stages 1–4's outgoing transition rows are independent of the prior — they're hard-coded from `persist_w/progress_w/skip_w`. The pre/post per-stage frequency delta (max 2.3 pp on BENIGN, 1.9 pp on RECON, 1.6 pp on MANEUVER) propagates only into those 5 BENIGN-row cells, with the rest of the 25-cell matrix untouched.

The Phase-2 exit gates G3 (KL between LSTM rollouts and ground-truth synthetic matrix) and G4 (cosine between LSTM and ground-truth rollouts) are **internal**: they compare the LSTM's rollouts to the *same* `EpisodeGenerator`'s ground-truth — both built from the same prior. Re-training on the corrected manifest would recompute both sides identically; G3 and G4 verdicts would not flip. G1 (i.i.d. gap) and G2 (token accuracy on holdout) measure the LSTM's self-consistency on synthetic episodes drawn from its own grammar; also unaffected.

**Why it's still a major finding.** The audit-trail invariant *"every figure manifest's inputs SHA matches on-disk"* is broken at the Phase-1→Phase-2 boundary. An examiner who notices the discrepancy will (rightly) ask whether downstream phases (3, 4, 5, 6, 7) inherit the same misalignment. They do not — Phase 4's detector training uses the post-fix `splits/manifest.json` directly via `RealizationEngine.from_split_manifest(..., exclude_ood=True)` (Step-1 §4 invariant table) — but the Phase-2 manifest itself doesn't reflect that.

**Recommended fix (two acceptable options; Step 7 owns the choice).**
- **(a)** *(preferred for honesty)* Re-run `python -m scripts.red_team.train_lstm --no-mlflow` against the post-fix manifest. With `seed=42` pinned and the `EpisodeGenerator` deterministic, this should reproduce F1+F2 with comparable gate values (the prior delta is small) but updated `git_sha`, updated input SHAs in the new manifest, and a PR that explicitly notes the re-run. Hash chain is freshly intact. Step 7 is the canonical place.
- **(b)** *(documentation-only, lower bar)* Add a single-paragraph note to `docs/results/02_red_team/RESULTS.md` (which doesn't yet exist — see Finding 4) explaining the temporal divergence, citing the cosine-similarity 0.99999 evidence, and quantifying the per-stage prior delta. The manifest itself stays as is. This preserves the audit-trail-as-historical-record principle but is weaker.

I recommend **(a)** at Step 7; in the meantime, the cosine-equality evidence in this memo is sufficient to retire the issue from blocker status.

**Suggested commit message at Step 7.** `fix(phase-2,manifest): re-run F1+F2 against post-3cd2fb9 splits manifest; update input hash chain`.

### Finding 2 — Model-selection metric mismatch: PLAN says macro-F1, code uses val_loss [severity: **minor**]

**Where.**
- `docs/results/02_red_team/PLAN.md:144-150` (G1 — train/val cross-entropy curves; G2 — token-level top-1 accuracy on synthetic val ≥ 0.55) does **not** specify which metric drives early stopping; it only specifies what the gates evaluate at the end.
- `docs/mentor_review/00_HANDOFF.md` and `docs/mentor_review/01_HANDOFF.md` Step-2 acceptance criteria (verbatim): *"balanced-val macro-F1 is the model-selection metric and matches saved best-epoch claim"*. That language was inherited from the Step-2 prompt and PLAN context.
- `scripts/red_team/train_lstm.py:337-355` configures `GeneratorTrainingConfig(... balanced_validation=True, ...)` but does **not** pass `use_macro_f1_stopping=True` — and that field defaults to `False` (`generator_trainer.py:83`). So the early-stopping branch taken at runtime is `generator_trainer.py:445-452`: minimize `val_loss`. The checkpoint saved at `_save_checkpoint()` (`generator_trainer.py:740-747`) is the one with lowest balanced-val cross-entropy.
- The saved checkpoint corresponds to **epoch 1** (`F1_summary.json::training.best_epoch=1`, `best_val_loss=0.854`). `val_macro_f1_max=0.444` was reached at a *different* (later) epoch and was **not** the criterion for which checkpoint was loaded.

**Why it matters.** The Step-2 acceptance criterion as worded would FAIL on this point if interpreted strictly ("balanced-val macro-F1 is the model-selection metric"). What the script *actually* does — minimum balanced-val cross-entropy — is also a defensible criterion (it's the standard model-selection rule for next-token language models, and macro-F1 on a balanced 5-class set is famously volatile when one class is absorbing/dominant). The G1–G4 gates all evaluate on the **best-loaded checkpoint** (epoch 1), so the gates' verdicts correctly correspond to what's actually deployed. This is a documentation/wording issue, not a correctness issue.

The `holdout_metrics` block in `F1_summary.json` (token accuracy 0.977, macro-F1 0.487) is computed by `trainer.evaluate(...)` (`train_lstm.py:361`) which calls `_load_best_checkpoint()` first; so the macro-F1 of 0.487 is the **best-checkpoint's macro-F1 on the holdout**, not the best macro-F1 ever observed.

**Recommended fix.** `docs(phase-2,§3.2): clarify in PLAN.md and the captions that the model-selection criterion is balanced-validation cross-entropy, not macro-F1`. PLAN.md is frozen audit trail and **must not** be edited; instead, a sibling note belongs in either `RESULTS.md` (when authored — see Finding 4) or in the `F1_learning_curves.caption.md` "What to look for" block. Concretely, replace the wording in 00_HANDOFF/01_HANDOFF and the Step-2 prompt's acceptance criterion (this very memo) with:

> "Best epoch is selected by minimum balanced-validation cross-entropy; macro-F1 is reported as a secondary diagnostic, not as the early-stopping criterion."

That makes future memos consistent with the code.

### Finding 3 — Stage-1 (RECON) F1 = 0.0 on the holdout [severity: **minor**]

**Where.** `F1_summary.json::holdout_metrics::f1_stage_1 = 0.0`, `recall_stage_1 = 0.0`, `precision_stage_1 = 0.0`. The model never predicts RECON on the natural-distribution holdout.

**Why it matters — and why it's *not* a Phase-5/6/7 blocker.** Phase 2 is a Red Team episode generator; Phase 5 (RL) consumes the LSTM only to **script attack episodes** (it samples episodes from the generator and replays them step-by-step into the environment). It does not consume the LSTM's per-token prediction logits as a feature for the RL agent. So F1=0 on RECON in *next-token prediction* doesn't directly degrade downstream. But the F1 caption truthfully calls this out (*"the balanced eval is unforgiving on the rare stages where the LSTM has very few exemplars"*) and it deserves an explicit committee-facing acknowledgement: **the LSTM prefers IMPACT over RECON when given a benign-history prefix**, because IMPACT is absorbing (`trans[4,4]=1.0`) and dominates the conditional likelihood whenever the history is ambiguous.

**Recommended fix.** `docs(phase-2): expand F1 caption with one sentence on stage-1 collapse and why downstream is unaffected`. One sentence in `F1_learning_curves.caption.md`. No code change. Already half-acknowledged in the existing caption, just needs to be sharper.

### Finding 4 — Phase 2 has no `RESULTS.md` [severity: **minor**]

**Where.** `docs/results/02_red_team/` contains `PLAN.md`, `manifest.json`, `F1_summary.json`, two PNGs, two `.caption.md` files. **No `RESULTS.md`**. Phases 03, 04, 05, 06, 07 all have a `RESULTS.md` *and* a `PLAN.md`; Phases 01 and 02 are the two outliers (Step 1 Finding 4 already flagged Phase 1).

**Why it matters.** Step 9 LaTeX rebuild needs canonical text for §4.1 (Red Team validation). The captions cover the figures but not the narrative (architecture, training protocol, gate verdicts, what the LSTM is for in the rest of the thesis). Without a `RESULTS.md`, the Step-9 author has to reconstruct that narrative from `PLAN.md` + captions + the Phase-2 producing commits — possible but lossy.

**Recommended fix.** *Same option as Step 1 Finding 4*: I recommend option **(b)** — add one paragraph to `docs/results/README.md` (or to a new `docs/results/02_red_team/RESULTS.md`) saying *"Phase 2's scientific narrative lives in the F1 + F2 captions and `F1_summary.json`; the architecture and training protocol are described in `PLAN.md` §1.1–§3.1"* — and accept the asymmetry. The retroactive option **(a)** (compose a full `RESULTS.md`) is also acceptable but more work; either is fine.

If we do compose a Phase-2 `RESULTS.md`, it's also the natural place to document Findings 1, 2, 3 of this memo as known limitations, in the candidate's own voice.

### Finding 5 — `tex/figs/lstm_*.png` carries three legacy figures that don't match F1+F2 [severity: minor]

**Where.** `tex/figs/lstm_train_accuracy_and_loss.png`, `tex/figs/lstm_validation_acc_and_loss.png`, `tex/figs/lstm_confusion_matrix.png`. These were the qualification-draft figures, produced before the Phase-2 v2 refactor (`283ca29`).

**Why it matters.** Step 9 (LaTeX rebuild) must replace these with `F1_learning_curves.png` and `F2_transition_matrix_comparison.png` — the qualification-era confusion-matrix figure has no equivalent under the Phase-2-v2 framing (PLAN.md §1.4: the LSTM is no longer evaluated as a real-flow classifier; that role moved to F11 in Phase 4).

**Recommended fix.** *Step 9 owns this.* Out of scope for Step 2; flagging as carry-forward.

### Finding 6 — `train_lstm.py` does not pin `torch.manual_seed` [severity: nit]

**Where.** `scripts/red_team/train_lstm.py:305-306` seeds `np.random.seed(args.seed)` and `np.random.default_rng(args.seed)`. PyTorch's RNG is not explicitly seeded.

**Why it's nit.** Independent of seed, the training is **CPU-only** (`device="cpu"`), the model is small (~10k parameters), and the training data is generated deterministically from the numpy-seeded `EpisodeGenerator`. The empirically observed `F1_summary.json` numbers (in particular the gate margins) are large enough that PyTorch's internal RNG variation (init weights of the LSTM and the linear head) cannot move them off the right side of the gate thresholds. But for full reproducibility, `torch.manual_seed(args.seed)` should be added immediately after numpy seeding. **Recommended fix:** `fix(phase-2,seed): also pin torch.manual_seed in train_lstm.py`. One line. Defer to Step 7 if a re-run is scheduled (Finding 1 option (a)); apply now if it isn't.

### Finding 7 — `manifest.json` does not pin the producing-script SHA [severity: nit]

**Where.** Same shape as Step 1 Finding 7 — the manifest carries `git_sha` of the producing commit but not the script's own content hash. **Defer to Step 8 cross-cutting audit** (already on the carry-forward list).

---

## 4. Validation: structural invariants that hold

| Invariant | Where enforced | Verified |
|---|---|---|
| LSTM never reads per-row features or row indices from the splits manifest | `scripts/red_team/train_lstm.py:302` calls `stage_distribution_from_split_manifest(...)` only; `src/generator/episode_generator.py:70-89` reads only `manifest["stage_counts"][split_name]` | ✅ |
| LSTM training prior is taken from the **train** split (not val/test/all/OOD) | `train_lstm.py:302` passes `"train"` literal | ✅ |
| Hash chain — F1 + F2 + summary outputs match on-disk | `manifest.json::outputs` SHAs match `shasum -a 256` for `F1_learning_curves.png` (`6064e50a…`), `F1_summary.json` (`1c7d26ea…`), `F2_transition_matrix_comparison.png` (`404127c4…`) | ✅ exact match |
| Hash chain — splits-manifest input | `manifest.json::inputs["…/splits/manifest.json"]` declared `82aa1214…`; on-disk SHA `1e99d596…` | ❌ **diverges — Finding 1** |
| Phase-2 exit gates G1–G4 all clear with strong margins | `F1_summary.json::gates_passed` = `{G1: true, G2: true, G3: true, G4: true}`, `gates_values` = (0.035, 0.977, 0.021, 0.99999) | ✅ |
| G5 (full pytest) passes on the current commit | 411 passed, 0 failed, 66.7 s | ✅ |
| MLflow run id captured | `train_lstm.py:280` (`--no-mlflow` flag) + `generator_trainer.py:324-326` (`mlflow.start_run(); mlflow.active_run().info.run_id` is logged). The Phase-2 figures in repo were produced with `--no-mlflow` per the F1 caption *"How it was generated"* block; that's intentional for the figure-shipping run, but the MLflow path exists and is exercised in tests. | ✅ structural |
| Balanced-validation split is stratified at 200 samples per stage | `train_lstm.py:348-349` (`balanced_validation=True, val_samples_per_class=200`); `generator_trainer.py:264-299` (`_build_balanced_validation_split`) | ✅ |
| `_load_best_checkpoint()` is invoked before `evaluate()` so the holdout metrics correspond to the saved checkpoint | `generator_trainer.py:467-468` (load best at end of train), `train_lstm.py:361` (`trainer.evaluate(holdout_episodes)` after `trainer.train(...)`) | ✅ |
| Best checkpoint is the **balanced-val cross-entropy minimum**, NOT the macro-F1 maximum | `generator_trainer.py:445-452` runs because `use_macro_f1_stopping=False` (default; not overridden by `train_lstm.py`); see **Finding 2** | ⚠️ correct, but doc-mismatched |
| LSTM rollouts vs ground-truth Markov: indistinguishable | F2 caption: max abs cell deviation = 0.012, mean per-row KL = 0.021; visually inspected in `F2_transition_matrix_comparison.png` (3-panel heatmap with diff column) | ✅ |
| `transition_mask.py` divergence (allows IMPACT→BENIGN, while EpisodeGenerator hard-codes IMPACT absorbing) | `transition_mask.py:79-80` vs `episode_generator.py:269-271`. **Not exercised** by Phase 2 — `train_lstm.py` does not call `set_transition_mask()`. Phase 5+ may exercise the mask; that's a Step-3/Step-5 question, noted as carry-forward. | ⚠️ noted |
| Determinism (numpy + python seeds) | `train_lstm.py:305-306` (`np.random.seed`, `np.random.default_rng`); `generator_trainer.py:191-193, 277` (`np.random.RandomState(seed)`) | ⚠️ partial — `torch.manual_seed` missing (Finding 6) |
| Phase-2 source-code coverage | `tests/test_red_team_helpers.py` (helpers), `test_episode_generator.py` (Markov sampling), `test_attack_sequence_generator.py` (LSTM forward + sampling), `test_generator_trainer.py` (training + evaluate), `test_transition_mask.py` (mask) — all 411 tests pass | ✅ |

---

## 5. F1 + F2 figure inspection

### F1 — LSTM Red Team learning curves
- 1902 × 716 px @ 160 DPI ⇒ implied print 11.9″ × 4.5″, 80 KB.
- Two panels: (left) train + val cross-entropy vs epoch, (right) val macro-F1 vs epoch with uniform-baseline 0.20 horizontal reference.
- Color choices: `#0072B2` (CB-blue, train), `#D55E00` (CB-orange, val), `#009E73` (CB-green, macro-F1) — all colour-blind-safe (Wong 2011 palette).
- Caption is internally consistent with `F1_summary.json::training` (best-epoch 1, best balanced-val loss 0.854, val_macro_f1_max 0.444). One subtlety: the right panel's 0.20 baseline is the **uniform-prediction** baseline, but on a 5-class balanced val with one absorbing class, the marginal-prediction baseline is closer to 0.30; this is a minor caption nuance not worth a separate finding.
- **Verdict:** publication-clean. Ship as is.

### F2 — Empirical 5×5 transition matrix vs ground truth
- 2020 × 678 px @ 160 DPI ⇒ implied print 12.6″ × 4.2″, 81 KB.
- Three heatmaps: (left) ground truth, (centre) LSTM empirical from 10 000 rollouts, (right) signed difference on a `coolwarm` colormap clipped to ±0.5.
- Per-cell numerical labels in white-on-dark / black-on-light contrast — readable.
- Stage labels rotated 35° on the x-axis; stage labels at fontsize 8 — borderline tight at thesis-page width but acceptable.
- Diff panel is the most informative: it reveals the LSTM's tiny over/under-prediction patterns. The "LSTM − Truth" cells are all in [−0.012, +0.012].
- **Verdict:** publication-clean. Ship as is.

The two PNGs at 160 DPI are below the customary 300 DPI bar; absolute pixel widths (1902, 2020) are large enough that at thesis-page width (≈6.5″) the effective resolution is ~290–310 ppi, which is sharp. **Do not regenerate** in Step 2 (read-only).

---

## 6. LSTM convergence narrative

The pre-registered concern was *"is the LSTM emitting just the marginal stage distribution rather than learning Markov structure?"* The answer is unambiguously **yes, it learned Markov structure**:

- **Marginal-only refute.** The marginal stage distribution of the train prior is `[0.226, 0.115, 0.084, 0.137, 0.439]`. If the LSTM were emitting only the marginal, every row of its empirical transition matrix would be ≈ that vector. It is not — see `F1_summary.json::lstm_transition_matrix`: row 0 (BENIGN→\*) is `[0.464, 0.099, 0.078, 0.109, 0.250]` (mass concentrated on BENIGN, falling roughly 1/distance to higher stages); row 1 (RECON→\*) is `[0.004, 0.318, 0.509, 0.110, 0.059]` (mass on persist + progress to ACCESS, near-zero regression). The 5×5 matrix has clear row-dependent structure that matches the ground-truth synthetic matrix to within 0.012 per cell.
- **Absorbing state.** Row 4 (IMPACT→\*) is `[2.0e-5, 0, 3.0e-5, 9.9e-5, 0.99985]`. Effectively absorbing. Tiny non-zero off-diagonal mass at row 4 is the LSTM's softmax floor — not a grammar violation that a sampler could exploit in practice.
- **No spurious regression.** The lower triangle of `lstm_transition_matrix` (entries (i,j) with j<i, i.e. backwards transitions) sums to 0.0036 — negligible. The model has learned the no-regression constraint without explicit masking.
- **Cosine vs marginal.** `cosine(stage_freq_lstm, stage_freq_train_prior) = 0.916` (computable from the summary). Same vectors compared against an *equal-sized* sample of the synthetic ground-truth rollouts: cosine = 1.0000 (G4 gate value). The G4 verdict against rollouts (1.0) being substantially higher than the marginal-projection cosine (0.916) is the cleanest single-number argument that the LSTM has learned more than the marginal.

The LSTM is fit-for-purpose for downstream consumption: Phase 5+'s `AdversarialEnv` samples episodes from this generator, and those episodes are statistically indistinguishable from canonical Kill-Chain progressions over the synthetic grammar.

The **honest framing** the dissertation should adopt for §4.1 is: *"the LSTM is a 5-token language model that compresses the synthetic Kill-Chain grammar; it is **not** a real-flow classifier, and any committee question of the form 'why is its real-flow accuracy so low' is a category error — that role is played by the supervised stage detector in Phase 4."* This is exactly the F2 caption's *"why this matters"* paragraph; just keep that clarity in §4.1 of the rebuilt LaTeX.

---

## 7. Actions taken in this session

### Files added
- `docs/mentor_review/02_red_team.md` — this memo.
- `docs/mentor_review/02_HANDOFF.md` — Step-3 resume handoff.

### Files edited
None. Per the operating rule *"Step 2 is read-only audit; no model retraining or plot regeneration"*. All proposed fixes (Findings 1–6) are deferred to follow-up `docs(phase-2,§…)` and `fix(phase-2,§…)` commits if/when the candidate accepts them. No PNG, JSON, or manifest under `docs/results/02_red_team/` was touched.

### Files deleted
None.

### Tests / scripts / models
None modified. Test count unchanged at **411 passed** (66.7 s).

### Results re-runs
None. No model trained, no plot regenerated, no JSON or PNG overwritten. The output-side hash chain is intact; the input-side splits-manifest divergence (Finding 1) is documented and explained but not re-resolved.

### Git hygiene applied (per phase, the standard flow)
This step's **Phase G1** (open the step) executed before the audit:
1. `git checkout main && git pull --ff-only origin main`
2. `git merge --no-ff mentor-review/step-1-dataset` (preserves the 5 Step-1 doc-fix commits as individual atoms in `main`'s history) — produced merge commit `90e5195`.
3. `git push origin main`.
4. `git branch -d mentor-review/step-1-dataset && git push origin --delete mentor-review/step-1-dataset`.
5. `git checkout -b mentor-review/step-2-red-team`.
6. Verified `git tag -l` empty, `git branch -a` shows only `main`, `origin/main`, `mentor-review/step-2-red-team`.
7. Ran `pytest -q` to confirm 411 passed before any audit work.

End-of-step **Phase G2** (close the step) is symmetric and runs only after candidate sign-off; it's listed in `02_HANDOFF.md` §6.

A one-time pager fix was also applied: configured `git config --global core.pager cat` and `pager.{tag,branch,log}=cat`, eliminating the `less`-pager hijack that interrupted earlier shell turns.

---

## 8. Open questions for the candidate

1. **Finding 1 (manifest input-hash divergence).** Option (a) re-run F1+F2 at Step 7 with the post-fix manifest (cleaner, regenerates hashes), or option (b) document-only in `RESULTS.md`. My recommendation is (a), at Step 7. Option (b) is acceptable if there's a reason to keep the original April-28 figures (e.g. external citation), but I see none.
2. **Finding 2 (model-selection metric).** Confirm that **balanced-val cross-entropy** (what the code does) is the intended criterion, and we'll fix the Step-2 prompt + 00/01 handoff wording. If macro-F1 + recall gates were the actual intent, that's a `fix(phase-2,trainer)` commit setting `use_macro_f1_stopping=True, min_recall_stage_1=0.5, min_recall_stage_2=0.5` and a re-run at Step 7. The PLAN.md §3.2 gates as written are agnostic to which metric drives early stopping, so either choice is faithful to the audit trail.
3. **Finding 4 (RESULTS.md asymmetry).** Same recommendation as Step-1 Finding 4: option (b) one-paragraph asymmetry note, not a backfilled retroactive document. Confirm which option you want.
4. **Carry-forward to Step 3.** The `transition_mask.py:79-80` divergence (mask permits IMPACT→BENIGN; episode_generator hard-codes absorbing) is irrelevant for Phase 2 (mask is unused). Flagging here so Step 3 can verify whether Phase 3+ (RL env) ever sets `use_transition_mask=True`. If yes, the absorbing-state divergence is a Step-3 finding; if no, it's a Step-8 cross-cutting cleanup.

---

## 9. Sign-off

This memo locks the Step-2 verdict at **PASS-WITH-FIXES**. The accompanying handoff `docs/mentor_review/02_HANDOFF.md` records the resume point for Step 3 (Phase 3 Environment review: MDP correctness, reward shape, env gates G3.x). Step 3 may not begin until the candidate signs off this step (a commit, a comment, or out-of-band confirmation), at which point Phase G2 (merge `mentor-review/step-2-red-team` into `main`, delete branch local + remote) executes before cutting `mentor-review/step-3-env`.

— mentor-review agent, 2026-05-06
