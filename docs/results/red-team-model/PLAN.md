# Phase 2 — Red Team v2: audit & plan

> Mentor's plan, **written before any code changes**, derived from a full
> read of `src/generator/`, `src/training/generator_trainer.py`, the IoT
> Warden paper, and the Phase-0 diagnosis.

## 1 — Audit of the current Red Team

### 1.1 What the LSTM actually is

`AttackSequenceGenerator` is a **next-token language model over 5 Kill
Chain stage IDs**:

```python
# src/generator/attack_sequence_generator.py L84-102
self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=32)
self.lstm      = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, ...)
self.fc        = nn.Linear(64, 5)   # logits over the 5 stages
```

Input shape: `(batch, seq_len)` of integers in `{0,1,2,3,4}`.
Output: a probability distribution over the same 5 stage IDs.

There is **no path** from the 29-D CICIoT flow feature vector to this
model. `features.npy` is not referenced anywhere in
`src/generator/` or `src/training/generator_trainer.py`.

### 1.2 What it is trained on

`EpisodeGenerator` synthesizes integer sequences from a 5×5 transition
matrix:

```python
# src/generator/episode_generator.py L161-211
trans[0,0] = 0.4
trans[0,j] = 0.6 * P_dataset(stage=j)   # weighted by real distribution
trans[i,i] = persist_w        # 0.3
trans[i,j] = progress_w       # 0.5  for j = i+1
trans[i,j] = skip_w/distance  # 0.2 / d for j > i+1
trans[4,4] = 1.0              # IMPACT is absorbing
```

The trainer (`prepare_data`, L163-261) takes those integer sequences,
slides a window of length 5 across them, and builds `(X, y)` pairs of
shape `(N, 5)` → `(N,)`. **The dataset stage distribution is used only
as a prior on the synthetic transition matrix** — never as a direct
training signal.

### 1.3 What "macro-F1 = 0.59" in the pre-restart artifacts measured

Given §1.1 + §1.2, the pre-restart number was the LSTM's accuracy at
predicting *the next synthetic stage* given *the previous five
synthetic stages*. It is a self-consistency check on the generator's
own grammar, not a measurement of attack-stage detection from real
network traffic. The IMPACT-biased confusion matrix (Phase 0
diagnosis §2.1) is exactly what one expects from an absorbing-state
language model: once the synthetic chain hits stage 4, it stays there,
so the model learns "predict 4 with high probability".

### 1.4 The architectural decision

The IoT Warden paper (and every related work in `docs/papers/`)
splits the responsibility into **two distinct components**:

1. **A trigger / stage detector** — sees flow features, outputs stage
   probabilities. This is a supervised classifier with a real
   evaluation regime (precision, recall, F1 per stage on real flows).
2. **A temporal model of attack progression** — independent of
   detector outputs, used to *script* attack episodes for the
   environment and to provide context to the agent. This is what our
   `AttackSequenceGenerator` actually is.

We currently have only #2. The pre-restart project tried to use #2
*as if it were #1*, which is the root cause of the bad results.

## 2 — Decision: keep the split, fix each piece in its proper phase

After weighing both alternatives:

- **Option A** *(chosen)*: keep `AttackSequenceGenerator` as a pure
  Red Team episode generator. Add the supervised stage detector as a
  new module in **Phase 4**. This matches IoT Warden, preserves
  ~2 360 lines of working code, and gives clean per-phase deliverables.
- **Option B** *(rejected)*: rewrite the LSTM to consume 29-D flow
  vectors. Conflates two jobs into one model, contradicts the
  paper-of-record, and would invalidate every test we have.

### 2.1 What this means for thesis figures

The original plan had F1/F2 reporting macro-F1 ≥ 0.75 on the LSTM. With
Option A that is the wrong thing to measure. Re-aligned figures:

| Figure | Phase | What it actually shows |
|--------|-------|------------------------|
| **F1** *(Phase 2)* | LSTM Red Team learning curves: train/val cross-entropy loss + token-accuracy as a function of epoch on **synthetic episodes**. The success criterion is *low overfitting* and a *smooth* curve, not high accuracy. |
| **F2** *(Phase 2)* | Diagnostic: an empirical transition matrix (5×5) of *generated* sequences from the trained LSTM, compared side-by-side with the empirical transition matrix of the **real CICIoT2023 stage sequences within attack windows** (recovered via stage indices). High agreement = the Red Team scripts realistic attack flows. |
| **F11** *(Phase 4)* | The supervised stage detector's **real** confusion matrix on `test_balanced` — this is the figure that has to clear macro-F1 ≥ 0.75. |
| **F12** *(Phase 4)* | Detector calibration / per-stage recall on `test_balanced` and on the OOD-attack splits. |

This split is more honest with examiners than the old plan and is
fully consistent with how IoT Warden reports.

## 3 — Concrete deliverables for Phase 2

### 3.1 Code changes

1. **No change** to `AttackSequenceGenerator` (architecture is
   correct for its job).
2. **`EpisodeGenerator` enhancement** *(new method)*: derive the
   stage-distribution prior **from the train split only**, never from
   `val`/`test`/OOD. Today the trainer pulls counts from
   `metadata.json → sampling_info` which counts the entire snapshot.
   This is a small bug — a closed-form leak from val into the
   transition matrix.
3. **`GeneratorTrainer` cleanup**:
   - Add a `train_split_indices: np.ndarray | None` argument so the
     stage prior can be recomputed from a specified split.
   - Drop ~80 lines of MLflow boilerplate that re-implements
     functionality already in `TrainingManager`.
   - Stop using `EpisodeGenerator()` *with default config* inside
     `prepare_data` solely to call `to_numpy` (L176). Use the static
     helper instead — that one default is silently ignoring the user's
     stage distribution.
4. **New script** `scripts/red_team/train_lstm.py` — thin CLI wrapper
   that:
   - Loads `splits/manifest.json` (refuses to run if hashes drift).
   - Builds `EpisodeGenerator` with the **train-split prior**.
   - Trains, evaluates, writes F1 + F2 to
     `docs/results/red-team-model/`, logs run to MLflow with a `figure_id`
     tag of `F1` / `F2`.
5. **New tests** (`tests/test_red_team_v2.py`):
   - The episode-generator prior matches the train-split distribution
     to within Laplace tolerance.
   - The trained LSTM's empirical transition matrix on 10 000 generated
     sequences agrees with the synthetic ground-truth matrix in
     KL-divergence ≤ 0.05.
   - `train_lstm.py` is importable and `--dry-run` exits with code 0.

### 3.2 Success criteria for Phase 2

These are the exit gates *for the LSTM as a Red Team*, not for stage
detection:

- **G1.** Train/val cross-entropy curves are monotonically decreasing
  on a moving average and **never differ by more than 25 %** at any
  epoch (no overfitting collapse).
- **G2.** Token-level top-1 accuracy on synthetic val ≥ 0.55 *(uniform
  baseline = 0.20)*. We do **not** target high macro-F1 here because
  the IMPACT absorbing state is part of the grammar.
- **G3.** KL-divergence between the LSTM's empirical 5×5 transition
  matrix (estimated from 10 000 generated sequences) and the
  ground-truth synthetic matrix ≤ 0.05.
- **G4.** Cosine similarity between the LSTM's stage-frequency vector
  on generated sequences and the train-split stage distribution ≥ 0.90.
- **G5.** All 266 existing tests still pass, plus the ~6 new ones
  added in §3.1.

If any gate fails, the figure is *not* produced — we go back and fix
the model rather than ship a misleading curve.

## 4 — Out-of-scope for Phase 2 (deliberately)

- Real-flow classification — moves to Phase 4.
- Reward-shaping, episode-lifecycle bugs, MTTC fix — Phase 3.
- Adversarial training of the LSTM against the RL agent — Phase 5+.
- Transformer encoder ablation — Phase 5.

## 5 — Sequencing inside Phase 2

| Step | Output | Estimated cost |
|------|--------|----------------|
| 5.1  | This PLAN.md (committed) | done |
| 5.2  | Refactor: train-split prior, drop MLflow duplication, fix `prepare_data` default-config bug | 1 commit |
| 5.3  | New CLI `scripts/red_team/train_lstm.py` + ~6 tests | 1 commit |
| 5.4  | Train run on the full synthetic dataset (train-split prior) — produce F1 figure | 1 commit (figure + manifest) |
| 5.5  | Evaluate empirical transition matrix vs ground-truth — produce F2 figure | 1 commit (figure + manifest) |
| 5.6  | CHANGELOG entry, dataset-card cross-link | 1 commit |

Total expected: **5 commits**, ~600–900 lines added/modified.

## 6 — Risks I'm watching

- **R1** — the train-split prior may differ from the snapshot prior by
  enough that a trained-on-snapshot model gets discarded. Mitigation:
  the rebuild is reproducible via `make build-split-indices`.
- **R2** — `WeightedRandomSampler` + `class_weights` interaction was
  written defensively but is brittle. We will *not* rely on it for
  Phase 2; uniform sampling is correct here because the synthetic
  episode distribution is already controlled by the
  `EpisodeGenerator` config.
- **R3** — KL-divergence threshold (0.05) is judgmental. If the
  unconstrained LSTM cannot hit it, we relax to 0.10 *and document
  the relaxation in the figure caption*. Examiners care more that the
  threshold is published before the experiment than that it's small.

## 7 — What I will *not* re-architect now

- `TransitionMask` — orthogonal, used only at inference for grammar
  enforcement. Keep as is.
- The `realization_engine` — that's how `(stage, row)` pairs are
  rendered into actual flow vectors for the RL environment. It will
  be revisited in Phase 3 / Phase 4 because of OOD-class exclusion,
  but is fine for Phase 2.

---

**Open question for the next session:** are you happy with re-aligning
F2 from "confusion matrix on real flows" to "empirical transition
matrix vs ground truth"? It is the technically correct figure for the
LSTM as it stands; the original F2 is moved to F11 in Phase 4 where it
belongs. If you'd rather keep the old labeling, swap names in
`docs/thesis_results_map.md`; the substance is identical.
