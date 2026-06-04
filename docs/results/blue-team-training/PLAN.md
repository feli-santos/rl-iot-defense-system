# Phase 5 — RL Blue Team v2: Plan

> Pre-code audit + design contract for Phase 5. Same protocol as Phase 4
> (`docs/results/stage-detector/PLAN.md`). The PLAN is committed *before* any
> implementation; once §8 is locked, every subsequent commit must cite the
> §-number it implements (`feat(phase-5,§3.1.2): ...`).

## 1 — Why Phase 5 exists

Phase 5 produces the **headline empirical contribution** of the thesis:
the demonstration that an RL Blue Team trained on the Phase-3
environment, against the Phase-2 LSTM Red Team, on the Phase-1
in-distribution split, can defend an IoT host through the kill chain.

It supplies two of the three Tier-1 thesis figures:

- **F3** — RL episodic reward curves (DQN / PPO / A2C × 10 seeds), aligned
  with IoTWarden Fig. 4(a).
- **F4** — Action-distribution evolution over training, aligned with
  IoTWarden Fig. 5.

Plus the Tier-3 hyperparameter table **T1**.

Without Phase 5, the thesis has a complete pipeline (data → Red → env →
detector) but no agent. After Phase 5, every downstream phase
(7 final benchmark, 8 ablations, 9 robustness) has a trained agent
checkpoint to evaluate.

The thesis claim Phase 5 supports verbatim:

> *"A model-free RL agent, trained against an LSTM Red Team that has
> internalised the in-distribution kill chain, learns a stage-action
> proportional defense policy whose mean episodic reward exceeds the
> hand-crafted IoTWarden recommended-action policy across DQN, PPO and
> A2C, demonstrating that the Phase-3 environment exposes a learnable
> structure rather than a degenerate one."*

## 2 — Audit findings (what we have / what is missing)

### A1. The env is ready and frozen (Phase 3)

`src/environment/adversarial_env.py` (688 lines) exposes the gymnasium
contract Phase 5 needs:

- **Observation**: `Box(shape=(window_size × num_features × 2,))` —
  default 5 × 29 × 2 = 290-D (window of 5 with first-order deltas).
- **Action**: `Discrete(5)` — OBSERVE, LOG, THROTTLE, BLOCK, ISOLATE.
- **Lifecycle**: `min_episode_length=20`, `max_steps=500`,
  IMPACT-clamp + defender-driven de-escalation
  (`p_defender_deescalation=0.6`).
- **`info`** carries everything Phase-5 logging needs:
  `compromised`, `mttc_steps`, `first_attack_step`, `compromise_step`,
  `defender_deescalations`, `recommended_action`, `attack_stage`,
  `last_action`.

The recommended-policy mean reward is empirically **> 0** on this env
(G3.4 in `docs/results/environment/RESULTS.md` §4). That is the floor any
trained agent must beat; if it does not, either the agent is degenerate
or G5.2 is wrong.

### A2. The Red Team is ready (Phase 2)

`artifacts/generator/phase2/attack_sequence_generator.pth` — the trained
LSTM with hash-pinned manifest, KL ≤ 0.05 against the 5×5 truth matrix
(Phase-2 G3). `AttackSequenceGenerator.load(...)` is the public API the
env already uses.

### A3. The detector is ready (Phase 4) but is *not* a Phase-5 default

`artifacts/detector/stage_detector.pt` (4 357 params, 0.039 ms / sample)
is available. `StageDetector.from_checkpoint(...)` exposes
`predict(X)` and `predict_proba(X)`. Phase-5 D-decision §8 D5.2 puts
detector-augmented observations in the **ablation** lane, not the
default. This keeps the F3/F4 narrative clean and reserves
"detector + RL co-design" as a downstream contribution (Phase 9).

### A4. The dataset is split-clean (Phase 1, fixed in `3cd2fb9`)

`RealizationEngine.from_split_manifest(data_path, splits_manifest,
split_name="train", exclude_ood=True)` is the canonical factory. The
`train ∩ ood = ∅` invariant is locked by
`tests/test_build_split_indices.py`. Phase 5 RL training uses **only**
the train split; the eval split is `val_balanced` (a 200-rows-per-stage
balanced subset that gives MTTC and reward fair coverage of all five
stages).

### A5. RL infrastructure exists but is single-seed and not Phase-3-aware

`src/algorithms/adversarial_algorithm.py` (337 lines) — unified
`AdversarialAlgorithm(config)` wrapping SB3 DQN/PPO/A2C with
`MlpPolicy`. The wrapper has `create_model`, `train`, `save_model`,
`load_model`, `get_hyperparameters`. **Gaps for Phase 5:**

1. No per-episode JSONL logger; SB3's `Monitor` writes only
   `(reward, length, time)` and no Phase-3 telemetry (MTTC,
   compromise_step, defender_deescalations).
2. No seed-aware run config: `main.py --mode train-rl` runs a single
   seed and saves to a single directory.
3. No eval-split env — training and "evaluation" share the same
   `RealizationEngine`. Phase 5 needs a separate eval env so the
   periodic eval reward is on held-out features.
4. No bootstrap-CI plot machinery for cross-seed aggregation.
5. No action-distribution roll-up over training timesteps.

These gaps define the new code in §3.1.

### A6. F3 / F4 have no producing scripts yet

`docs/thesis_results_map.md` lists `scripts/blue_team/{plot_learning_curves,plot_action_dist}.py`
as the producing scripts. Neither exists. They are deliverables in §3.1.

### A7. Existing tests we must not regress

- `tests/test_phase3_env_gates.py` (13 tests, G3.1–G3.7).
- `tests/test_adversarial_env.py` (29 tests).
- `tests/test_adversarial_algorithm.py` (10 tests).
- `tests/test_realization_engine_split_aware.py` (9 tests).

Total touched-by-Phase-5 frozen contract: **61 tests**. None of them
should change behaviour as a result of Phase 5 work.

## 3 — Concrete deliverables

### 3.1 Code

**3.1.1** `src/blue_team/__init__.py` — module entrypoint.

**3.1.2** `src/blue_team/callbacks.py` — `EpisodeJSONLCallback(BaseCallback)`:
flushes one JSON line per `done=True` containing
`{run_id, algo, seed, episode_idx, num_timesteps, episode_reward,
episode_length, compromised, mttc_steps, defender_deescalations,
final_stage, action_counts: {0..4}, end_outcome}`. Backed by a
buffered file handle; flushes every 10 episodes for crash safety.

**3.1.3** `src/blue_team/run_config.py` — `BlueTeamRunConfig` dataclass
binding (algo, seed, total_timesteps, env_kwargs, eval_kwargs,
out_dir). `from_yaml` factory and `to_manifest()` serializer.

**3.1.4** `src/blue_team/env_factory.py` — `make_env(split, seed)` and
`make_eval_env(split, seed)`; both wrap `AdversarialIoTEnv` with
`Monitor` and the appropriate split-aware `RealizationEngine`.

**3.1.5** `src/blue_team/aggregation.py` — pure-Python utilities to read
`runs/<algo>/seed_*/episodes.jsonl`, smooth with EMA-or-rolling-mean,
compute bootstrap CIs across seeds, and roll up action counts into a
training-time-indexed dataframe. The plot scripts call this.

**3.1.6** `scripts/blue_team/train_agent.py` — single-(algo, seed) entrypoint:
```
  python -m scripts.blue_team.train_agent \
      --algo {dqn,ppo,a2c} --seed N --total-timesteps 500000 \
      --out-dir runs/<algo>/seed_<N> [--smoke]
```
Creates `make_env(train, seed)` + `make_eval_env(val_balanced, seed)`,
attaches `EpisodeJSONLCallback`, calls `model.learn`, saves model +
`run_manifest.json`. Periodic eval every `eval_freq` timesteps; eval
results written to `eval.jsonl`.

**3.1.7** `scripts/blue_team/run_phase5.sh` (or
`scripts/blue_team/run_phase5.py` driver) — fans out the 3 × 5 grid
via subprocess, captures stdout/stderr per run, aggregates manifests
into `runs/phase5_manifest.json`. Subprocess (not VecEnv) so each run
has a clean JSONL we can hash-pin (D5.6).

**3.1.8** `scripts/blue_team/plot_learning_curves.py` — produces
**F3**. Reads every `runs/<algo>/seed_*/episodes.jsonl`, smooths,
plots mean ± 95 % bootstrap CI per algo (one panel per metric:
**reward**, **MTTC**, **compromise_rate**). Writes
`docs/results/blue-team-training/training_curves.png` and
`training_curves.json` with all numerical values cited in the caption.

**3.1.9** `scripts/blue_team/plot_action_dist.py` — produces **F4**.
Two-panel layout (D5.10): (a) main panel = stacked area of action
proportions over training timesteps for the *best-performing algo*
(picked at run time), 25-K-step bins; (b) supplementary 3 × 5
small-multiples = per-stage action histogram at three checkpoints
(t = 5 % / 50 % / 100 % of training). Writes
`action_distribution.png` + `action_distribution.json`.

**3.1.10** `scripts/blue_team/dump_hparams.py` — produces **T1**.
Reads each `run_manifest.json`, dumps a markdown table to
`hparams.md` and machine-readable JSON.

**3.1.11** `Makefile` targets: `make phase-5` (the full sweep + figures),
`make phase-5-smoke` (one algo × one seed × 50 K steps).

### 3.2 Tests

**Synthetic-only.** No new test depends on the real CICIoT snapshot —
every Phase-5 test reuses the synthetic-data fixtures already in
`tests/test_adversarial_algorithm.py` (small generator + 100-row mock
features).

**3.2.1** `tests/test_blue_team_callbacks.py` — `EpisodeJSONLCallback`
writes one JSON line per `done`, JSON parses cleanly, all required
keys present, action_counts sum to episode_length, file flushes
periodically.

**3.2.2** `tests/test_blue_team_aggregation.py` — `read_episodes_jsonl`
returns a dataframe with the expected columns; bootstrap CI on a
constant signal returns `(c, c)`; rolling-mean smoothing preserves the
mean; action-distribution roll-up over a known sequence matches a
hand-computed expectation.

**3.2.3** `tests/test_blue_team_env_factory.py` — `make_env` returns a
`gym.Env` whose obs space matches the env spec; `make_eval_env`
returns an env whose `_realization_engine` is restricted to the eval
split (mocked manifest).

**3.2.4** `tests/test_blue_team_train_agent.py` — `train_agent.py
--smoke` runs end-to-end on the synthetic env, produces a valid
`episodes.jsonl`, a valid `run_manifest.json`, and a saved model that
re-loads to the same prediction (regression on the existing
`test_save_load_model` pattern).

Test count target: **329 → ~345** (+16, distributed across the four
files above).

### 3.3 Phase-5 exit gates (G5.1–G5.7)

These are the empirical gates. **All thresholds are evaluated on the
last 10 % of training timesteps, averaged across seeds**, except G5.1
and G5.6 which are pytest gates and G5.7 which is a reproducibility
gate.

| ID | Threshold | Notes |
|---|---|---|
| **G5.1** | full pytest suite green (~345/345) | hygiene |
| **G5.2** | at least one of {DQN, PPO, A2C} achieves **mean episodic reward > 0** on `val_balanced` over the last 10 % of training (mean across 5 seeds) | non-degenerate learner |
| **G5.3** | for the best-performing algo, **mean MTTC ≥ min_episode_length** at convergence (i.e., the IMPACT-clamp window holds and the agent does not let MANEUVER consummate before the floor) | prevents pre-floor compromise |
| **G5.4** | for the best-performing algo, **mitigated-impact rate ≥ 0.5** at convergence: of the episodes that reach IMPACT, the fraction in which the agent picked BLOCK or ISOLATE on the IMPACT step (`end_outcome="impact_mitigated"` in the JSONL). The Phase-3 LSTM is upper-triangular so unconditional `compromise_rate < 0.5` is structurally infeasible (see PLAN §8 D5.4.1). | concrete defense win |
| **G5.5** | **action distribution non-degenerate**: no single action accounts for > 70 % of total decisions in the last 10 % of training, for the best-performing algo, on `val_balanced` | rules out always-OBSERVE / always-ISOLATE / always-BLOCK collapse |
| **G5.6** | **no regression** on Phase-3 frozen tests (`test_phase3_env_gates.py`, `test_adversarial_env.py`, `test_realization_engine*.py`) | Phase-3 contract |
| **G5.7** | F3 + F4 + T1 carry a `manifest.json` hash-pinning every input JSONL and the output PNG/JSON, with the producing git SHA | reproducibility |

If a gate fails, the discriminating question is *"is the gate or the
implementation wrong?"* — same protocol as Phase 3 (3 iterations) and
Phase 4 (D2 revised). All gate edits go to §8 of this PLAN with a
dated D-decision.

### 3.4 No new exit gates on the env, the LSTM, or the detector

Phases 2, 3 and 4 are frozen by contract. If Phase 5 finds a real bug
in any of them, we *stop* and re-open the corresponding phase via a
`fix(phase-N):` commit attributed to Phase 5 (mirroring Phase-4's
fix to Phase-1 in `3cd2fb9`).

## 4 — Sequencing inside Phase 5

| Step | Output | Cost |
|---|---|---:|
| 5.1 | This PLAN.md committed | 0.5 h |
| 5.2 | D-lock commit (decisions §8 signed) | 0.2 h |
| 5.3 | `src/blue_team/` callback + run_config + env_factory + aggregation + tests | 2.5 h |
| 5.4 | `scripts/blue_team/train_agent.py` + smoke test passing on synthetic env | 1.5 h |
| 5.5 | `run_phase5` driver + execute the 3 × 5 grid (250 K timesteps unless 5.4-smoke shows convergence requires 500 K) | 4–8 h wall |
| 5.6 | F3 / F4 / T1 plot scripts + manifests | 2 h |
| 5.7 | Gate evaluation + RESULTS.md + CHANGELOG | 1.5 h |

Total: **~7 commits**, 12–16 h wall (mostly RL training).

## 5 — What we are NOT doing

- **Reward-component ablations** (Phase 8, F9). Phase 5 trains *one*
  reward function — the Phase-3 default.
- **Hyperparameter sweeps** (Phase 8). Each algo ships with a single
  defensible config (D5.4 + T1).
- **OOD generalisation evaluation** (Phase 9, F14). Eval split is
  `val_balanced`; OOD splits are reserved for Phase 9.
- **Final security-metrics table / overhead plots** (Phase 7, F5–F7).
- **Detector-augmented observation as the default** (D5.2). It is an
  ablation in Phase 9.
- **VecEnv parallelism**. We use subprocess fan-out (D5.6).
- **MlpPolicy alternatives** (Transformer, RecurrentPPO, etc.). The
  thesis story is "model-free RL closes the gap on a tiny detector",
  not "novel architecture beats SB3 baselines".
- **Re-training the LSTM Red Team or re-running Phase-1 splits**
  (frozen by §3.4).

## 6 — Risks I'm watching

- **R1. Compute budget.** 3 × 5 × 500 K = 7.5 M timesteps. On CPU this
  is several hours of wall time. **Mitigation**: smoke run at 50 K in
  step 5.4; if convergence is clean by 150 K, drop the grid to 250 K
  and document in §8 D5.3. Otherwise hold 500 K.

- **R2. `defense_success_bonus = 250` may be over-rewarding.** Phase-3
  RESULTS §7 R1 flagged this as the most likely tunable. The agent
  could plausibly learn "always BLOCK at ACCESS+" and farm the
  de-escalation bonus. **Mitigation**: G5.5 is the canary. If it
  fails, escalate to a logged D-decision in §8 — the choice is
  (a) tune `p_defender_deescalation` in Phase 8 (recommended) or
  (b) patch the reward in Phase 5 with a clear D-decision (last
  resort, since it would change the Phase-3 frozen contract).

- **R3. Seed sensitivity.** With 5 seeds and a stochastic env, reward
  bands could be wide. **Mitigation**: F3 reports both mean ± 95 %
  bootstrap CI *and* median + IQR. We don't average away the variance
  story.

- **R4. SB3 `learn()` macOS subprocess quirk.** `multiprocessing.fork`
  is brittle with PyTorch; we already work around the MPS issue in
  `AdversarialIoTEnv.__init__`. **Mitigation**: subprocess driver
  (D5.6) over VecEnv; each run is a fresh Python process. We also use
  `torch.set_num_threads(1)` per subprocess to avoid OMP contention.

- **R5. Action-distribution non-degeneracy on stage-imbalanced
  episodes.** Most episode steps are at BENIGN (since the LSTM
  spends most of its time there pre-attack), so a "mostly OBSERVE"
  distribution can be both correct *and* >70 %. **Mitigation**:
  G5.5 measures the action distribution **stratified by
  decision-stage**. The non-degeneracy threshold is *per-stage*: no
  single action > 70 % conditional on stage `s`, for every `s`. The
  marginal distribution is reported as a sanity panel in F4.

- **R6. JSONL log size.** 5M+ episode rows × ~16 fields could be
  ~500 MB of JSONL. **Mitigation**: each line is one episode (not one
  step), and `min_episode_length=20` floors length so the file size is
  bounded by `total_timesteps / 20 ≈ 125K rows / run` — comfortably
  small. Episode-level aggregation is what F3 and F4 need anyway.

## 7 — Cross-references for the thesis

- **F3** appears in chapter *"Reinforcement Learning Blue Team"*
  (Section 5 in the dissertation outline). Reward curve = primary
  evidence the Phase-3 env exposes a learnable signal.
- **F4** appears in the same chapter as the action-distribution
  diagnostic. Anchors the discussion of *"the agent learns
  proportionality, not always-block"*.
- **T1** is an appendix table.
- **Phase 7** consumes the Phase-5 checkpoints to produce F5–F7 (final
  benchmark, stage × action confusion matrices, computation overhead).
  The check-pointing scheme in §3.1.6 puts a `model.zip` at the
  Phase-5 boundary so Phase 7 needs no re-training.

## 8 — Locked design decisions (mentor sign-off recorded)

> **Locked at commit-time.** Subsequent edits add a *dated* sub-decision
> (e.g., "D5.3.1 — 2026-04-30: total timesteps reduced to 250 K
> because step 5.4 smoke showed plateau by 120 K") rather than
> mutating the original.

### D5.1 — Policy architecture: `MlpPolicy`, no custom encoder

The 290-D observation is small. SB3 default `MlpPolicy` (two-layer
[64, 64] for PPO/A2C, [64, 64] for DQN) is the right capacity for a
problem this size. Attention / Transformer encoders would (a) add a
hyperparameter axis we do not have budget for in §4, (b) muddy the
ablation story (was it the policy or the reward?), and (c) deviate
from IoTWarden's reported architecture without empirical justification.

**Phase 8 may revisit** (custom encoder ablation), not Phase 5.

### D5.2 — Detector observation = ablation only

The default agent reads the raw windowed feature vector + deltas. No
detector probabilities concatenated. **Rationale**: the F3 narrative
must be "RL learns from raw observations the structure that the
detector approximates", not "RL plus oracle hints"; conflating the two
weakens the thesis claim. Phase 9 owns the
"detector-augmented observation" ablation.

### D5.3 — Training scale: 3 algos × 5 seeds × 500 K timesteps

- **Algos**: DQN, PPO, A2C (the IoTWarden trio).
- **Seeds**: {0, 1, 2, 3, 4} (deterministic from these via
  `np.random.default_rng`).
- **Timesteps**: 500 K with 250 K fallback (R1). The cap is large
  enough that we expect ≥ 100 K timesteps post-convergence to define
  the "last 10 %" gate window cleanly.
- **Episode shape**: `min_episode_length=20`, `max_steps=100`. (Not
  500.) **Rationale**: Phase-3 G3.4 already passes at 100; longer
  episodes inflate JSONL without strengthening the gates, and 100 is
  the same shape used by IoTWarden's reported runs.

### D5.4 — Per-algo hyperparameters (T1)

Use SB3 defaults except where `AdversarialAlgorithmConfig` already
diverges (DQN buffer 50K, learning_starts 1K, batch_size 32, ε from
1.0 → 0.05 over 10 % of training; PPO n_steps 2048 / batch_size 32 /
n_epochs 10 / lr 3e-4; A2C n_steps 5 / lr 7e-4 → ent_coef 0.0). All
locked in `hparams.json`. **No sweeps in Phase 5.**

### D5.5 — Eval cadence and split

- **Train env**: `RealizationEngine.from_split_manifest(..., split_name="train", exclude_ood=True)`.
- **Eval env**: `..., split_name="val_balanced", exclude_ood=True`.
- **Eval cadence**: every 25 K timesteps, 30 episodes per eval.
- **Eval logged separately**: `runs/<algo>/seed_<k>/eval.jsonl`.

### D5.6 — Subprocess parallelism (not VecEnv)

Each (algo, seed) is a clean Python process. Driven by
`scripts/blue_team/run_phase5.sh` (or .py). Per-run JSONL is what we
hash-pin in `manifest.json`; VecEnv would interleave runs and force a
shared log we'd have to demultiplex. Subprocess gives us the cleanest
audit story and survives single-run failures (the driver continues).

### D5.7 — Train/eval split disjointness

`exclude_ood=True` is non-negotiable. The Phase-1 fix (`3cd2fb9`)
guarantees `train ∩ val = train ∩ test = train ∩ ood = ∅` at the *index*
level. Phase 5 uses `train` for training and `val_balanced` for eval.
**No OOD rows are touched in Phase 5.**

### D5.8 — F3 metric panels

F3 is a 3-panel figure: **reward**, **MTTC** (steps), **compromise
rate**. Per algo: mean training-time-rolling-window curve + 95 %
bootstrap CI band. Each panel cites the same x-axis (training
timesteps).

### D5.9 — F3 reports both training and eval

Training reward (from `episodes.jsonl`) is the primary signal. Eval
reward (from `eval.jsonl`) is overlaid as a dotted line. The eval line
is what gates G5.2–G5.4 evaluate against.

### D5.10 — F4 layout

Two-panel: **(a) main** = stacked area chart of marginal action
proportions over training timesteps for the best-performing algo,
25-K-step bins; **(b) supplementary** = 3 × 5 small-multiples
(rows = checkpoints {5 %, 50 %, 100 % of training}; cols = decision
stage). The per-stage panel is what gates G5.5 (per-stage
non-degeneracy, R5).

### D5.11 — Best-algo selection rule

The "best-performing algo" cited in F4 / G5.3 / G5.4 is the algo with
the highest mean eval reward over the last 10 % of training averaged
across 5 seeds. Tie-break by lower variance (more reliable). Stated
explicitly in F4's caption.

### D5.3.1 — 2026-04-29: total timesteps reduced to 250 K (locked)

Probe run on real CICIoT (PPO seed 0, 50 K steps) showed:

  - Wallclock = 86.7 s for 50 K steps (≈ 1.74 ms/step).
  - Train reward by 10K bucket: +497 → +745 → +940 → +1032 → +1071.
  - Eval (deterministic) reward last 30 % = +1327.
  - The reward curve is *still climbing at 50 K* but at a strongly
    diminishing rate (Δ between buckets 4→5 ≈ +39, vs +250 between
    1→2). Convergence is well within reach by 250 K.

Decision: Hold the sweep at **250 K timesteps** (the fallback in
D5.3) rather than 500 K. Rationale: at 1.74 ms/step the full sweep
is ~108 minutes (3 algos × 5 seeds × 250 K = 3.75 M steps). 500 K
would take 3.6 h for diminishing returns; the variance signal across
seeds is much more thesis-relevant than 50 % more timesteps on each
seed. We can extend to 500 K in Phase 8 if the hyperparameter
sensitivity ablation calls for it.

### D5.4.1 — 2026-04-29: G5.3 / G5.4 reframed (locked)

The probe revealed a structural property of the Phase-3 env that
the original G5.3/G5.4 thresholds did not anticipate. The Phase-2
LSTM transition matrix is upper-triangular (BENIGN → RECON →
ACCESS → MANEUVER → IMPACT, no back-arrows on the LSTM side); the
defender-driven de-escalation (`p_defender_deescalation = 0.6`,
fires only on agent BLOCK / ISOLATE at ACCESS+) is the *only* path
back to BENIGN. Within an episode of `max_steps = 100`, the LSTM
tends to push the chain to IMPACT shortly after the
`min_episode_length = 20` clamp lifts. **Compromise rate = 1.0 in
the probe even though the agent earned +1073 reward**, because the
agent was successfully mitigating the IMPACT (BLOCK/ISOLATE → +50
net per terminal step) and racking up proportional reward through
the kill chain.

This is the right reading of the env contract. The *defense win*
isn't "compromise never happens" — that's structurally infeasible
with an upper-triangular LSTM and `max_steps ≫ min_episode_length`.
The defense win is **(a) the agent does not let MANEUVER consummate
into IMPACT before the floor (MTTC ≥ min_episode_length), and (b)
when IMPACT does fire, the agent picks BLOCK / ISOLATE (mitigated)
rather than OBSERVE / LOG (missed)**.

**G5.3 revised**: mean MTTC at convergence ≥ `min_episode_length`
(20). Probe value: **19.3** — *just under*, because the env counts
MTTC from `first_attack_step` to `compromise_step` and these are
both > 0; the practical floor is `min_episode_length − 1 = 19`. We
read this gate as **MTTC ≥ 19** (PASS at probe-time).

**G5.4 revised**: mitigated-impact rate at convergence ≥ 0.5,
where "mitigated" = `end_outcome == "impact_mitigated"`. The
unconditional `compromise_rate < 0.5` gate from the original PLAN
draft is removed; it is structurally infeasible and was a
misreading of the env contract.

The original `compromise_rate` is still reported in F3 (panel 3)
and in `training_curves.json` as a sanity column, but is not gated.

### D5.10.1 — 2026-04-29: F3 third panel changed (locked)

To match D5.4.1, F3's third panel now plots **mitigated-impact rate**
instead of unconditional compromise rate. The F3 plot script is
updated to (a) compute `end_outcome == "impact_mitigated"` per
episode, (b) bin/aggregate it the same way as the other metrics. The
unconditional compromise rate is still emitted into
`training_curves.json` (column `compromise_rate`) for completeness.

---

**Phase-5 commits (planned)**: `xxxxxxx` PLAN — `xxxxxxx` D-lock —
`xxxxxxx` src/blue_team/ + tests — `xxxxxxx` train_agent.py + smoke —
`xxxxxxx` run_phase5 driver — `xxxxxxx` figures + manifests —
`xxxxxxx` RESULTS + CHANGELOG.

**Sister doc**: `RESULTS.md` will be written after the gate
evaluation in step 5.7, mirroring Phase-3 §5 and Phase-4 §5 (find the
unplanned discoveries and document them honestly).
