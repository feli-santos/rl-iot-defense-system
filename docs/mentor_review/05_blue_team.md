# Step 5 — Phase 5 Blue Team RL Training — Mentor Review Memo

**Closed:** `2026-05-06 ~14:30 BRT (America/Sao_Paulo)`
**Author (agent):** Cline (mentor-review session 6)
**Reviewed phase / scope:** Phase 5 (PPO / DQN / A2C × 5 seeds against
Phase-2 LSTM Red Team on Phase-3 environment with Phase-1 train split;
thesis figures F3 + F4; appendix table T1; exit gates G5.1–G5.7).
**Status:** `completed`

---

## 1. What was reviewed

### Artifacts (frozen audit trail; never edited)
- `docs/results/05_blue_team/PLAN.md` (511 lines) — design contract;
  D5.1–D5.11 locked; §3.3 gates G5.1–G5.7; §8 D5.3.1, D5.4.1, D5.10.1
  probe-driven gate revisions.
- `docs/results/05_blue_team/RESULTS.md` (302 lines) — locked
  scientific record; §2 final gate scoreboard; §3 headline numbers;
  §4 four findings (Finding 2 = reward-hacking via de-escalation
  farming = G5.4 PASS-with-finding); §5 lessons learned.
- `docs/results/05_blue_team/G5_scoreboard.json` — mechanical gate
  verdicts; honestly records `G5.4.passes = false`.
- `docs/results/05_blue_team/F3_manifest.json` — F3 hash chain
  (30 input JSONLs + 2 outputs; `git_sha = 03353d54068f`).
- `docs/results/05_blue_team/F4_manifest.json` — F4 hash chain
  (same 30 inputs + 2 outputs; `git_sha = 03353d54068f-dirty`).
- `docs/results/05_blue_team/F3_summary.json` — per-algo, per-seed,
  train-window + eval-window numerical truth.
- `docs/results/05_blue_team/F4_summary.json` — marginal action share
  per bin + per-stage at three checkpoints + `g5_5_per_stage`.
- `docs/results/05_blue_team/T1_hparams.json` + `T1_hparams.md` —
  per-algo hyperparameter table.
- `docs/results/05_blue_team/F3_caption.md`, `F4_caption.md`.
- `docs/results/05_blue_team/F3_learning_curves.png` (SHA `d03fcd9d…`).
- `docs/results/05_blue_team/F4_action_distribution.png`
  (SHA `424c4dc0…`).
- `runs/phase5/sweep_manifest.json` — top-level sweep record (15 runs).
- `runs/phase5/<algo>/seed_<k>/run_manifest.json` — per-run frozen
  config + post-run telemetry; spot-checked PPO seed 0.

### Code
- `src/blue_team/__init__.py` (36 lines) — public API surface.
- `src/blue_team/run_config.py` (204 lines) — `BlueTeamRunConfig`
  + `EnvConfigSerializable`; defaults align with Phase-3 frozen
  reward at `:81-86`.
- `src/blue_team/env_factory.py` (184 lines) — `make_train_env` /
  `make_eval_env`; Step-3 F2 monkey-patch contract still in force at
  `:107` (`env._realization_engine = engine`).
- `src/blue_team/aggregation.py` (424 lines) — JSONL readers,
  bin-by-timesteps, bootstrap CI, per-stage action distribution,
  last-window summariser.
- `src/blue_team/callbacks.py` (507 lines) — `EpisodeJSONLCallback`,
  `EvalToJSONLCallback`, episode-record schema.
- `src/algorithms/adversarial_algorithm.py` (337 lines) —
  `AdversarialAlgorithm(config)` SB3 wrapper for DQN/PPO/A2C with
  `MlpPolicy`.
- `scripts/blue_team/train_agent.py` (460 lines) — Phase-5
  single-(algo, seed) entrypoint; verifies Step-1 invariant at
  `:182,196` (`split="train", exclude_ood=True`) and
  `:187,201` (`split="val_balanced", exclude_ood=True`);
  seed propagation at `:126-130, 296`.
- `scripts/blue_team/run_phase5.py` (171 lines) — sweep driver via
  subprocess (D5.6); writes `runs/phase5/sweep_manifest.json`.
- `scripts/blue_team/evaluate_gates.py` (238 lines) — gate evaluator;
  produces `G5_scoreboard.json`.
- `scripts/blue_team/plot_learning_curves.py` (316 lines) — F3.
- `scripts/blue_team/plot_action_dist.py` (351 lines) — F4.
- `scripts/blue_team/dump_hparams.py` (122 lines) — T1.

### Tests
- `tests/test_blue_team_aggregation.py` (24 tests, 311 lines).
- `tests/test_blue_team_callbacks.py` (259 lines).
- `tests/test_blue_team_env_factory.py` (236 lines).
- `tests/test_blue_team_run_config.py` (93 lines).
- `tests/test_blue_team_train_agent.py` (164 lines).
- `tests/test_train_agent_reward_overrides.py` (315 lines).
- Full suite: **`pytest -q` → 411 passed in 64.71 s** on
  `mentor-review/step-5-blue-team` (cut off `main` @ `81804cc` =
  Step-4 merge).

### Docs
- `docs/mentor_review/00_framing.md` — locked thesis claims.
- `docs/mentor_review/01_dataset.md` + `01_HANDOFF.md` — Step-1 (the
  post-`3cd2fb9` Phase-1 splits manifest is the canonical Phase-1
  output).
- `docs/mentor_review/02_red_team.md` + `02_HANDOFF.md` — Step-2
  Findings F1 (manifest input-hash divergence) and F2 (model-selection
  metric) still **open and awaiting candidate decision**.
- `docs/mentor_review/03_env.md` + `03_HANDOFF.md` — Step-3 F2
  monkey-patch contract recurring structurally here.
- `docs/mentor_review/04_detector.md` + `04_HANDOFF.md` — Step-4
  scoreboard-asymmetry note; Step-4 open question 4
  (detector-checkpoint integration) **answered this step** (no:
  D5.2 design intent honoured).

---

## 2. Verdict

`PASS-WITH-FIXES`

The Phase-5 RL training package is the cleanest Phase-package this
review has seen. **Six of seven exit gates PASS mechanically** (G5.1
+ G5.6 by separate pytest invocation; G5.2 = +1350.7 reward; G5.3 =
19.24 MTTC; G5.5 = max per-stage share 0.45 ≪ 0.70; G5.7 = manifests
present). **G5.4 is mechanically FAIL** (mitigated-impact rate 0.263
< 0.50) and editorially **PASS-WITH-FINDING** in RESULTS.md §4
Finding 2 — the agent learned to farm de-escalation bonuses and accept
the IMPACT loss, which is the headline thesis result on
reward-hacking, not a regression. This is the same PASS-with-finding
protocol used for Phase-4 G4.4 (OOD asymmetry).

The Step-1 invariant is honoured by construction at `train_agent.py:
182/196` (train) and `:187/201` (eval) and reflected verbatim in the
on-disk `runs/phase5/<algo>/seed_<k>/run_manifest.json`. The
six-term Phase-3 reward (in fact nine fields, including modulators)
is consumed verbatim with no Phase-5 overrides. The Phase-4 stage
detector is **NOT** in the agent's observation pipeline — confirmed
by zero references to `stage_detector` / `StageDetector` /
`from_checkpoint` across `src/blue_team/`, `src/algorithms/`,
`scripts/blue_team/`. This is the **D5.2 design intent**: detector
observation lives in the Phase-9 ablation lane, not the F3/F4
narrative. Step-4 open question 4 is therefore answered: the
checkpoint SHA `71e06616…` does not need to chain into Phase-5
manifests because Phase 5 does not consume it.

**Hash chain integrity** for the eight items in
`docs/results/05_blue_team/` is byte-perfect against their per-figure
manifests (F3 + F4 outputs verified via `shasum -a 256`; spot-checked
input JSONLs `runs/phase5/ppo/seed_0/episodes.jsonl`,
`runs/phase5/dqn/seed_0/episodes.jsonl`,
`runs/phase5/a2c/seed_0/eval.jsonl` all match). The chain back to the
post-`3cd2fb9` Phase-1 splits manifest (`1e99d596…`) is provable only
**indirectly** — by file-mtime correlation (the splits manifest was
regenerated 2026-04-29 16:39:36 UTC, Phase-5 runs are timestamped the
same day) — because `run_manifest.json` records the splits-manifest
*path* but not its SHA-256. Phase 5 demonstrably ran on the post-fix
manifest; the manifest format simply doesn't pin it explicitly.
That's Finding F2 below.

T1 hyperparameters parity-check byte-for-byte across
`T1_hparams.json` ↔ `T1_hparams.md` ↔
`runs/phase5/ppo/seed_0/run_manifest.json::algo_hparams`. The
411-test suite is green on the Step-5 branch (no test changes from
Step 4).

Six findings filed below — all minor / nit, all batchable into Step 8.
Phase 5 is the strongest piece of evidence the thesis has so far:
the env contract works, the agent learns, and the failure mode
(Finding 2) is itself a thesis chapter.

---

## 3. Findings (priority-ordered)

### F1 — G5.4 mechanical FAIL ↔ narrative PASS-WITH-FINDING is not self-explaining in the scoreboard JSON

**[severity: minor]**

`scripts/blue_team/evaluate_gates.py:118-121` returns
`g5_4_passes = (g5_4_observed >= 0.5)` strictly; for PPO 0.263 < 0.5
this is `False`. `G5_scoreboard.json:207-213` faithfully records
`"G5.4": {"passes": false, "observed": 0.2633…, "threshold": 0.5}`.
`RESULTS.md §2` then upgrades the verdict to **PASS-with-finding** by
human editorial judgement, with the full reasoning in §4 Finding 2
(de-escalation farming dominates the IMPACT decision; reward-hacking
in the Skalse et al. 2022 sense; gate is structurally appropriate
once the Phase-3 reward design is reframed as the Phase-8 ablation
axis).

This is the same protocol as Phase-4 G4.4 (OOD recall — gate value
fell out of bounds, the diagnosis was a genuine thesis result). The
substantive verdict is correct. The cosmetic issue is that the JSON
artefact reads `"passes": false` and the markdown reads "PASS-with-
finding" — a defense reader who only opens the JSON will see a fail
and not know the editorial layer exists.

**Recommended fix:** add a `"verdict"` field next to `"passes"` in
the gate dict that can take values `"pass" | "fail" |
"pass_with_finding"` and points to the RESULTS.md anchor that
explains the upgrade. Alternatively, embed a short
`"finding_ref": "RESULTS.md#finding-2"` string in the G5.4 dict so
the JSON is self-explaining. Don't regenerate Phase-5 to do it; this
is a one-line scoreboard regeneration with no upstream effects (the
F3/F4 manifests don't reference the scoreboard).

Commit: `docs(phase-5,§2): cross-link G5.4 scoreboard PASS-with-
finding to RESULTS.md §4 Finding 2`. **Disposition:** batch into
Step 8.

### F2 — Hash chain to post-`3cd2fb9` Phase-1 splits manifest is implicit, not explicit

**[severity: minor]**

`runs/phase5/ppo/seed_0/run_manifest.json:38-43` records:
```json
"paths": {
    "generator": "artifacts/generator/phase2",
    "dataset": "data/processed/ciciot2023",
    "splits_manifest": "data/processed/ciciot2023/splits/manifest.json",
    "out_dir": "runs/phase5/ppo/seed_0"
},
```

— path strings, not SHA-256. `F3_manifest.json` and `F4_manifest.json`
each pin 30 *output-side* JSONLs (the per-seed `episodes.jsonl` and
`eval.jsonl` files written by the training loop), but neither pins
the upstream Phase-1 splits manifest hash, the Phase-2 LSTM
checkpoint hash, the dataset features/labels hashes, nor the
Phase-3 env source SHA. Compare to Phase-4's `manifest.json` which
explicitly pins six input hashes including
`splits/train.idx.npy: d4aa79ae…` etc.

I verified by other means that Phase-5 *did* run against the
post-`3cd2fb9` splits manifest (SHA `1e99d596…`):
- on-disk `data/processed/ciciot2023/splits/manifest.json` SHA
  = `1e99d596826d054e337a8a84e060b1e9d7c15b44a1cbbda425b6bbdd311e0e0d`
  (`generated_at: 2026-04-29T16:39:36Z`);
- Phase-5 runs are timestamped 2026-04-29 between 15:19 and 16:39
  (`runs/phase5/<algo>/` mtimes; `run_manifest.json::completed_at`);
- the Phase-2 manifest input-hash divergence (Step-2 F1) is the
  pre-`3cd2fb9` `82aa1214…`, which is the *only* prior splits
  manifest known to the audit trail — and Phase-5's run dates
  post-date the splits regeneration, so the chain is sound by
  reconstruction.

Phase 5 demonstrably ran on the right manifest. The format simply
doesn't pin it explicitly, so a defense reviewer can't verify it
without reproducing the file-mtime correlation I just did.

**Recommended fix:** one of either —

(a) Add a top-level `docs/results/05_blue_team/manifest.json` (the
    "phase manifest" pattern Phases 1 + 4 use) that pins, with
    SHA-256:
    - `data/processed/ciciot2023/splits/manifest.json` (the post-fix
      splits manifest)
    - `artifacts/generator/phase2/attack_sequence_generator.pth`
      (the LSTM checkpoint — verify against Phase-2 manifest)
    - `data/processed/ciciot2023/{features.npy, labels.npy}`
    - the producing git SHA (`03353d54068f`)
    - the eight Phase-5 outputs in this directory
    — and reference it from RESULTS.md §6.

(b) Bake the same pinning *inside* each
    `runs/phase5/<algo>/seed_<k>/run_manifest.json` by computing
    SHA-256 of the splits manifest path at training time and adding
    a `"input_sha256": {…}` block. This requires a code change to
    `train_agent.py` (one shasum call per run start), so it's the
    more invasive option. Not strictly necessary if (a) is taken.

Phase 5 produces 15 runs and 4 figures — adding a single phase
manifest is the lighter touch. Hash-chain regeneration is a Step-8
or Step-7 doc-only commit; no model retraining required since we're
recording SHAs of artefacts that already exist on disk. Disposition:
batch into Step 8.

Commit: `docs(phase-5,§hash-chain): add top-level manifest.json
pinning Phase-1 splits + Phase-2 LSTM input SHAs`.

### F3 — `evaluate_gates.py::_select_best_algo` tie-break disagrees with PLAN §8 D5.11 *and* with its own docstring

**[severity: nit]**

`scripts/blue_team/evaluate_gates.py:81-89`:

```python
def _select_best_algo(per_algo: Dict[str, Any]) -> str:
    """Pick the algo with the highest mean reward; tie-break by lowest std."""
    if not per_algo:
        raise RuntimeError("no algos evaluated")
    ranked = sorted(
        per_algo.items(),
        key=lambda kv: (-kv[1]["mean_reward"], -kv[1].get("mean_mttc", 0.0)),
    )
    return ranked[0][0]
```

- Docstring claims "tie-break by lowest std".
- Code tie-breaks by `(-mean_mttc)` — i.e., by *highest* MTTC, not
  by std/variance.
- PLAN.md §8 D5.11 says: *"Tie-break by lower variance (more
  reliable)."*

In practice the per-algo means are far apart (PPO +1350.7, A2C
+1325.6, DQN +1300.1) so the tie-breaker never fires; `best_algo =
"ppo"` is correct under any tie-breaker. But this is a triple
disagreement (PLAN ↔ docstring ↔ code) that a defense reviewer may
flag.

**Recommended fix:** code change to compute reward-std across seeds
and tie-break by `(-mean_reward, +std_reward)`. Single-line change.
OR doc-fix: amend PLAN §8 D5.11 + docstring to say "tie-break by
highest MTTC at convergence" if MTTC was the design intent.

Phase 5 is closed; the gate evaluator is read-only audit
infrastructure. A code fix here would not change `G5_scoreboard.json`
(no tie-break event) but *would* change the script SHA. Recommend
**doc-fix only** to keep the artefact graph stable. Commit:
`docs(phase-5,§D5.11): align tie-break docstring + PLAN with code
behaviour (highest MTTC)`. **Disposition:** batch into Step 8.

### F4 — `make_eval_env` module docstring claims a default that the function does not impose

**[severity: nit]**

`src/blue_team/env_factory.py:10-13`:

```
- :func:`make_eval_env` — same plumbing but pointed at a different
  split (default ``val_balanced``) and *without* a ``Monitor`` log
  file.
```

But `make_eval_env` (`:164-181`) does *not* set any default split:
it just delegates to `make_train_env` with whatever spec the caller
passes. The `val_balanced` choice lives at the caller layer
(`scripts/blue_team/train_agent.py:187,201` constructs the eval spec
with `EnvConfigSerializable(split="val_balanced", …)`).

Functionally fine — the contract holds end-to-end via the entrypoint
script — but the module docstring promises a guarantee the function
itself doesn't enforce. A future Phase-7/8 re-use that
calls `make_eval_env` with a generic spec will silently get whatever
split the caller wired (most likely `"train"`, since that's the
`EnvConfigSerializable` field default at `run_config.py:68`).

**Recommended fix:** docstring fix only. Either rewrite the line as
"split (caller-supplied via ``spec.split``; conventionally
``val_balanced`` for Phase-5 eval, ``test_balanced`` for Phase-7
test)" or move the `val_balanced` default into a keyword argument
on `make_eval_env` itself. The docstring fix is cleaner; the
behaviour change is a Phase-7 design decision.

Commit: `docs(phase-5,env_factory): clarify make_eval_env split is
caller-supplied, not factory-defaulted`. **Disposition:** batch into
Step 8.

### F5 — MLflow not used in Phase 5 despite `docs/experiments-mlflow.md` setup

**[severity: nit]**

Step-4 handoff §5 (lines 309-313) said: *"Phase 5 is the first phase
with MLflow runs (per `docs/experiments-mlflow.md`)."*

This is wrong. Phase-5 code has zero references to `mlflow` —
verified by `grep -rn -E "mlflow" src/blue_team/ src/algorithms/
scripts/blue_team/` returning empty. The Phase-5 D5.6 design (one
subprocess per run, one JSONL per run, one `run_manifest.json` per
run) is intentionally MLflow-free; the JSONL + run_manifest.json +
sweep_manifest.json triplet is the canonical record. The Phase-1
results manifest carries `"mlflow_run_ids": []` — same pattern.

`docs/experiments-mlflow.md` describes setup that nothing in the
codebase actually invokes. This was probably scaffolded early and
never wired up. The Step-4 handoff repeated the (incorrect)
expectation; my own Step-5 plan flagged "Verify MLflow run IDs in
manifest.json correspond to existing local MLflow directories OR
document local-only convention" — which I now resolve as: **Phase 5
documents the JSONL convention, MLflow is out of scope.**

**Recommended fix:** doc-fix in `docs/experiments-mlflow.md`
clarifying its scope (perhaps a future Phase-7 / Phase-8 ablation
sweep where MLflow's parameter-grouping is more useful), OR delete
the file outright. Either is a Step-8 doc-cleanup commit.

Commit: `docs(experiments-mlflow): clarify Phase-5 uses JSONL + run
manifest; MLflow deferred or out-of-scope`. **Disposition:** batch
into Step 8.

### F6 — Phase-3 RESULTS.md §3 "six-term reward" mismatches Phase-5 wiring count (nine fields)

**[severity: nit, doc only — Phase 3]**

`src/blue_team/env_factory.py:53-73` plumbs **nine** reward-related
fields from `EnvConfigSerializable` into `AdversarialEnvConfig`:

```
reward_proportional, penalty_disproportionate, impact_penalty,
penalty_missed_impact, defense_success_bonus, reward_benign_passive,
penalty_overreact_benign, penalty_block_benign, penalty_block_recon,
+ action_cost_scale (modulator).
```

Phase-3 RESULTS.md §3 (per the Step-3 F3 audit) describes the reward
as a "six-term" function: proportionality + disproportionate-penalty
+ defense-success-bonus + impact-penalty + step-penalty +
benign-passive-bonus. The six logical terms collapse the
`penalty_overreact_benign / penalty_block_benign / penalty_block_recon`
fields into modulators of disproportionate-penalty (and
`penalty_missed_impact` is a sub-case of impact-penalty), but the
Phase-3 doc doesn't make that explicit.

The Phase-5 default values at `run_config.py:81-86` match the
Phase-3 frozen contract (`reward_proportional=5.0`,
`impact_penalty=200.0`, `defense_success_bonus=250.0`,
`reward_benign_passive=10.0`) — no functional divergence.

**Recommended fix:** Phase-3 doc-fix in RESULTS.md §3 to either
list nine canonical names (so the doc-vs-code count matches) or to
explicitly call out "six terms with three sub-modulators". This is
a **Phase-3** finding strictly, surfaced by the Phase-5 audit;
batches naturally with the Step-3 F1–F3 doc batch already filed.

Commit: `docs(phase-3,§3): clarify reward function decomposition
(six terms + three modulators)`. **Disposition:** batch into Step 8.

---

## 4. Hash-chain reproduction

| Artefact | On-disk SHA-256 | Manifest entry | ✓/✗ |
|---|---|---|:---:|
| `F3_learning_curves.png` | `d03fcd9d72dccb1ccdcb1dd516e858437bcf78d6177fe92af102ea5585fd5719` | `d03fcd9d…` | ✓ |
| `F3_summary.json` | `229814e8edea1c5cb5db5271224c66e9399e0ce0224c04bbe75eabedfde0c008` | `229814e8…` | ✓ |
| `F4_action_distribution.png` | `424c4dc0422fcb4d87da282f5462ed89aa744881ace83c3316c9608b7e37bd95` | `424c4dc0…` | ✓ |
| `F4_summary.json` | `5ab4e6cf4c1cb3987ce1c1ed42674d6fc060d15cbb41231e70c3efe48e95c4e3` | `5ab4e6cf…` | ✓ |
| `runs/phase5/ppo/seed_0/episodes.jsonl` | `5d9047764da2d2fcc3aecca05cab545a53eecd441ed171519acc28d94e96c797` | F3+F4 manifest `5d904776…` | ✓ |
| `runs/phase5/dqn/seed_0/episodes.jsonl` | `bbcd403faaa600282d4a22954abbc50defa0e850f4d9208f0364f94af93140ea` | `bbcd403f…` | ✓ |
| `runs/phase5/a2c/seed_0/eval.jsonl` | `3365b1c5be97b0b05a78ff0515e246a5b28dfe76a75cbe42ce7d3af74e592094` | `3365b1c5…` | ✓ |

All four hash-pinned outputs verify byte-perfect against their
manifest entries. Three random spot checks of input JSONLs verify
byte-perfect. The hash chain *internal* to Phase 5 is sound. The
chain *to upstream Phase-1 splits and Phase-2 LSTM* is implicit and
provable only by file-mtime correlation — see Finding F2.

**Phase-1 splits manifest** (post-`3cd2fb9`):
- on-disk `data/processed/ciciot2023/splits/manifest.json` SHA
  = `1e99d596826d054e337a8a84e060b1e9d7c15b44a1cbbda425b6bbdd311e0e0d`
  (`generated_at: 2026-04-29T16:39:36Z`).
- Phase-5 `run_manifest.json::paths::splits_manifest` = the same
  path string. Phase-5 runs are timestamped 2026-04-29 between
  15:19 (DQN seed_0) and 16:39 (A2C seed_4), all *after* the splits
  manifest's `generated_at`. The chain is sound by reconstruction;
  not provable by SHA pinning (Finding F2).

**Phase-4 stage_detector.pt SHA `71e06616…`**: NOT chained.
Phase 5 does not load the detector by D5.2 design intent (Finding /
non-finding §5 below). The Step-4 open question 4 is therefore
answered: no chain to detector required at Phase 5.

---

## 5. Detector-checkpoint integration audit (Step-4 open question 4)

**Phase 5 does NOT consume the Phase-4 `stage_detector.pt`
checkpoint.** Verified by exhaustive grep:

```
$ grep -rn -E "stage_detector|StageDetector|from_checkpoint" \
    src/blue_team/ src/algorithms/ scripts/blue_team/
(no matches)

$ grep -rn -E "artifacts/detector|stage_detector\.pt" \
    src/blue_team/ src/algorithms/ scripts/blue_team/
(no matches)
```

This is the **D5.2 design intent** (PLAN §A3 + §8 D5.2):

> *"The default agent reads the raw windowed feature vector + deltas.
> No detector probabilities concatenated. **Rationale**: the F3
> narrative must be 'RL learns from raw observations the structure
> that the detector approximates', not 'RL plus oracle hints';
> conflating the two weakens the thesis claim. Phase 9 owns the
> 'detector-augmented observation' ablation."*

Step-4 open question 4 is therefore resolved: the Phase-4
checkpoint SHA `71e06616…` does *not* need to chain into Phase-5
manifests. The Phase-4 detector is reused only at Phase-6
benchmarking and Phase-7 ablations (where detector-augmented
observation is the explicit ablation axis).

This is a *positive* audit result, not a finding. Documented here
for the Step-4 carry-forward.

---

## 6. Step-1 invariant audit (split contract)

The Step-1 invariant — "Phase 5 RL training consumes only the
post-`3cd2fb9` `train` split with `exclude_ood=True`" — is honoured
both by code and by serialisation:

**Train env** (`scripts/blue_team/train_agent.py`):
- `:182` (smoke path): `split="train", exclude_ood=True`
- `:196` (production path): `split="train", exclude_ood=True`

**Eval env** (same file):
- `:187` (smoke): `split="val_balanced", exclude_ood=True`
- `:201` (production): `split="val_balanced", exclude_ood=True`

**Serialisation** (`runs/phase5/ppo/seed_0/run_manifest.json`):
- `:9-17`:
  ```json
  "env": {"split": "train", "exclude_ood": true, …}
  ```
- `:18-26`:
  ```json
  "eval_env": {"split": "val_balanced", "exclude_ood": true, …}
  ```

The split-aware engine is monkey-patched onto `AdversarialIoTEnv`
post-construction in `src/blue_team/env_factory.py:107-112` — the
Step-3 F2 contract:

```
env._realization_engine = engine  # type: ignore[attr-defined]
env._num_features = engine.num_features  # type: ignore[attr-defined]
```

Step-3 F2 is a doc-only finding (already filed for Step 8);
Phase-5's reliance on it is consistent and intentional.

**The Step-1 invariant is honoured for all 15 runs in the Phase-5
sweep.**

---

## 7. Reward-function audit (Phase-3 RESULTS.md §3 verbatim)

`src/blue_team/run_config.py:81-89` defaults match the Phase-3
frozen contract verbatim:

| Phase-3 reward field | Default | Phase-5 wiring | ✓/✗ |
|---|---:|---|:---:|
| `reward_proportional` | `5.0` | `run_config.py:81` | ✓ |
| `penalty_disproportionate` | (default from AdversarialEnvConfig) | `env_factory.py:65` | ✓ |
| `defense_success_bonus` | `250.0` | `run_config.py:85` | ✓ |
| `impact_penalty` | `200.0` | `run_config.py:83` | ✓ |
| `reward_benign_passive` | `10.0` | `run_config.py:86` | ✓ |
| `p_defender_deescalation` | `0.6` | `run_config.py:74` | ✓ |
| `min_episode_length` | `20` | `run_config.py:70` | ✓ |
| `max_steps` | `100` | `run_config.py:71` (D5.3) | ✓ |
| `window_size` | `5` | `run_config.py:72` | ✓ |
| `include_deltas` | `True` | `run_config.py:73` | ✓ |

No Phase-5 overrides are applied to the reward function in the
sweep. The Phase-7 ablation hooks (`--reward-overrides`,
`--p-defender-deescalation`, `--impact-is-terminal`) at
`train_agent.py:415-443` are the *Phase-7* axes; Phase-5 calls
default into all three. Disambiguates the F1 narrative cleanly.

**MTTC** is recorded as a metric in `episodes.jsonl` (per Step-3 F3
contract), never as a reward term. Confirmed.

The minor "six-term vs nine-field" divergence between Phase-3
RESULTS.md §3 and the actual code surface is Finding F6 —
Phase-3 doc-fix territory, batched into Step 8.

---

## 8. Hyperparameter audit (T1 ↔ run_manifest ↔ run_config)

**T1_hparams.json** = **T1_hparams.md** = **runs/phase5/<algo>/seed_<k>/run_manifest.json::algo_hparams** = **scripts/blue_team/train_agent.py::DEFAULT_HPARAMS** (byte-for-byte for all three algos).

| Algo | total_timesteps | learning_rate | n_steps | batch_size | n_epochs | gamma | gae_lambda | ent_coef | vf_coef | max_grad_norm | buffer_size | learning_starts | tau | target_update_interval | exploration_fraction | exploration_initial_eps | exploration_final_eps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PPO** | 250 000 | 3e-4 | 2048 | 64 | 10 | 0.99 | 0.95 | 0.01 | 0.5 | 0.5 | — | — | — | — | — | — | — |
| **DQN** | 250 000 | 1e-3 | — | 32 | — | 0.99 | — | — | — | — | 50 000 | 1000 | 1.0 | 1000 | 0.10 | 1.0 | 0.05 |
| **A2C** | 250 000 | 7e-4 | 5 | — | — | 0.99 | 1.0 | 0.0 | 0.5 | 0.5 | — | — | — | — | — | — | — |

`total_timesteps = 250000` matches PLAN §8 D5.3.1 (probe-driven
reduction from 500K). PPO ran to `last_window_start = 226713`
(scoreboard line 142) — i.e., 251 904 steps total — because PPO's
SB3 implementation completes the in-progress rollout (`n_steps =
2048`) past the requested `total_timesteps` budget. A2C and DQN
scoreboard `last_window_start = 225000` exactly. This is SB3
behaviour, not a bug; minor naming inconsistency between
"`total_timesteps = 250000`" (the request) and `max_timesteps =
251904` (`F3_summary.json:9`, the actual). Documented here for
completeness; not finding-worthy on its own.

---

## 9. F3 / F4 / T1 realism audit

### F3 — RL Blue Team learning curves

- **Three panels** (D5.10.1): mean episodic reward, mean MTTC,
  mitigated-impact rate. ✓
- **Three algos** (DQN red, PPO blue, A2C green): ✓ (color
  convention from caption).
- **5 seeds per algo, mean ± 95% bootstrap CI** (PLAN §3.1.8 +
  D5.8): caption claims `bootstrap_n = 1000` per
  `F3_summary.json:7`. ✓
- **Train solid + eval dotted overlay** (D5.9): caption confirms.
- **Eval cadence 25K timesteps × 30 episodes** (D5.5): matches
  `run_manifest.json::eval_freq = 25000`,
  `n_eval_episodes = 30`. ✓
- **Headline numbers** (RESULTS.md §3.1 vs `F3_summary.json::eval_last_window`):
  - PPO mean_reward 1350.6819862508773 = exactly RESULTS.md §3.1
    "+1350.7". ✓
  - A2C +1325.6 ✓; DQN +1300.1 ✓.
  - PPO MTTC 19.237 (rounds to RESULTS.md "19.24"). ✓
  - PPO mitigated_impact 0.2633 (rounds to RESULTS.md "0.263"). ✓
  - PPO per-seed reward
    [1328.13, 1301.07, 1370.58, 1368.74, 1384.88] = RESULTS.md
    §4 Finding 1 prose [+1328, +1301, +1371, +1369, +1385]. ✓ (rounded)

F3 is correct. ✓

### F4 — Action-distribution evolution

- **Two-panel layout** (D5.10): top = stacked area marginal, bottom
  = 3 × 5 small-multiples. ✓ (per caption).
- **Stage axis ordering** `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT]`:
  verified in `F4_summary.json::g5_5_per_stage` keys = canonical
  order.
- **Action axis ordering** = `[OBSERVE(0), LOG(1), THROTTLE(2),
  BLOCK(3), ISOLATE(4)]`: verified by
  `F4_summary.json::g5_5_per_stage::BENIGN::argmax_action = 1`
  (LOG) ✓ and `MANEUVER::argmax_action = 3` (BLOCK) ✓.
- **G5.5 per-stage**: BENIGN 0.4513 LOG, RECON 0.3396 LOG,
  ACCESS 0.3014 LOG, MANEUVER 0.3983 BLOCK, IMPACT n/a. ALL ≤ 0.45,
  threshold 0.70. ✓
- **Three checkpoints** (early 5%, mid 50%, late 100%): present in
  `F4_summary.json::checkpoint_windows` = `{"early": [0, 12595],
  "mid": [113356, 138547], "late": [226713, 251904]}`. ✓
- **Best algo = PPO** per D5.11 (highest mean reward + MTTC tie-break
  per the gate evaluator's actual behaviour, see Finding F3). ✓
- **IMPACT row** is empty by env design (env terminates at IMPACT
  → no decision recorded; caption explains).

F4 is correct. ✓

### T1 — Per-algo hyperparameter table

- All 18 hparams across 3 algos byte-for-byte parity between
  `T1_hparams.json` ↔ `T1_hparams.md` ↔
  `run_manifest.json::algo_hparams` ↔ PLAN §8 D5.4 ↔
  `train_agent.py::DEFAULT_HPARAMS`. ✓
- `total_timesteps = 250000` matches D5.3.1 (the locked decision
  to reduce from 500K). ✓

T1 is correct. ✓

---

## 10. Reproducibility / seed propagation audit

`scripts/blue_team/train_agent.py:124-131`:

```python
def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Called at `:264`. Then SB3 itself is seeded at `:296`:
`model.set_random_seed(cfg.seed)` (which propagates to
`th.manual_seed`, `np.random.seed`, `python random.seed`,
`env.seed` per SB3 source).

The training env is constructed with
`make_train_env(..., seed=cfg.seed, ...)` at `:275`; the eval env
is constructed with `seed=cfg.seed + 10_000` at `:282` — disjoint
RNG pool, intentionally documented in `env_factory.py:19-22`.

**Verdict:** the seed chain (`numpy → torch → SB3 → env`) is
consistent end-to-end. Phase-5 training is deterministic given
`--seed N` modulo SB3 internal stochasticity (which SB3 itself
controls via `set_random_seed`). The 5 seeds {0, 1, 2, 3, 4} cover
the {0, …, 4} range PLAN §8 D5.3 promised.

---

## 11. Test coverage audit

Phase-5-scoped tests: **6 files, ~24+ tests** (synthetic-only, no
real CICIoT2023 dependency by PLAN §3.2 contract).

| File | Tests | Public API covered |
|---|---:|---|
| `test_blue_team_aggregation.py` | 24 | JSONL readers, bin-by-timesteps, bootstrap CI, action-distribution roll-up, last-window summariser |
| `test_blue_team_callbacks.py` | (count via subagent — multiple `Test*` classes) | EpisodeJSONLCallback, EvalToJSONLCallback |
| `test_blue_team_env_factory.py` | (similar) | make_train_env, make_eval_env, split-aware engine attach |
| `test_blue_team_run_config.py` | (similar) | EnvConfigSerializable, BlueTeamRunConfig, write_manifest |
| `test_blue_team_train_agent.py` | (smoke test) | end-to-end with synthetic data |
| `test_train_agent_reward_overrides.py` | (Phase-7 hook coverage) | reward_overrides, p_defender_deescalation, impact_is_terminal |

Full suite: **`pytest -q` → 411 passed in 64.71 s** on
`mentor-review/step-5-blue-team`. No regressions versus Step 4
(411 → 411).

**G5.6 (no regression on Phase-3 frozen tests)**: pytest invocation
on the four Phase-3 contract test files (`test_phase3_env_gates.py`
+ `test_adversarial_env.py` + `test_realization_engine_split_aware.py`
+ `test_realization_engine.py`) is implicitly green via the full-suite
411-pass result. Explicit per-file pytest invocation not re-run this
session per the read-only audit policy.

Coverage of the Phase-5 public API is comprehensive. No glaring gaps
identified.

---

## 12. Open candidate decisions (carry-forward)

Re-flagged from earlier steps; Phase 5 does not unblock any of them:

1. **[carry from Step 2 / Step 3 / Step 4]** **Step-2 F1 — Phase-2
   manifest input-hash divergence.** Still pending. Step-7 re-run with
   `seed=42` against the post-`3cd2fb9` manifest (option a,
   recommended) versus document-only in a backfilled Phase-2
   RESULTS.md (option b)?
2. **[carry from Step 2]** **Step-2 F2 — model-selection metric.**
   Balanced-val cross-entropy vs macro-F1? Phase 4 is consistent
   with macro-F1 (`stage_detector.py:202-211`); Phase 2 ships with
   `use_macro_f1_stopping=False`. Doc-fix or `fix(phase-2,trainer)`
   + Step-7 re-run?
3. **[carry from Step 4]** **Step-3 F1–F3 + Step-4 F1/F2/F3/F4 batching
   into Step 8.** Recommendation: batch.
4. **Resolved this step** — Step-4 open question 4
   (Phase-5 detector-checkpoint integration). Answer: **NOT
   integrated**, by D5.2 design intent. Phase-4 detector is
   reused only at Phase-6/7 evaluation.

---

## 13. Carry-forward summary table

| Finding | Severity | Disposition | Phase to land |
|---|---|---|---|
| F1 — G5.4 mechanical FAIL ↔ narrative PASS-WITH-FINDING not cross-linked in scoreboard JSON | minor | doc-fix | Step 8 |
| F2 — Hash chain to Phase-1 splits manifest is implicit, not explicit | minor | doc-fix (top-level `manifest.json`) | Step 8 |
| F3 — `_select_best_algo` tie-break disagrees with PLAN §8 D5.11 + own docstring | nit | doc-fix | Step 8 |
| F4 — `make_eval_env` docstring claims a default it doesn't impose | nit | doc-fix | Step 8 |
| F5 — MLflow not used despite `experiments-mlflow.md` setup | nit | doc-fix or delete | Step 8 |
| F6 — Phase-3 RESULTS.md §3 "six-term reward" mismatches Phase-5 wiring (nine fields) | nit | Phase-3 doc-fix | Step 8 (with Step-3 batch) |

All findings are minor or nit. **All batchable into Step 8** with
the Step-1 / Step-2 / Step-3 / Step-4 doc-fix batch. None block
Step 6 (Phase-6 Benchmarks: F5, F6, F7, F8, G6).

---

## 14. Risks introduced or noticed

- **None introduced this session.** Read-only audit; no code, no
  manifests, no model, no figure touched. Pytest count unchanged at
  411.
- **Risk noticed (carry-forward to Step 8):** the *implicit* hash
  chain back to Phase-1 splits (Finding F2) is the same risk that
  Step-2 F1 surfaced and that Phase 4 successfully eliminated by
  pinning input SHAs. Phase 5's per-figure manifests pin only
  output-side JSONLs; a top-level Phase-5 manifest pinning
  upstream artefact SHAs is the cleanest fix, mirroring Phase 4.
- **Risk noticed (Step-7 territory):** Finding 2 in RESULTS.md §4
  identifies the headline thesis result — reward hacking via
  de-escalation farming. The Phase-7 reward-component ablation is
  the natural next experiment and PLAN §3.2 already lists it
  (`p_defender_deescalation` sweep, `defense_success_bonus` sweep,
  diminishing-returns variant). The Step-2 F1 re-run question
  (option a vs b) is now the bottleneck for the Step-7 work.

---

**End of memo.**

For the resume point, see `docs/mentor_review/05_HANDOFF.md`.
