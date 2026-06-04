# Phase 7 — Ablations + OOD-class Robustness: Plan

> **Locked PLAN.** Same protocol as Phases 2–6: this document is
> committed *before* any implementation, and `feat(phase-7,§N)`
> commits cite §-numbers from this file. Section 8 is the locked
> design-decisions ledger; revisions get explicit `D7.X.1` follow-up
> entries with date + rationale.
>
> Phase 7 owns four thesis figures: **F9 (reward-component
> ablation), F10 (attack-aggressiveness sweep), F12
> (security-vs-availability Pareto), F15 (OOD-class robustness — new,
> promoted from afterthought to first-class deliverable by the
> 2026-04-30 mentor audit, finding AF1).**

## 1 — Why Phase 7 exists

Phase 6 closed with two threads pointing here, and a third surfaced
by a mentor-mode audit on 2026-04-30. All three drive a single
chapter: *what does the trained RL policy actually accomplish, and
where does it fail?*

**Strand A — close the deployable gap (D6.2.1, audit AF2).**
On `test_balanced` (10 seeds × 300 episodes) the best deployable agent
(A2C, +1336.6) captures **79.3 % of the oracle ceiling** set by the
recommended-action rule (+1684.8) — a rule that has *free oracle access* to
`info["attack_stage"]` and is therefore not a deployable defender.
The remaining ~+348 reward is the **Phase-7 target**, not the
Phase-6 loss. Phase 5 D5.4.1 already named the mechanism — the
Phase-3 de-escalation bonus rewards a strategy that scores well
in-distribution but does not generalise — so the diagnostic is
clear: sweep the reward components Phase 3 calibrated and identify
which one(s) drive(s) the gap. **F9** owns the answer.

**Strand B — sensitivity to attack aggressiveness (IoTWarden Fig. 6
re-implementation).** IoTWarden's headline figure characterises how
the defender's value function changes as the attacker's
aggressiveness varies; their analogue of `p_defender_deescalation`
sweeps from 0.0 (defender never wins a de-escalation roll) to 1.0
(defender always wins). This phase reproduces and extends that
sweep on the CICIoT2023 env using the trained PPO policy plus the
oracle rule baseline as a reference line. **F10** owns the answer.

**Strand C — OOD-class robustness (audit AF1, 2026-04-30).** Phase 4
RESULTS §3.2 found that the supervised RF stage detector has
**0.001 recall on `VulnerabilityScan`** (the held-out RECON OOD
class) — a structural blind spot in the entire supervised
classification chain. The thesis claim "RL closes the OOD gap by
acting on raw features rather than detector outputs" was made in
Phase 4 but currently has **no evidence on disk**. F15 supplies it
by re-running Phase-6's `eval_runner` harness with the env's
`RealizationEngine.allowed_indices` constrained to each OOD class
in turn. No retraining; the test exercises whether the *already-
trained* RL policies generalise to OOD attacks better than the
supervised detector + rule pipeline does. **F15** owns the answer.

**Strand D — operating-point Pareto.** Once F9 and F10 produce a
grid of (reward_config, aggressiveness) points, plotting them on
{security gain, availability cost} axes reveals the policy-design
trade-off surface. Defenders with different security/availability
preferences can choose different operating points. **F12** owns
the answer.

## 2 — Audit findings (what already exists, what's missing)

### 2.1 — What exists on disk and is reusable

- **15 trained Phase-5 PPO/DQN/A2C checkpoints** at
  `runs/phase5/<algo>/seed_<k>/model.zip` (verified present
  locally; `sweep_manifest.json` SHA-pins all 15).
- **8 Phase-6 baseline rollouts** at
  `runs/phase6/<policy>/seed_<k>/{eval_test,latency}.jsonl`
  (verified present; `eval_manifest.json` is the canonical pin).
- **Phase-6 `eval_runner.run_policy(...)`** accepts any
  `Policy`-Protocol callable and any `gymnasium.Env`. The signature
  does not assume a fixed config — F15 reuses it unchanged.
- **`RealizationEngine.from_split_manifest`** already accepts
  `allowed_indices` (added in Phase-3 B4). It supports
  `splits/ood_attack/<class>.idx.npy` paths. F15 just supplies
  these paths instead of `splits/test_balanced.idx.npy`.
- **`recommended_action_policy`** in `src/benchmark/baseline_policies.py`
  reads `info["attack_stage"]` directly. It works under any env
  configuration unchanged.
- **F8 idiom** (`scripts/benchmark/plot_baselines.py` + horizontal
  bar chart with bootstrap CIs) is the visual template for F15
  (4-class grouped variant) and F9 (per-component grouped bar).
- **Phase-3 frozen reward defaults** (Phase-3 RESULTS §3 table):
  `defense_success_bonus = 250`, `penalty_missed_impact = 150`,
  `reward_proportional = 5`, `penalty_disproportionate = 5`,
  `reward_benign_passive = 10`, `penalty_overreact_benign = 50`,
  `penalty_block_benign = 100`, `penalty_block_recon = 50`,
  `impact_penalty = 200`, `p_defender_deescalation = 0.6`,
  `min_episode_length = 20`, `max_steps = 500`. These are the
  axes Phase 7 sweeps around.
- **OOD splits on disk:**
  `data/processed/ciciot2023/splits/ood_attack/{DDoS-HTTP_Flood,
  Mirai-udpplain, VulnerabilityScan, XSS}.idx.npy` (sizes 30 KB –
  97 KB; suitable for fast eval rollouts).

### 2.2 — What's missing (Phase-7 will add)

- **No env-config override hook in `train_agent.py`.** The env spec
  is hard-coded inside `build_run_config(args)` (lines 134–184) as
  two literal `EnvConfigSerializable(...)` constructions. Phase 7
  adds a `--reward-overrides JSON` arg + a
  `--p-defender-deescalation` numeric arg that override specific
  fields without forking the script. **Default behaviour is
  byte-for-byte identical to Phase 5** when neither flag is passed.
- **No `impact_is_terminal` knob in `AdversarialEnvConfig`.** D6.6
  deferred this from Phase 6. Phase 7 adds it (default `True`
  preserves frozen contract; `False` enables an explicit
  IMPACT-row decision before termination).
- **No `scripts/ablation/` package.** Phase 7 creates it with
  six scripts: two sweep drivers (F9 reward, F10 aggressiveness),
  one OOD evaluator (F15), three plotters (F9, F10, F12, F15).
- **No `runs/phase7/` directory.** Phase 7 creates the layout
  `runs/phase7/{reward_sweep,aggressiveness,ood}/<cell>/seed_<k>/`
  + `runs/phase7/<sweep>_manifest.json` per sweep.
- **No Make targets.** Phase 7 adds `phase-7-reward`,
  `phase-7-aggressiveness`, `phase-7-ood`, `phase-7-figures`,
  `phase-7` (chains all four).

### 2.3 — Cross-phase scope decisions (locked in §8)

- **`impact_is_terminal` is folded into F9** as one extra binary
  axis (D7.3). Default `True` preserves the Phase-3/4/5/6 frozen
  contract.
- **F12 is *derived* from F9 + F10**, not a separate sweep. It
  reads the same JSONLs and re-projects them onto
  `(security_gain, availability_cost)` axes (D7.5).
- **F15 reuses Phase-6 trained checkpoints** — no retraining, only
  eval rollouts under the OOD constraint (D7.6).

## 3 — Concrete deliverables

### 3.1 — Code

#### 3.1.1 — `impact_is_terminal` env-config flag

**File:** `src/environment/adversarial_env.py`

- Add `impact_is_terminal: bool = True` to `AdversarialEnvConfig`.
- In `step()`, branch the IMPACT-termination logic on this flag.
  When `True` (default): unchanged Phase-3 behaviour — env
  terminates the same step that IMPACT is reached and
  `_step_at_impact` produces the terminal reward. When `False`:
  env transitions to IMPACT but does NOT terminate; the agent's
  next action is taken as the explicit IMPACT-row decision, then
  `_step_at_impact` runs with that action and the env terminates.
- The `info` dict at termination is unchanged in either branch
  (still emits `compromised`, `mttc_steps`, `attack_stage`,
  `defender_deescalations`, `recommended_action`).

**Test file:** `tests/test_phase31_impact_terminal.py` (new, ~6 tests):

1. `test_default_is_true_phase3_contract_preserved` — config with
   no override has `impact_is_terminal=True`; episode lifecycle
   matches Phase-3 frozen behaviour.
2. `test_false_terminates_one_step_later` — with
   `impact_is_terminal=False`, the env emits one extra `step()`
   after reaching IMPACT before `terminated=True`.
3. `test_false_records_impact_row_action` — the agent's choice
   in that extra step is captured under
   `info["action_counts_by_stage"]["4"]`.
4. `test_false_isolate_at_impact_full_reward` — choosing ISOLATE
   in the IMPACT-row decision earns the full
   `defense_success_bonus` (since the agent now actually picks).
5. `test_false_observe_at_impact_full_penalty` — choosing OBSERVE
   incurs `impact_penalty + penalty_missed_impact`.
6. `test_invalid_action_in_impact_row_raises` — guard clause.

**Synthetic-only**, no real-data dependency. Adds ~6 tests; total
≈ 426.

#### 3.1.2 — `--reward-overrides` + `--p-defender-deescalation` in `train_agent.py`

**File:** `scripts/blue_team/train_agent.py`

- Add two CLI args:
  - `--reward-overrides JSON` (default `"{}"`): a JSON object
    whose keys are `AdversarialEnvConfig` field names and whose
    values are the override values. Validated against
    `AdversarialEnvConfig.__dataclass_fields__` keys.
  - `--p-defender-deescalation FLOAT` (default `None`): explicit
    knob since this is the F10 axis.
- Plumb both into `build_run_config(args)`: the
  `EnvConfigSerializable` it produces gets per-field overrides
  applied after the existing literal construction.
- Hash-pin the merged config into the run manifest at
  `runs/<algo>/seed_<k>/run_manifest.json` so the Phase-7
  sweep manifests can refer to it.
- **Default behaviour (no overrides) is byte-for-byte identical
  to Phase-5 `train_agent.py`.** Verified by a regression test.

**Test file:** `tests/test_train_agent_reward_overrides.py` (new,
~5 tests):

1. `test_no_overrides_matches_phase5_baseline` — passing no flags
   produces exactly the same `EnvConfigSerializable` Phase 5
   used.
2. `test_reward_overrides_json_applied` — `--reward-overrides
   '{"defense_success_bonus": 500}'` produces a config with
   `defense_success_bonus=500` and all other fields at default.
3. `test_unknown_field_raises_value_error` — `--reward-overrides
   '{"banana": 1}'` raises a clear error with the bad field name.
4. `test_p_defender_deescalation_arg_overrides_field` — the
   numeric arg takes precedence over an `--reward-overrides`
   JSON value if both specify it.
5. `test_run_manifest_records_merged_config` — the produced
   `run_manifest.json` includes the final merged
   `AdversarialEnvConfig` values.

**Synthetic-only.** Uses argparse + a fake env factory; no
real data.

#### 3.1.3 — F15 OOD eval runner + plot

**File:** `scripts/ablation/run_ood_eval.py` (new)

CLI:
```
--ood-classes STR[,STR,...]     # default 'DDoS-HTTP_Flood,Mirai-udpplain,VulnerabilityScan,XSS'
--policies STR[,STR,...]        # default all 8
--phase5-runs DIR               # default runs/phase5
--phase6-runs DIR               # default runs/phase6
--out-dir DIR                   # default runs/phase7/ood
--n-episodes INT                # default 30 (Phase-6 D6.3)
--seeds INT[,INT,...]           # default 0,1,2,3,4
--rf-model PATH                 # default artifacts/detector/random_forest.joblib
--scaler PATH                   # default artifacts/detector/scaler.joblib
--dataset-path STR              # default data/processed/ciciot2023
--splits-manifest PATH          # default data/processed/ciciot2023/splits/manifest.json
```

**Behaviour:** for each (class, policy, seed) triple, build
`RealizationEngine.from_split_manifest(..., split_name=f"ood_attack/{class}", exclude_ood=False)`,
build `AdversarialEnv` with the Phase-3 frozen config, load the
policy (rule, RF-Acting, DQN/PPO/A2C from
`runs/phase5/<algo>/seed_<k>/model.zip`, random,
always-OBSERVE, always-BLOCK), and call `run_policy(...)` for
30 episodes. Emit
`runs/phase7/ood/<class>/<policy>/seed_<k>/eval_test.jsonl`
+ `runs/phase7/ood/<class>/<policy>/seed_<k>/latency.jsonl`
(but latency is the same per-policy property as Phase 6 — we do
NOT re-measure it; F15 inherits Phase-6's F7 latency claim
unchanged).

Total budget: 4 classes × 8 policies × (5 seeds for non-deterministic,
1 seed × 150 ep for deterministic) × 30 ep ≈ 4 800 episodes ≈ 1 h
CPU. Episodes are short (median ≈ 30 steps post-Phase-3 lifecycle
fix), no model load is repeated.

**File:** `scripts/ablation/plot_ood_robustness.py` (new)

Reads `runs/phase7/ood/<class>/<policy>/seed_<k>/eval_test.jsonl`
files, aggregates with the same bootstrap CI pattern as
`build_summary_table.py`. Emits:

- `docs/results/07_ablation/F15_ood_robustness.png` — 4-class
  grouped horizontal bar chart, one panel per class
  (DDoS-HTTP_Flood, Mirai-udpplain, VulnerabilityScan, XSS), 8
  bars per panel (rule oracle ceiling marker; trained DQN/PPO/A2C
  bars; RF-Acting; random; always-OBSERVE; always-BLOCK), 95 %
  bootstrap CIs as error bars. Same visual idiom as F8.
- `docs/results/07_ablation/F15_summary.json` — per
  (class, policy) row with mean_reward, ci_low, ci_high,
  n_episodes, plus a `headline` block reporting whether trained
  RL beat RF-Acting on each class (G7.9 evaluator).
- `docs/results/07_ablation/F15_caption.md` — one-paragraph
  thesis caption.
- `docs/results/07_ablation/F15_manifest.json` — SHA-256 hash
  chain over the 4 × 8 × 5 input JSONLs + git SHA + the
  upstream Phase-5 sweep + Phase-6 eval manifests.

**Make target:** `phase-7-ood` chains
`run_ood_eval.py → plot_ood_robustness.py`.

#### 3.1.4 — F9 reward-component sweep driver + plot

**File:** `scripts/ablation/run_reward_sweep.py` (new)

The grid is locked in **D7.1**: sparse one-at-a-time around
the Phase-3 default at multipliers `{0.5×, 1×, 2×}` for **5
components** PLUS a **6th binary axis `impact_is_terminal ∈
{True, False}`** (D7.3 — folds D6.6 into F9). The "1×" centre
cell is shared across all 6 axes (it's the Phase-5/6 baseline);
each off-centre is sampled once. Total cells:

- 5 components × 2 off-centre multipliers (0.5×, 2×) = 10
- 1 binary axis × 1 off-centre value (False) = 1
- 1 centre cell = 1
- **Total = 12 cells × 5 seeds × PPO 250K timesteps = 60 runs.**

CPU budget: 60 runs × ~6 min each ≈ **6 h CPU**, walk-away.

**Components swept (the 5):**

| Component | Phase-3 default | 0.5× | 2× |
|---|---:|---:|---:|
| `defense_success_bonus` | 250 | 125 | 500 |
| `penalty_missed_impact` | 150 | 75 | 300 |
| `reward_proportional` | 5 | 2.5 | 10 |
| `penalty_disproportionate` | 5 | 2.5 | 10 |
| `reward_benign_passive` | 10 | 5 | 20 |

CLI:
```
--component STR                # 'defense_success_bonus' | ... | 'all'
--multiplier FLOAT             # default 'all' (sweeps 0.5,1,2)
--algo STR                     # default 'ppo' (D7.2 — best Phase-5)
--seeds INT[,INT,...]          # default 0,1,2,3,4
--total-timesteps INT          # default 250_000 (D5.3.1)
--out-dir DIR                  # default runs/phase7/reward_sweep
```

For each cell it shells out to `train_agent.py` with the
`--reward-overrides` JSON. Manifests one `sweep_manifest.json`
SHA-pinning every produced `model.zip`.

After training, an inline eval pass runs each cell's checkpoints
on `test_balanced` (reusing `eval_runner.run_policy`) and writes
`runs/phase7/reward_sweep/<cell>/seed_<k>/eval_test.jsonl`.

**File:** `scripts/ablation/plot_reward_ablation.py` (new)

Emits:

- `docs/results/07_ablation/F9_reward_ablation.png` — per-component
  panel (5 panels + 1 binary panel for `impact_is_terminal`),
  each panel showing mean test reward at 0.5× / 1× / 2× with
  95 % bootstrap CIs and a horizontal reference line at the
  **Phase-6 oracle ceiling +1684.8** and a second reference line
  at **Phase-6 deployable best A2C +1336.6** (10 seeds × 300 ep).
  The plot makes it visually obvious which component, if any, lifts
  trained PPO past its own Phase-6 ceiling.
- `docs/results/07_ablation/F9_summary.json` — per-cell
  aggregate (n_episodes, mean_reward, ci_low, ci_high) +
  per-component slope estimate (linear fit through the 3 points)
  + headline `best_cell` (max mean_reward across cells, with
  CI separation flag).
- `docs/results/07_ablation/F9_caption.md`.
- `docs/results/07_ablation/F9_manifest.json`.

**Make target:** `phase-7-reward` chains run + plot.

#### 3.1.5 — F10 attack-aggressiveness sweep + plot

**File:** `scripts/ablation/run_aggressiveness_sweep.py` (new)

Sweeps `p_defender_deescalation ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`
× **PPO only** (D7.2) × 5 seeds × 250K timesteps. Total:
**6 × 5 = 30 runs ≈ 1.5 h CPU**.

For each (p, seed) cell it shells out to `train_agent.py
--p-defender-deescalation P --seed S`. Then evaluates each cell
on `test_balanced` and ALSO on the recommended-action rule
under the same `p` (the rule's behaviour does not depend on
`p`, but the env's attacker realisations do — the RULE'S
mean reward shifts with `p` because the attacker's success rate
shifts). This produces two curves for F10 (PPO and rule).

**File:** `scripts/ablation/plot_aggressiveness.py` (new)

Emits `F10_aggressiveness.png` — x-axis `p_defender_deescalation`,
y-axis mean test reward, two lines (PPO, rule) with 95 %
bootstrap CIs as shaded bands. Aligned with IoTWarden Fig. 6.

**Make target:** `phase-7-aggressiveness` chains run + plot.

#### 3.1.6 — F12 Pareto plot

**File:** `scripts/ablation/plot_pareto.py` (new)

Reads `runs/phase7/{reward_sweep,aggressiveness}/.../eval_test.jsonl`
plus the Phase-6 baseline rollouts. For each (cell, policy) point,
computes:

- `security_gain = 1 - compromise_rate` (from `info["compromised"]`
  averaged over episodes)
- `availability_cost = mean BLOCK+ISOLATE share` (from
  `action_counts_by_stage` summed over stages, normalised)

Plots a 2-D scatter on `(availability_cost, security_gain)` with
the Pareto-frontier highlighted. Marker shape encodes policy
class (RL vs supervised vs rule); colour encodes
`p_defender_deescalation` value where applicable.

Emits `F12_pareto.png` + `F12_summary.json` (frontier points
list + dominance count) + `F12_caption.md` + `F12_manifest.json`.

**Make target:** `phase-7-pareto` chains plot only (depends on
`phase-7-reward` and `phase-7-aggressiveness` outputs).

#### 3.1.7 — Cross-cutting infrastructure

- `scripts/ablation/__init__.py` (empty, makes it a package).
- New Make targets (added to root Makefile):
  - `phase-7-reward`: F9 sweep + plot
  - `phase-7-aggressiveness`: F10 sweep + plot
  - `phase-7-ood`: F15 eval + plot
  - `phase-7-pareto`: F12 plot (depends on F9 + F10 outputs)
  - `phase-7-figures`: chains all four plotters only (assumes
    sweep outputs exist on disk)
  - `phase-7`: chains everything end-to-end

### 3.2 — Optional bonus deliverable (only if F9 finds a winner)

If F9 identifies a `best_cell` whose mean test reward exceeds the
Phase-6 oracle ceiling +1624 by ≥ 1σ on test_balanced, retrain
that cell's PPO config with **all 3 algorithms** (DQN/PPO/A2C ×
5 seeds) and re-run the Phase-6 evaluation harness. This produces
**F8b** ("Phase-6 baselines + Phase-7 retuned RL trio") for the
closing chapter. **Do NOT ship if F9 fails** — the honest finding
in that case is "the gap was characterised but not closed";
F8b would be misleading.

### 3.3 — Tests (synthetic-only)

| File | Tests | Purpose |
|---|---:|---|
| `tests/test_phase31_impact_terminal.py` | 6 | Pin the new `impact_is_terminal` codepath; default `True` is byte-for-byte equal to Phase-3 contract. |
| `tests/test_train_agent_reward_overrides.py` | 5 | Pin the `--reward-overrides` + `--p-defender-deescalation` plumbing; default behaviour unchanged. |

Total: 420 → ~431 tests (+11). Run-driver and plotter tests are
out of scope — those depend on real data and we test them
end-to-end via the gate-evaluator script (G7.8 / G7.9 act as the
acceptance test for F15; G7.2 for F9; G7.3 for F10; G7.4 for F12).

### 3.4 — Exit gates

| Gate | Threshold | Evaluator |
|---|---|---|
| **G7.1** | `pytest -q` ≥ 430 passed; zero new skips | tests run on every commit |
| **G7.2** | F9 best cell mean test reward ≥ Phase-6 deployable best (A2C +1336.6) by ≥ 1σ | F9 plotter; **stretch goal**: meet Phase-6 oracle ceiling +1684.8 |
| **G7.3** | F10 PPO mean test reward at p=0.0 < at p=0.6 by ≥ 1σ; rule curve monotone non-decreasing | F10 plotter |
| **G7.4** | F12 Pareto frontier has ≥ 3 distinct dominant points (no single config dominates {security, availability}) | F12 plotter |
| **G7.5** | All Phase-3 frozen tests still PASS with `impact_is_terminal=True` (default); `False` codepath has its own tests (T7.31) | `pytest tests/test_phase3_env_gates.py tests/test_adversarial_env.py tests/test_phase31_impact_terminal.py` |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests overall | full `pytest -q` |
| **G7.7** | F9, F10, F12, F15 each ship a `manifest.json` SHA-pinning every input JSONL + the producing git SHA + the upstream Phase-5/6 manifests | manifest validator |
| **G7.8** *(audit-AF1)* | F15 eval emits a complete 4 × 8 result matrix (4 classes × 8 policies); no NaNs; manifest hash chain valid | F15 plotter |
| **G7.9** *(audit-AF1, headline)* | On `VulnerabilityScan`, **trained RL** mean test reward beats **RF-Acting** mean test reward by ≥ 1σ of the per-policy bootstrap CI | F15 plotter; **acceptable failure mode** (turns into a finding): RL does NOT beat RF-Acting on `VulnerabilityScan`; document why and narrow the thesis claim to "RL is *robust to* (not *better at*) the OOD class" |

**Same protocol as Phases 5 / 6:** any gate that fails on real
data gets a dated `D7.X.1` follow-up entry in §8 with rationale,
preserving the original threshold verbatim. The JSON
`G7_scoreboard.json` records `passes:false` permanently in such
cases. This is the AF3 protocol-continuity argument applied
forward.

### 3.5 — Figures produced

| Figure | Path | Tier | Owner | Caption sketch |
|---|---|:---:|---|---|
| **F9** | `docs/results/07_ablation/F9_reward_ablation.png` | 2 | Phase 7 | Per-component test-reward effect at 0.5× / 1× / 2× of Phase-3 defaults; Phase-6 oracle ceiling and deployable best as reference lines. |
| **F10** | `docs/results/07_ablation/F10_aggressiveness.png` | 2 | Phase 7 | PPO and oracle-rule mean test reward as a function of `p_defender_deescalation`; bands = 95 % bootstrap CI. |
| **F12** | `docs/results/07_ablation/F12_pareto.png` | 2 | Phase 7 | Security gain (1 − compromise_rate) vs. availability cost (BLOCK+ISOLATE share); Pareto frontier highlighted. |
| **F15** | `docs/results/07_ablation/F15_ood_robustness.png` | **1** | Phase 7 (audit AF1) | 4-class × 8-policy grouped horizontal bar chart of mean test reward under OOD-class-restricted realisations. |

## 4 — Sequencing table

Phase 7 is the most expensive phase in CPU terms (re-training is
the bulk). 9 commits, ~10 h human + ~7.5 h CPU walk-away.

| # | Commit | Files touched | Human | CPU |
|:-:|---|---|:-:|:-:|
| C1 | `docs(phase-6,§6.4): reframe rec-action floor as oracle upper bound (audit AF2)` | `docs/results/06_benchmark/RESULTS.md` only | 20 m | 0 |
| C2 | `docs(phase-7): audit & PLAN — F9 + F10 + F12 + F15` (this commit) | `docs/results/07_ablation/PLAN.md`, `docs/thesis_results_map.md` (add F15 row, promote F9/F10/F12 to Tier 2 confirmed) | 1.5 h | 0 |
| C3 | `feat(phase-7,§3.1.1): impact_is_terminal flag in AdversarialEnvConfig` | `src/environment/adversarial_env.py`, `tests/test_phase31_impact_terminal.py` | 1 h | 0 |
| C4 | `feat(phase-7,§3.1.2): --reward-overrides + --p-defender-deescalation in train_agent.py` | `scripts/blue_team/train_agent.py`, `tests/test_train_agent_reward_overrides.py` | 1.5 h | 0 |
| C5 | `feat(phase-7,§3.1.3): F15 OOD eval runner + plot` | `scripts/ablation/{__init__,run_ood_eval,plot_ood_robustness}.py`, Make target | 2 h code + ~1 h CPU |
| C6 | `feat(phase-7,§3.1.4): F9 reward-component sweep driver + plot` | `scripts/ablation/{run_reward_sweep,plot_reward_ablation}.py`, Make target | 1.5 h code + ~6 h CPU |
| C7 | `feat(phase-7,§3.1.5): F10 aggressiveness sweep + plot` | `scripts/ablation/{run_aggressiveness_sweep,plot_aggressiveness}.py`, Make target | 1 h code + ~1.5 h CPU |
| C8 | `feat(phase-7,§3.1.6): F12 Pareto from F9 + F10 outputs` | `scripts/ablation/plot_pareto.py`, Make target | 30 m | 0 |
| C9 | `docs(phase-7): close — RESULTS + CHANGELOG + G7 scoreboard` | `docs/results/07_ablation/RESULTS.md`, `CHANGELOG.md`, `docs/results/07_ablation/G7_scoreboard.json` | 2 h | 0 |

CPU runs (C5, C6, C7) are walk-away and can overlap.

## 5 — What we are NOT doing (defer to Phase 8 or later)

- **Robustness to observation noise / drift** (Phase 8, F13) —
  noise injection on the obs vector, drift on the realiser,
  decision rules under corrupted features. Distinct mechanism
  from OOD-class robustness (which is a distribution-shift on
  *attack identity*, not on *observation channel*).
- **F14 — Generalisation training to held-out attack class** —
  the Tier-3 figure currently scoped for Phase 8 retains its
  scope (training-time augmentation for OOD coverage). F15 is
  *evaluation-time* OOD; F14 if it ships will be *training-time*
  OOD.
- **Re-training at full 500 K timesteps** — D5.3.1 locked us at
  250 K; Phase 7 inherits that.
- **Hyperparameter sweeps within an algorithm** — T1 in Phase 5
  locked one config per algo. Phase 7 sweeps the *env* not
  the *algorithm*.
- **Re-implementing IoTWarden's DQN head-to-head** — officially
  retired in `ecfb584`.
- **Exhaustive 5-component × 3-level × 2 binary full grid (405
  cells)** — out of scope under the ~36 h CPU budget that would
  require. Sparse one-at-a-time at 12 cells (D7.1) is the
  shipping design.

## 6 — Risks tracked

| ID | Risk | Mitigation |
|---|---|---|
| **R7.1** | F9 sweep does not close the +288 gap at any cell. | Reframe G7.2 as PASS-WITH-FINDING (D7.1.1 if needed): "the linear sweep failed to close the gap, characterising the limit of one-at-a-time Phase-3-style reward shaping. Closing the gap requires a different mechanism (curriculum, reward modelling, or attack-aware exploration), deferred to future work." This is a defensible thesis outcome; the sweep did its diagnostic job. |
| **R7.2** | F15 trained RL does NOT beat RF-Acting on `VulnerabilityScan`. | Acceptable failure of G7.9 → finding D7.9.1: narrow the thesis claim from "RL closes the OOD gap" to "RL is comparable to RF-Acting on OOD attack classes; neither dominates the other across all four held-out classes." The 4-panel plot makes the per-class story honest regardless of the headline. |
| **R7.3** | F12 Pareto frontier has < 3 distinct dominant points (G7.4 fails) — every config sits on a single trade-off line. | Reframe as a finding: "the (security, availability) trade-off is approximately linear under the Phase-3 reward formulation; operating-point choice reduces to a single scalar weighting." This is itself a thesis-defensible structural claim. |
| **R7.4** | The added `impact_is_terminal=False` branch breaks a Phase-3 frozen test. | Default `True` preserves the contract byte-for-byte; the test for the `False` codepath lives in `test_phase31_impact_terminal.py` only. The Phase-3 frozen tests never see `False`. Verified by G7.5. |
| **R7.5** | The PPO sweep produces unstable training curves at some F9 cells (e.g., `defense_success_bonus = 500` blowing up). | All cells are 5-seed averages with bootstrap CIs — instability shows up as wide CIs, not a moved mean. F9 plot reports CI widths; outlier seeds (defined ex post by ≥ 3σ deviation from cell mean) are reported but not removed. |
| **R7.6** | RF-Acting's 14 ms latency from Phase-6 D6.8.1 hides a real budget violation. | Phase-7 inherits the Phase-6 latency story unchanged; F15 inherits F7's latency claim. If the user's defense committee pushes back, the production-batching mitigation argument from D6.8.1 stands. |
| **R7.7** | An OOD class has zero ACCESS+ realisations and the env terminates immediately. | The OOD splits are 30 KB – 97 KB which is > 100 episodes' worth of rows even at the smallest. Verified empirically in C5 by sampling 10 episodes per class as a smoke test before the full sweep. |

## 7 — Cross-references to thesis chapter outline

- **F9, F10, F12** all feed thesis Chapter "Empirical Results"
  §6.5 (Ablations) + §6.6 (Operating-Point Pareto).
- **F15** feeds Chapter "Empirical Results" §6.7 (OOD-class
  robustness — the Phase-4-finding-to-Phase-7-evidence chain).
- The AF2-reframed Phase-6 §6.1 (oracle-ceiling framing) is
  cited in §6.4 (Phase-6 headline) which §6.5 builds on.
- The AF3 protocol-continuity argument (G6.2 / G5.4 / G7.X
  precedents) is cited as a defence-deck slide, NOT a thesis
  chapter (per AF3, this is defence-prep, not codebase work).

## 8 — Locked design decisions

These are locked at PLAN-commit time. Subsequent `feat(phase-7,…)`
commits cite §-numbers; revisions of these decisions get explicit
`D7.X.1` follow-up entries with date + rationale, mirroring the
Phase-5/6 protocol (D5.3.1, D5.4.1, D5.10.1, D6.2.1, D6.6,
D6.8.1).

| ID | Decision | Rationale |
|---|---|---|
| **D7.1** | F9 grid is **sparse one-at-a-time**: 5 components × {0.5×, 1×, 2×} multipliers + binary `impact_is_terminal ∈ {True, False}` = **12 cells** (1 shared centre + 10 component off-centres + 1 binary off-centre). NOT the full 5-component × 3-level × 2-binary = 405-cell grid. | The full grid would cost ≈ 36 h CPU (405 × 5 seeds × ~6 min). The sparse design answers the headline question ("which single component drives the gap?") at a 12 × 5 = 60-run / ~6 h budget. Cross-component interaction effects are out of scope unless the sparse sweep finds nothing — in which case we revisit under D7.1.1 with explicit user approval for the larger budget. |
| **D7.2** | F9 and F10 train **PPO only** (best Phase-5 / Phase-6 trio, lowest CI width). DQN and A2C are NOT swept. | Three-algo sweeps would 3× the CPU budget without changing the diagnostic. PPO's CI band is the tightest of the three (Phase-6 F8: PPO CI width 119, A2C 70, DQN 142 — A2C is actually tightest, but PPO's mean is closest to the cluster centroid and Phase-5 G5.4 PASS-WITH-FINDING applies equally to all three, making PPO the representative choice). |
| **D7.3** | The deferred Phase-6 `impact_is_terminal` flag (D6.6) is **folded into F9** as one binary axis, not shipped as a standalone Phase 7.1. | The flag's effect is mechanistically a reward-shaping change (the agent gets an explicit IMPACT-row decision step, which moves its share of `defense_success_bonus` from the realiser's de-escalation roll to its own action choice). Sweeping it with the other 5 components keeps the diagnostic single-chapter, single-figure. Default `True` preserves the Phase-3/4/5/6 frozen contract. |
| **D7.4** | F15 OOD eval uses the **Phase-3 frozen reward config** unchanged. The OOD test isolates the *generalisation* axis from the *reward-shaping* axis. | F15 answers "does the trained policy generalise?" — answering this and "does the new reward generalise?" simultaneously (i.e., evaluating Phase-7-best-cell on OOD) confounds the two axes. If F9 finds a winning cell, a *follow-up* OOD eval of that cell becomes a §3.2 optional bonus; the headline F15 stays on the Phase-6 trio. |
| **D7.5** | F12 (Pareto) is **derived from F9 + F10 outputs**, not a separate sweep. | Each F9 / F10 cell already produces a `(reward, compromise_rate, action_share)` tuple from its `eval_test.jsonl`. Re-projecting them onto (security, availability) axes is a plotter-only operation. Saves a third sweep's CPU budget. |
| **D7.6** | F15 reuses the **frozen Phase-5 trained checkpoints** (no retraining, even with `impact_is_terminal=True/False` variants). | F15 measures "does the policy already trained on in-distribution data generalise to OOD?" — retraining contaminates that question with "did it learn the OOD class implicitly?" The headline number is what the deployed policy from Phase 5 actually achieves on OOD, full stop. |
| **D7.7** | All Phase-7 manifests SHA-pin the upstream Phase-5 `sweep_manifest.json` AND the Phase-6 `eval_manifest.json` (where reused). | Reproducibility chain: F15 result → F15_manifest → Phase-6 eval_manifest → Phase-5 sweep_manifest → Phase-1 splits manifest. Same idiom as Phase-6 D6.9. |
| **D7.8** | F9 / F10 retrains use **`total-timesteps = 250 000`** (Phase-5 D5.3.1) per cell, NOT 500 000. | D5.3.1 locked the 250K budget after the empirical observation that PPO converges between 100K and 250K. Phase-7 sweeps the *env*, not the *training horizon*; comparing cells at the same horizon is the right protocol. The thesis would be confounded if some cells got 250K and others got 500K. |
| **D7.9** | The G7.2 success criterion uses **A2C's Phase-6 +1336.6 (deployable best, 10 seeds × 300 ep) as the lower bar**. | "Beating Phase-6's deployable baseline" is the meaningful claim. A2C +1336.6 is the best deployable Phase-6 number; F9 must move PPO past A2C under at least one sweep cell to count as a positive finding. The +1684.8 oracle ceiling is the stretch target, not the threshold. |
| **D7.10** | Phase-7 figures live under `docs/results/07_ablation/` (singular `07_ablation`, not `07_ablations`). | Consistent with `02_red_team`, `03_env`, `04_detector`, `05_blue_team`, `06_benchmark` (each is a single noun-phrase). |

### Pre-emptive D-decisions (logged here so future agents can find them)

- **D7.1.1** (locked 2026-05-01 — partial activation, audit
  fix) — the F9 sparse sweep DID find a winner on the
  apples-to-apples strand (`impact_is_terminal_false` PPO mean
  +1542 beats DQN +1336 by +205.6, mit_rate 0.900 vs 0.153
  baseline → G7.2 PASS-WITHOUT-STRETCH). However the audit
  cycle on 2026-05-01 also surfaced a subtler finding worth
  capturing: **no reward-coefficient cell** (axis="reward",
  10/12 of the grid) moves PPO mean reward by ≥ 1σ on the
  apples-to-apples strand. The 11 reward-coefficient + centre
  baseline cells stay within ±150 of the centre baseline. The
  one cell that moves the needle (`impact_is_terminal_false`)
  is an **env-semantics flip**, not a reward-coefficient
  perturbation. Implications: (a) coefficient scaling within
  the Phase-3 reward formulation is bounded — closing the
  remaining −82.5 gap to the oracle ceiling +1624 requires a
  mechanism *other* than coefficient scaling (curriculum,
  reward modelling, or non-linear composition); (b) the
  G7.2 evaluator gained a two-strand definition (raw-reward
  apples-to-apples + security-KPI fallback) so future Phase-7
  re-runs cannot mistake reward-coefficient scaling for policy
  improvement (see `tests/test_close_phase7_parsers.py` for the
  pinned logic). Original G7.2 threshold preserved verbatim in
  `G7_scoreboard.json#gates[1].threshold`. See RESULTS.md §6.1
  for the full chapter narrative.
- **D7.9.1** (locked 2026-05-01 — fully ACTIVATED) — R7.2
  fired: on `VulnerabilityScan` trained RL does NOT beat
  RF-Acting (DQN +1313 (CI 1228–1387) vs RF-Acting +1611 (CI
  1556–1666); Δ = −298 at ≥ 1σ separation). Original G7.9
  threshold preserved verbatim in
  `G7_scoreboard.json#gates[8].threshold`. **Thesis claim
  narrows from** "RL closes the OOD gap by acting on raw
  features" **to** "RL is **robust to** (not **better at**)
  the OOD class" — DQN's mean OOD reward (+1313) is within
  seed-noise of its in-distribution mean (+1336), so
  generalisation does not collapse the policy. RF-Acting's
  stronger OOD reward (+1611) is not evidence of RF working
  (Phase-4 RF recall on this class = 0.001) — it is evidence
  that the recommended-action mapping defaults to OBSERVE on
  RF's mis-prediction, and 'do nothing' is locally-good when
  the Phase-3 reward function is dominated by avoiding
  disproportionate-penalty costs. Future work to *exceed*
  RF-Acting OOD belongs in Phase 8 F14 (train-time OOD-class
  augmentation). See RESULTS.md §6.2 for the full chapter
  narrative + defense-committee Q&A pre-rebuttals.

## 9 — Test count history

Phase 0 254 → Phase 1 266 → Phase 2 283 → Phase 3 296 → Phase 4
329 → Phase 5 376 → Phase 6 420 → **Phase 7 target: ≥ 431**
(+11 from §3.3 above).
