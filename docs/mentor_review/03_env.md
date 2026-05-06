# Step 03 — Phase 3 Environment Review (MDP, reward, gates G3.1–G3.7)

**Mentor memo. Audits the adversarial RL environment for the MSc defense
at Unicamp/FEEC.**

---

## Verdict

`PASS-WITH-FIXES`

The Phase-3 v2 environment is a clean rewrite that fixes every bug B1–B6 from
the pre-restart audit (`PLAN.md` §1) and ships a complete regression suite
that mechanically reproduces all seven exit gates G3.1–G3.7. Read against the
locked thesis claims:

- **P3 (structural lever, +288 → +1542 mean reward, mit-rate 0.153 → 0.900)**
  is supported by `impact_is_terminal: bool = True` at
  `src/environment/adversarial_env.py:200` (Phase-3 frozen default) plus the
  `=False` branch wired through `src/blue_team/run_config.py:77` and
  `src/blue_team/env_factory.py:61`. Both branches are exercised by
  `tests/test_phase31_impact_terminal.py` (6 tests, all passing).
- **R2 (linear Pareto)** is supported by the **eleven** reward fields the env
  factory now forwards (`adversarial_env.py:204-223`,
  `env_factory.py:53-73`), so the F9 reward sweep can override every reward
  component independently.
- **F10 (aggressiveness sweep)** is supported by `p_defender_deescalation:
  float = 0.6` at `adversarial_env.py:196`, plumbed through the run config
  and overridable per-cell.

`pytest -q` is green: **411 passed in 68.10 s** on `mentor-review/step-3-env`
(off `main` @ `d4acfca`); the Phase-3-scoped slice (G3 gates +
impact_is_terminal lever + split-aware engine) is **30/30 in 4.09 s**.

Three findings must land before binding:

1. **(F1, minor)** Phase 3 has *no* `manifest.json` and *no*
   `G3_scoreboard.json` under `docs/results/03_env/`. The G3 verdicts live as
   a Markdown table in `RESULTS.md` §4. This is the same audit-trail
   asymmetry already filed for Phase 1 (Step-1 F4) and Phase 2 (Step-2 F4);
   recommended doc-fix is a single one-paragraph note in
   `docs/results/README.md` clarifying which phases produce JSON
   scoreboards/manifests and which do not.
2. **(F2, minor)** The env's `__init__` constructs a *non-split-aware*
   `RealizationEngine(dataset_path)` at
   `src/environment/adversarial_env.py:305`. Split-awareness is enforced
   by `src/blue_team/env_factory.py:107` via direct attribute assignment
   (`env._realization_engine = engine`). The pattern works in production
   because Phase-5/6/7 always go through the factory, but a future caller
   that instantiates `AdversarialIoTEnv` directly silently bypasses the
   OOD-exclusion + train-only restriction. Recommended doc-fix: a class-level
   docstring note. Optional code-fix: add an injection seam (`engine:
   Optional[RealizationEngine] = None` kwarg).
3. **(F3, minor)** PLAN.md §3.1 step 2 documents a four-component reward
   formula
   (`R = R_proportional - C_action - P_overreact - P_underreact + R_progress_blocked`);
   the as-built reward (RESULTS.md §3 + `_calculate_reward` at
   `adversarial_env.py:659-716`) implements **six** terms — the four planned
   plus `+reward_benign_passive` (10.0) on BENIGN×{OBSERVE,LOG} and the
   `defense_success_bonus` recalibration from 10 → 250 (Iteration 3 in
   RESULTS.md §5). PLAN is the design contract, RESULTS.md is the as-built
   record, and the divergence is itself documented in RESULTS.md §5
   ("iterations & lessons learned"). Nothing is wrong on the science side.
   But the candidate's task prompt for Step 3 cited "the four documented
   reward components — proportionality reward, mitigation,
   disproportionate-penalty, MTTC" — a fourth-incorrect summary in which
   MTTC (which is a **metric**, not a reward component) replaces the actual
   `reward_benign_passive`/asymmetric-guardrails terms. Recommended doc-fix:
   one sentence in `docs/reward-shaping.md` (or wherever the thesis-facing
   English description of the reward lives) reconciling the planned vs
   as-built component list, and explicitly noting MTTC is a per-episode
   metric (`info["mttc_steps"]`), never a reward term.

Plus one nit (F4) and the now-resolved Step-2 carry-forward (F5).

The Step-2 transition_mask carry-forward (Step-2 Finding 8) is **resolved
benign** here: Phase 3 does **not** call `set_transition_mask()` anywhere
(grep confirms: zero matches in `src/environment/`, `src/blue_team/`,
`src/algorithms/`, or `scripts/`; the only calls are inside
`src/generator/attack_sequence_generator.py` itself and the dedicated
`tests/test_transition_mask.py:172`). The `transition_mask.py:79-80` ↔
`episode_generator.py:269-271` IMPACT-state divergence therefore remains a
Step-8 cross-cutting cleanup, never a Step-3 correctness bug. See §6.

---

## 1. What was reviewed

### Frozen audit trail
- `docs/results/03_env/PLAN.md` (234 lines) — the design contract:
  bugs B1–B6, the rewrite decision, deliverables 3.1, exit gates G3.1–G3.7,
  sequencing 4.1–4.6, risks R1–R3.
- `docs/results/03_env/RESULTS.md` (196 lines) — the as-built record:
  bug-fix table, lifecycle pseudocode, reward formula, default constants,
  per-gate scoreboard table, Iteration 1–3 lessons, and Phase-3 commits
  (`482299e` PLAN, `3a6b13a` split-aware engine, `2a526af` env rewrite,
  `36fec22` gates + calibration).
- `docs/results/03_env/manifest.json` — **does not exist** (Finding F1).
- `docs/results/03_env/G3_scoreboard.json` — **does not exist** (Finding F1).

### Code under review
- `src/environment/adversarial_env.py` (716 lines) — the env. Re-read in
  full: imports, `ACTION_NAMES`/`ACTION_COSTS`, `_recommended_action`
  mapping, `AdversarialEnvConfig` dataclass, `AdversarialIoTEnv.__init__`,
  `reset`, `step`, `_step_at_impact`, `_maybe_defender_deescalation`,
  `_advance_attack`, `_build_observation`, `_build_info`,
  `_calculate_reward`.
- `src/blue_team/env_factory.py` (184 lines) — split-aware env construction:
  `make_train_env`, `make_eval_env`, `_build_env_config`, `_build_env`.
  This is where the `RealizationEngine.from_split_manifest(...)` call lives
  and where the env's internal engine is monkey-patched (line 107).
- `src/blue_team/run_config.py` — `EnvConfigSerializable` dataclass
  (lines 50–96) forwards every reward field plus `impact_is_terminal`,
  `p_defender_deescalation`, `split`, `exclude_ood`. Default
  `impact_is_terminal: bool = True` (line 77), `exclude_ood: bool = True`
  (line 69), `p_defender_deescalation: float = 0.6` (line 74).
- `src/utils/realization_engine.py` (already audited at Step 1) — selective
  re-read of `from_split_manifest` (lines 104–168). Defaults
  `exclude_ood=True` (line 110); the OOD-stripping is at line 165.
- `src/algorithms/adversarial_algorithm.py` — selective read; no impact on
  Phase-3 MDP semantics (orchestrates DQN/PPO/A2C training on top of the
  env, does not interact with `set_transition_mask`, `RealizationEngine`,
  or env construction in a way that bypasses the env factory).

### Tests
- `tests/test_adversarial_env.py` (29 tests) — generic env unit tests:
  config, init, observation/action spaces, reset, step, MTTC, action costs.
- `tests/test_phase3_env_gates.py` (13 tests) — the **G3.1–G3.7 mechanical
  gates** (PLAN §3.2). Every gate has a named test; see §4 of this memo.
- `tests/test_phase31_impact_terminal.py` (6 tests) — the
  `impact_is_terminal` lever (Phase-7 D7.3 / F9 reward-ablation axis):
  pins the default-True Phase-3 frozen contract (3 tests) and the
  False branch (3 tests). All passing.
- `tests/test_realization_engine.py` (19 tests) — engine basics.
- `tests/test_realization_engine_split_aware.py` (9 tests) — Step-1
  invariant: `from_split_manifest`, `exclude_ood`, train/val/test
  isolation. All passing.
- `tests/test_train_agent_reward_overrides.py` — confirms the reward-field
  forwarding from CLI overrides → `EnvConfigSerializable` → env config is
  byte-correct (the F9/F10 sweep substrate). Passes.
- `tests/test_blue_team_env_factory.py` — confirms `make_train_env`/`make_eval_env`
  pass `split="train"/"val_balanced"`, `exclude_ood=True` to
  `RealizationEngine.from_split_manifest(...)`.

### Docs (skim, reference only)
- `00_framing.md` — IoTWarden recommended-action mapping (BENIGN→OBSERVE,
  RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK, IMPACT→ISOLATE) confirmed
  on disk at `_RECOMMENDED_ACTION_BY_STAGE = [0, 1, 2, 3, 4]`
  (`adversarial_env.py:116`).
- `02_red_team.md` Finding 8 — resolved benign, see §6.
- `01_dataset.md` — the `from_split_manifest`/`exclude_ood` invariant is
  the same one audited and signed off at Step 1.

---

## 2. MDP correctness (PLAN §1, RESULTS §3)

| Element | PLAN.md / RESULTS.md spec | Code (file:line) | Match? |
|---|---|---|---|
| State space | window of `window_size=5` realised feature vectors (29-D) ± first-order deltas | `adversarial_env.py:310-318` (`obs_dim = window_size × num_features × 2`); `_build_observation` at 611-620 | ✓ |
| Action space | `Discrete(5)`: OBSERVE=0, LOG=1, THROTTLE=2, BLOCK=3, ISOLATE=4 | `adversarial_env.py:321` (`spaces.Discrete(num_actions=5)`); names at lines 59-65 | ✓ |
| Hidden attack-stage | agent does **not** observe the true stage | `adversarial_env.py:10-12` doc; observation built only from realised features; `_build_info` exposes the stage but it is not part of the obs | ✓ |
| Transition function | LSTM (`AttackSequenceGenerator.sample_next`) with last 5-stage history | `adversarial_env.py:599-609` (`_advance_attack`) | ✓ |
| IMPACT-clamp (lifecycle floor) | LSTM transitions to IMPACT before `min_episode_length` are downgraded to MANEUVER | `adversarial_env.py:458-466` | ✓ |
| Defender-driven de-escalation | action≥BLOCK & previous_stage≥ACCESS & `rng < p_defender_deescalation` ⇒ next stage = BENIGN, +`defense_success_bonus` | `adversarial_env.py:561-580` (`_maybe_defender_deescalation`); reward bump at line 444 | ✓ |
| Terminal condition (default) | `impact_is_terminal=True`: terminate the same step the env transitions to IMPACT *and* `step_count ≥ min_episode_length` | `adversarial_env.py:493-514` | ✓ |
| Terminal condition (Phase-7 lever) | `impact_is_terminal=False`: defer to next-step explicit IMPACT-row decision via `_step_at_impact` | `adversarial_env.py:432-433`, 515-524, 534-559 | ✓ |
| Truncation | `step_count >= max_steps` (default 500) | `adversarial_env.py:525` | ✓ |
| Recommended-action mapping | `BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK, IMPACT→ISOLATE` (IoTWarden) | `adversarial_env.py:116` (`_RECOMMENDED_ACTION_BY_STAGE = [0,1,2,3,4]`) + lookup at 119-121 | ✓ |
| MTTC | `mttc_steps = compromise_step − first_attack_step`, both tracked per-episode; exposed in `info` | `adversarial_env.py:469-478` (bookkeeping) + 638-642 (computation) | ✓ |

The env is a faithful, well-documented, test-locked rewrite of the v1
environment whose four hard bugs (B1, B2, B4, B5) and two soft bugs
(B3, B6) the Phase-0 diagnosis flagged.

---

## 3. Reward function audit (PLAN §3.1 step 2, RESULTS §3)

The B2 fix — **stage-action proportionality computed solely from
`decision_stage` and `action`, with no dependence on the agent's previous
move** — is implemented faithfully. The reward formula at
`adversarial_env.py:659-716` (`_calculate_reward`) has **six** components:

| # | Component | Trigger | Implementation |
|---|---|---|---|
| 1 | Action cost | always | `reward -= ACTION_COSTS[action] × action_cost_scale` (line 690) |
| 2 | Overreact-on-BENIGN | `stage=BENIGN ∧ action≥THROTTLE` | `−penalty_overreact_benign` (line 694), plus `−penalty_block_benign` if `action≥BLOCK` (line 696) |
| 3 | Block-on-RECON | `stage=RECON ∧ action≥BLOCK` | `−penalty_block_recon` (line 698) |
| 4 | Underreact-at-IMPACT (safety net) | `stage=IMPACT ∧ action≤LOG` | `−penalty_missed_impact` (line 702); usually `_step_at_impact` handles IMPACT, this is the safety net referenced in line 700 |
| 5 | Benign-passive bonus | `stage=BENIGN ∧ action≤LOG` | `+reward_benign_passive` (line 706); the small consistent positive signal that prevents always-OBSERVE from netting zero |
| 6 | **Proportionality core (B2)** | always | `+reward_proportional` if `\|action − recommended(stage)\| ≤ 1`, else `−penalty_disproportionate` (lines 711-714) |

Plus the **terminal-IMPACT inline accounting** at `adversarial_env.py:497-514`
(only fires when `impact_is_terminal=True`, the default Phase-3 frozen
contract):

```
reward -= impact_penalty                    # always at IMPACT termination
if action >= BLOCK:    reward += defense_success_bonus    # ≥3
elif action <= LOG:    reward -= penalty_missed_impact    # ≤1
```

The same bookkeeping is duplicated in `_step_at_impact` for the
`impact_is_terminal=False` branch (lines 540-547).

**Default reward constants** (lines 194-223) match RESULTS.md §3's table:
`reward_proportional=5.0`, `penalty_disproportionate=5.0`,
`impact_penalty=200.0`, `penalty_missed_impact=150.0`,
`defense_success_bonus=250.0`, `reward_benign_passive=10.0`,
`penalty_overreact_benign=50.0`, `penalty_block_benign=100.0`,
`penalty_block_recon=50.0`, `action_cost_scale=1.0`.

**Stage-action mapping** is the IoTWarden mapping locked in `00_framing.md`
§2: `[BENIGN, RECON, ACCESS, MANEUVER, IMPACT] → [OBSERVE, LOG, THROTTLE,
BLOCK, ISOLATE]` (`adversarial_env.py:101-116`).

The B6 calibration story (Iteration 3 in RESULTS.md §5) — bumping
`defense_success_bonus` from 10 to 250 so ISOLATE@IMPACT nets +49 — is
documented inline in the dataclass docstring at lines 210-217 with cross-ref
to PLAN §B6. Nice trace.

---

## 4. Exit gates G3.1–G3.7 — reproduction on current `main`

Pytest re-run on `mentor-review/step-3-env` (off `main` @ `d4acfca`):
**30/30 passed in 4.09 s** (Phase-3-scoped slice: G3 gates +
impact_is_terminal lever + split-aware engine).

| Gate | RESULTS.md §4 threshold | Test name (file:line) | Status |
|---|---|---|---|
| G3.1.a | recommended action net-positive at every stage | `test_phase3_env_gates.py::TestG3_1_RegressionTests::test_recommended_action_yields_positive_reward_per_step` | ✓ |
| G3.1.b | overreact on BENIGN net-negative | `test_overreaction_on_benign_yields_negative_reward` | ✓ |
| G3.1.c | underreact at IMPACT net-negative | `test_underreaction_on_impact_yields_negative_reward` | ✓ |
| G3.1.d | always-BLOCK survives ≥ 5 steps | `test_block_does_not_terminate_episode_early` | ✓ |
| G3.1.e | MTTC fields present in `info` | `test_mttc_fields_present_in_info` | ✓ |
| G3.1.f | MTTC fields = `None` at reset | `test_mttc_is_none_at_reset` | ✓ |
| G3.1.g | defender de-escalation fires at ACCESS+ | `test_defender_deescalation_resets_to_benign` | ✓ |
| G3.1.h | defender de-escalation does **not** fire below ACCESS | `test_defender_deescalation_does_not_fire_below_access` | ✓ |
| G3.2 | median random-action episode length ≥ 15 | `test_g3_2_random_action_median_episode_length[100]` | ✓ |
| G3.3 | median always-BLOCK episode length ≥ 10 | `test_g3_3_always_block_median_episode_length[100]` | ✓ |
| G3.4 | recommended-policy mean reward > 0 | `test_g3_4_recommended_action_mean_reward_positive[50]` | ✓ |
| G3.5 | always-OBSERVE mean reward < 0 | `test_g3_5_always_observe_mean_reward_negative[50]` | ✓ |
| G3.6 | always-ISOLATE mean reward < 0 | `test_g3_6_always_isolate_mean_reward_negative[50]` | ✓ |
| G3.7 | full test suite green (PLAN target ~ 296 passed; **as of Step 3 main is 411 passed** — phases 4–7 added tests on top of Phase-3's 296) | `pytest -q` → **411 passed in 68.10 s** | ✓ |

Note on G3.7: PLAN.md §3.2 G3.7 wrote "274 (Phase 2) + ~9 new tests + ~5
updated tests". RESULTS.md §4 reports 296 on commit `36fec22`. Today,
`main` reports 411. The growth is monotonically explained by Phases 4–7
adding their own tests; it is not a regression.

---

## 5. OOD-leakage boundary check (carry-forward from Step 1)

The Step-1 invariant — *"the env never loads val/test/OOD rows during
training-time `step()` calls"* — holds at the **factory layer** but is not
enforced at the env-constructor layer. Concretely:

- `src/blue_team/env_factory.py:99-116`: when `splits_manifest` is non-None
  (production path), the factory builds
  `RealizationEngine.from_split_manifest(data_path, splits_manifest,
  split_name=spec.split, exclude_ood=spec.exclude_ood, seed=seed)` and
  **monkey-patches** the env's internal engine (`env._realization_engine =
  engine`). `EnvConfigSerializable` defaults are `split="train"` for the
  training env (`run_config.py:20`), `split="val_balanced"` for the eval
  env (`run_config.py:23`), and `exclude_ood=True` (line 69). Phase-5
  training (`scripts/blue_team/train_agent.py:182`) and Phase-7 sweeps
  (`scripts/ablation/run_*.py`) consume the factory.
- `src/environment/adversarial_env.py:305`: bare
  `RealizationEngine(dataset_path)` with no `allowed_indices`. **A direct
  caller of `AdversarialIoTEnv(...)` (no factory) silently gets the full
  442 237-row snapshot, including val/test/OOD.** The 78 Phase-3-scoped
  tests rely on this happily — they use synthetic mocks — so the bug is
  latent.

This is **F2** in §6 (minor finding, doc + optional code-fix). The
production training path is correct because Phase 5 always goes through
the factory; the only risk is a future caller bypassing the factory.

The split-aware engine itself has 9 dedicated tests in
`tests/test_realization_engine_split_aware.py` covering: default = all
rows, allowed-indices subset, empty-set raises, empty-stage drop,
sample-only-from-allowed, train-split-only,
`exclude_ood=True`-by-default, unknown-split-raises,
missing-file-raises. All passing.

---

## 6. Step-2 Finding 8 carry-forward (transition_mask) — RESOLVED BENIGN

The Step-2 handoff §5 / §8 question 5 directed Step 3 to verify whether
`AdversarialEnv` (or any Phase-3 consumer of `AttackSequenceGenerator`)
calls `set_transition_mask()`. If yes, the divergence between
`src/generator/transition_mask.py:79-80` (allows IMPACT→BENIGN) and
`src/generator/episode_generator.py:269-271` (IMPACT absorbing) becomes a
**Step-3 correctness finding**.

**Grep across `src/`, `tests/`, `scripts/`:**

```
src/generator/transition_mask.py            (definition)
src/generator/attack_sequence_generator.py:50    use_transition_mask: bool = False
src/generator/attack_sequence_generator.py:107   self._transition_mask = None  # set via set_transition_mask
src/generator/attack_sequence_generator.py:126   def set_transition_mask(...)
src/generator/attack_sequence_generator.py:226   if self._transition_mask is not None and len(history) > 0:
tests/test_transition_mask.py:172                model.set_transition_mask(mask)        ← only consumer outside the class
```

**Zero references** in `src/environment/`, `src/blue_team/`,
`src/algorithms/`, `scripts/blue_team/`, `scripts/ablation/`,
`scripts/benchmark/`, or `scripts/red_team/`.

The env loads the LSTM via `AttackSequenceGenerator.load(...)` at
`adversarial_env.py:300` and never calls `set_transition_mask` afterwards.
`_advance_attack` (line 599) calls `self._generator.sample_next(...)`,
which falls through the `if self._transition_mask is not None` branch
because the mask is `None` by default.

**Verdict:** Step-2 Finding 8 is benign for Phase 3. The `transition_mask.py`
↔ `episode_generator.py` IMPACT-state divergence remains a
**Step-8 cross-cutting cleanup** (decide single source of truth for
"is IMPACT absorbing?"; reconcile or delete one of the two).

---

## 7. Findings (priority-ordered)

### F1. **[severity: minor]** Phase 3 has no `manifest.json` and no `G3_scoreboard.json`

**Where.** `docs/results/03_env/` contains only `PLAN.md` and
`RESULTS.md`. The Phase-3 G3 verdicts live as a Markdown table in
`RESULTS.md` §4.

**Why it matters.** It breaks the audit-trail symmetry the candidate
established for Phases 5/6/7/10 (each of which has at least one
`G<N>_scoreboard.json`). It also breaks the input-side hash-chain
referenced in the Step-3 task prompt's verification recipe ("verify hash
chain via `shasum -a 256` against on-disk artefacts; confirm input SHAs
chain back to Phase-1 outputs and Phase-2 LSTM checkpoint").

**Caveat.** PLAN.md §3.3 explicitly states "no thesis figures in Phase 3"
— the Phase 3 deliverable is a working environment, not a figure. So the
absence of an *output-side* manifest is by design. The absence of a
**G3 scoreboard JSON** is the gap.

**Recommended fix.** This is the same shape as Step-1 Finding 4
("no Phase-1 RESULTS.md") and Step-2 Finding 4 ("no Phase-2 RESULTS.md").
Resolve all three with a single one-paragraph note in
`docs/results/README.md` documenting the asymmetry: *"Phase 3 is
infrastructure-only and produces no thesis figures, hence no
`manifest.json`. The G3 scoreboard is a Markdown table in `RESULTS.md` §4
because Phase 3's gates are mechanical pytest assertions; the canonical
verdict is `pytest -q tests/test_phase3_env_gates.py`."*

**Commit message (suggested).**
`docs(phase-3,§audit-trail): document Phase-3 PLAN/RESULTS/scoreboard asymmetry`

### F2. **[severity: minor]** Bare `RealizationEngine(dataset_path)` in env constructor; split-awareness only enforced via factory monkey-patch

**Where.** `src/environment/adversarial_env.py:303-308`:

```python
# Load Realization Engine (for feature sampling)
dataset_path = Path(dataset_path)
self._realization_engine = RealizationEngine(dataset_path)
```

vs `src/blue_team/env_factory.py:99-116`:

```python
if splits_manifest is not None:
    engine = RealizationEngine.from_split_manifest(
        data_path=dataset_path, splits_manifest=splits_manifest,
        split_name=spec.split, exclude_ood=spec.exclude_ood, seed=seed,
    )
    env._realization_engine = engine  # type: ignore[attr-defined]
    env._num_features = engine.num_features  # type: ignore[attr-defined]
```

**Why it matters.** A direct caller of `AdversarialIoTEnv(...)` (without
the factory) silently bypasses the OOD-exclusion + train-only
restriction that Step-1's `from_split_manifest` shipped. The `# type:
ignore[attr-defined]` on the assignment confirms the pattern is
deliberate, but it relies on private-attribute mutation rather than a
typed injection seam. Phase 5/6/7 production paths happen to always go
through the factory, so the bug is latent. Future readers who skim the
env's `__init__` will misread the OOD-leakage invariant.

**Recommended fix (preferred).** Add a class-level docstring note on
`AdversarialIoTEnv` (around line 230) reading roughly: *"For training
runs, use `src.blue_team.env_factory.make_train_env` /
`make_eval_env` — they restrict the internal `RealizationEngine` to the
named split with OOD attacks excluded. Direct construction
(`AdversarialIoTEnv(...)`) loads the full dataset snapshot and is
intended for tests with synthetic data only."*

**Recommended fix (optional, deeper).** Add an injection seam to
`__init__`: `realization_engine: Optional[RealizationEngine] = None`. If
non-None, use it as-is; if None, fall back to the current bare ctor.
Then the factory at line 100 builds the engine first and passes it in,
eliminating the `# type: ignore` and making the OOD invariant
type-safe. This is a 5-line code change but I would defer it to Step 7
re-run territory — it touches a frozen module and no gate would change.

**Commit message (suggested).**
`docs(phase-3,§env-init): clarify RealizationEngine injection contract`

### F3. **[severity: minor]** PLAN-vs-RESULTS-vs-prompt reward-component mismatch; MTTC is a metric, not a reward term

**Where.** Three documents disagree on the reward component count:

- **PLAN.md §3.1 step 2** lists *four* components: `R_proportional`,
  `C_action`, `P_overreact`, `P_underreact`, plus `+R_progress_blocked`
  ("the env de-escalates" bonus, applied in `step` not `_calculate_reward`).
- **RESULTS.md §3** lists *six* terms in `_calculate_reward` (matching
  the code): action cost, overreact-on-benign, block-on-benign,
  block-on-recon, missed-impact safety net, benign-passive bonus,
  proportionality core. The two new terms (`reward_benign_passive` and
  the calibrated `defense_success_bonus = 250`) are documented in §5
  Iteration 2/3 as "real correctness bug, not just calibration"
  fixes.
- **The Step-3 task prompt** the candidate handed me says: *"Verify the
  four documented components — proportionality reward, mitigation,
  disproportionate-penalty, MTTC — match PLAN.md's formulation."* This
  list is wrong: MTTC is a per-episode telemetry metric exposed via
  `info["mttc_steps"]`, not a reward term, and it is not part of the
  PLAN.md formulation either.

**Why it matters.** PLAN.md is frozen audit-trail and we don't edit it.
RESULTS.md is the as-built record and is correct. The thesis-facing
prose in `docs/reward-shaping.md` and the eventual LaTeX (§3.4 in the
chapter outline) need to track RESULTS.md's six-term version, not the
abbreviated PLAN.md or the further-abbreviated task prompt.

**Recommended fix.** One sentence in `docs/reward-shaping.md` (or a
new "Reward function" subsection if the doc isn't structured for it)
listing the six implemented terms with the as-built default values, and
a one-line clarification: *"MTTC (Mean Time To Compromise) is a
**per-episode telemetry metric** — exposed in `info["mttc_steps"]` —
not a reward component."* The thesis chapter (§3.4) needs the same
correction during the Step 9 LaTeX rebuild — flag as carry-forward.

**Commit message (suggested).**
`docs(phase-3,§reward): align reward-component list with as-built code; clarify MTTC is metric not reward`

### F4. **[severity: nit]** RESULTS.md §7 R2 acknowledges MTTC bias; thesis prose should propagate the caveat

**Where.** `docs/results/03_env/RESULTS.md` §7 risk R2: *"The clamp's
MANEUVER-substitution might bias MTTC downward (compromise always
happens at exactly `min_episode_length` if it happens at all). Phase 7
should report MTTC restricted to 'natural' IMPACT events (i.e., where
the LSTM produced IMPACT *after* step 20), not clamped ones."*

**Why it matters.** The IMPACT-clamp at `adversarial_env.py:458-466`
is essential for the lifecycle floor (B1 fix), but it makes
`mttc_steps` boundary-clipped at `min_episode_length=20` because every
"natural" IMPACT before step 20 gets re-routed to MANEUVER. So the
distribution of MTTC has a hard left wall at 20. If the thesis quotes
mean MTTC anywhere, the reader needs to know the floor is structural,
not empirical.

**Recommended fix.** No code change. When the candidate writes the
thesis chapter §3.4 (Step 9 LaTeX rebuild), include the R2 caveat in a
footnote on the MTTC definition. Optional: add a Phase-7-side filter to
report mean MTTC restricted to episodes where IMPACT arrived after
step 20 (the "natural" set). I'd defer this to Step 7 (ablations) review
or Step 9 (LaTeX rebuild) — not a Phase-3 fix.

**Commit message (suggested, deferred to Step 9).**
`docs(phase-3,§mttc): note IMPACT-clamp lifecycle-floor bias in MTTC definition`

### F5. **[carry-forward, resolved benign here]** Step-2 Finding 8: transition_mask vs episode_generator IMPACT divergence

Phase 3 does not call `set_transition_mask` (see §6 above). The Step-2
carry-forward is **not a Step-3 finding**; it remains a Step-8
cross-cutting cleanup task. Step 8 should reconcile (or delete one of)
`src/generator/transition_mask.py:79-80` (mask permits IMPACT→BENIGN)
vs `src/generator/episode_generator.py:269-271` (episode generator
hard-codes IMPACT absorbing).

---

## 8. What's intentionally NOT a finding

### "G3.7 says 296 passed, current main has 411"
PLAN.md predicted ~296. RESULTS.md confirms 296 on the Phase-3 closeout
commit `36fec22`. Today's count is 411 because Phases 4–7 each shipped
their own tests on top. Monotone growth, no regressions.

### "Reward magnitudes are large compared to action costs"
PLAN.md §B6 and Risk R3 explicitly defer reward-magnitude tuning to
Phase 5 (training reveals it) or Phase 8 (ablation). RESULTS.md
Iteration 3 documents the calibration choice (`defense_success_bonus =
250`). The committee question on this is well-handled by the docs.

### "MTTC isn't computed in `_step_at_impact`'s natural-IMPACT case"
False alarm: `_step_at_impact` does set `self._compromise_step =
self._step_count` if it wasn't already (line 553-554), so the MTTC
value remains computable in the `impact_is_terminal=False` branch as
well. Verified by `test_phase31_impact_terminal::test_false_explicit_impact_row_decision_*` tests.

### "OOD leakage at the env layer"
Already F2. Production path (factory) is correct — and Phase 5+ all go
through the factory.

---

## 9. Acceptance criteria check (per Step-3 PASS criterion)

- [x] **MDP semantics correct.** State space, action space, transition
      function, terminal-condition logic all match PLAN.md and RESULTS.md
      (see §2 table).
- [x] **Reward function** matches the documented formulation (six terms
      verified line-by-line vs RESULTS.md §3, see §3 of this memo).
      F3 is a doc-only mismatch in the abbreviated descriptions, not in the
      code-vs-RESULTS.md correspondence.
- [x] **All Phase-3 exit gates G3.1–G3.7 PASS.** 30/30 in
      `tests/test_phase3_env_gates.py + test_phase31_impact_terminal.py +
      test_realization_engine_split_aware.py` on `mentor-review/step-3-env`
      (off `main` @ `d4acfca`); see §4.
- [x] **No OOD-leakage at the Phase-3 boundary.** Production path (env
      factory) restricts to `train` split with `exclude_ood=True`. Latent
      bypass at the env-ctor layer is **F2** (minor).
- [-] **Hash chain intact for `docs/results/03_env/`.** No `manifest.json`
      to verify (**F1**, minor; by design per PLAN §3.3 — Phase 3 is
      infrastructure, no figures).
- [x] **Findings filed against documentation** (`docs(phase-3,§...)`), no
      correctness bugs requiring `fix(phase-3,§...)`.

Verdict locked: **PASS-WITH-FIXES**.

---

## 10. Recommended commit sequence (deferred until candidate sign-off)

These are **doc-only** commits the candidate may opt into during Step 3 or
batch into Step 8 (cross-cutting audit). All are minor; none gate the Step 4
review.

1. `docs(phase-3,§audit-trail): document Phase-3 manifest/scoreboard absence`
   — single paragraph in `docs/results/README.md` covering Phase-1, Phase-2,
   and Phase-3 audit-trail asymmetries (F1, plus rolling up Step-1 F4 and
   Step-2 F4 into one place).
2. `docs(phase-3,§env-init): clarify RealizationEngine injection contract`
   — class-level docstring note on `AdversarialIoTEnv` per F2.
3. `docs(phase-3,§reward): align reward-component list with as-built code;
   clarify MTTC is metric not reward` — `docs/reward-shaping.md` per F3.

F4 (R2 MTTC bias) is a Step-9 LaTeX-rebuild concern; F5 is Step-8
cross-cutting. Neither needs a Step-3 commit.

If the candidate prefers to defer F1–F3 to Step 8 (where the cross-cutting
audit will batch all three asymmetry findings), that is also acceptable.

---

## 11. Sign-off

This memo locks the Step-3 verdict at **PASS-WITH-FIXES**. The Phase-3
v2 environment is correctly architected, faithfully implemented,
well-documented, and mechanically verified by 30 tests covering every
gate G3.1–G3.7 plus the Phase-7 `impact_is_terminal` lever and the
Step-1 OOD-exclusion invariant.

The next session executes **Step 4 — Phase 4 Stage Detector review**
(F11, realism, kill-chain confusion matrix). See `03_HANDOFF.md` for the
context-loading recipe and outstanding-actions checklist.

— mentor-review agent, 2026-05-06
