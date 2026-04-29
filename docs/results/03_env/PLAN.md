# Phase 3 — Environment v2: audit & plan

> Mentor's plan, written **before any code changes**, based on a full read of
> `src/environment/adversarial_env.py` (528 lines), `src/utils/realization_engine.py`,
> and the Phase-0 diagnosis. The Red Team transition matrix from Phase 2 (F2)
> is also a key input — its shape determines what the environment can actually
> produce.

## 1 — Audit findings

### B1. Episodes terminate too early (often at step ≤ 2)

`step()` has *two* early-termination paths that fire on the very first step
the LSTM produces a non-trivial transition:

1. **L313-332** — "If `_current_attack_stage == IMPACT`: terminate". With
   the Phase-2 Red Team, `T[BENIGN, IMPACT] ≈ 0.25` (intentional: that's the
   IMPACT-frequency design choice). So one in four episodes jumps straight
   from BENIGN to IMPACT at step 1, terminates at step 2.
2. **L335-348** — "If `is_active_attack and action ≥ BLOCK`: terminate".
   The agent can win an entire episode by picking BLOCK exactly once.

Combined effect: median episode length pre-restart was ~3 steps (Phase-0
diagnosis §2.2). The thesis claims "agent learns multi-stage Kill Chain
defence" but the agent never sees a multi-stage trajectory.

### B2. Reward measures "did the agent change action" instead of "is the action proportionate to the stage"

L468-484: the four "R_defense" branches gate on `action_escalated`,
`action_deescalated` (vs *previous* action) and `attack_escalated`,
`attack_deescalated` (vs *previous* attack stage). What we actually want
to reward is **stage-action proportionality**, not action-change behaviour.

Concrete example: an agent that flipped between `OBSERVE → BLOCK → OBSERVE
→ BLOCK → ...` while the attack stage rose monotonically would collect
`+correct_escalation_reward` on every odd step, regardless of whether
BLOCK was the right call.

### B3. The de-escalation reward is dead code

Phase-2 confirmed the LSTM's transition matrix is upper-triangular — no
"attack stage decreasing" event can ever be sampled in a natural rollout.
Therefore `correct_de_escalation_reward` (L478-480) is unreachable.
Either we remove it, or we introduce a *defender-driven de-escalation*
mechanism (see §3, B3 fix).

### B4. Realization Engine reads the *full snapshot*, including val/test/OOD

`RealizationEngine.__init__` (`src/utils/realization_engine.py` L93-130)
loads `state_indices.json` over the entire 442 237 rows. The RL Blue Team
trains on features that include val/test rows — which is data leakage:
when Phase 7 evaluates on `test_balanced`, the agent has *already seen*
those exact rows during training.

Fix is straightforward: the engine must honour the Phase-1 split manifest
and only sample from indices that are *also* in the train split. The OOD
rows must additionally be excluded.

### B5. MTTC is not computed anywhere

Phase-0 diagnosis listed `mttc = 0.0` as a metric. Search of the codebase
confirms the metric is never produced — neither the env nor the benchmark
runner has a definition. We need to add it explicitly.

The standard definition adapted to our environment:

> **MTTC (Mean Time To Compromise)** = average number of environment steps
> from the start of a non-BENIGN attack until the attack reaches IMPACT
> (terminated successfully) **or** the agent successfully terminates the
> attack via BLOCK/ISOLATE.

A high MTTC under attack is good (agent is delaying or preventing
compromise); a low MTTC is bad. We will compute this both per-episode
(in `info["mttc_steps"]`) and aggregate over a rollout.

### B6. Action-cost-reward asymmetry: action costs are tiny vs penalties

`ACTION_COSTS = [0.0, 0.1, 0.3, 0.5, 0.8]` but
`impact_penalty = 200.0`, `penalty_block_benign = 100.0`. Costs are 100×
to 1 000× smaller than penalties. This is fine in principle but invites a
degenerate strategy: "always BLOCK" wins on attacks at the price of
~50 false positives = -5 000 reward, which is dominated by avoiding even
*one* IMPACT (-200 - 150 missed = -350). The agent should be willing to
be slightly trigger-happy. We'll keep the structure but rebalance after
Phase 3 if Phase 5 reveals it's misaligned.

## 2 — Decision: rewrite the reward function and the lifecycle, keep the action space

After studying the IoTWarden paper and the Tharewal et al. survey:

- **Action space** (`Discrete(5)`: OBSERVE / LOG / THROTTLE / BLOCK /
  ISOLATE) — *keep unchanged*. This is canonical and the thesis claim is
  "force-continuum policy", not "novel action design".
- **Observation space** (window of feature vectors + deltas) — *keep
  unchanged*. The 29-feature CICIoT vector is the right granularity.
- **Episode lifecycle** — *rewrite*. New rules in §3.
- **Reward function** — *rewrite*. New formulation in §3.
- **Realization engine** — *change*. Honour split manifest, exclude OOD.
- **MTTC** — *add*.

## 3 — Concrete deliverables

### 3.1 Code changes

1. `src/utils/realization_engine.py`:
   - Accept an optional `allowed_row_indices: np.ndarray` argument that
     restricts sampling to a subset of rows. Defaults to "all rows".
   - Sampling becomes `intersect(state_indices[stage], allowed_row_indices)`.
   - Add `RealizationEngine.from_split_manifest(splits_manifest_path,
     split_name="train", exclude_ood=True)` factory.

2. `src/environment/adversarial_env.py` — new behaviour:
   - **Lifecycle (B1 fix).** Drop the "BLOCK = win" early termination
     entirely. The episode runs for **`min_episode_length` ≤ N ≤ `max_steps`**
     steps. Default `min_episode_length = 20`.
     - At any step, if `current_stage == IMPACT`, the agent gets one
       *defensive* turn and the episode terminates. (Same as today.)
     - When the agent picks BLOCK or ISOLATE on an active attack, the env
       does **not** terminate; instead it forces a *defender-driven
       de-escalation*: with probability `p_defender_deescalation` (default
       0.6) the next stage is reset to BENIGN. The agent earned its
       defence reward without breaking the temporal structure.
     - Truncation still fires at `max_steps` (default 500).
   - **Reward (B2 fix).** Replace the action-change-based reward with a
     stage-action proportionality matrix. The recommended action for each
     stage (the IoTWarden mapping) is:
       - BENIGN  → OBSERVE   (0)
       - RECON   → LOG       (1)
       - ACCESS  → THROTTLE  (2)
       - MANEUVER → BLOCK    (3)
       - IMPACT  → ISOLATE   (4)
     Per step, `R_t = R_proportional - C_action - P_overreact - P_underreact + R_progress_blocked`
     where:
       - `R_proportional = +reward_proportional` if `|action - recommended| ≤ 1`,
         else `-penalty_disproportionate`.
       - `C_action = ACTION_COSTS[action] * action_cost_scale`.
       - `P_overreact` fires when stage = BENIGN and action ≥ THROTTLE
         (preserved from current code).
       - `P_underreact` fires when stage ≥ ACCESS and action ≤ LOG
         (preserved from current code).
       - `R_progress_blocked = +defense_success_bonus` whenever the env
         de-escalates the attack from MANEUVER/IMPACT to BENIGN due to
         agent's BLOCK/ISOLATE.
     Crucially, this reward is computed **purely from `decision_stage`
     and `action`** — no dependency on the agent's *previous* action, no
     dependency on the `attack_escalated`/`attack_deescalated` indicator.
   - **MTTC (B5 fix).** Track `_first_attack_step` (the step at which a
     non-BENIGN stage was first observed) and `_compromise_step` (the
     step at which IMPACT terminated the episode, or `None` if not).
     `info["mttc_steps"] = compromise_step - first_attack_step` if
     compromised, otherwise `None`. Aggregated by the benchmark runner.

3. `tests/test_adversarial_env.py` — extend with regression tests
   (these are the *exit gates* for Phase 3; see §3.2):
   - `test_episode_length_at_least_20_steps_under_random_actions`
   - `test_episode_length_at_least_15_steps_under_always_block`
   - `test_recommended_action_yields_positive_reward`
   - `test_overreaction_on_benign_yields_negative_reward`
   - `test_underreaction_on_impact_yields_negative_reward`
   - `test_realization_engine_honours_train_split`
   - `test_realization_engine_excludes_ood_rows`
   - `test_mttc_is_finite_and_positive_when_compromised`
   - `test_mttc_is_none_when_no_compromise`

### 3.2 Phase-3 exit gates

These are **mechanical** (run by pytest), unlike the empirical gates of
Phase 2:

- **G3.1** All new regression tests in `tests/test_adversarial_env.py` pass.
- **G3.2** Median episode length over 200 random-action rollouts ≥ 15 steps
  (proves B1 is fixed).
- **G3.3** Median episode length over 200 always-BLOCK rollouts ≥ 10 steps
  (proves the "BLOCK = instant win" bug is gone).
- **G3.4** A fresh agent given the recommended action for the current
  stage on every step achieves average reward > 0 over 100 rollouts
  (proves the new reward function rewards correct policy).
- **G3.5** A fresh agent that always picks OBSERVE achieves average
  reward < 0 over 100 rollouts (no degenerate "do nothing" exploit).
- **G3.6** A fresh agent that always picks ISOLATE achieves average
  reward < 0 over 100 rollouts (no degenerate "always blast" exploit).
- **G3.7** Test suite total: 274 (Phase 2) + ~9 new tests + ~5 updated
  tests, all passing.

### 3.3 No new figures in Phase 3

This phase does *not* produce a thesis figure. Its output is a working
environment + regression suite that downstream phases can rely on.
The next thesis figures (F11/F12 in Phase 4) will be the first to consume
the new reward/lifecycle.

## 4 — Sequencing inside Phase 3

| Step | Output | Estimated cost |
|------|--------|----------------|
| 4.1  | This PLAN.md (committed) | ~0.5 h |
| 4.2  | Refactor RealizationEngine: split-aware factory + tests | 1 commit, 1 h |
| 4.3  | Rewrite AdversarialIoTEnv: lifecycle, reward, MTTC | 1 commit, 2 h |
| 4.4  | Update + extend `tests/test_adversarial_env.py` | 1 commit, 1 h |
| 4.5  | Run G3.1–G3.7 gate-evaluation script (`scripts/env/check_gates.py`) | 1 commit, 1 h |
| 4.6  | CHANGELOG entry; cross-link from PLAN.md | 1 commit, 0.5 h |

Total: **5 commits**, ~6 h.

## 5 — What we will *not* re-architect now

- The action space (`Discrete(5)`).
- The observation space (29-D × window × delta).
- `AttackSequenceGenerator` itself (Phase-2 done).
- `EpisodeGenerator`'s synthetic transition matrix (we may *re-use* it for
  testing in Phase 3, not change it).

## 6 — Risks I'm watching

- **R1.** `p_defender_deescalation = 0.6` is a free hyperparameter. Phase 8
  will sweep this in the sensitivity ablation; for Phase 3 we just need
  G3.2/G3.3 to pass. The choice is documented in the env config.
- **R2.** Some existing `test_adversarial_env.py` tests likely encode
  the *old* "BLOCK = win" behaviour. We will update them, not delete them;
  any deletion will be flagged in the commit message with an explicit
  rationale.
- **R3.** The new reward function's absolute scale may be off. Phase 5 RL
  training will reveal this in the learning-curve shape; we'll rebalance
  rewards there, not here. *Phase 3's job is to make the reward
  meaningful, not to tune its magnitude.*

---

**Open question for you:** are you OK with the IoTWarden recommended-action
mapping (§3.1 step 2) — BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE,
MANEUVER→BLOCK, IMPACT→ISOLATE — or do you want a different alignment?
This is the most consequential design choice in Phase 3 because it defines
the policy the agent is being graded against. Default is fine if you don't
have a strong opinion.
