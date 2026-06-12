# Environment

`AdversarialIoTEnv` (`src/environment/adversarial_env.py`) — a Gymnasium POMDP where
a Blue-Team agent defends against a fixed 5x5 Markov attacker walking the Cyber Kill
Chain under a finite intrusion budget.

## Observation

- Sliding window of `window_size=5` per-stage feature rows, each 29 CICIoT2023
  features -> `(5, 29)`.
- First-order deltas appended along the feature axis (`include_deltas=True`) ->
  `(5, 58)`; flattened to **float32**.
- **`obs_dim = window_size * num_features * 2 = 5 * 29 * 2 = 290`** (`__init__` L422).
- Optional `include_stage_pred` appends a one-hot stage prediction from a frozen
  detector (size `num_actions`).
- `observation_space = Box(-inf, +inf, (290,), float32)` (L428).

## Actions (force continuum)

`action_space = Discrete(5)`. There is **no enum** — actions are index-based parallel
lists (`ACTION_NAMES` L61, `ACTION_COSTS` L69).

| idx | action | cost | recommended for stage |
|---|---|---|---|
| 0 | OBSERVE | 0.0 | BENIGN |
| 1 | LOG | 0.1 | RECON |
| 2 | RESTRICT | 0.3 | ACCESS |
| 3 | BLOCK | 0.5 | MANEUVER |
| 4 | ISOLATE | 0.8 | IMPACT |

> **RESTRICT is a first-class proportional action (tug-of-war).** Under the tug-of-war
> dynamics a *proportionate* action (`|action - recommended| == 0`) de-escalates the
> attacker one stage w.p. `p_down=0.90`, so RESTRICT at ACCESS is non-dominated. ISOLATE
> is strictly stronger than BLOCK (`p_down_isolate=0.98` vs `p_down=0.90`), giving a real
> RESTRICT < BLOCK < ISOLATE force gradient. (The old "THROTTLE dominated action" caveat
> is obsolete: it described the pre-redesign de-escalation rule that only fired for
> `action >= BLOCK`.)

## Kill-chain stages

`KillChainStage(IntEnum)` (`src/utils/label_mapper.py:19`): `BENIGN=0, RECON=1,
ACCESS=2, MANEUVER=3, IMPACT=4`. `_RECOMMENDED_ACTION_BY_STAGE = [0,1,2,3,4]`.

## Reactive tug-of-war attacker (headline)

The headline attacker is **reactive and strictly sequential** (`tug_of_war=True`,
`skip_weight=0`). Two mechanisms (`_advance_tug_of_war`):

- **Autonomous onset from BENIGN** (no defender dependence, no budget drain): single
  roll — `p_onset=0.35` -> RECON, `p_onset_access=0.10` -> ACCESS (mid-chain start),
  else dormant. No skip-ahead onset (this is why always-block is a beatable, costly
  baseline rather than leaky).
- **Tug-of-war once active** (RECON..MANEUVER), on the signed force gap
  `d = action - recommended(stage)`:
  - `d <= -1` (under-force): attacker **escalates** one stage w.p. `p_up=0.90`, else holds.
  - `d == 0` (proportional): attacker **de-escalates** one stage w.p. `p_down=0.90`
    (`p_down_isolate=0.98` for ISOLATE), else holds.
  - `d >= 1` (over-force): attacker **holds** (penalised as disproportionate).
- IMPACT is **absorbing**; the attacker never regresses autonomously — all de-escalation
  is defender-driven.
- The legacy autonomous-Markov path (`_advance_attack`, skip distribution,
  `_maybe_defender_deescalation`, `evasion_prob`, `retreat_prob`) is retained **only**
  on the `tug_of_war=False` / `skip_weight>0` ablation path.

## Step lifecycle (`step` L526)

1. Reward at the decision-time stage: `_calculate_reward(action, prev_stage)`.
2. Progression: `_advance_tug_of_war(action, prev_stage)` (signed rule above). A
   proportionate push-back drains `budget_step_cost + budget_reset_cost` and grants a
   capped `reward_deescalation`; any active step drains `budget_step_cost`.
3. **Grace clamp**: any IMPACT before `min_episode_length=20` is downgraded to MANEUVER
   (early "preventions" are not defender-attributable; prevention-rate is reported
   conditioned on `step >= min_episode_length`).

## Finite attacker budget (prevention model)

- Drains: `budget_step_cost=1` per active progression step (only when `stage >= RECON`);
  an additional `budget_reset_cost=2` per defender-forced de-escalation.
- **Exhaustion-before-IMPACT => prevented** (L639): if budget `<= 0` and `stage < IMPACT`,
  the episode terminates with `+prevention_bonus`, outcome `"prevented"` — checked
  **before** the IMPACT branch, fires **regardless of `impact_is_terminal`**.
  Tie-break favors the attacker (budget hitting 0 on the same step IMPACT arrives lets
  IMPACT win).

## Termination matrix (L626-677)

- `attacker_exhausted_now` -> prevented, terminate.
- elif `impact_arrived` (stage==IMPACT and step>=20) AND `impact_is_terminal` -> inline
  terminal reward (`-impact_penalty`; `action>=3` => `+defense_success_bonus`
  "impact_mitigated"; `action<=1` => `-penalty_missed_impact` "impact_missed"; else
  "compromised").
- else `terminated=False` — with `impact_is_terminal=False` (primary contract), IMPACT
  does not terminate; the agent gets an explicit IMPACT-row decision next step via
  `_step_at_impact` L689.
- `truncated = step >= max_steps`.

## Reward (`_calculate_reward` L860)

1. action cost (L891).
2. `reward_mode='outcome_only'` ablation returns after the action cost (L896) —
   strips all stage-conditioned shaping.
3. Asymmetric guardrails (L899): `penalty_overreact_benign=50`, `penalty_block_benign=100`,
   `penalty_block_recon=50`, `penalty_missed_impact=150`.
4. Benign-passive bonus `+reward_benign_passive=10` (L912).
5. Proportionality core (L915): `|action - recommended| <= 1` => `+reward_proportional=5`
   else `-penalty_disproportionate=5`.

Outcome / tug-of-war signals applied in `step`/`_step_at_impact`:
`defense_success_bonus=250` (terminal-IMPACT survival only), `impact_penalty=200`,
`prevention_bonus=50` (budget exhausted before IMPACT), `reward_deescalation=15`
(per proportionate push-back, capped 150/ep), `proportional_bonus_cap=100/ep`. The
proportionality and de-escalation caps remove the reward-farming loopholes that the
redesign fixed.
