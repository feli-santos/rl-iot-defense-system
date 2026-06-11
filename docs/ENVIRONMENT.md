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
| 2 | THROTTLE | 0.3 | ACCESS |
| 3 | BLOCK | 0.5 | MANEUVER |
| 4 | ISOLATE | 0.8 | IMPACT |

> **C7 — THROTTLE is a dominated action.** De-escalation only fires for `action >= BLOCK`
> (L739), so THROTTLE earns the proportionality bonus on ACCESS but can never trigger
> de-escalation. Disclosed as a limitation in the thesis.

## Kill-chain stages

`KillChainStage(IntEnum)` (`src/utils/label_mapper.py:19`): `BENIGN=0, RECON=1,
ACCESS=2, MANEUVER=3, IMPACT=4`. `_RECOMMENDED_ACTION_BY_STAGE = [0,1,2,3,4]`.

## Markov attacker

Fixed 5x5 first-order transition matrix (`MarkovAttacker._build_transition_matrix` L66):
- BENIGN (row 0): stay 0.4; uniform onset `0.6 * 0.25` into each attack stage.
- Attack rows (i>=1): persistence 0.3; one-step progression `trans[i,i+1]=0.5`;
  longer skips `0.2/distance`; **no regression** (lower triangle 0).
- IMPACT (row 4) is **absorbing** (`trans[4,4]=1.0`).
- **Evasion-before-commit** (`_advance_attack` L782): if `evasion_prob>0`, the defender
  just forced a BLOCK/ISOLATE (`_recent_block`), and stage in {RECON, ACCESS}, the
  attacker stalls (`next_stage = current`) with prob `evasion_prob` — the one
  defender-coupled adaptive axis. `retreat_prob` (default 0) is a non-monotonic
  stress-test override.

## Step lifecycle (`step` L526)

1. Reward at the decision-time stage: `_calculate_reward(action, prev_stage)` L562.
2. Record `_recent_block = action >= 3` L567 (for evasion coupling).
3. **De-escalation** (`_maybe_defender_deescalation` L571): fires if `action >= BLOCK`
   AND `prev_stage >= ACCESS` AND `rng < p_defender_deescalation` (default 0.6). On
   success: stage -> BENIGN, `+defense_success_bonus`, drain budget by
   `budget_reset_cost`.
4. Else `_advance_attack` L576 + drain `budget_step_cost` if `stage >= RECON` (L582).
5. **Grace clamp** L596: any IMPACT before `min_episode_length=20` is downgraded to
   MANEUVER (early "preventions" are not defender-attributable; report prevention-rate
   conditioned on `step >= min_episode_length`).

## Finite attacker budget (prevention model)

- Drains: `budget_step_cost=1` per active progression step (only when `stage >= RECON`);
  `budget_reset_cost=5` per defender-forced de-escalation.
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

Outcome signals applied in `step`/`_step_at_impact`: `defense_success_bonus=250`,
`impact_penalty=200`, `prevention_bonus=0`, terminal FPR penalty (`fpr_penalty_beta=0`).
