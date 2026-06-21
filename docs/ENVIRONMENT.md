# Environment

`AdversarialIoTEnv` (`src/environment/adversarial_env.py`) — a Gymnasium POMDP
where a Blue-Team agent defends against a reactive tug-of-war attacker walking
the Cyber Kill Chain under proximity-coupled escalation (no fixed intrusion
budget).

## Observation

- Sliding window of `window_size=5` per-stage feature rows, each 29 CICIoT2023
  features → `(5, 29)`.
- First-order deltas appended along the feature axis (`include_deltas=True`) →
  `(5, 58)`; flattened to **float32**.
- **`obs_dim = window_size * num_features * 2 = 5 * 29 * 2 = 290`**.
- Optional `include_stage_pred` appends a one-hot stage prediction from a frozen
  detector (size `num_actions`).
- `observation_space = Box(-inf, +inf, (290,), float32)`.

**Session-coherent sampling.** The realization engine draws contiguous
same-stage runs (a session proxy) rather than i.i.d. rows. CICIoT2023 ships
pre-aggregated flow records with no recoverable session key, so session
coherence is a modelling abstraction.

**Observation aliasing (rate α).** With probability α a step emits a feature
row from a stage **adjacent** to the true one (clamped at the BENIGN and IMPACT
endpoints), so adjacent stages overlap in feature space and no single row
identifies the stage. This is the mechanism that turns the task into a genuine
sequential-inference problem rather than a disguised per-flow classification.

## Actions (force continuum)

`action_space = Discrete(5)`. Actions are index-based parallel lists
(`ACTION_NAMES`, `ACTION_COSTS`).

| idx | action | cost | recommended for stage |
|---|---|---|---|
| 0 | OBSERVE | 0.0 | BENIGN |
| 1 | LOG | 0.1 | RECON |
| 2 | RESTRICT | 0.3 | ACCESS |
| 3 | BLOCK | 0.5 | MANEUVER |
| 4 | ISOLATE | 0.8 | IMPACT |

A proportionate action (`|action - recommended| == 0`) de-escalates the
attacker one stage w.p. `p_down=0.90` (`p_down_isolate=0.98` for ISOLATE),
giving a real RESTRICT < BLOCK < ISOLATE force gradient.

## Kill-chain stages

`KillChainStage(IntEnum)` (`src/utils/label_mapper.py`): `BENIGN=0, RECON=1,
ACCESS=2, MANEUVER=3, IMPACT=4`. `_RECOMMENDED_ACTION_BY_STAGE = [0,1,2,3,4]`.

## Reactive tug-of-war attacker

Two mechanisms (`_advance_tug_of_war`):

- **Autonomous onset from BENIGN** (no defender dependence): single roll —
  `p_onset=0.35` → RECON, `p_onset_access=0.10` → ACCESS (mid-chain start),
  else dormant.
- **Tug-of-war once active** (RECON..MANEUVER), on the signed force gap
  `d = action - recommended(stage)`:
  - `d <= -1` (under-force): attacker **escalates** one stage w.p. `p_up_eff`,
    else holds. The escalation probability is proximity-coupled:
    `p_up_eff = p_up * (sigma_min + (1-sigma_min) * lambda)`, where
    `lambda = stage/4` and `sigma_min = 0.4`.
  - `d == 0` (proportional): attacker **de-escalates** one stage w.p.
    `p_down=0.90` (`p_down_isolate=0.98` for ISOLATE), else holds.
  - `d >= 1` (over-force): attacker **holds** (penalised as disproportionate).
- IMPACT is **absorbing**; the attacker never regresses autonomously.

## Prevention model

Prevention = holding the attacker below IMPACT for the full episode horizon.
There is no finite intrusion budget to drain; the proximity-coupled escalation
rule replaces it. A prevented episode earns `+prevention_bonus = 50`.

## Grace clamp

Any IMPACT before `min_episode_length=20` is downgraded to MANEUVER (early
"preventions" are not defender-attributable).

## Reward

Two reward contracts:

- **`outcome`** (primary deployment contract): the per-step reward is **only**
  the action cost; every stage-conditioned shaping term is stripped. The
  learning signal comes exclusively from realised outcomes: the terminal
  prevention bonus, the IMPACT penalty, and the action cost. This de-couples
  the objective from the per-step `recommended_action(stage)` label, removing
  the privileged-shaping advantage of a supervised classifier.
- **`coupled`** (reward-shaping ablation cell): the per-step reward includes a
  proportionality term keyed on `d = action - recommended_action(stage)` — the
  same quantity that drives the attacker's tug-of-war transition. Under this
  contract the value-maximising policy reduces to "infer the stage, emit the
  matching action", i.e. a 5-way stage classifier.

Reward components (per-step, under `coupled`):
1. Action cost (always applied).
2. Proportionality bonus `+reward_proportional=5` if `|action - recommended| <= 1`,
   else `−penalty_disproportionate=5`.
3. Asymmetric guardrails: `penalty_overreact_benign=50`, `penalty_block_benign=100`,
   `penalty_block_recon=50`.
4. Benign-passive bonus `+reward_benign_passive=10`.

Outcome signals applied in `step`/`_step_at_impact`:
`prevention_bonus=50` (prevented), `impact_penalty=200` (compromised),
`defense_success_bonus=250` (terminal-IMPACT survival), `reward_deescalation=15`
(per proportionate push-back, capped 150/ep), `proportional_bonus_cap=100/ep`.
