# Reward Shaping

> Updated in Step 8 (Step-3 F3 / Step-5 F6 doc-fix) to reflect the
> as-built Phase-3 v2 reward function. The pre-Step-8 version of this
> document described a v1 reward (action-vs-previous-action change,
> `correct_escalation_reward`, `patience_bonus`) that was retired
> at the **B2 fix** in Phase 3 — see
> `docs/results/03_env/RESULTS.md` §2.B2 for the rationale.

This system uses a piecewise reward to balance **security** and
**availability**. The reward is computed in
`AdversarialIoTEnv._calculate_reward` (per-step) and
`AdversarialIoTEnv.step` / `_step_at_impact` (terminal). The
canonical numerical reference is
`docs/results/03_env/RESULTS.md` §3 (formulae) + §4 (gate-tested
calibration).

## Decomposition: six reward signals + three asymmetric guardrails

The reward has **six independent reward signals**:

1. **Action cost** ``- C_action(action) * action_cost_scale`` — the
   force-continuum cost in `ACTION_COSTS`.
2. **Per-step proportionality bonus / penalty** —
   ``+ reward_proportional`` if ``|action - rec(decision_stage)| ≤ 1``,
   else ``- penalty_disproportionate``. Graded against the IoTWarden
   recommended-action mapping ``rec(stage) = stage`` (BENIGN→OBSERVE,
   RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK, IMPACT→ISOLATE).
3. **Terminal-IMPACT penalty** ``- impact_penalty`` — applied
   unconditionally when the episode terminates at IMPACT (either
   via the inline-terminal path with `impact_is_terminal=True`
   [reward-mis-specification case study], or via `_step_at_impact` when
   the env passes through IMPACT and grants the agent a final mitigation
   turn [primary contract: `impact_is_terminal=False`]).
4. **Terminal-IMPACT mitigation bonus** ``+ defense_success_bonus``
   — earned when the agent picks BLOCK or ISOLATE at IMPACT *and*
   when the env de-escalates from MANEUVER/IMPACT to BENIGN
   because the agent's strong action triggered the
   `_maybe_defender_deescalation` branch (B3 fix).
5. **Terminal-IMPACT miss penalty** ``- penalty_missed_impact`` —
   added when the agent picks OBSERVE/LOG at IMPACT.
6. **Benign-passive bonus** ``+ reward_benign_passive`` — for
   OBSERVE/LOG on truly BENIGN traffic. Small but consistent
   positive signal so the "do nothing" baseline does not net to
   zero.

Plus **three asymmetric guardrails** for cocked-trigger
classification policies:

- ``- penalty_overreact_benign`` (action ≥ THROTTLE on BENIGN);
- ``- penalty_block_benign`` (additional, action ≥ BLOCK on BENIGN);
- ``- penalty_block_recon`` (action ≥ BLOCK on RECON).

The three guardrails are exposed as independent CLI / sweep targets
in `src/blue_team/env_factory.py:53-73` (Phase 5 plumbed nine reward
fields plus `action_cost_scale` for this reason). They are *not*
additional reward terms — they are sub-modulators that collapse
into signal (1) above for the purposes of the §3 RESULTS narrative.
**Step-5 F6** surfaced this count divergence ("six terms" in the
narrative vs. "nine fields" in the wiring); **Step-3 F3** documented
that MTTC is a metric (not a reward term) — see below.

## MTTC is a metric, not a reward term

`info["mttc_steps"]` is a per-episode telemetry field exposed by
`AdversarialIoTEnv._build_info` (B5 fix in Phase 3). It is **not**
consumed by `_calculate_reward`. Earlier mentor-review prompts that
listed "MTTC" alongside `reward_proportional` / `defense_success_bonus`
/ `penalty_disproportionate` as a fourth reward component were
mis-stated — MTTC is reported in Phase 5/6/7 as a security KPI,
**not optimised against** by the agent.

## Phase-3 default constants (`AdversarialEnvConfig`)

| Field | Value | Where it fires |
|---|---:|---|
| `action_cost_scale` | 1.0 | All steps; multiplier on `ACTION_COSTS[action]`. |
| `reward_proportional` | 5.0 | All non-IMPACT steps where `|action - rec(stage)| ≤ 1`. |
| `penalty_disproportionate` | 5.0 | All non-IMPACT steps where `|action - rec(stage)| ≥ 2`. |
| `impact_penalty` | 200.0 | Terminal-IMPACT step (always). |
| `defense_success_bonus` | 250.0 | Terminal-IMPACT BLOCK/ISOLATE; defender-driven de-escalation BLOCK/ISOLATE @ ACCESS+. |
| `penalty_missed_impact` | 150.0 | Terminal-IMPACT OBSERVE/LOG. |
| `reward_benign_passive` | 10.0 | Non-terminal BENIGN + OBSERVE/LOG. |
| `penalty_overreact_benign` | 50.0 | Non-terminal BENIGN + action ≥ THROTTLE. |
| `penalty_block_benign` | 100.0 | Additional, on top of overreact, when action ≥ BLOCK on BENIGN. |
| `penalty_block_recon` | 50.0 | Non-terminal RECON + action ≥ BLOCK. |
| `max_steps` | 500 | Truncation cap. |
| `min_episode_length` | 20 | IMPACT-clamp window (PLAN B1 fix). |
| `p_defender_deescalation` | 0.6 | Probability the env honours `_maybe_defender_deescalation`. Phase-7 F10 sweeps this. |
| `impact_is_terminal` | True | Phase-3 frozen contract (default); Phase-7 F9 sweeps this. |

Calibration check: optimal-policy net reward over a 20-step episode
is ``(20 × +5) + (-200 + 250) + (-0.8 ISOLATE cost) ≈ +49``, against
``always-OBSERVE`` ``(20 × +10 -200 -150) ≈ -150``. The asymmetry
between win and miss is preserved and the gate G3.4 holds (PLAN
required mean reward > 0 for the recommended policy).

## Phase-7 ablation axes that touch the reward

`scripts/blue_team/train_agent.py` exposes the reward function via
two CLI mechanisms:

- ``--reward-overrides JSON`` — arbitrary key/value overrides on
  the nine reward fields above + `action_cost_scale`.
- ``--p-defender-deescalation FLOAT`` — direct override on the
  lifecycle parameter that controls B3.
- ``--impact-is-terminal BOOL`` — direct override on the
  Phase-3 frozen lifecycle contract; flips `_step_at_impact`
  pathway visibility (Phase-7 F9 / D7.3).

Phase 7 swept all three under controlled conditions; see
`docs/results/07_ablation/RESULTS.md` §6.1 (F9 reward-component
sweep) and §6.3 (F10 aggressiveness sweep).

## Configuration keys (`config.yml`)

The Phase-3 frozen defaults are mirrored into `config.yml` under
`adversarial_environment.reward.*` for runtime override:

- `action_cost_scale`
- `reward_proportional`
- `penalty_disproportionate`
- `impact_penalty`
- `defense_success_bonus`
- `penalty_missed_impact`
- `reward_benign_passive`
- `penalty_overreact_benign`
- `penalty_block_benign`
- `penalty_block_recon`

Phase 5 / 6 / 7 read these via `EnvConfigSerializable` in
`src/blue_team/run_config.py`; the defaults match
`AdversarialEnvConfig` byte-for-byte (verified by
`tests/test_blue_team_run_config.py`).

## Notes

- The reward is evaluated against the **decision-time stage** (the
  stage the agent saw when it chose `action`), not against the
  *next* stage. This preserves causal credit assignment — the agent
  is graded on what it knew when it acted.
- The terminal-IMPACT branch is now inlined into `step()` when
  `impact_is_terminal=True` (Phase-3 frozen contract); the
  `_step_at_impact` helper is reachable only when
  `impact_is_terminal=False` (Phase-7 F9 ablation cell).
- Pre-Phase-3 v1 reward was action-vs-previous-action; this was
  retired at B2 because a flip-flop `OBSERVE→BLOCK→OBSERVE→BLOCK`
  policy could farm the `correct_escalation_reward` on a monotone
  attack. The current proportionality formulation depends only on
  `(decision_stage, action)`, never on the agent's previous move.
