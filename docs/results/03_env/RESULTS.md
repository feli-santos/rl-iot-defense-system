# Phase 3 — Environment v2: Results

> Sister doc to `PLAN.md`. The PLAN is the *audit + design contract* written
> before any code; this doc is the *as-built record* of what actually
> shipped, including the two iterations that were needed to satisfy every
> exit gate. Read both side-by-side if you want to understand the full
> reasoning chain.

## 1 — Summary

| | |
|---|---|
| **Goal** | Replace the buggy v1 environment (4 hard bugs, 2 soft) with one whose lifecycle and reward function actually produce a learnable RL signal. |
| **Output** | A rewritten `AdversarialIoTEnv` + a split-aware `RealizationEngine` + a regression suite that locks the new contract. **No thesis figures** — Phase 3 is infrastructure. |
| **Status** | All 7 exit gates pass. 296 tests green. Ready to feed Phase 4 (detector head + supervised baselines). |

## 2 — Bug fixes (cross-ref to PLAN §1)

| ID | Symptom | Fix |
|----|---------|-----|
| **B1** | Pre-fix median episode length = 3 steps. BLOCK on any active attack ended the episode. BENIGN→IMPACT one-shots from the LSTM ended it even faster. | Dropped the BLOCK-=-win early termination. Added `min_episode_length` (default 20) and an *IMPACT-clamp*: any LSTM transition to IMPACT before that step is downgraded to MANEUVER. Real-world IMPACT requires time-to-execute, so this matches the IoTWarden threat model. |
| **B2** | Reward measured *action-vs-previous-action change* (e.g., `action_escalated` flag) so a flip-flop policy `OBSERVE→BLOCK→OBSERVE→BLOCK` could farm rewards on a monotone attack. | Replaced with **stage-action proportionality** against the IoTWarden recommended-action mapping (`_recommended_action`). Reward now depends only on `decision_stage` and `action`, never on the agent's previous move. |
| **B3** | `correct_de_escalation_reward` was dead code: the LSTM transition matrix is upper-triangular. | Added `_maybe_defender_deescalation`: when the agent picks BLOCK or ISOLATE at an ACCESS+ stage, the env resets the attack to BENIGN with probability `p_defender_deescalation = 0.6`. The agent earns `+defense_success_bonus`. This makes the de-escalation branch reachable and rewards offensive defense. |
| **B4** | `RealizationEngine` read the full 442 237-row snapshot, including val/test/OOD. RL training would consume features that Phase 7 evaluates on → leakage. | Added `RealizationEngine(allowed_indices=...)` and `RealizationEngine.from_split_manifest(...)`. The factory loads the Phase-1 manifest, restricts sampling to the named split, and (by default) excludes the OOD-attack rows. Verified on real data: train pool ∩ val.idx = ∅. |
| **B5** | MTTC was referenced in metrics but never computed. | Per-episode tracking via `_first_attack_step`, `_compromise_step`, `_defender_deescalations`. `info` now exposes `compromised`, `mttc_steps`, `recommended_action`, etc. |
| **B6** | Action-cost ↔ penalty asymmetry: ISOLATE@IMPACT used to net only −190.8, swamping the proportional reward. | Bumped `defense_success_bonus` from 10 to **250** so a *correct* IMPACT response nets +49 (vs −350 for OBSERVE@IMPACT). The asymmetry between win and miss is preserved — and now actually winnable. |

## 3 — Lifecycle & reward formulae (as-built)

### Episode lifecycle

```
reset()                          stage = BENIGN
                                 step_count = 0

while not (terminated or truncated):
    step_count += 1
    previous_stage = current_stage

    if previous_stage == IMPACT:               # 1. final mitigation turn
        return _step_at_impact(action)         #    -> terminate immediately

    reward = _calculate_reward(action, prev)   # 2. proportionality reward

    if (action ≥ BLOCK and prev ≥ ACCESS       # 3a. defender-driven de-escalation
        and rng.random() < p_de_escalation):
        new_stage = BENIGN
        reward += defense_success_bonus
    else:
        new_stage = generator.sample_next(...) # 3b. natural LSTM transition

    if new_stage == IMPACT and step_count < min_episode_length:
        new_stage = MANEUVER                   # 3c. lifecycle-floor clamp

    if new_stage == IMPACT:                    # 4. terminal IMPACT accounting
        terminated = True
        reward -= impact_penalty               #    -200, unconditional
        if action ≥ BLOCK:    reward += defense_success_bonus  # +250
        elif action ≤ LOG:    reward -= penalty_missed_impact  # -150
```

### Reward formula (`_calculate_reward(action, decision_stage)`)

```
R = - C_action(action) * action_cost_scale

    + (reward_benign_passive if decision_stage == BENIGN and action ≤ LOG else 0)
    - (penalty_overreact_benign if decision_stage == BENIGN and action ≥ THROTTLE else 0)
    - (penalty_block_benign if decision_stage == BENIGN and action ≥ BLOCK else 0)
    - (penalty_block_recon if decision_stage == RECON and action ≥ BLOCK else 0)
    - (penalty_missed_impact if decision_stage == IMPACT and action ≤ LOG else 0)

    + (reward_proportional if |action - rec(decision_stage)| ≤ 1
       else - penalty_disproportionate)
```

with the IoTWarden mapping `rec(stage) = stage`:

| stage | recommended action |
|---|---|
| BENIGN | OBSERVE |
| RECON | LOG |
| ACCESS | THROTTLE |
| MANEUVER | BLOCK |
| IMPACT | ISOLATE |

### Default constants (`AdversarialEnvConfig`)

| Field | Value | Notes |
|---|---:|---|
| `max_steps` | 500 | Truncation cap. |
| `min_episode_length` | 20 | IMPACT-clamp window. |
| `p_defender_deescalation` | 0.6 | Phase 8 sweeps this. |
| `reward_proportional` | 5.0 | Per-step proportionality bonus. |
| `penalty_disproportionate` | 5.0 | Symmetric. |
| `impact_penalty` | 200.0 | Always applied at IMPACT termination. |
| `penalty_missed_impact` | 150.0 | Adds to impact_penalty for OBSERVE/LOG. |
| `defense_success_bonus` | 250.0 | Calibrated so ISOLATE@IMPACT nets +49. |
| `reward_benign_passive` | 10.0 | The "do-nothing on benign" reward. |
| `penalty_overreact_benign` | 50.0 | Action ≥ THROTTLE on BENIGN. |
| `penalty_block_benign` | 100.0 | Adds to overreact for action ≥ BLOCK on BENIGN. |
| `penalty_block_recon` | 50.0 | Action ≥ BLOCK on RECON. |

## 4 — Exit-gate scoreboard (PLAN.md §3.2)

All 13 tests in `tests/test_phase3_env_gates.py` pass on commit `36fec22`.

| Gate | Threshold | Observed | Test |
|---|---|---|---|
| G3.1.a | recommended action net-positive at every stage | ✓ | `test_recommended_action_yields_positive_reward_per_step` |
| G3.1.b | overreact on BENIGN net-negative | ✓ | `test_overreaction_on_benign_yields_negative_reward` |
| G3.1.c | underreact at IMPACT net-negative | ✓ | `test_underreaction_on_impact_yields_negative_reward` |
| G3.1.d | always-BLOCK survives ≥ 5 steps | ✓ | `test_block_does_not_terminate_episode_early` |
| G3.1.e | MTTC fields present in info | ✓ | `test_mttc_fields_present_in_info` |
| G3.1.f | MTTC fields = None at reset | ✓ | `test_mttc_is_none_at_reset` |
| G3.1.g | defender de-escalation fires at ACCESS+ | ✓ | `test_defender_deescalation_resets_to_benign` |
| G3.1.h | defender de-escalation does **not** fire below ACCESS | ✓ | `test_defender_deescalation_does_not_fire_below_access` |
| G3.2 | median random-action episode length ≥ 15 | ✓ | `test_g3_2_…` |
| G3.3 | median always-BLOCK episode length ≥ 10 | ✓ | `test_g3_3_…` |
| G3.4 | recommended-policy mean reward > 0 | ✓ | `test_g3_4_…` |
| G3.5 | always-OBSERVE mean reward < 0 | ✓ | `test_g3_5_…` |
| G3.6 | always-ISOLATE mean reward < 0 | ✓ | `test_g3_6_…` |
| G3.7 | full test suite green | ✓ | `pytest -q` → **296 passed** |

## 5 — Iterations & lessons learned

The first cut of the env failed three gates. Each failure pointed to a real
design omission, not a flaky test. Documenting them so future readers
understand the reasoning:

1. **Iteration 1 — lifecycle floor (G3.2/G3.3 fail).** The untrained LSTM
   used by the test fixture had ~uniform transition probabilities, so
   1-in-5 rollouts went BENIGN→IMPACT in two steps. `min_episode_length`
   alone was not enough — I had to *clamp* IMPACT transitions back to
   MANEUVER until the floor elapses. This is principled: real-world IMPACT
   is the *consummation* of MANEUVER, not an instantaneous jump from
   RECON. The clamp is also what enables MTTC to be a meaningful signal:
   it's now the time the attacker spends in MANEUVER trying to break out.

2. **Iteration 2 — terminal-step accounting (G3.5 fail).** Even with the
   clamp, always-OBSERVE was netting +31 (should be < 0). Tracing one
   episode revealed the bug: when the env terminated due to IMPACT, only
   the per-step proportionality reward was applied — `_step_at_impact`
   (which contains the IMPACT penalty) was never reached because the
   rollout exits the loop on `terminated=True`. Inlined the IMPACT
   terminal accounting directly in `step()`. This was a real correctness
   bug, not just a calibration issue.

3. **Iteration 3 — `defense_success_bonus` calibration (G3.4 fail).** With
   the inlined terminal penalty, *even the optimal policy* netted -42.
   Per-step proportionality earns ~+5/step × 20 steps = +100, but
   ISOLATE@IMPACT was -200 - 0.8 + 10 = -190.8. The agent could play
   perfectly and still lose. The fix was to make winning at IMPACT actually
   *winnable*: `defense_success_bonus = 250` so ISOLATE@IMPACT = +49. The
   asymmetry is preserved — OBSERVE@IMPACT still loses -350 — but the
   optimal policy is now strictly net-positive. This matches the IoTWarden
   threat model in which quarantine *before* damage spreads is a *win*,
   not a partial mitigation.

The three iterations show why the gate suite was worth writing: each
"failed" gate caught a real bug or design hole that would have silently
poisoned RL training in Phase 5.

## 6 — What this enables for Phase 4+

The new env exposes:

- **Per-episode telemetry** (`info["mttc_steps"]`, `compromised`,
  `defender_deescalations`) that Phase 7's benchmark plots can consume
  directly. No further env work needed there.
- **Split-aware features** (`RealizationEngine.from_split_manifest`) that
  prevent the test set from leaking into RL training. Phase 4 will use
  this for the detector-head training set; Phase 5 will use it for the
  RL agent.
- **A reward function with a known-positive target** (the recommended
  policy nets > 0). RL training in Phase 5 has a meaningful upper-bound
  reference: any agent worse than the recommended policy is provably
  doing the wrong thing.

## 7 — Risks carried forward

- **R1** (PLAN §6). The `defense_success_bonus = 250` is large by
  comparison to per-step rewards. If Phase 5 finds the agent is
  hyper-aggressive (always BLOCK at the first sign of trouble), Phase 8's
  sensitivity ablation will sweep this parameter. We log it as the most
  likely candidate for downstream re-tuning.
- **R2** The clamp's MANEUVER-substitution might bias MTTC downward
  (compromise always happens at exactly `min_episode_length` if it
  happens at all). Phase 7 should report MTTC restricted to "natural"
  IMPACT events (i.e., where the LSTM produced IMPACT *after*
  step 20), not clamped ones.

---

**Phase-3 commits**: `482299e` (PLAN), `3a6b13a` (split-aware engine),
`2a526af` (env rewrite), `36fec22` (gates + calibration).
