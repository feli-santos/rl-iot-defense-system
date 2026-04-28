# Reward Shaping

This system uses a piecewise reward to balance **security** and **availability**. The reward is computed in `AdversarialIoTEnv._calculate_reward`.

## Base formulation

$$R_t = R_{defense} - C_{action} - P_{impact}$$

Where:

- $C_{action}$ is the action cost scaled by `action_cost_scale`.
- $P_{impact}$ is applied when the attack reaches IMPACT (with a one-step mitigation window).
- $R_{defense}$ rewards correct escalation/de-escalation and appropriate persistence.

## Action cost

$$C_{action} = c(a) \cdot \text{action\_cost\_scale}$$

Costs are defined in `adversarial_env.py` and tuned in `config.yml`.

## Benign and low-risk behavior

- **Overreaction penalty** (BENIGN with aggressive actions):
  - `penalty_overreact_benign`
  - Additional `penalty_block_benign` for BLOCK/ISOLATE
- **Passive benign bonus** (OBSERVE/LOG on BENIGN):
  - `reward_benign_passive`
- **Patience bonus** on low-risk stages (RECON) with passive actions:
  - `patience_bonus`

## Escalation matching

- Escalate when attack escalates → `correct_escalation_reward`
- De-escalate when attack de-escalates → `correct_de_escalation_reward`
- Maintain appropriate level when attack persists → `maintained_defense_reward`

## IMPACT handling (one-step mitigation window)

When the attack stage is IMPACT at decision time:

1. **Full impact penalty** is applied
2. **BLOCK/ISOLATE** earns a partial mitigation bonus
3. **OBSERVE/LOG** applies additional `penalty_missed_impact`

This creates a strong preference for earlier intervention without making the last step meaningless.

## Configuration keys

From `config.yml`:

- `adversarial_environment.reward.action_cost_scale`
- `adversarial_environment.reward.impact_penalty`
- `adversarial_environment.reward.defense_success_bonus`
- `adversarial_environment.reward.false_positive_penalty`
- `adversarial_environment.reward.penalty_overreact_benign`
- `adversarial_environment.reward.penalty_block_benign`
- `adversarial_environment.reward.penalty_block_recon`
- `adversarial_environment.reward.penalty_missed_impact`
- `adversarial_environment.reward.reward_benign_passive`
- `adversarial_environment.reward.patience_bonus`
- `adversarial_environment.reward.defense_reward.*`

## Notes

- The reward is evaluated against the **previous** stage to preserve causal credit assignment.
- In the IMPACT branch, reward is computed before early termination and includes a final action outcome.
