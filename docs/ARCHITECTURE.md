# Architecture

Adversarial-RL IoT defense system. Python, CPU-first, Gymnasium +
Stable-Baselines3. The headline attacker is a **reactive tug-of-war** on the
kill chain: at each active stage it reacts to the defender's force
`d = action − recommended(stage)`, and escalation pressure is **coupled to the
attacker's proximity to the impact stage** rather than to a fixed intrusion
budget. The defender NEVER observes the true stage (POMDP / partial
observability = central thesis).

## Adversarial loop (one `env.step`)

```
MarkovAttacker --(stage)--> RealizationEngine.sample --> observation window
   --> SB3 Blue-Team agent --> action --> env.step --> reward + next stage
```

- **Tug-of-war attacker** (on `MarkovAttacker`) advances or de-escalates one
  stage per step according to the signed force gap. Escalation probability
  rises with proximity to IMPACT: `p_up_eff = p_up * (sigma_min + (1-sigma_min)
  * lambda)`, where `lambda = stage/4` and `sigma_min = 0.4`. IMPACT is
  absorbing; all de-escalation is defender-driven.
- **RealizationEngine** maps that stage to a sampled real CICIoT2023 feature
  row, drawn **session-coherently** (contiguous same-stage runs) with
  **adjacent-stage observation aliasing** at rate α (with probability α a step
  emits a row from an adjacent stage, clamped at the BENIGN/IMPACT endpoints).
- **AdversarialIoTEnv** (Gymnasium) builds the 290-dim windowed observation,
  applies the agent's action, computes the kill-chain reward, and advances or
  de-escalates the attacker. Prevention = holding the attacker below IMPACT
  for the full horizon (no budget to drain).
- **Blue-Team agent** is an SB3 `MlpPolicy` DQN/PPO/A2C.

## `src/` module map

| Module | Role |
|---|---|
| `src/environment/adversarial_env.py` | `AdversarialIoTEnv(gym.Env)`; `AdversarialEnvConfig` dataclass; 5 actions (OBSERVE/LOG/RESTRICT/BLOCK/ISOLATE); reward modes `coupled` (proportionality shaping) and `outcome` (sparse, primary contract); proximity-coupled escalation; session-coherent + aliasing-aware sampling. |
| `src/generator/markov_attacker.py` | `MarkovAttacker`; strictly sequential kill-chain progression (skip_weight=0); absorbing IMPACT. The runtime tug-of-war dynamics live in `AdversarialIoTEnv` (signed proportionality rule), not in this matrix. |
| `src/utils/realization_engine.py` | `RealizationEngine`; `from_split_manifest` (split-aware, `exclude_ood`); `sample(stage)` draws session-coherent rows with adjacent-stage aliasing at rate α. |
| `src/utils/label_mapper.py` | `KillChainStage(IntEnum)` (BENIGN=0..IMPACT=4); `_LABEL_TO_STAGE` (34 CICIoT labels → 5 stages). |
| `src/algorithms/adversarial_algorithm.py` | `AdversarialAlgorithm` (DQN/PPO/A2C factory, always `MlpPolicy`). |
| `src/detector/` | `random_forest.py` (sklearn `RandomForestClassifier` stage detector: `train_random_forest`, save/load, `RandomForestConfig`) + `evaluation.py`. Baseline apparatus, not a contribution. |
| `src/blue_team/` | `env_factory.py` (`make_train_env` / `make_eval_env`); `run_config.py` (`BlueTeamRunConfig` + `EnvConfigSerializable` → `run_manifest.json`); `callbacks.py` (`EpisodeJSONLCallback`); `aggregation.py` (`bootstrap_ci`). |
| `src/benchmark/` | `baseline_policies.py` (random/always_observe/always_block/recommended_action/`RFActingPolicy`/`SB3PolicyAdapter`); `eval_runner.run_policy`. |

## Config flow

The canonical training entry is `scripts/blue_team/train_agent.py`:
`build_run_config` → `EnvConfigSerializable` → `env_factory._build_env_config`
→ `AdversarialEnvConfig`.

Defaults live in two synced copies (`AdversarialEnvConfig` and
`EnvConfigSerializable`): `aliasing_rate=0.0`, `session_coherent=False`,
`no_post_transition_leak=False`, `proximity_coupled=False`,
`proximity_min_escalation=0.4`, `reward_mode='proportional'` (legacy alias for
`coupled`; normalised in `__post_init__`). The redesign regime (outcome reward,
session-coherent, aliasing-aware, proximity-coupled) is injected at runtime
via Make/CLI parameters.

## Caveats

- `main.py:train_rl` is stale/broken; the canonical training entry is
  `scripts/blue_team/train_agent.py`.
- Stale "LSTM"/"Red-Team" docstrings remain in `config.yml`,
  `episode_generator.py`, and `run_config.py`. Harmless; off the live loop.

See `docs/ENVIRONMENT.md` for observation/action/reward mechanics.
