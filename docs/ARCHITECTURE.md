# Architecture

Adversarial-RL IoT defense system. Python, CPU-first, Gymnasium + Stable-Baselines3.
The Red-Team LSTM has been deleted; the attacker is now a fixed 5x5 first-order
Markov process (`MarkovAttacker`). A finite `attacker_budget` (the prevention model)
governs how long the attacker can climb before exhaustion.

## Adversarial loop (one `env.step`)

```
MarkovAttacker --(stage)--> RealizationEngine.sample --> observation window
   --> SB3 Blue-Team agent --> action --> env.step --> reward + next stage
```

- **MarkovAttacker** emits the next kill-chain stage from a fixed 5x5 transition
  matrix (upper-triangular, absorbing IMPACT).
- **RealizationEngine** maps that stage to a sampled real CICIoT2023 feature row.
- **AdversarialIoTEnv** (Gymnasium) builds the 290-dim observation, applies the
  agent's action, computes the kill-chain reward, advances/deescalates the attacker,
  and drains the attacker budget.
- **Blue-Team agent** is an SB3 `MlpPolicy` DQN/PPO/A2C.

## `src/` module map (phase = thesis chapter)

| Module | Role |
|---|---|
| `src/environment/adversarial_env.py` | `AdversarialIoTEnv(gym.Env)` (`__init__` L380, `reset` L469, `step` L526); `AdversarialEnvConfig` dataclass L165; `ACTION_NAMES` L61 / `ACTION_COSTS` L69; `_RECOMMENDED_ACTION_BY_STAGE=[0,1,2,3,4]` L152; attacker `self._attacker=MarkovAttacker()` L407 (`generator_path` accepted-but-ignored). |
| `src/generator/markov_attacker.py` | `MarkovAttacker` L34; `_build_transition_matrix` L66 (BENIGN row `trans[0,0]=0.4`, attack onset `0.6*0.25` uniform; attack rows persistence 0.3 / progression `trans[i,i+1]=0.5` / skip 0.2/distance, no regression; IMPACT row absorbing `trans[4,4]=1.0`); `sample_next` L107. |
| `src/generator/episode_generator.py` | **Legacy** — off the runtime loop; feeds the dataset-prior stage distribution and mirrors the canonical Markov grammar. Stale "LSTM" docstrings remain. |
| `src/utils/realization_engine.py` | `RealizationEngine` L38; `from_split_manifest` L104 (split-aware, `exclude_ood`); `num_features` L252 (=29); `sample(stage)` L271. |
| `src/utils/label_mapper.py` | `KillChainStage(IntEnum)` L19 (BENIGN=0..IMPACT=4); `_LABEL_TO_STAGE` L44 (34 CICIoT labels -> 5 stages). |
| `src/algorithms/adversarial_algorithm.py` | `AdversarialAlgorithm` L88 (DQN/PPO/A2C factory, always `MlpPolicy`). |
| `src/detector/` | `StageDetector` (MLP 29->64->32->5), `random_forest` (`RFActingPolicy`), `cnn1d`, `evaluation` (`NUM_STAGES=5`). Baseline apparatus, not a contribution. |
| `src/blue_team/` | `env_factory.py` (`make_train_env` L141 / `make_eval_env` L186 / `_build_env` L96 / `_build_env_config` L45); `run_config.py` (`BlueTeamRunConfig` L117 + `EnvConfigSerializable` L47 mirror written to `run_manifest.json`); `callbacks.py` (`EpisodeJSONLCallback`); `aggregation.py` (`bootstrap_ci`). |
| `src/benchmark/` | `baseline_policies.py` (random/always_observe/always_block/recommended_action/`RFActingPolicy`/`SB3PolicyAdapter`); `eval_runner.run_policy` L67; `latency.py`; `model_stats.py`. |
| `src/training/training_manager.py` | `TrainingManager` + `MLflowCallback` (used by `main.py`). |

## Config flow — two paths

- **Path A (LEGACY, broken): `config.yml` -> `main.py`.** `main.py:train_rl`
  constructs `AdversarialEnvConfig(...)` with kwargs that no longer exist on the
  dataclass (`false_positive_penalty`, `patience_bonus`, `correct_*_reward`,
  `maintained_defense_reward`) and requires a deleted `attack_sequence_generator.pth`.
  This path raises `TypeError` — do not use it.
- **Path B (LIVE): `scripts/blue_team/train_agent.py`.** `build_run_config` L173 ->
  `EnvConfigSerializable` -> `env_factory._build_env_config` L45 -> `AdversarialEnvConfig`.

Defaults live in two synced copies (`AdversarialEnvConfig` L165 and
`EnvConfigSerializable` L47): `attacker_budget=None` (code default; **40 is injected
at runtime** via Make/CLI `BLUE_TEAM_REWARD_OVERRIDES` / `BENCHMARK_ATTACKER_BUDGET`),
`budget_step_cost=1`, `budget_reset_cost=5`, `budget_cost_model='hybrid'`,
`prevention_bonus=0`, `impact_is_terminal=True` (code default; the locked primary
contract **False** is passed explicitly by scripts), `evasion_prob=0`, `retreat_prob=0`,
`reward_mode='proportional'`.

## Caveats (for future agents)

- `main.py:train_rl` is stale/broken; the canonical training entry is
  `scripts/blue_team/train_agent.py`.
- `attacker_budget=40` is an experiment contract injected at runtime, **not** a code
  default (code default is `None` / unbounded).
- Numerous "LSTM" / "Red Team" docstrings/comments remain stale across `config.yml`,
  `episode_generator.py`, `run_config.py`, and `config_loader.py` (`_validate_config`
  still requires a legacy `attack_generator` section).

See `docs/ENVIRONMENT.md` for observation/action/reward/budget mechanics and
`docs/RESULTS.md` for the budget=40 headline + gate scoreboard.
