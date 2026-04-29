"""Phase-5 Blue Team training infrastructure.

Public API:

- :class:`EpisodeJSONLCallback` — SB3 callback that flushes one JSON
  line per terminated/truncated episode, capturing Phase-3 telemetry
  (MTTC, compromised, defender_deescalations, action histogram).
- :class:`BlueTeamRunConfig` — dataclass binding for a single
  ``(algo, seed, total_timesteps, ...)`` run; serialises to / from a
  ``run_manifest.json``.
- :func:`make_train_env` / :func:`make_eval_env` — split-aware
  environment factories that wire :class:`AdversarialIoTEnv` up to a
  :class:`RealizationEngine` restricted to a Phase-1 split.
- :mod:`src.blue_team.aggregation` — reading + smoothing + bootstrap-CI
  helpers consumed by ``scripts/blue_team/plot_*.py``.

See ``docs/results/05_blue_team/PLAN.md`` §3.1 for the contract.
"""

from src.blue_team.callbacks import (
    EpisodeJSONLCallback,
    EpisodeRecord,
    EvalToJSONLCallback,
)
from src.blue_team.env_factory import make_eval_env, make_train_env
from src.blue_team.run_config import BlueTeamRunConfig, EnvConfigSerializable

__all__ = [
    "BlueTeamRunConfig",
    "EnvConfigSerializable",
    "EpisodeJSONLCallback",
    "EpisodeRecord",
    "EvalToJSONLCallback",
    "make_eval_env",
    "make_train_env",
]
