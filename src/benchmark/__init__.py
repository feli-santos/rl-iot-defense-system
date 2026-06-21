"""Held-Out Benchmark RL Algorithm Benchmark — public package surface.

The benchmark package consumes the **frozen Blue-Team Training checkpoints** at
``runs/blue_team/<algo>/seed_<k>/model.zip`` and rolls them — together with
a fixed set of non-RL baselines — on the held-out ``test_balanced``
split, emitting JSONL files in the same v1.0 schema as
:class:`src.blue_team.callbacks.EpisodeJSONLCallback` so all downstream
aggregation utilities (Blue-Team Training's ``aggregation.py``) work unmodified.

Public symbols (PLAN §3.1):

- :func:`random_policy`, :func:`always_observe`, :func:`always_block`,
  :func:`recommended_action_policy` — closed-form non-RL baselines.
- :class:`RFActingPolicy` — Stage Detector RandomForest stage classifier
  composed with the recommended-action mapping (D6.5).
- :class:`SB3PolicyAdapter` — wraps a stable-baselines3 ``BaseAlgorithm``
  to expose the ``(obs, info) -> int`` Policy protocol.
- :func:`run_policy` — rolls any Policy on a VecEnv for ``n_episodes``
  episodes, writing schema-v1.0 episode JSONL plus an optional sidecar
  ``latency.jsonl``.
- :func:`measure_inference_latency` — standalone CDF-ready latency
  benchmark for F7.

The Held-Out Benchmark deliberately re-uses ``EpisodeRecord`` from
:mod:`src.blue_team.callbacks`; we do **not** redefine the schema here
(D6.4 — schema v1.0 stays frozen).
"""

from __future__ import annotations

from src.benchmark.baseline_policies import (
    Policy,
    RFActingPolicy,
    SB3PolicyAdapter,
    always_block,
    always_observe,
    random_policy,
    recommended_action_policy,
)
from src.benchmark.eval_runner import LatencyRecord, run_policy
from src.benchmark.latency import measure_inference_latency

__all__ = [
    # baseline_policies
    "Policy",
    "RFActingPolicy",
    "SB3PolicyAdapter",
    "always_block",
    "always_observe",
    "random_policy",
    "recommended_action_policy",
    # eval_runner
    "LatencyRecord",
    "run_policy",
    # latency
    "measure_inference_latency",
]
