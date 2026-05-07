"""Phase-6 generic policy roll-out harness (PLAN §3.1.2).

:func:`run_policy` is the Phase-6 analogue of Phase-5's
:class:`EvalToJSONLCallback._run_eval_block`. It rolls any
:class:`Policy` on a vectorised env for ``n_episodes`` complete
episodes, emitting:

1. A schema-**v1.0** ``episodes.jsonl`` (same dataclass as
   :class:`src.blue_team.callbacks.EpisodeRecord`) so Phase-5's
   ``aggregation.py`` reads Phase-6 outputs unchanged.
2. An optional sidecar ``latency.jsonl`` with one row per step
   (``{"step_idx", "duration_ns"}``) — opt-in via ``latency_path``.
   Per D6.4 we deliberately do **not** extend EpisodeRecord with a
   latency field; schema v1.0 stays frozen.

The rollout assumes a single-env :class:`DummyVecEnv` (Phase-5's
default — see ``env_factory.make_eval_env``). ``n_envs > 1`` is
unsupported here and would produce ambiguous per-step latency rows;
use multiple sequential calls instead.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from src.blue_team.callbacks import (
    EpisodeRecord,
    _SCHEMA_VERSION,
    _EpisodeAccumulator,
)
from src.utils.label_mapper import KillChainStage

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """One row of the sidecar ``latency.jsonl`` (Phase-6, PLAN §3.1.3).

    Per-step records are intentionally minimal: an episode-bounding
    ``episode_idx`` plus the in-episode ``step_idx`` so plotters can
    reconstruct trajectories if needed, plus the wall duration in
    nanoseconds.

    The ``run_id`` and ``policy_name`` are echoed once so cross-file
    aggregation is unambiguous.
    """

    schema_version: str
    run_id: str
    policy_name: str
    episode_idx: int
    step_idx: int
    duration_ns: int

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def run_policy(
    policy: Callable[[np.ndarray, Dict[str, Any]], int],
    env: Any,
    *,
    n_episodes: int,
    jsonl_path: Union[str, Path],
    run_id: str,
    policy_name: Optional[str] = None,
    latency_path: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """Roll ``policy`` on ``env`` for ``n_episodes`` and write the JSONL.

    Args:
        policy: Anything that satisfies the
            ``(obs: np.ndarray, info: Dict) -> int`` Protocol.
        env: A single-env ``DummyVecEnv`` (or any VecEnv with
            ``num_envs == 1``). Phase-5's
            :func:`src.blue_team.env_factory.make_eval_env` is the
            canonical producer.
        n_episodes: Number of complete episodes to roll. The rollout
            terminates after the ``n_episodes``-th episode finishes,
            regardless of any further env autoreset.
        jsonl_path: Output path for the EpisodeRecord-v1.0 JSONL.
            Parent directory is created if missing. **Overwritten** —
            callers that want append semantics must rotate paths
            themselves (matches Phase-5 behaviour).
        run_id: Stable identifier echoed in every record (e.g.
            ``"random_seed_2"`` or ``"ppo_seed_3_test"``). Phase-6's
            convention is ``"<policy_name>_seed_<k>_test"``.
        policy_name: Free-form label (e.g. ``"random"``, ``"ppo"``,
            ``"rf_acting"``). Echoed in latency records and used by F5
            / F8 plot scripts to group rows. Defaults to ``run_id``
            when ``None``.
        latency_path: Optional sidecar path. When provided, every
            policy call's wall duration (``time.perf_counter_ns``) is
            logged as one row. ``None`` disables latency capture (the
            default for the F5/F6/F8 sweeps; F7 turns it on).
        seed: **No-op at the env layer (Step-6 F4 / Step-8 doc-fix).**
            Pre-Step-8 the docstring claimed this was forwarded to
            ``env.reset(seed=...)``; in fact both branches of the
            implementation below call identical ``env.reset()`` (SB3
            ``DummyVecEnv`` 1.x's ``reset()`` does not accept a
            ``seed`` kwarg; the seed plumbing was never wired through).
            The parameter is preserved for backwards compatibility but
            has zero effect on rollout behaviour at this layer.
            Reproducibility is delivered by *caller-side* env-construction
            seeding — see ``scripts.benchmark.run_test_eval`` which
            constructs each per-run env with an explicit seed via
            ``make_eval_env(spec=..., seed=seed)``. Phase-6 numbers
            are deterministic on a given checkpoint × split because
            the agent runs ``deterministic=True`` and the random
            baseline is seeded externally; this parameter does not
            influence either path.

    Returns:
        Dict with bookkeeping totals: ``{"n_episodes_written",
        "n_steps_total", "n_latency_rows"}``. Useful for the manifest.

    Notes on the per-step bookkeeping (mirrors EvalToJSONLCallback):

        - ``decision_stage`` for the *first* step of an episode is
          ``BENIGN`` (the env's reset contract).
        - For subsequent steps, ``decision_stage`` is the
          ``info["attack_stage"]`` written by the *previous* env.step
          — i.e., what the agent saw when it picked the upcoming
          action. Capturing this correctly is what makes the per-stage
          action histogram meaningful for F6.
    """
    pol_name = policy_name if policy_name is not None else run_id
    out_path = Path(jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lat_fh = None
    lat_path = Path(latency_path) if latency_path is not None else None
    if lat_path is not None:
        lat_path.parent.mkdir(parents=True, exist_ok=True)
        lat_fh = lat_path.open("w", encoding="utf-8")

    n_steps_total = 0
    n_latency_rows = 0
    t_run_start = time.time()

    with out_path.open("w", encoding="utf-8") as ep_fh:
        # First reset is the only place we may pass `seed` — Gym/SB3
        # contract: subsequent resets are auto-seeded by the wrapper.
        if seed is not None:
            obs = env.reset()  # SB3 DummyVecEnv ignores `seed=` kwarg pre-1.x;
            # fall back to setting the action_space seed via env's RNG
            # by calling reset with a numpy-style call when supported.
        else:
            obs = env.reset()

        for episode_idx in range(n_episodes):
            acc = _EpisodeAccumulator()
            decision_stage = 0  # BENIGN at reset, see env.reset()
            step_idx = 0
            done = False
            # `obs` carries over: on episode 0 from the initial reset
            # above; on subsequent episodes from the autoreset that
            # SB3 does inside the previous env.step (the obs returned
            # by step() *is* the post-reset obs).
            while not done:
                info_for_policy = _info_seed(decision_stage)
                t0 = time.perf_counter_ns() if lat_fh is not None else 0
                action = int(policy(_squeeze(obs), info_for_policy))
                t1 = time.perf_counter_ns() if lat_fh is not None else 0

                obs, reward, dones, infos = env.step(np.asarray([action]))
                r = float(np.asarray(reward).reshape(-1)[0])
                acc.update(action, r, decision_stage)
                done = bool(np.asarray(dones).reshape(-1)[0])
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                if not done:
                    decision_stage = int(info.get("attack_stage", 0))
                else:
                    _emit_episode(
                        ep_fh, acc, info, episode_idx, run_id, pol_name,
                        n_steps_total + step_idx + 1, t_run_start,
                    )

                if lat_fh is not None:
                    lat = LatencyRecord(
                        schema_version=_SCHEMA_VERSION,
                        run_id=run_id,
                        policy_name=pol_name,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        duration_ns=int(t1 - t0),
                    )
                    lat_fh.write(lat.to_jsonl())
                    lat_fh.write("\n")
                    n_latency_rows += 1

                step_idx += 1
            n_steps_total += step_idx

    if lat_fh is not None:
        lat_fh.flush()
        lat_fh.close()

    logger.info(
        "run_policy completed: run_id=%s policy=%s episodes=%d steps=%d latency_rows=%d",
        run_id, pol_name, n_episodes, n_steps_total, n_latency_rows,
    )
    return {
        "n_episodes_written": n_episodes,
        "n_steps_total": n_steps_total,
        "n_latency_rows": n_latency_rows,
    }


# ----------------------------------------------------------------- helpers


def _squeeze(obs: Any) -> np.ndarray:
    """Return ``obs`` with a leading 1-batch dim removed if present.

    Phase-5's DummyVecEnv emits ``(1, obs_dim)``; baselines like
    :func:`recommended_action_policy` ignore the obs entirely, but
    :class:`RFActingPolicy` needs the 1-D vector. SB3 models
    re-add the batch dim themselves inside :class:`SB3PolicyAdapter`,
    so squeezing here is the right shape for *all* callees.
    """
    arr = np.asarray(obs)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _info_seed(decision_stage: int) -> Dict[str, Any]:
    """Build the *pre-step* info dict the policy sees at decision time.

    The Phase-3 env's ``info["recommended_action"]`` is a function of
    the *current* attack stage at decision time (not the post-step
    stage). We reconstruct it here from ``decision_stage`` so policies
    that depend on it (``recommended_action_policy``) still work even
    on the very first step (where the env hasn't emitted info yet).

    The mapping must stay in sync with
    ``src/environment/adversarial_env.py::_recommended_action``; the
    constant lives in :mod:`baseline_policies` to keep one canonical
    definition.
    """
    from src.benchmark.baseline_policies import _RECOMMENDED_BY_STAGE

    return {
        "attack_stage": int(decision_stage),
        "recommended_action": int(_RECOMMENDED_BY_STAGE[int(decision_stage)]),
    }


def _emit_episode(
    fh: Any,
    acc: _EpisodeAccumulator,
    info: Dict[str, Any],
    episode_idx: int,
    run_id: str,
    policy_name: str,  # noqa: ARG001 — policy_name lives in latency records, not EpisodeRecord
    num_timesteps: int,
    t_start: float,
) -> None:
    """Write one EpisodeRecord-v1.0 line to ``fh``.

    ``num_timesteps`` is the cumulative step count *across* episodes in
    this run — Phase-5 used the SB3 global step counter; here we use
    the analogous "step number since the rollout started" so the
    Phase-5 :func:`bin_by_timesteps` aggregator still produces sensible
    bucket assignments if the rollout JSONL is fed through it (Phase 6
    aggregations don't bin by timestep, but the schema invariant
    matters).
    """
    # On done, SB3 DummyVecEnv packs the pre-reset (terminal) info under
    # info["terminal_info"]; if absent (e.g., custom env wrappers that
    # don't add it), fall back to the live info — the env's
    # ``_step_at_impact`` writes the right values into both branches.
    src = info.get("terminal_info") if isinstance(info, dict) else None
    if not isinstance(src, dict):
        src = info if isinstance(info, dict) else {}

    final_stage = int(src.get("attack_stage", 0))
    mttc_steps = src.get("mttc_steps")
    compromised = bool(src.get("compromised", False))
    defender_deescalations = int(src.get("defender_deescalations", 0))
    outcome = str(src.get("outcome", "unknown"))

    # When SB3 packs Monitor's "episode" subdict, its totals include the
    # terminal step's reward and length — same precedence rule as
    # EpisodeJSONLCallback._emit_record.
    monitor = info.get("episode") if isinstance(info, dict) else None
    ep_reward = (
        float(monitor["r"]) if monitor and "r" in monitor
        else float(acc.cumulative_reward)
    )
    ep_length = (
        int(monitor["l"]) if monitor and "l" in monitor
        else int(acc.length)
    )

    record = EpisodeRecord(
        schema_version=_SCHEMA_VERSION,
        run_id=run_id,
        # Phase-6 EpisodeRecord still carries `algo` because the
        # aggregator reads it — for non-RL baselines we put the policy
        # name there. The F5 plotter groups by policy_name (matching
        # `run_id` parsing), so this stays consistent.
        algo=_algo_field_from_run_id(run_id),
        seed=_seed_field_from_run_id(run_id),
        episode_idx=episode_idx,
        num_timesteps=int(num_timesteps),
        wallclock_seconds=time.time() - t_start,
        episode_reward=ep_reward,
        episode_length=ep_length,
        compromised=compromised,
        mttc_steps=int(mttc_steps) if mttc_steps is not None else None,
        defender_deescalations=defender_deescalations,
        final_stage=final_stage,
        final_stage_name=KillChainStage(final_stage).name,
        end_outcome=outcome,
        action_counts=list(acc.action_counts),
        action_counts_by_stage={
            str(s): list(c) for s, c in acc.action_counts_by_stage.items()
        },
    )
    fh.write(record.to_jsonl())
    fh.write("\n")


def _algo_field_from_run_id(run_id: str) -> str:
    """Extract the algo / policy field from a Phase-6 run_id.

    Convention (PLAN §3.1.4): ``"<policy_name>_seed_<k>_test"`` or
    ``"<policy_name>_seed_<k>"``. The first underscore-token before
    ``"seed"`` is the algo / policy label.
    """
    head = run_id.split("_seed_", 1)[0]
    return head if head else run_id


def _seed_field_from_run_id(run_id: str) -> int:
    """Extract the integer seed from a Phase-6 run_id; 0 if absent."""
    parts = run_id.split("_seed_", 1)
    if len(parts) != 2:
        return 0
    tail = parts[1]
    # Tail may be "3" or "3_test"; take leading digits.
    digits: list[str] = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    return int("".join(digits)) if digits else 0


__all__ = ["LatencyRecord", "run_policy"]
