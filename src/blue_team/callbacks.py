"""SB3 callbacks for blue-team Blue-Team training.

The single point of integration with stable-baselines3 is
:class:`EpisodeJSONLCallback`, which writes one JSON line per terminated
or truncated episode, capturing the environment-design ``info`` telemetry plus a
per-episode action histogram. The schema is intentionally flat so that
``aggregation.read_episodes_jsonl`` can deserialise it with a single
pandas call.

Schema of one line in ``episodes.jsonl``::

    {
      "schema_version": "1.0",
      "run_id": "ppo_seed_3",
      "algo": "ppo",
      "seed": 3,
      "episode_idx": 412,
      "num_timesteps": 41200,
      "wallclock_seconds": 78.421,
      "episode_reward": 47.62,
      "episode_length": 100,
      "compromised": false,
      "mttc_steps": null,
      "defender_deescalations": 2,
      "final_stage": 0,
      "final_stage_name": "BENIGN",
      "end_outcome": "ongoing",
      "action_counts": [38, 25, 15, 14, 8],
      "action_counts_by_stage": {
        "0": [30, 20, 5, 0, 0],
        "1": [4, 4, 4, 4, 4],
        ...
      }
    }

``num_timesteps`` is the SB3 global step counter at the moment the
episode terminated. ``wallclock_seconds`` is wall time since the
callback's ``_on_training_start`` hook.

Note that SB3 may run ``n_envs > 1`` (when wrapped in a VecEnv); the
callback iterates over each sub-env's ``infos[i]`` / ``dones[i]`` and
maintains one accumulator per sub-env. blue-team only uses a single
DummyVecEnv (one env), but the implementation is correct for the
multi-env case so the same callback survives ablation.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback

from src.utils.label_mapper import KillChainStage

logger = logging.getLogger(__name__)


# blue-team callback schema version. Bump whenever a field is added,
# removed, or renamed; aggregation.read_episodes_jsonl asserts on this
# so silent schema drift cannot poison the F3/F4 figures.
_SCHEMA_VERSION = "1.0"


@dataclass
class EpisodeRecord:
    """One row in ``episodes.jsonl``.

    Mirrors the schema above. We use a dataclass so the
    ``EpisodeJSONLCallback`` can build records type-safely and so
    aggregation can refer to a single canonical struct.
    """

    schema_version: str
    run_id: str
    algo: str
    seed: int
    episode_idx: int
    num_timesteps: int
    wallclock_seconds: float
    episode_reward: float
    episode_length: int
    compromised: bool
    mttc_steps: Optional[int]
    defender_deescalations: int
    final_stage: int
    final_stage_name: str
    end_outcome: str
    action_counts: List[int]
    action_counts_by_stage: Dict[str, List[int]]

    def to_jsonl(self) -> str:
        """Render the record as a single JSON line (no trailing newline)."""
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class _EpisodeAccumulator:
    """Per-sub-env running accumulator. Reset at every episode boundary."""

    cumulative_reward: float = 0.0
    length: int = 0
    action_counts: List[int] = field(default_factory=lambda: [0] * 5)
    action_counts_by_stage: Dict[int, List[int]] = field(
        default_factory=lambda: {s: [0] * 5 for s in range(5)}
    )

    def update(self, action: int, reward: float, decision_stage: int) -> None:
        """Record one (action, reward, decision-stage) triple.

        ``decision_stage`` is the ``info["attack_stage"]`` *before* the
        env executed the action — i.e., what the agent saw when it
        picked ``action``. We bookkeep this on the callback side
        because ``info`` already reflects the post-step stage; see
        ``EpisodeJSONLCallback._on_step``.
        """
        self.cumulative_reward += float(reward)
        self.length += 1
        self.action_counts[int(action)] += 1
        self.action_counts_by_stage[int(decision_stage)][int(action)] += 1

    def reset(self) -> None:
        self.cumulative_reward = 0.0
        self.length = 0
        self.action_counts = [0] * 5
        self.action_counts_by_stage = {s: [0] * 5 for s in range(5)}


class EpisodeJSONLCallback(BaseCallback):
    """SB3 callback that writes one JSON line per finished episode.

    Args:
        out_path: Where to write the JSONL file. Parent directory is
            created if missing.
        run_id: Stable identifier for this run, e.g. ``"ppo_seed_3"``.
            Echoed in every record so cross-run aggregation is
            unambiguous.
        algo: Algorithm name (``"dqn"``, ``"ppo"``, or ``"a2c"``).
        seed: The seed this run was started with.
        flush_every: Flush the file handle every N completed episodes.
            10 is a good balance between crash safety and I/O cost.
            Set to 1 in tests to make assertions on disk content
            deterministic.
        verbose: SB3 verbosity (passed through).

    Example:
        >>> cb = EpisodeJSONLCallback(
        ...     out_path="runs/ppo/seed_3/episodes.jsonl",
        ...     run_id="ppo_seed_3", algo="ppo", seed=3,
        ... )
        >>> model.learn(total_timesteps=500_000, callback=cb)
    """

    def __init__(
        self,
        out_path: Union[str, Path],
        run_id: str,
        algo: str,
        seed: int,
        flush_every: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._out_path = Path(out_path)
        self._run_id = str(run_id)
        self._algo = str(algo)
        self._seed = int(seed)
        if flush_every < 1:
            raise ValueError(f"flush_every must be >= 1, got {flush_every}")
        self._flush_every = int(flush_every)

        self._fh: Optional[Any] = None
        self._t_start: float = 0.0
        self._accumulators: Dict[int, _EpisodeAccumulator] = {}
        self._pre_step_stages: Dict[int, int] = {}
        self._episode_idx: int = 0
        self._unflushed: int = 0

    # ------------------------------------------------------------------ SB3 API

    def _on_training_start(self) -> None:
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._out_path.open("w", encoding="utf-8")
        self._t_start = time.time()
        # Each sub-env gets its own accumulator; populated lazily on
        # the first _on_step call once we know n_envs.
        self._accumulators = {}
        self._pre_step_stages = {}
        self._episode_idx = 0
        self._unflushed = 0
        if self.verbose > 0:
            logger.info(
                "EpisodeJSONLCallback opened %s for run_id=%s algo=%s seed=%d",
                self._out_path, self._run_id, self._algo, self._seed,
            )

    def _on_step(self) -> bool:  # noqa: D401 — SB3 API
        infos = self.locals.get("infos") or []
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        # On the very first step the env-side reset has already run,
        # so info["attack_stage"] == decision-time stage. After that,
        # info["attack_stage"] is the *post-step* stage; we therefore
        # cache the previous step's post-step stage as the next
        # decision-time stage. Bootstrap from the env's reset_info on
        # first observation.
        for env_idx, info in enumerate(infos):
            if env_idx not in self._accumulators:
                self._accumulators[env_idx] = _EpisodeAccumulator()
                # First decision was made on info["attack_stage"] of the
                # reset; but by the time this callback fires we're
                # post-step. Using info["attack_stage"] from the previous
                # iteration is the right thing — we cache it below.
                self._pre_step_stages[env_idx] = 0  # BENIGN at reset

            decision_stage = self._pre_step_stages[env_idx]
            action_i = self._extract_action(actions, env_idx)
            reward_i = self._extract_reward(rewards, env_idx)
            self._accumulators[env_idx].update(action_i, reward_i, decision_stage)

            done_i = self._extract_done(dones, env_idx)
            if done_i:
                self._emit_record(env_idx, info)
                self._accumulators[env_idx].reset()
                # After SB3's auto-reset the next step's decision stage
                # is BENIGN (the env starts fresh). Keep this in sync
                # with adversarial_env.reset() which sets
                # _current_attack_stage = 0.
                self._pre_step_stages[env_idx] = 0
            else:
                # info["attack_stage"] is the post-step stage; that
                # becomes the decision stage for the *next* step.
                self._pre_step_stages[env_idx] = int(info.get("attack_stage", 0))
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _extract_action(actions: Any, env_idx: int) -> int:
        """Pull the action that produced ``infos[env_idx]``.

        SB3 stores actions either as a list (on-policy: PPO/A2C) or as
        a numpy array (off-policy: DQN). Both are addressable by index;
        we coerce to int for the histogram.
        """
        try:
            return int(actions[env_idx])
        except (TypeError, IndexError):
            return int(actions)

    @staticmethod
    def _extract_reward(rewards: Any, env_idx: int) -> float:
        try:
            return float(rewards[env_idx])
        except (TypeError, IndexError):
            return float(rewards)

    @staticmethod
    def _extract_done(dones: Any, env_idx: int) -> bool:
        try:
            return bool(dones[env_idx])
        except (TypeError, IndexError):
            return bool(dones)

    def _emit_record(self, env_idx: int, info: Dict[str, Any]) -> None:
        """Build an EpisodeRecord from the current accumulator + info,
        write it as a JSONL line, and increment counters."""
        acc = self._accumulators[env_idx]
        # SB3 wraps each env in Monitor, which puts an "episode" subdict
        # in info on done=True. Prefer Monitor's totals when available
        # (they include the terminal-step reward, which our running
        # tally also includes). Fall back to the accumulator otherwise.
        monitor = info.get("episode") if isinstance(info, dict) else None
        ep_reward = (
            float(monitor["r"]) if monitor and "r" in monitor
            else float(acc.cumulative_reward)
        )
        ep_length = (
            int(monitor["l"]) if monitor and "l" in monitor
            else int(acc.length)
        )

        # When SB3 auto-resets a Monitor-wrapped env, ``info["terminal_observation"]``
        # is set and ``info["attack_stage"]`` already reflects the post-reset
        # stage (BENIGN). Pull the *terminal* stage from
        # ``terminal_info`` if present (DummyVecEnv adds it), otherwise
        # fall back to whatever the env reported.
        terminal_info = info.get("terminal_info") if isinstance(info, dict) else None
        if isinstance(terminal_info, dict):
            final_stage = int(terminal_info.get("attack_stage", 0))
            mttc_steps = terminal_info.get("mttc_steps")
            compromised = bool(terminal_info.get("compromised", False))
            defender_deescalations = int(terminal_info.get("defender_deescalations", 0))
            outcome = str(terminal_info.get("outcome", "unknown"))
        else:
            final_stage = int(info.get("attack_stage", 0))
            mttc_steps = info.get("mttc_steps")
            compromised = bool(info.get("compromised", False))
            defender_deescalations = int(info.get("defender_deescalations", 0))
            outcome = str(info.get("outcome", "unknown"))

        record = EpisodeRecord(
            schema_version=_SCHEMA_VERSION,
            run_id=self._run_id,
            algo=self._algo,
            seed=self._seed,
            episode_idx=self._episode_idx,
            num_timesteps=int(self.num_timesteps),
            wallclock_seconds=time.time() - self._t_start,
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
                str(s): list(counts)
                for s, counts in acc.action_counts_by_stage.items()
            },
        )
        assert self._fh is not None  # invariant after _on_training_start
        self._fh.write(record.to_jsonl())
        self._fh.write("\n")
        self._episode_idx += 1
        self._unflushed += 1
        if self._unflushed >= self._flush_every:
            self._fh.flush()
            self._unflushed = 0


class EvalToJSONLCallback(BaseCallback):
    """Periodic eval rollouts on a held-out env, written to ``eval.jsonl``.

    On every ``eval_freq``-th SB3 callback step, runs ``n_eval_episodes``
    deterministic rollouts on the supplied ``eval_env`` and writes one
    JSONL line per episode in the same v1.0 schema as
    :class:`EpisodeJSONLCallback`. Distinguishing fields:

    - ``run_id``: ``"<algo>_seed_<seed>_eval"``
    - ``num_timesteps`` set to the model's training-time global step at
      eval time (so all eval episodes for one eval-block share an x-axis
      value).

    Why a custom class and not SB3's :class:`EvalCallback`? We need
    environment-design telemetry (mttc_steps, compromised, defender_deescalations,
    per-stage action counts) on the eval side too, and SB3's eval log
    only stores reward + length + success-rate. Writing our own keeps
    aggregation symmetric.

    Args:
        eval_env: VecEnv (single-env DummyVecEnv) to roll out on.
        out_path: Where to write ``eval.jsonl``.
        run_id: Stable identifier (suffixed ``_eval`` by convention).
        algo, seed: Echoed in every record.
        eval_freq: Run an eval block every ``eval_freq`` callback ticks.
        n_eval_episodes: Episodes per eval block.
        deterministic: Pass through to ``model.predict``.
    """

    def __init__(
        self,
        eval_env: Any,
        out_path: Union[str, Path],
        run_id: str,
        algo: str,
        seed: int,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 30,
        deterministic: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._out_path = Path(out_path)
        self._run_id = str(run_id)
        self._algo = str(algo)
        self._seed = int(seed)
        self._eval_freq = int(eval_freq)
        self._n_eval_episodes = int(n_eval_episodes)
        self._deterministic = bool(deterministic)

        self._fh: Optional[Any] = None
        self._t_start: float = 0.0
        self._eval_block_idx: int = 0
        self._n_evals: int = 0  # cumulative eval episodes written

    def _on_training_start(self) -> None:
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._out_path.open("w", encoding="utf-8")
        self._t_start = time.time()
        self._eval_block_idx = 0
        self._n_evals = 0

    def _on_step(self) -> bool:  # noqa: D401 — SB3 API
        if self._eval_freq <= 0:
            return True
        if self.n_calls % self._eval_freq != 0:
            return True
        self._run_eval_block()
        self._eval_block_idx += 1
        return True

    def _on_training_end(self) -> None:
        # One last eval block at the end of training so the last 10 %
        # gate window has fresh post-convergence data.
        if self._fh is not None:
            try:
                self._run_eval_block(final=True)
            finally:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    def _run_eval_block(self, *, final: bool = False) -> None:
        """Roll ``n_eval_episodes`` complete episodes deterministically.

        Each episode is bookkept by a fresh accumulator so we can
        reuse the same per-stage histogram logic as
        :class:`EpisodeJSONLCallback`.
        """
        env = self._eval_env
        model = self.model
        for ep in range(self._n_eval_episodes):
            acc = _EpisodeAccumulator()
            decision_stage = 0  # BENIGN at reset, see env.reset()
            obs = env.reset()
            done = False
            while not done:
                action_arr, _ = model.predict(obs, deterministic=self._deterministic)
                obs, reward, dones, infos = env.step(action_arr)
                a = int(action_arr[0]) if hasattr(action_arr, "__len__") else int(action_arr)
                r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
                acc.update(a, r, decision_stage)
                done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
                # info["attack_stage"] may be the post-reset stage on
                # auto-reset; guard against that with terminal_info.
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                if not done:
                    decision_stage = int(info.get("attack_stage", 0))
                else:
                    # Terminal record: pull telemetry from
                    # info["terminal_info"] when SB3 packs it, otherwise
                    # from the live info before reset.
                    self._emit_eval_record(ep, acc, info)

    def _emit_eval_record(
        self, ep: int, acc: _EpisodeAccumulator, info: Dict[str, Any]
    ) -> None:
        terminal_info = info.get("terminal_info") if isinstance(info, dict) else None
        if isinstance(terminal_info, dict):
            src = terminal_info
        else:
            src = info if isinstance(info, dict) else {}

        final_stage = int(src.get("attack_stage", 0))
        mttc_steps = src.get("mttc_steps")
        compromised = bool(src.get("compromised", False))
        defender_deescalations = int(src.get("defender_deescalations", 0))
        outcome = str(src.get("outcome", "unknown"))

        record = EpisodeRecord(
            schema_version=_SCHEMA_VERSION,
            run_id=f"{self._run_id}_eval",
            algo=self._algo,
            seed=self._seed,
            episode_idx=self._n_evals,
            num_timesteps=int(self.num_timesteps),
            wallclock_seconds=time.time() - self._t_start,
            episode_reward=float(acc.cumulative_reward),
            episode_length=int(acc.length),
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
        assert self._fh is not None
        self._fh.write(record.to_jsonl())
        self._fh.write("\n")
        self._fh.flush()  # eval blocks are sparse, flush eagerly
        self._n_evals += 1


__all__ = ["EpisodeJSONLCallback", "EpisodeRecord", "EvalToJSONLCallback"]
