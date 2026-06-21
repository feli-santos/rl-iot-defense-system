"""Tests for src.benchmark.eval_runner.run_policy (PLAN §3.3).

The harness here is a tiny stub VecEnv that emits scripted episodes —
real Adversarial Environment rollouts would require the full Dataset/Split artefact
chain, which the tests file is forbidden from depending on. The stub
is structurally identical to SB3's DummyVecEnv-with-Monitor contract:

- ``reset()`` returns batched obs of shape ``(1, obs_dim)``.
- ``step(action_arr)`` returns ``(obs, reward, dones, infos)`` with
  every entry batched along axis 0. On the terminal step ``dones[0]``
  is ``True`` and ``infos[0]`` carries
  ``{"episode": {"r", "l"}, "compromised", "mttc_steps",
    "defender_deescalations", "outcome", "attack_stage"}`` exactly as
    SB3's Monitor + the Adversarial Environment would.
- After autoreset, ``obs`` is the post-reset obs of the next episode
  (caller-transparent).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.benchmark.baseline_policies import always_observe, recommended_action_policy
from src.benchmark.eval_runner import (
    LatencyRecord,
    _algo_field_from_run_id,
    _seed_field_from_run_id,
    run_policy,
)
from src.blue_team.aggregation import read_episodes_jsonl

# ----------------------------------------------------------------- stub env


class _StubEpisode:
    """One scripted episode: a list of (post_step_attack_stage, reward,
    final_outcome_kwargs) triples. The last triple is the terminal step.
    """

    def __init__(
        self,
        *,
        stages: list[int],
        rewards: list[float],
        compromised: bool,
        mttc_steps: int | None,
        outcome: str,
        defender_deescalations: int = 0,
    ) -> None:
        assert len(stages) == len(rewards)
        self.stages = stages
        self.rewards = rewards
        self.compromised = compromised
        self.mttc_steps = mttc_steps
        self.outcome = outcome
        self.defender_deescalations = defender_deescalations


class _StubVecEnv:
    """Minimal SB3-shaped VecEnv playing back a list of ``_StubEpisode``."""

    def __init__(self, episodes: list[_StubEpisode], obs_dim: int = 8) -> None:
        self._episodes = list(episodes)
        self._obs_dim = obs_dim
        self._ep_idx = 0
        self._step_in_ep = 0
        self.actions_received: list[int] = []

    # ----------------- VecEnv API ----------------- #

    def reset(self) -> np.ndarray:
        # the Held-Out Benchmark's run_policy calls reset() at episode 0 only;
        # subsequent episodes start from the autoreset obs returned by
        # the previous step(). Ensure idempotency: if called mid-rollout
        # we silently restart from the current cursor.
        return np.zeros((1, self._obs_dim), dtype=np.float32)

    def step(self, action_arr: np.ndarray):
        self.actions_received.append(int(np.asarray(action_arr).reshape(-1)[0]))
        ep = self._episodes[self._ep_idx]
        i = self._step_in_ep
        post_stage = ep.stages[i]
        reward = ep.rewards[i]
        terminal = i == len(ep.stages) - 1

        if terminal:
            # Pre-reset (terminal) info — what the env "would have"
            # written before the autoreset bumps attack_stage back to 0.
            terminal_info: dict[str, Any] = {
                "attack_stage": int(post_stage),
                "compromised": ep.compromised,
                "mttc_steps": ep.mttc_steps,
                "defender_deescalations": ep.defender_deescalations,
                "outcome": ep.outcome,
            }
            # SB3 Monitor packs total reward + length under "episode".
            info: dict[str, Any] = {
                "attack_stage": 0,  # post-reset
                "compromised": False,
                "mttc_steps": None,
                "defender_deescalations": 0,
                "outcome": "ongoing",
                "episode": {"r": float(sum(ep.rewards)), "l": len(ep.stages)},
                "terminal_info": terminal_info,
            }
            self._ep_idx = min(self._ep_idx + 1, len(self._episodes) - 1)
            self._step_in_ep = 0
        else:
            info = {
                "attack_stage": int(post_stage),
                "compromised": False,
                "mttc_steps": None,
                "defender_deescalations": 0,
                "outcome": "ongoing",
            }
            self._step_in_ep += 1

        obs = np.full((1, self._obs_dim), float(post_stage), dtype=np.float32)
        rewards = np.asarray([reward], dtype=np.float64)
        dones = np.asarray([terminal], dtype=bool)
        infos = [info]
        return obs, rewards, dones, infos


# ------------------------------------------------------------ helper episodes


def _two_episodes() -> list[_StubEpisode]:
    return [
        _StubEpisode(
            stages=[1, 2, 3],
            rewards=[1.0, 2.0, 3.0],
            compromised=False,
            mttc_steps=None,
            outcome="ongoing",
            defender_deescalations=0,
        ),
        _StubEpisode(
            stages=[1, 2, 3, 4],
            rewards=[0.5, 1.5, 2.5, -10.0],
            compromised=True,
            mttc_steps=3,
            outcome="impact_missed",
            defender_deescalations=1,
        ),
    ]


# ----------------------------------------------------------------- tests


class TestRunPolicy:
    def test_round_trip_jsonl_loads_via_aggregation(
        self,
        tmp_path: Path,
    ) -> None:
        env = _StubVecEnv(_two_episodes())
        out = tmp_path / "eval.jsonl"
        stats = run_policy(
            always_observe,
            env,
            n_episodes=2,
            jsonl_path=out,
            run_id="random_seed_2_test",
            policy_name="random",
        )
        assert stats["n_episodes_written"] == 2
        assert stats["n_steps_total"] == 3 + 4
        assert stats["n_latency_rows"] == 0
        # The on-disk JSONL must round-trip through Blue-Team Training's
        # aggregation reader without complaint — proves schema-v1.0
        # compliance (G6.7 / D6.4).
        records = read_episodes_jsonl(out)
        assert len(records) == 2
        # First episode telemetry.
        r0 = records[0]
        assert r0["schema_version"] == "1.0"
        assert r0["run_id"] == "random_seed_2_test"
        assert r0["compromised"] is False
        assert r0["episode_length"] == 3
        assert sum(r0["action_counts"]) == 3
        assert r0["episode_reward"] == pytest.approx(6.0)
        # Second episode telemetry.
        r1 = records[1]
        assert r1["compromised"] is True
        assert r1["mttc_steps"] == 3
        assert r1["end_outcome"] == "impact_missed"
        assert r1["episode_length"] == 4
        assert r1["defender_deescalations"] == 1

    def test_latency_sidecar_emits_one_row_per_step(
        self,
        tmp_path: Path,
    ) -> None:
        env = _StubVecEnv(_two_episodes())
        out = tmp_path / "eval.jsonl"
        lat = tmp_path / "latency.jsonl"
        stats = run_policy(
            always_observe,
            env,
            n_episodes=2,
            jsonl_path=out,
            run_id="always_observe_seed_0_test",
            policy_name="always_observe",
            latency_path=lat,
        )
        assert stats["n_latency_rows"] == 3 + 4
        rows = [json.loads(ln) for ln in lat.read_text().splitlines() if ln.strip()]
        assert len(rows) == 7
        for r in rows:
            assert r["schema_version"] == "1.0"
            assert r["policy_name"] == "always_observe"
            assert r["run_id"] == "always_observe_seed_0_test"
            assert r["duration_ns"] >= 0
            assert "step_idx" in r and "episode_idx" in r
        # In-episode step_idx resets per episode.
        ep0_rows = [r for r in rows if r["episode_idx"] == 0]
        ep1_rows = [r for r in rows if r["episode_idx"] == 1]
        assert [r["step_idx"] for r in ep0_rows] == [0, 1, 2]
        assert [r["step_idx"] for r in ep1_rows] == [0, 1, 2, 3]

    def test_recommended_action_policy_consumes_info_seed(
        self,
        tmp_path: Path,
    ) -> None:
        # The eval_runner reconstructs `info["recommended_action"]` from
        # the (decision-time) attack stage, not from the env's post-step
        # info. recommended_action_policy depends on this; we therefore
        # assert that running it produces a meaningful action histogram.
        env = _StubVecEnv(_two_episodes())
        out = tmp_path / "eval.jsonl"
        run_policy(
            recommended_action_policy,
            env,
            n_episodes=2,
            jsonl_path=out,
            run_id="recommended_seed_0_test",
            policy_name="recommended_action",
        )
        records = read_episodes_jsonl(out)
        # First decision is at BENIGN (action = 0); the next decisions
        # see the previous post-step attack_stage. Episode 0 stages
        # post-step are [1,2,3], so decision stages are [0, 1, 2] and
        # the recommended actions are [0, 1, 2]. Episode 1 stages are
        # [1,2,3,4], so decisions are [0, 1, 2, 3] and recommended
        # actions are [0, 1, 2, 3]. The resulting action_counts:
        #   ep0: [1, 1, 1, 0, 0]
        #   ep1: [1, 1, 1, 1, 0]
        assert records[0]["action_counts"] == [1, 1, 1, 0, 0]
        assert records[1]["action_counts"] == [1, 1, 1, 1, 0]

    def test_action_counts_by_stage_uses_decision_stage(
        self,
        tmp_path: Path,
    ) -> None:
        env = _StubVecEnv(_two_episodes())
        out = tmp_path / "eval.jsonl"
        run_policy(
            recommended_action_policy,
            env,
            n_episodes=1,
            jsonl_path=out,
            run_id="recommended_seed_0_test",
            policy_name="recommended_action",
        )
        records = read_episodes_jsonl(out)
        # ep0 decision stages are [0,1,2], actions [0,1,2]; histogram by
        # stage: 0→[1,0,0,0,0], 1→[0,1,0,0,0], 2→[0,0,1,0,0], 3,4 zeros.
        cm = records[0]["action_counts_by_stage"]
        assert cm["0"] == [1, 0, 0, 0, 0]
        assert cm["1"] == [0, 1, 0, 0, 0]
        assert cm["2"] == [0, 0, 1, 0, 0]
        assert cm["3"] == [0, 0, 0, 0, 0]
        assert cm["4"] == [0, 0, 0, 0, 0]


class TestLatencyRecord:
    def test_to_jsonl_round_trips(self) -> None:
        rec = LatencyRecord(
            schema_version="1.0",
            run_id="ppo_seed_0_test",
            policy_name="ppo",
            episode_idx=4,
            step_idx=12,
            duration_ns=987654,
        )
        loaded = json.loads(rec.to_jsonl())
        assert loaded == {
            "schema_version": "1.0",
            "run_id": "ppo_seed_0_test",
            "policy_name": "ppo",
            "episode_idx": 4,
            "step_idx": 12,
            "duration_ns": 987654,
        }


class TestRunIdParsers:
    @pytest.mark.parametrize(
        "rid, algo, seed",
        [
            ("ppo_seed_0_test", "ppo", 0),
            ("dqn_seed_3", "dqn", 3),
            ("random_seed_12", "random", 12),
            ("rf_acting_seed_0_test", "rf_acting", 0),
            ("always_observe_seed_4_test", "always_observe", 4),
            ("noseed_run", "noseed_run", 0),
        ],
    )
    def test_extraction(self, rid: str, algo: str, seed: int) -> None:
        assert _algo_field_from_run_id(rid) == algo
        assert _seed_field_from_run_id(rid) == seed
