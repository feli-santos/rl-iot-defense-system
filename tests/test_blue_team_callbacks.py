"""Tests for the Blue-Team Training EpisodeJSONLCallback (3.2.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.blue_team.callbacks import _SCHEMA_VERSION, EpisodeJSONLCallback, EpisodeRecord

# --------------------------------------------------------------------- harness


class _FakeModel:
    """Minimal duck-typed model so BaseCallback.init_callback works."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.verbose = 0


class _CallbackHarness:
    """Drive an EpisodeJSONLCallback by hand-feeding ``locals``.

    The callback's only contact with SB3 is via ``self.locals`` (via
    ``BaseCallback.locals``) and ``self.num_timesteps``. We can poke
    both here without spinning up an SB3 model + env, which makes the
    tests fast and deterministic.
    """

    def __init__(self, callback: EpisodeJSONLCallback) -> None:
        self.cb = callback
        self.cb.model = _FakeModel()  # type: ignore[assignment]
        self.cb.locals = {}  # type: ignore[assignment]
        self.cb._on_training_start()

    def step(
        self,
        *,
        actions: list[int],
        rewards: list[float],
        dones: list[bool],
        infos: list[dict[str, Any]],
    ) -> None:
        # Increment SB3 timestep counter the way SB3 itself does.
        self.cb.model.num_timesteps += 1
        self.cb.locals = {  # type: ignore[assignment]
            "actions": np.asarray(actions),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "dones": np.asarray(dones, dtype=bool),
            "infos": list(infos),
        }
        # BaseCallback.on_step bumps n_calls then calls _on_step.
        self.cb.n_calls += 1
        self.cb.num_timesteps = self.cb.model.num_timesteps  # type: ignore[assignment]
        self.cb._on_step()

    def end(self) -> None:
        self.cb._on_training_end()


def _run_episode(
    h: _CallbackHarness,
    *,
    actions: list[int],
    rewards: list[float],
    stages: list[int],
    final_info_extra: dict[str, Any] | None = None,
) -> None:
    """Drive a single complete episode through the callback.

    ``stages[i]`` is the **post-step** attack stage that the env would
    have written into ``info["attack_stage"]`` after applying
    ``actions[i]``. The first decision is always at BENIGN (stage 0).
    """
    assert len(actions) == len(rewards) == len(stages)
    n = len(actions)
    for i, (a, r, s) in enumerate(zip(actions, rewards, stages)):
        is_last = i == n - 1
        info: dict[str, Any] = {"attack_stage": s}
        if is_last:
            info["episode"] = {"r": float(np.sum(rewards)), "l": n}
            info.update(
                {
                    "compromised": False,
                    "mttc_steps": None,
                    "defender_deescalations": 0,
                    "outcome": "ongoing",
                }
            )
            if final_info_extra:
                info.update(final_info_extra)
        h.step(actions=[a], rewards=[r], dones=[is_last], infos=[info])


# ------------------------------------------------------------------ tests


class TestEpisodeRecordSchema:
    def test_to_jsonl_round_trips(self) -> None:
        rec = EpisodeRecord(
            schema_version="1.0",
            run_id="ppo_seed_3",
            algo="ppo",
            seed=3,
            episode_idx=12,
            num_timesteps=1234,
            wallclock_seconds=4.5,
            episode_reward=42.0,
            episode_length=20,
            compromised=False,
            mttc_steps=None,
            defender_deescalations=0,
            final_stage=0,
            final_stage_name="BENIGN",
            end_outcome="ongoing",
            action_counts=[5, 5, 5, 3, 2],
            action_counts_by_stage={
                "0": [5, 0, 0, 0, 0],
                "1": [0, 5, 0, 0, 0],
                "2": [0, 0, 5, 0, 0],
                "3": [0, 0, 0, 3, 0],
                "4": [0, 0, 0, 0, 2],
            },
        )
        loaded = json.loads(rec.to_jsonl())
        assert loaded["schema_version"] == "1.0"
        assert loaded["episode_reward"] == 42.0
        assert sum(loaded["action_counts"]) == 20


class TestEpisodeJSONLCallback:
    @pytest.fixture
    def callback(self, tmp_path: Path) -> EpisodeJSONLCallback:
        return EpisodeJSONLCallback(
            out_path=tmp_path / "episodes.jsonl",
            run_id="ppo_seed_0",
            algo="ppo",
            seed=0,
            flush_every=1,
        )

    def test_one_record_per_episode(self, callback: EpisodeJSONLCallback, tmp_path: Path) -> None:
        h = _CallbackHarness(callback)
        # Episode 1: 3 steps.
        _run_episode(
            h,
            actions=[0, 1, 2],
            rewards=[1.0, 1.0, 1.0],
            stages=[0, 1, 2],
        )
        # Episode 2: 2 steps.
        _run_episode(
            h,
            actions=[3, 4],
            rewards=[2.0, 2.0],
            stages=[3, 4],
        )
        h.end()

        lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
        assert len(lines) == 2
        ep1 = json.loads(lines[0])
        ep2 = json.loads(lines[1])
        assert ep1["episode_idx"] == 0
        assert ep2["episode_idx"] == 1
        assert ep1["episode_length"] == 3
        assert ep2["episode_length"] == 2

    def test_action_counts_sum_to_episode_length(
        self, callback: EpisodeJSONLCallback, tmp_path: Path
    ) -> None:
        h = _CallbackHarness(callback)
        actions = [0, 0, 1, 2, 3, 3, 4, 4, 4, 4]
        _run_episode(
            h,
            actions=actions,
            rewards=[1.0] * 10,
            stages=[0] * 10,
        )
        h.end()
        rec = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
        assert sum(rec["action_counts"]) == rec["episode_length"] == 10
        assert rec["action_counts"] == [2, 1, 1, 2, 4]

    def test_per_stage_action_counts_track_decision_stage(
        self, callback: EpisodeJSONLCallback, tmp_path: Path
    ) -> None:
        h = _CallbackHarness(callback)
        # Decision stages, in order: BENIGN, BENIGN, RECON, ACCESS.
        # After step 1, env reports stage=BENIGN; after step 2, env
        # reports stage=RECON; etc. So info["attack_stage"] sequence
        # is [BENIGN, RECON, ACCESS, ACCESS] for 4 actions.
        # Decision-time stages are therefore [0, 0, 1, 2].
        _run_episode(
            h,
            actions=[0, 0, 1, 2],
            rewards=[0.0, 0.0, 0.0, 0.0],
            stages=[0, 1, 2, 2],
        )
        h.end()
        rec = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
        per_stage = rec["action_counts_by_stage"]
        # 2 decisions at stage 0 -> action 0 (twice).
        assert per_stage["0"] == [2, 0, 0, 0, 0]
        # 1 decision at stage 1 -> action 1.
        assert per_stage["1"] == [0, 1, 0, 0, 0]
        # 1 decision at stage 2 -> action 2.
        assert per_stage["2"] == [0, 0, 1, 0, 0]

    def test_terminal_telemetry_pulled_from_info(
        self, callback: EpisodeJSONLCallback, tmp_path: Path
    ) -> None:
        h = _CallbackHarness(callback)
        # Episode that compromises: env reports IMPACT (stage=4) on
        # the final step plus the Adversarial Environment MTTC fields.
        _run_episode(
            h,
            actions=[0, 0, 0],
            rewards=[0.0, 0.0, -350.0],
            stages=[1, 3, 4],
            final_info_extra={
                "compromised": True,
                "mttc_steps": 2,
                "defender_deescalations": 0,
                "outcome": "impact_missed",
            },
        )
        h.end()
        rec = json.loads((tmp_path / "episodes.jsonl").read_text().splitlines()[0])
        assert rec["compromised"] is True
        assert rec["mttc_steps"] == 2
        assert rec["final_stage"] == 4
        assert rec["final_stage_name"] == "IMPACT"
        assert rec["end_outcome"] == "impact_missed"

    def test_flush_every_buffers_correctly(self, tmp_path: Path) -> None:
        cb = EpisodeJSONLCallback(
            out_path=tmp_path / "episodes.jsonl",
            run_id="dqn_seed_0",
            algo="dqn",
            seed=0,
            flush_every=3,
        )
        h = _CallbackHarness(cb)
        for _ in range(5):
            _run_episode(
                h,
                actions=[0, 1],
                rewards=[1.0, 1.0],
                stages=[0, 1],
            )
        h.end()
        # After end (which flushes + closes) all 5 records present.
        lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
        assert len(lines) == 5

    def test_run_id_seed_algo_echoed_in_every_record(
        self, callback: EpisodeJSONLCallback, tmp_path: Path
    ) -> None:
        h = _CallbackHarness(callback)
        for _ in range(3):
            _run_episode(
                h,
                actions=[0, 1],
                rewards=[1.0, 1.0],
                stages=[0, 1],
            )
        h.end()
        lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
        for line in lines:
            rec = json.loads(line)
            assert rec["run_id"] == "ppo_seed_0"
            assert rec["algo"] == "ppo"
            assert rec["seed"] == 0
            assert rec["schema_version"] == _SCHEMA_VERSION

    def test_flush_every_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            EpisodeJSONLCallback(
                out_path=tmp_path / "x.jsonl",
                run_id="x",
                algo="ppo",
                seed=0,
                flush_every=0,
            )
