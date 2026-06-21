"""Tests for src/blue_team/run_config.py (3.2.3 sub-test)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.blue_team.run_config import (
    _SCHEMA_VERSION,
    BlueTeamRunConfig,
    EnvConfigSerializable,
)


class TestBlueTeamRunConfig:
    def test_run_id_format(self) -> None:
        cfg = BlueTeamRunConfig(algo="ppo", seed=3)
        assert cfg.run_id == "ppo_seed_3"

    def test_default_eval_split_is_val_balanced(self) -> None:
        cfg = BlueTeamRunConfig(algo="dqn", seed=0)
        assert cfg.eval_env.split == "val_balanced"
        assert cfg.env.split == "train"

    def test_to_dict_round_trip(self) -> None:
        cfg = BlueTeamRunConfig(
            algo="a2c",
            seed=2,
            total_timesteps=100_000,
            algo_hparams={"lr": 7e-4, "ent_coef": 0.0},
            notes="smoke",
        )
        d = cfg.to_dict()
        assert d["schema_version"] == _SCHEMA_VERSION
        assert d["algo"] == "a2c"
        assert d["seed"] == 2
        assert d["env"]["split"] == "train"
        assert d["eval_env"]["split"] == "val_balanced"
        assert d["algo_hparams"]["lr"] == 7e-4
        assert d["paths"]["dataset"] == "data/processed/ciciot2023"
        assert "completed_at" not in d  # not yet completed

    def test_to_dict_completed_adds_telemetry(self) -> None:
        cfg = BlueTeamRunConfig(algo="ppo", seed=1)
        d = cfg.to_dict(
            completed=True,
            wallclock_seconds=42.0,
            n_episodes_train=100,
            n_episodes_eval=30,
            git_sha="deadbeef",
        )
        assert d["wallclock_seconds"] == 42.0
        assert d["n_episodes_train"] == 100
        assert d["git_sha"] == "deadbeef"
        assert "completed_at" in d
        assert d["completed_at"].endswith("Z")

    def test_write_manifest_round_trip(self, tmp_path: Path) -> None:
        cfg = BlueTeamRunConfig(
            algo="ppo",
            seed=1,
            total_timesteps=10_000,
            env=EnvConfigSerializable(split="train", max_steps=50),
            eval_env=EnvConfigSerializable(split="val_balanced", max_steps=50),
            algo_hparams={"lr": 3e-4},
        )
        path = tmp_path / "run_manifest.json"
        cfg.write_manifest(path, wallclock_seconds=1.0, n_episodes_train=5, n_episodes_eval=2)
        loaded = BlueTeamRunConfig.from_manifest(path)
        assert loaded.algo == "ppo"
        assert loaded.seed == 1
        assert loaded.total_timesteps == 10_000
        assert loaded.env.max_steps == 50
        assert loaded.eval_env.split == "val_balanced"
        assert loaded.algo_hparams == {"lr": 3e-4}

    def test_from_dict_rejects_bad_schema(self) -> None:
        bad = {
            "schema_version": "0.5",
            "algo": "ppo",
            "seed": 0,
        }
        with pytest.raises(ValueError, match="schema_version"):
            BlueTeamRunConfig.from_dict(bad)

    def test_atomic_write_does_not_leave_tmp_file(self, tmp_path: Path) -> None:
        cfg = BlueTeamRunConfig(algo="ppo", seed=0)
        path = tmp_path / "manifest.json"
        cfg.write_manifest(path)
        # Only the canonical manifest should exist.
        assert path.exists()
        assert not path.with_suffix(".json.tmp").exists()
        # And it should parse as valid JSON.
        json.loads(path.read_text())


class TestTrainEvalParity:
    """The eval number must measure the same MDP the agent trained on."""

    def test_default_config_passes_parity(self) -> None:
        # Default eval split differs from train split but every contract field
        # matches — construction must succeed.
        BlueTeamRunConfig(algo="ppo", seed=0)

    @pytest.mark.parametrize(
        "field,train_val,eval_val",
        [
            ("attacker_budget", 40, None),
            ("impact_is_terminal", False, True),
            ("reward_mode", "outcome", "coupled"),
            ("p_down", 0.90, 0.80),
            ("action_cost_scale", 1.0, 2.0),
            ("max_steps", 100, 50),
            # Partial-observability redesign contract fields: a train/eval
            # mismatch on any of these means the agent is evaluated on a
            # different MDP than it trained on.
            ("aliasing_rate", 0.4, 0.0),
            ("session_coherent", True, False),
            ("no_post_transition_leak", True, False),
            ("proximity_coupled", True, False),
            ("proximity_min_escalation", 0.4, 0.6),
        ],
    )
    def test_contract_mismatch_raises(self, field, train_val, eval_val) -> None:
        with pytest.raises(ValueError, match="contract mismatch"):
            BlueTeamRunConfig(
                algo="ppo",
                seed=0,
                env=EnvConfigSerializable(split="train", **{field: train_val}),
                eval_env=EnvConfigSerializable(
                    split="val_balanced", **{field: eval_val}
                ),
            )

    def test_reward_mode_alias_difference_does_not_trip_parity(self) -> None:
        # "proportional" and "coupled" are the same contract after
        # normalisation, so this must NOT raise.
        BlueTeamRunConfig(
            algo="ppo",
            seed=0,
            env=EnvConfigSerializable(split="train", reward_mode="proportional"),
            eval_env=EnvConfigSerializable(
                split="val_balanced", reward_mode="coupled"
            ),
        )

    def test_matching_budget_and_mode_passes(self) -> None:
        BlueTeamRunConfig(
            algo="ppo",
            seed=0,
            env=EnvConfigSerializable(
                split="train", attacker_budget=40, reward_mode="outcome"
            ),
            eval_env=EnvConfigSerializable(
                split="val_balanced", attacker_budget=40, reward_mode="outcome"
            ),
        )
