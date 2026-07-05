"""Benchmark eval-manifest provenance regression guard.

The benchmark ``eval_manifest.json`` ``eval_env`` block is built from the
*actual* eval spec used for the rollouts. Previously it was hand-rolled from a
field-omitting eval_env block and enumerated only seven fields, so
``evasion_prob`` / ``impact_is_terminal`` / ``proximity_coupled`` were absent
from the manifest even when set — the metadata gap. These tests lock the
fix: the manifest must faithfully record the full field set the eval actually
ran with.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scripts.benchmark.run_test_eval import _assert_train_eval_contract, _eval_env_spec
from src.blue_team.run_config import BlueTeamRunConfig, EnvConfigSerializable


class TestEvalEnvSpecProvenance:
    def test_impact_is_terminal_recorded_false(self) -> None:
        """The benchmark eval contract pins impact_is_terminal=False."""
        d = dataclasses.asdict(_eval_env_spec())
        assert d["impact_is_terminal"] is False

    def test_manifest_fields_are_complete(self) -> None:
        """asdict() must expose the full field set (incl. the metadata-gap trio).

        Guards against a future regression that re-introduces a hand-rolled,
        field-omitting eval_env block.
        """
        d = dataclasses.asdict(_eval_env_spec())
        for field in ("evasion_prob", "impact_is_terminal", "proximity_coupled"):
            assert field in d, f"manifest eval_env must include {field!r}"

    def test_default_reward_mode_is_outcome(self) -> None:
        """The benchmark eval contract defaults to the deployment reward mode."""
        d = dataclasses.asdict(_eval_env_spec())
        assert d["reward_mode"] == "outcome"

    def test_default_is_proximity_coupled(self) -> None:
        """The benchmark eval contract defaults to proximity-coupled escalation."""
        d = dataclasses.asdict(_eval_env_spec())
        assert d["proximity_coupled"] is True


class TestCrossScriptContractParity:
    """The F5 eval contract must match each checkpoint's training manifest."""

    def _write_training_manifest(
        self, run_root: Path, *, reward_mode, proximity_coupled=True
    ) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        cfg = BlueTeamRunConfig(
            algo="ppo",
            seed=0,
            env=EnvConfigSerializable(
                split="train",
                reward_mode=reward_mode,
                proximity_coupled=proximity_coupled,
                impact_is_terminal=False,
            ),
            eval_env=EnvConfigSerializable(
                split="val_balanced",
                reward_mode=reward_mode,
                proximity_coupled=proximity_coupled,
                impact_is_terminal=False,
            ),
        )
        (run_root / "run_manifest.json").write_text(json.dumps(cfg.to_dict()))

    def test_matching_contract_passes(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(run_root, reward_mode="outcome")
        _assert_train_eval_contract(
            _eval_env_spec(reward_mode="outcome"),
            run_root,
            algo="ppo",
            seed=0,
        )

    def test_proximity_mismatch_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(run_root, reward_mode="outcome", proximity_coupled=False)
        with pytest.raises(ValueError, match="different MDP"):
            _assert_train_eval_contract(
                _eval_env_spec(reward_mode="outcome", proximity_coupled=True),
                run_root,
                algo="ppo",
                seed=0,
            )

    def test_reward_mode_mismatch_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(run_root, reward_mode="coupled")
        with pytest.raises(ValueError, match="different MDP"):
            _assert_train_eval_contract(
                _eval_env_spec(reward_mode="outcome"),
                run_root,
                algo="ppo",
                seed=0,
            )

    def test_missing_manifest_warns_not_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        run_root.mkdir(parents=True, exist_ok=True)
        # No run_manifest.json => warn, do not fail (pre-manifest checkpoint).
        _assert_train_eval_contract(
            _eval_env_spec(reward_mode="outcome"),
            run_root,
            algo="ppo",
            seed=0,
        )
