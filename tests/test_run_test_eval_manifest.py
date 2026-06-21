"""Benchmark eval-manifest provenance (caveat C10 regression guard).

The benchmark ``eval_manifest.json`` ``eval_env`` block is built from the
*actual* eval spec used for the rollouts. Previously it was hand-rolled from a
bare ``_eval_env_spec()`` (no ``attacker_budget`` argument) and enumerated only
seven fields, so ``attacker_budget`` / ``evasion_prob`` / ``impact_is_terminal``
were absent from the manifest even when a finite budget was applied — the C10
metadata gap. These tests lock the fix: the manifest must faithfully record the
finite attacker budget (and the full field set) that the eval actually ran with.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scripts.benchmark.run_test_eval import (
    _assert_train_eval_contract,
    _eval_env_spec,
)
from src.blue_team.run_config import BlueTeamRunConfig, EnvConfigSerializable


class TestEvalEnvSpecProvenance:
    def test_finite_budget_is_recorded(self) -> None:
        """A finite ``--attacker-budget`` survives into the serialised spec."""
        d = dataclasses.asdict(_eval_env_spec(40))
        assert d["attacker_budget"] == 40, (
            "C10: the benchmark eval spec must record the finite attacker "
            "budget it was built with, not a default None."
        )

    def test_default_budget_is_none(self) -> None:
        """No budget arg => unbounded control cell (attacker_budget=None)."""
        d = dataclasses.asdict(_eval_env_spec())
        assert d["attacker_budget"] is None

    def test_impact_is_terminal_recorded_false(self) -> None:
        """The benchmark eval contract pins impact_is_terminal=False."""
        d = dataclasses.asdict(_eval_env_spec(40))
        assert d["impact_is_terminal"] is False

    def test_manifest_fields_are_complete(self) -> None:
        """asdict() must expose the full field set (incl. the C10 trio).

        Guards against a future regression that re-introduces a hand-rolled,
        field-omitting eval_env block.
        """
        d = dataclasses.asdict(_eval_env_spec(40))
        for field in ("attacker_budget", "evasion_prob", "impact_is_terminal"):
            assert field in d, f"C10: manifest eval_env must include {field!r}"

    def test_default_reward_mode_is_outcome(self) -> None:
        """The benchmark eval contract defaults to the deployment reward mode."""
        d = dataclasses.asdict(_eval_env_spec(40))
        assert d["reward_mode"] == "outcome"


class TestCrossScriptContractParity:
    """The F5 eval contract must match each checkpoint's training manifest."""

    def _write_training_manifest(
        self, run_root: Path, *, attacker_budget, reward_mode
    ) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        cfg = BlueTeamRunConfig(
            algo="ppo",
            seed=0,
            env=EnvConfigSerializable(
                split="train",
                attacker_budget=attacker_budget,
                reward_mode=reward_mode,
                impact_is_terminal=False,
            ),
            eval_env=EnvConfigSerializable(
                split="val_balanced",
                attacker_budget=attacker_budget,
                reward_mode=reward_mode,
                impact_is_terminal=False,
            ),
        )
        (run_root / "run_manifest.json").write_text(json.dumps(cfg.to_dict()))

    def test_matching_contract_passes(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(
            run_root, attacker_budget=40, reward_mode="outcome"
        )
        _assert_train_eval_contract(
            _eval_env_spec(40, reward_mode="outcome"),
            run_root,
            algo="ppo",
            seed=0,
        )

    def test_budget_mismatch_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(
            run_root, attacker_budget=40, reward_mode="outcome"
        )
        with pytest.raises(ValueError, match="different MDP"):
            _assert_train_eval_contract(
                _eval_env_spec(None, reward_mode="outcome"),
                run_root,
                algo="ppo",
                seed=0,
            )

    def test_reward_mode_mismatch_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        self._write_training_manifest(
            run_root, attacker_budget=40, reward_mode="coupled"
        )
        with pytest.raises(ValueError, match="different MDP"):
            _assert_train_eval_contract(
                _eval_env_spec(40, reward_mode="outcome"),
                run_root,
                algo="ppo",
                seed=0,
            )

    def test_missing_manifest_warns_not_raises(self, tmp_path: Path) -> None:
        run_root = tmp_path / "ppo" / "seed_0"
        run_root.mkdir(parents=True, exist_ok=True)
        # No run_manifest.json => warn, do not fail (pre-manifest checkpoint).
        _assert_train_eval_contract(
            _eval_env_spec(40, reward_mode="outcome"),
            run_root,
            algo="ppo",
            seed=0,
        )
