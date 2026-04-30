"""Phase-7 ablation: ``--reward-overrides`` + ``--p-defender-deescalation``
+ ``--impact-is-terminal`` plumbing in ``scripts.blue_team.train_agent``.

PLAN §3.1.2 / D7.3. Pins the per-field override mechanism that the F9
reward-component sweep uses to fan out one ``train_agent.py`` invocation
per cell. Default behaviour (no overrides) is byte-for-byte identical to
Phase 5; this test file is what enforces that invariant going forward.

Synthetic-only — no real-data dependency. Tests only the CLI parsing +
override plumbing into ``EnvConfigSerializable``; the actual training
smoke is already covered by ``tests/test_blue_team_train_agent.py`` and
the Phase-3 frozen tests cover the env-side semantics.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from scripts.blue_team.train_agent import (
    _apply_env_overrides,
    _build_argparser,
    build_run_config,
)
from src.blue_team import EnvConfigSerializable


# ---------------------------------------------------------------------------
# _apply_env_overrides — direct unit tests
# ---------------------------------------------------------------------------


class TestApplyEnvOverrides:
    """Unit tests for the override-merge helper."""

    def test_no_overrides_returns_equal_spec(self) -> None:
        """No flags → output spec equals input spec field-by-field."""
        spec = EnvConfigSerializable()
        out = _apply_env_overrides(spec)
        assert dataclasses.asdict(out) == dataclasses.asdict(spec), (
            "Default behaviour (no overrides) must be byte-for-byte "
            "identical to the input spec — this is what guarantees "
            "Phase-7 retraining without --reward-overrides matches "
            "Phase-5 behaviour."
        )

    def test_reward_overrides_dict_applied(self) -> None:
        """A reward-overrides dict is applied to the named fields only."""
        spec = EnvConfigSerializable()
        out = _apply_env_overrides(
            spec,
            reward_overrides={"defense_success_bonus": 500.0},
        )
        assert out.defense_success_bonus == 500.0
        # All other fields unchanged.
        assert out.penalty_missed_impact == spec.penalty_missed_impact
        assert out.reward_proportional == spec.reward_proportional
        assert out.p_defender_deescalation == spec.p_defender_deescalation
        assert out.impact_is_terminal == spec.impact_is_terminal

    def test_unknown_field_raises_value_error(self) -> None:
        """Typos / unknown keys raise ValueError with the bad key."""
        spec = EnvConfigSerializable()
        with pytest.raises(ValueError, match="banana"):
            _apply_env_overrides(spec, reward_overrides={"banana": 1})

    def test_p_defender_deescalation_explicit_overrides_dict(self) -> None:
        """The explicit kwarg takes precedence over the same field in
        the reward-overrides dict (PLAN §3.1.2 precedence rule)."""
        spec = EnvConfigSerializable()
        out = _apply_env_overrides(
            spec,
            reward_overrides={"p_defender_deescalation": 0.2},
            p_defender_deescalation=0.8,
        )
        assert out.p_defender_deescalation == 0.8, (
            "When both --reward-overrides and --p-defender-deescalation "
            "specify p_defender_deescalation, the explicit kwarg wins."
        )

    def test_impact_is_terminal_explicit_overrides_dict(self) -> None:
        """Same precedence rule for impact_is_terminal."""
        spec = EnvConfigSerializable()
        out = _apply_env_overrides(
            spec,
            reward_overrides={"impact_is_terminal": True},
            impact_is_terminal=False,
        )
        assert out.impact_is_terminal is False

    def test_multiple_reward_fields_applied(self) -> None:
        """Multi-field override is honoured."""
        spec = EnvConfigSerializable()
        out = _apply_env_overrides(
            spec,
            reward_overrides={
                "defense_success_bonus": 125.0,
                "penalty_missed_impact": 75.0,
                "reward_proportional": 10.0,
            },
        )
        assert out.defense_success_bonus == 125.0
        assert out.penalty_missed_impact == 75.0
        assert out.reward_proportional == 10.0


# ---------------------------------------------------------------------------
# build_run_config — end-to-end CLI parsing
# ---------------------------------------------------------------------------


def _parse(argv: list[str]):
    """Helper: parse CLI args via the production argparser."""
    return _build_argparser().parse_args(argv)


class TestBuildRunConfigOverrides:
    """End-to-end: CLI args → BlueTeamRunConfig with merged env spec."""

    def _base_argv(self) -> list[str]:
        """Minimal valid CLI for build_run_config()."""
        return ["--algo", "ppo", "--seed", "0", "--splits-manifest", ""]

    def test_no_overrides_matches_phase5_baseline(self) -> None:
        """No flags → env spec uses Phase-5 defaults verbatim.

        This is the **invariant that lets Phase-5 trained checkpoints
        and Phase-7 untrained-cell checkpoints be compared apples-to-
        apples**. If this test ever flips, Phase-5 numbers and Phase-7
        baselines are no longer directly comparable.
        """
        args = _parse(self._base_argv())
        cfg = build_run_config(args)

        # Every reward field at the Phase-3 / Phase-5 default.
        assert cfg.env.defense_success_bonus == 250.0
        assert cfg.env.penalty_missed_impact == 150.0
        assert cfg.env.reward_proportional == 5.0
        assert cfg.env.penalty_disproportionate == 5.0
        assert cfg.env.reward_benign_passive == 10.0
        assert cfg.env.impact_penalty == 200.0
        assert cfg.env.action_cost_scale == 1.0
        # Lifecycle defaults too.
        assert cfg.env.p_defender_deescalation == 0.6
        assert cfg.env.impact_is_terminal is True
        # Same for the eval spec (overrides apply to both).
        assert cfg.eval_env.defense_success_bonus == 250.0
        assert cfg.eval_env.p_defender_deescalation == 0.6
        assert cfg.eval_env.impact_is_terminal is True

    def test_reward_overrides_json_applied_to_both_specs(self) -> None:
        """``--reward-overrides`` modifies both train and eval env specs."""
        args = _parse(
            self._base_argv()
            + ["--reward-overrides", '{"defense_success_bonus": 500}']
        )
        cfg = build_run_config(args)
        assert cfg.env.defense_success_bonus == 500.0
        assert cfg.eval_env.defense_success_bonus == 500.0
        # Other fields unchanged.
        assert cfg.env.penalty_missed_impact == 150.0
        assert cfg.eval_env.penalty_missed_impact == 150.0

    def test_p_defender_deescalation_arg_applied_to_both_specs(self) -> None:
        """``--p-defender-deescalation`` modifies both env and eval_env."""
        args = _parse(
            self._base_argv() + ["--p-defender-deescalation", "0.0"]
        )
        cfg = build_run_config(args)
        assert cfg.env.p_defender_deescalation == 0.0
        assert cfg.eval_env.p_defender_deescalation == 0.0
        # Other defaults preserved.
        assert cfg.env.defense_success_bonus == 250.0
        assert cfg.env.impact_is_terminal is True

    def test_impact_is_terminal_false_arg(self) -> None:
        """``--impact-is-terminal false`` flips the env-config flag.

        Pins the F9 binary axis (D7.3): the value is a boolean parsed
        from the string "false" / "true".
        """
        args = _parse(
            self._base_argv() + ["--impact-is-terminal", "false"]
        )
        cfg = build_run_config(args)
        assert cfg.env.impact_is_terminal is False
        assert cfg.eval_env.impact_is_terminal is False

    def test_unknown_reward_field_raises(self) -> None:
        """Bad field name in ``--reward-overrides`` JSON → ValueError."""
        args = _parse(
            self._base_argv() + ["--reward-overrides", '{"banana": 1}']
        )
        with pytest.raises(ValueError, match="banana"):
            build_run_config(args)

    def test_run_manifest_records_merged_config(
        self, tmp_path: Path
    ) -> None:
        """The serialised ``run_manifest.json`` includes the merged
        env config, so downstream Phase-7 sweep manifests can SHA-pin
        the per-cell config without re-parsing the CLI."""
        args = _parse(
            self._base_argv()
            + [
                "--reward-overrides",
                '{"defense_success_bonus": 125.0, "reward_proportional": 2.5}',
                "--p-defender-deescalation",
                "0.4",
                "--impact-is-terminal",
                "false",
            ]
        )
        cfg = build_run_config(args)

        # Check the in-memory config carries the overrides.
        assert cfg.env.defense_success_bonus == 125.0
        assert cfg.env.reward_proportional == 2.5
        assert cfg.env.p_defender_deescalation == 0.4
        assert cfg.env.impact_is_terminal is False

        # Check the serialised manifest carries them too. We only test
        # the dict shape (write_manifest's atomic file I/O is already
        # tested in test_blue_team_run_config.py).
        d = cfg.to_dict()
        assert d["env"]["defense_success_bonus"] == 125.0
        assert d["env"]["reward_proportional"] == 2.5
        assert d["env"]["p_defender_deescalation"] == 0.4
        assert d["env"]["impact_is_terminal"] is False
        # Eval env carries them too (overrides apply to both).
        assert d["eval_env"]["defense_success_bonus"] == 125.0
        assert d["eval_env"]["impact_is_terminal"] is False

    def test_smoke_mode_still_honours_overrides(self) -> None:
        """``--smoke`` builds a smaller env spec; reward overrides are
        still applied on top of that."""
        args = _parse(
            self._base_argv()
            + [
                "--smoke",
                "--reward-overrides",
                '{"defense_success_bonus": 500.0}',
            ]
        )
        cfg = build_run_config(args)
        # Smoke shrinks the lifecycle.
        assert cfg.env.min_episode_length == 5
        assert cfg.env.max_steps == 20
        # But the reward override is still applied.
        assert cfg.env.defense_success_bonus == 500.0


class TestBackwardCompatibility:
    """Deserialising a pre-Phase-7 ``run_manifest.json`` (which lacks the
    new reward-shaping fields) must still round-trip cleanly."""

    def test_old_manifest_deserialises_with_defaults(
        self, tmp_path: Path
    ) -> None:
        """A Phase-5-era manifest (only the original 7 env fields) must
        deserialise into a config whose new fields are at default."""
        from src.blue_team import BlueTeamRunConfig

        old_manifest = {
            "schema_version": "1.0",
            "run_id": "ppo_seed_0",
            "algo": "ppo",
            "seed": 0,
            "total_timesteps": 250_000,
            "eval_freq": 25_000,
            "n_eval_episodes": 30,
            "env": {
                "split": "train",
                "exclude_ood": True,
                "min_episode_length": 20,
                "max_steps": 100,
                "window_size": 5,
                "include_deltas": True,
                "p_defender_deescalation": 0.6,
            },
            "eval_env": {
                "split": "val_balanced",
                "exclude_ood": True,
                "min_episode_length": 20,
                "max_steps": 100,
                "window_size": 5,
                "include_deltas": True,
                "p_defender_deescalation": 0.6,
            },
            "algo_hparams": {},
            "paths": {
                "generator": "g",
                "dataset": "d",
                "splits_manifest": "",
                "out_dir": "o",
            },
            "notes": "",
        }
        manifest_path = tmp_path / "old_manifest.json"
        manifest_path.write_text(json.dumps(old_manifest))

        cfg = BlueTeamRunConfig.from_manifest(manifest_path)

        # Pre-Phase-7 fields preserved.
        assert cfg.env.p_defender_deescalation == 0.6
        # New Phase-7 fields default to AdversarialEnvConfig defaults
        # (== Phase-3 frozen contract). This is what makes Phase-5
        # checkpoints loadable + comparable under Phase-7 evaluation.
        assert cfg.env.defense_success_bonus == 250.0
        assert cfg.env.penalty_missed_impact == 150.0
        assert cfg.env.reward_proportional == 5.0
        assert cfg.env.impact_is_terminal is True
