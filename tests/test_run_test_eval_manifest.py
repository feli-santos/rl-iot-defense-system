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

from scripts.benchmark.run_test_eval import _eval_env_spec


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
