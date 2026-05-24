"""Synthetic tests for the ablation closer parsers (audit fix 2026-05-01).

These pin the regression where the auto-finalizer reported G7.1
``passes: false`` despite "442 passed, 2 warnings", which then
cascaded false-fail through G7.5 and G7.6 (per the audit findings
captured in PLAN §8 / RESULTS §6.5).

Two parsers are exercised:

  - ``_parse_pytest_summary``: pytest summary line → counts dict.
  - ``_evaluate_g72`` (in ``plot_reward_ablation``): F9 rows →
    two-strand G7.2 verdict (apples-to-apples reward strand +
    security-KPI fallback strand per D7.1.1).

The tests are pure-Python (no pytest invocation, no figure render),
so they do not extend the real-data smoke surface — they pin the
parsers' shape only.
"""

from __future__ import annotations

import math

from scripts.ablation.close_ablation import _parse_pytest_summary
from scripts.ablation.plot_reward_ablation import _evaluate_g72


# ----------------------------------------------------------- pytest summary


def test_parse_pytest_summary_passed_warnings_only():
    line = "================== 442 passed, 2 warnings in 84.49s (0:01:24) =================="
    counts = _parse_pytest_summary(line)
    assert counts["passed"] == 442
    assert counts["failed"] == 0
    assert counts["warnings"] == 2
    assert counts["skipped"] == 0
    assert counts["errors"] == 0


def test_parse_pytest_summary_passed_only():
    line = "442 passed in 64.00s"
    counts = _parse_pytest_summary(line)
    assert counts["passed"] == 442
    assert counts["failed"] == 0


def test_parse_pytest_summary_with_skipped():
    line = "================== 440 passed, 2 skipped in 60s =================="
    counts = _parse_pytest_summary(line)
    assert counts["passed"] == 440
    assert counts["skipped"] == 2
    assert counts["failed"] == 0


def test_parse_pytest_summary_with_failures():
    line = "439 passed, 3 failed, 1 warning in 90s"
    counts = _parse_pytest_summary(line)
    assert counts["passed"] == 439
    assert counts["failed"] == 3
    assert counts["warnings"] == 1


def test_parse_pytest_summary_empty_line():
    counts = _parse_pytest_summary("")
    assert counts["passed"] == 0
    assert counts["failed"] == 0


def test_parse_pytest_summary_singular_warning():
    line = "100 passed, 1 warning in 5s"
    counts = _parse_pytest_summary(line)
    assert counts["passed"] == 100
    assert counts["warnings"] == 1


# ----------------------------------------------------------- G7.2 two-strand


def _row(
    cell_id: str,
    axis: str,
    mean_reward: float,
    *,
    component: str = None,
    multiplier: float = None,
    impact_is_terminal: bool = True,
    mit: float = 0.3,
    ci_half_width: float = 50.0,
):
    return {
        "cell_id": cell_id,
        "axis": axis,
        "component": component,
        "multiplier": multiplier,
        "impact_is_terminal": impact_is_terminal,
        "n_seeds": 5,
        "n_episodes": 750,
        "mean_reward": mean_reward,
        "ci_low": mean_reward - ci_half_width,
        "ci_high": mean_reward + ci_half_width,
        "compromise_rate": 1.0,
        "mitigated_impact_rate": mit,
    }


def test_g72_strand1_passes_when_reward_comparable_cell_beats_dqn():
    """impact_is_terminal=False (axis='impact_terminal') beats DQN
    +1336 by ≥ 1σ → strand-1 PASS."""
    rows = [
        _row("baseline", "baseline", 1300.0, mit=0.27),
        _row(
            "impact_is_terminal_false", "impact_terminal", 1500.0,
            impact_is_terminal=False, mit=0.90,
        ),
        # A reward-coefficient cell that scaled the bonus 2× and
        # has a huge raw reward — must NOT be the apples-to-apples
        # winner.
        _row(
            "defense_success_bonus_x2p0", "reward", 2900.0,
            component="defense_success_bonus", multiplier=2.0, mit=0.55,
        ),
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is True
    assert result["best_reward_comparable_cell"] == "impact_is_terminal_false"
    # Raw-reward winner is reported but not the headline.
    assert result["raw_reward_winner_cell"] == "defense_success_bonus_x2p0"


def test_g72_strand1_fails_strand2_passes_activates_d7_1_1():
    """No reward-comparable cell beats DQN, but security-KPI
    strand passes → FAIL-WITH-FINDING (D7.1.1)."""
    rows = [
        _row("baseline", "baseline", 1300.0, mit=0.27),
        _row(
            "impact_is_terminal_false", "impact_terminal", 1330.0,
            impact_is_terminal=False, mit=0.90,
        ),
        _row(
            "defense_success_bonus_x2p0", "reward", 2900.0,
            component="defense_success_bonus", multiplier=2.0, mit=0.55,
        ),
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is False  # strand-1 failed
    assert result["security_kpi_strand_passes"] is True  # strand-2 passed
    assert result["best_security_kpi_cell"] == "impact_is_terminal_false"
    assert "D7.1.1" in result["interpretation"]


def test_g72_both_strands_fail():
    """Pure null-result case: nothing improves on either metric."""
    rows = [
        _row("baseline", "baseline", 1300.0, mit=0.15),
        _row(
            "reward_proportional_x2p0", "reward", 1310.0,
            component="reward_proportional", multiplier=2.0, mit=0.18,
        ),
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is False
    assert result["security_kpi_strand_passes"] is False
    assert "FAIL-WITH-FINDING" in result["interpretation"]


def test_g72_oracle_stretch_met():
    """Strand-1 passes AND best reward-comparable cell exceeds the
    oracle ceiling +1624 → stretch goal met."""
    rows = [
        _row("baseline", "baseline", 1300.0),
        _row(
            "impact_is_terminal_false", "impact_terminal", 1800.0,
            impact_is_terminal=False, mit=0.95,
        ),
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is True
    assert result["meets_oracle_ceiling_stretch"] is True
    assert "STRETCH MET" in result["interpretation"]


def test_g72_no_finite_rows_returns_failure():
    rows = [
        {
            "cell_id": "broken",
            "axis": "baseline",
            "mean_reward": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "mitigated_impact_rate": math.nan,
        }
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is False
    assert "no candidate" in result.get("reason", "")


def test_g72_reward_coefficient_cell_does_not_count_for_strand1():
    """Even if a reward-coefficient cell has a huge mean reward,
    it does NOT make strand-1 pass; only environment-design-reward-fn-
    preserving cells qualify (axis baseline / impact_terminal)."""
    rows = [
        _row("baseline", "baseline", 1100.0, mit=0.20),
        _row(
            "defense_success_bonus_x2p0", "reward", 3000.0,
            component="defense_success_bonus", multiplier=2.0, mit=0.55,
        ),
    ]
    result = _evaluate_g72(rows)
    assert result["passes"] is False
    # The reward-coefficient cell is the raw winner but not the
    # apples-to-apples winner.
    assert result["raw_reward_winner_cell"] == "defense_success_bonus_x2p0"
    assert result["best_reward_comparable_cell"] == "baseline"
