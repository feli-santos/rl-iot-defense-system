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
from scripts.ablation.run_ood_eval import _ood_eval_env_spec

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
            "impact_is_terminal_false",
            "impact_terminal",
            1500.0,
            impact_is_terminal=False,
            mit=0.90,
        ),
        # A reward-coefficient cell that scaled the bonus 2× and
        # has a huge raw reward — must NOT be the apples-to-apples
        # winner.
        _row(
            "defense_success_bonus_x2p0",
            "reward",
            2900.0,
            component="defense_success_bonus",
            multiplier=2.0,
            mit=0.55,
        ),
    ]
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
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
            "impact_is_terminal_false",
            "impact_terminal",
            1330.0,
            impact_is_terminal=False,
            mit=0.90,
        ),
        _row(
            "defense_success_bonus_x2p0",
            "reward",
            2900.0,
            component="defense_success_bonus",
            multiplier=2.0,
            mit=0.55,
        ),
    ]
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
    assert result["passes"] is False  # strand-1 failed
    assert result["security_kpi_strand_passes"] is True  # strand-2 passed
    assert result["best_security_kpi_cell"] == "impact_is_terminal_false"
    assert "D7.1.1" in result["interpretation"]


def test_g72_both_strands_fail():
    """Pure null-result case: nothing improves on either metric."""
    rows = [
        _row("baseline", "baseline", 1300.0, mit=0.15),
        _row(
            "reward_proportional_x2p0",
            "reward",
            1310.0,
            component="reward_proportional",
            multiplier=2.0,
            mit=0.18,
        ),
    ]
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
    assert result["passes"] is False
    assert result["security_kpi_strand_passes"] is False
    assert "FAIL-WITH-FINDING" in result["interpretation"]


def test_g72_oracle_stretch_met():
    """Strand-1 passes AND best reward-comparable cell exceeds the
    oracle ceiling +1624 → stretch goal met."""
    rows = [
        _row("baseline", "baseline", 1300.0),
        _row(
            "impact_is_terminal_false",
            "impact_terminal",
            1800.0,
            impact_is_terminal=False,
            mit=0.95,
        ),
    ]
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
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
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
    assert result["passes"] is False
    assert "no candidate" in result.get("reason", "")


def test_g72_reward_coefficient_cell_does_not_count_for_strand1():
    """Even if a reward-coefficient cell has a huge mean reward,
    it does NOT make strand-1 pass; only environment-design-reward-fn-
    preserving cells qualify (axis baseline / impact_terminal)."""
    rows = [
        _row("baseline", "baseline", 1100.0, mit=0.20),
        _row(
            "defense_success_bonus_x2p0",
            "reward",
            3000.0,
            component="defense_success_bonus",
            multiplier=2.0,
            mit=0.55,
        ),
    ]
    result = _evaluate_g72(
        rows,
        deployable_best=1336.0,
        oracle_ceiling=1624.0,
        deployable_best_mitigated=0.20,
    )
    assert result["passes"] is False
    # The reward-coefficient cell is the raw winner but not the
    # apples-to-apples winner.
    assert result["raw_reward_winner_cell"] == "defense_success_bonus_x2p0"
    assert result["best_reward_comparable_cell"] == "baseline"


# ----------------------------------------------------- OOD commensurability


def test_ood_eval_spec_carries_attacker_budget_and_reward_mode():
    """The zero-day OOD eval MUST run at the same operating point as the
    held-out benchmark: a finite attacker budget (so prevention can fire
    and bound the episode) and the same reward contract. Without this the
    OOD reward is on an incomparable, unbounded penalty-bleed axis.
    """
    spec = _ood_eval_env_spec(attacker_budget=40, reward_mode="outcome")
    assert spec.attacker_budget == 40
    assert spec.reward_mode == "outcome"
    # impact-row decision is kept explicit (primary training contract).
    assert spec.impact_is_terminal is False


def test_ood_eval_spec_normalises_reward_mode_alias():
    """Legacy aliases normalise so the OOD manifest is comparable to the
    benchmark manifest under the same canonical token."""
    assert _ood_eval_env_spec(reward_mode="outcome_only").reward_mode == "outcome"
    assert _ood_eval_env_spec(reward_mode="proportional").reward_mode == "coupled"


def test_ood_eval_spec_default_budget_is_unbounded_but_explicit():
    """Default (no budget passed) stays unbounded — but the Makefile and
    CLI now pass the benchmark budget explicitly, so production OOD runs
    are commensurable; this default only guards direct/legacy callers."""
    spec = _ood_eval_env_spec()
    assert spec.attacker_budget is None
    assert spec.reward_mode == "outcome"


# ------------------------------------------- detector-independence figure


def test_recall_vs_advantage_points_track_detector_blind_spot(tmp_path):
    """The detector-independence figure must report a LARGER RL-minus-RF
    advantage where the supervised detector's recall is LOWER.

    This pins the headline claim's mechanics: on a blind-spot class
    (recall ≈ 0) the detector-free RL policy should hold up while the
    detector-coupled RF-Acting baseline degrades, so the advantage is
    positive; on a class the detector handles well (recall ≈ 1) the RL
    policy has no such edge.
    """
    from scripts.ablation.plot_ood_robustness import _render_recall_vs_advantage

    def _row(cls, policy, prevention):
        return {
            "ood_class": cls,
            "policy": policy,
            "prevention_rate": prevention,
            "mean_reward": 0.0,
            "compromise_rate": 0.0,
        }

    rows = [
        _row("VulnerabilityScan", "rf_acting", 0.0),
        _row("VulnerabilityScan", "dqn", 0.55),
        _row("VulnerabilityScan", "ppo", 0.60),
        _row("VulnerabilityScan", "a2c", 0.50),
        _row("DDoS-HTTP_Flood", "rf_acting", 0.95),
        _row("DDoS-HTTP_Flood", "dqn", 0.55),
        _row("DDoS-HTTP_Flood", "ppo", 0.55),
        _row("DDoS-HTTP_Flood", "a2c", 0.55),
    ]
    recall = {"VulnerabilityScan": 0.001, "DDoS-HTTP_Flood": 0.95}

    out = _render_recall_vs_advantage(
        rows,
        recall,
        ["VulnerabilityScan", "DDoS-HTTP_Flood"],
        tmp_path / "f15b.png",
        metric="prevention_rate",
    )
    assert (tmp_path / "f15b.png").exists()
    pts = {p["ood_class"]: p for p in out["points"]}
    # Blind-spot class: best RL (0.60) beats RF (0.0) → positive advantage.
    assert pts["VulnerabilityScan"]["advantage"] > 0
    # Detector-covered class: RF (0.95) beats RL (0.55) → negative advantage.
    assert pts["DDoS-HTTP_Flood"]["advantage"] < 0
    # The advantage is strictly larger where detector recall is lower.
    assert (
        pts["VulnerabilityScan"]["advantage"] > pts["DDoS-HTTP_Flood"]["advantage"]
    )


def test_recall_vs_advantage_skips_cells_without_recall(tmp_path):
    """Classes with missing recall (None) or missing RF row are skipped
    gracefully so the plotter degrades on a partial checkout."""
    from scripts.ablation.plot_ood_robustness import _render_recall_vs_advantage

    rows = [
        {"ood_class": "XSS", "policy": "ppo", "prevention_rate": 0.5},
        # no rf_acting row for XSS
        {"ood_class": "Recon-OSScan", "policy": "rf_acting", "prevention_rate": 0.4},
        {"ood_class": "Recon-OSScan", "policy": "ppo", "prevention_rate": 0.6},
    ]
    recall = {"XSS": 0.9, "Recon-OSScan": None}
    out = _render_recall_vs_advantage(
        rows, recall, ["XSS", "Recon-OSScan"], tmp_path / "f.png", metric="prevention_rate"
    )
    # XSS skipped (no rf row); Recon-OSScan skipped (recall None).
    assert out["points"] == []


# ------------------------------------------- coupled-vs-decoupled reward ablation


def test_eval_spec_for_mode_carries_reward_mode_and_budget():
    from scripts.ablation.run_reward_coupling import _eval_spec_for_mode

    coupled = _eval_spec_for_mode("coupled", 40)
    assert coupled.reward_mode == "coupled"
    assert coupled.attacker_budget == 40
    assert coupled.split == "test_balanced"
    assert coupled.impact_is_terminal is False
    assert coupled.exclude_ood is True

    outcome = _eval_spec_for_mode("outcome", 40)
    assert outcome.reward_mode == "outcome"
    assert outcome.attacker_budget == 40

    # Aliases normalise through the spec.
    assert _eval_spec_for_mode("proportional", None).reward_mode == "coupled"
    assert _eval_spec_for_mode("outcome_only", None).reward_mode == "outcome"


def _write_coupling_eval(path, rewards):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, r in enumerate(rewards):
            rec = {
                "schema_version": "1.0",
                "run_id": "x",
                "algo": "ppo",
                "seed": 0,
                "episode_idx": i,
                "num_timesteps": 0,
                "wallclock_seconds": 0.0,
                "episode_reward": float(r),
                "episode_length": 10,
                "compromised": False,
                "mttc_steps": None,
                "defender_deescalations": 0,
                "final_stage": 0,
                "final_stage_name": "BENIGN",
                "end_outcome": "prevented",
                "action_counts": {},
                "action_counts_by_stage": {},
            }
            f.write(json.dumps(rec) + "\n")


def test_coupling_gap_sign_tracks_reward_mode(tmp_path):
    """Under coupled, RF-Acting must dominate the best RL agent (positive
    RF-minus-RL gap); under outcome the gap must shrink or reverse. This is
    the direct, synthetic check of the WS2 aggregation arithmetic."""
    from scripts.ablation.plot_reward_coupling import summarise_mode

    root = tmp_path
    # Coupled: RF dominates (RF 450 vs best RL 280).
    _write_coupling_eval(root / "coupled" / "ppo" / "seed_0" / "eval_test.jsonl", [280] * 20)
    _write_coupling_eval(root / "coupled" / "dqn" / "seed_0" / "eval_test.jsonl", [260] * 20)
    _write_coupling_eval(root / "coupled" / "a2c" / "seed_0" / "eval_test.jsonl", [270] * 20)
    _write_coupling_eval(root / "coupled" / "rf_acting" / "eval_test.jsonl", [450] * 20)
    # Outcome: RL wins (RF 120 vs best RL 300).
    _write_coupling_eval(root / "outcome" / "ppo" / "seed_0" / "eval_test.jsonl", [300] * 20)
    _write_coupling_eval(root / "outcome" / "dqn" / "seed_0" / "eval_test.jsonl", [250] * 20)
    _write_coupling_eval(root / "outcome" / "a2c" / "seed_0" / "eval_test.jsonl", [290] * 20)
    _write_coupling_eval(root / "outcome" / "rf_acting" / "eval_test.jsonl", [120] * 20)

    coupled = summarise_mode(root, "coupled", seeds=[0])
    outcome = summarise_mode(root, "outcome", seeds=[0])

    assert coupled["best_algo"] == "ppo"
    assert coupled["rf_minus_rl_gap"] > 100  # RF dominates under coupling
    assert outcome["rf_minus_rl_gap"] < -100  # RL wins under outcome
    # The gap collapses by a large margin from coupled to outcome.
    assert coupled["rf_minus_rl_gap"] - outcome["rf_minus_rl_gap"] > 300


def test_coupling_mean_lies_within_ci(tmp_path):
    """Regression guard: ``bootstrap_ci`` returns ``(low, mean, high)``. A
    mis-ordered unpack would put the mean outside its own CI. Use VARIED
    rewards so the bootstrap CI has non-zero width (constant inputs collapse
    the CI to a point and hide the swap)."""
    from scripts.ablation.plot_reward_coupling import summarise_mode

    root = tmp_path
    varied = [100.0 + 40.0 * (i % 7) for i in range(60)]  # spread, non-constant
    for algo in ("ppo", "dqn", "a2c"):
        _write_coupling_eval(root / "outcome" / algo / "seed_0" / "eval_test.jsonl", varied)
    _write_coupling_eval(root / "outcome" / "rf_acting" / "eval_test.jsonl", varied)

    s = summarise_mode(root, "outcome", seeds=[0])
    for algo, cell in s["per_algo"].items():
        assert cell["ci_low"] <= cell["mean_reward"] <= cell["ci_high"], (
            f"{algo}: mean {cell['mean_reward']} outside CI "
            f"[{cell['ci_low']}, {cell['ci_high']}]"
        )
    assert s["rf_acting_ci_low"] <= s["rf_acting_reward"] <= s["rf_acting_ci_high"]


def test_coupling_summary_handles_missing_cells(tmp_path):
    """Empty/missing cells degrade to None rather than raising."""
    from scripts.ablation.plot_reward_coupling import summarise_mode

    # No files written at all.
    s = summarise_mode(tmp_path, "outcome", seeds=[0])
    assert s["best_rl_reward"] is None
    assert s["rf_acting_reward"] is None
    assert s["rf_minus_rl_gap"] is None
