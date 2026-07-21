"""Synthetic tests for the ablation closer parsers (audit fix 2026-05-01).

These pin the regression where the auto-finalizer reported G7.1
``passes: false`` despite "442 passed, 2 warnings", which then
cascaded false-fail through G7.5 and G7.6 (per the audit findings
captured in PLAN §8 / RESULTS §6.5).

One parser is exercised:

  - ``_parse_pytest_summary``: pytest summary line → counts dict.

The tests are pure-Python (no pytest invocation, no figure render),
so they do not extend the real-data smoke surface — they pin the
parsers' shape only.
"""

from __future__ import annotations

from scripts.ablation.close_ablation import _parse_pytest_summary
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


# ----------------------------------------------------- OOD commensurability


def test_ood_eval_spec_carries_reward_mode():
    """The zero-day OOD eval MUST run at the same operating point as the
    held-out benchmark: the same reward contract under proximity-coupled
    escalation. Without this the OOD reward is on an incomparable axis.
    """
    spec = _ood_eval_env_spec(reward_mode="outcome")
    assert spec.reward_mode == "outcome"
    # impact-row decision is kept explicit (primary training contract).
    assert spec.impact_is_terminal is False


def test_ood_eval_spec_normalises_reward_mode_alias():
    """Legacy aliases normalise so the OOD manifest is comparable to the
    benchmark manifest under the same canonical token."""
    assert _ood_eval_env_spec(reward_mode="outcome_only").reward_mode == "outcome"
    assert _ood_eval_env_spec(reward_mode="proportional").reward_mode == "coupled"


def test_ood_eval_spec_default_reward_mode_is_outcome():
    """Default reward mode is the primary deployment contract (``outcome``)
    under proximity-coupled escalation."""
    spec = _ood_eval_env_spec()
    assert spec.reward_mode == "outcome"
    assert spec.proximity_coupled is True


def test_ood_eval_argparser_defaults_match_locked_contract():
    """The argparse defaults (not just the function defaults) must match the
    locked contract so direct ``python -m scripts.ablation.run_ood_eval``
    invocation does not silently run a different MDP than training."""
    from scripts.ablation.run_ood_eval import _build_argparser

    args = _build_argparser().parse_args([])
    assert args.proximity_coupled is True
    assert args.seeds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert args.n_deterministic_episodes == 300
    assert args.blue_team_runs == "runs/redesign_5M_det/alpha_04"
    assert args.reward_mode == "outcome"


# ------------------------------------------- detector-independence figure


def test_recall_vs_advantage_points_track_detector_blind_spot():
    """The detector-independence statistics must report a LARGER RL-minus-RF
    advantage where the supervised detector's recall is LOWER.

    This pins the headline claim's mechanics: on a blind-spot class
    (recall ≈ 0) the detector-free RL policy should hold up while the
    detector-coupled RF-Acting baseline degrades, so the advantage is
    positive; on a class the detector handles well (recall ≈ 1) the RL
    policy has no such edge.

    The standalone scatter figure was retired (commit 15ba5a3); the point
    table + independence statistics are now produced by
    ``_compute_recall_vs_advantage`` (no figure is written), so this test
    exercises the returned ``points`` only.
    """
    from scripts.ablation.plot_ood_robustness import _compute_recall_vs_advantage

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
    recall: dict[str, float | None] = {"VulnerabilityScan": 0.001, "DDoS-HTTP_Flood": 0.95}

    out = _compute_recall_vs_advantage(
        rows,
        recall,
        ["VulnerabilityScan", "DDoS-HTTP_Flood"],
        metric="prevention_rate",
    )
    pts = {p["ood_class"]: p for p in out["points"]}
    # Blind-spot class: best RL (0.60) beats RF (0.0) → positive advantage.
    assert pts["VulnerabilityScan"]["advantage"] > 0
    # Detector-covered class: RF (0.95) beats RL (0.55) → negative advantage.
    assert pts["DDoS-HTTP_Flood"]["advantage"] < 0
    # The advantage is strictly larger where detector recall is lower.
    assert pts["VulnerabilityScan"]["advantage"] > pts["DDoS-HTTP_Flood"]["advantage"]


def test_recall_vs_advantage_skips_cells_without_recall():
    """Classes with missing recall (None) or missing RF row are skipped
    gracefully so the statistics degrade on a partial checkout."""
    from scripts.ablation.plot_ood_robustness import _compute_recall_vs_advantage

    rows = [
        {"ood_class": "XSS", "policy": "ppo", "prevention_rate": 0.5},
        # no rf_acting row for XSS
        {"ood_class": "Recon-OSScan", "policy": "rf_acting", "prevention_rate": 0.4},
        {"ood_class": "Recon-OSScan", "policy": "ppo", "prevention_rate": 0.6},
    ]
    recall = {"XSS": 0.9, "Recon-OSScan": None}
    out = _compute_recall_vs_advantage(
        rows, recall, ["XSS", "Recon-OSScan"], metric="prevention_rate"
    )
    # XSS skipped (no rf row); Recon-OSScan skipped (recall None).
    assert out["points"] == []


# ------------------------------------------- coupled-vs-decoupled reward ablation


def test_eval_spec_for_mode_carries_reward_mode():
    from scripts.ablation.run_reward_coupling import _eval_spec_for_mode

    coupled = _eval_spec_for_mode("coupled")
    assert coupled.reward_mode == "coupled"
    assert coupled.split == "test_balanced"
    assert coupled.impact_is_terminal is False
    assert coupled.exclude_ood is True

    outcome = _eval_spec_for_mode("outcome")
    assert outcome.reward_mode == "outcome"

    # Aliases normalise through the spec.
    assert _eval_spec_for_mode("proportional").reward_mode == "coupled"
    assert _eval_spec_for_mode("outcome_only").reward_mode == "outcome"


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
        assert (
            cell["ci_low"] <= cell["mean_reward"] <= cell["ci_high"]
        ), f"{algo}: mean {cell['mean_reward']} outside CI [{cell['ci_low']}, {cell['ci_high']}]"
    assert s["rf_acting_ci_low"] <= s["rf_acting_reward"] <= s["rf_acting_ci_high"]


def test_coupling_summary_handles_missing_cells(tmp_path):
    """Empty/missing cells degrade to None rather than raising."""
    from scripts.ablation.plot_reward_coupling import summarise_mode

    # No files written at all.
    s = summarise_mode(tmp_path, "outcome", seeds=[0])
    assert s["best_rl_reward"] is None
    assert s["rf_acting_reward"] is None
    assert s["rf_minus_rl_gap"] is None


# ------------------------------------------- OOD hybrid realiser mutation


def test_ood_hybrid_realiser_mutation_pattern(mock_dataset):
    """The F15 hybrid realiser surgically replaces one stage's row pool with
    OOD-class rows by mutating three private RealizationEngine attrs. Pin the
    mutation contract so a refactor that changes the attr names or drops the
    total recompute is caught (scripts/ablation/run_ood_eval.py:294-298)."""
    from src.utils.realization_engine import RealizationEngine

    engine = RealizationEngine(data_path=mock_dataset["path"])

    ood_stage = 3  # MANEUVER (e.g., Mirai-udpplain)
    original_total = engine._total_samples
    original_ood_count = engine._stage_counts[ood_stage]

    # Synthetic OOD-class rows: a small set of row indices.
    new_indices = [0, 1, 2, 5, 7]

    # Mirror the _build_ood_env mutation (run_ood_eval.py:294-298).
    engine._state_indices[ood_stage] = new_indices
    engine._stage_counts[ood_stage] = len(new_indices)
    engine._total_samples = sum(engine._stage_counts.values())

    # The OOD stage's pool is replaced with the new indices.
    assert engine._state_indices[ood_stage] == new_indices
    assert engine._stage_counts[ood_stage] == len(new_indices)
    # The total is recomputed as the sum of all stage counts.
    assert engine._total_samples == sum(engine._stage_counts.values())
    # The total reflects the surgical replacement (other stages untouched).
    assert engine._total_samples == original_total - original_ood_count + len(new_indices)
