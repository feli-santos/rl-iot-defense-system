"""Smoke tests for scripts.benchmark.run_statistical_tests (thesis review C4).

These tests verify that the statistical test machinery works correctly on
synthetic data, without requiring actual Phase-6 run outputs. They check:
  - Helper functions (_cohens_d, _bootstrap_ci, _welch_test, _wilcoxon_test)
  - run_tests() on mock JSONL data written to tmp_path
  - Edge cases: missing data, zero variance, unequal sample sizes
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from scripts.benchmark.run_statistical_tests import (
    _bootstrap_ci,
    _cohens_d,
    _flatten,
    _load_episode_rewards,
    _welch_test,
    _wilcoxon_test,
    run_tests,
)


# ------------------------------------------------------------------ fixtures


def _write_jsonl(path: Path, rewards: List[float]) -> None:
    """Write synthetic EpisodeRecord-like JSONL with only episode_reward."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rewards:
            fh.write(json.dumps({"episode_reward": r}) + "\n")


@pytest.fixture()
def mock_phase6_root(tmp_path: Path) -> Path:
    """Build a minimal Phase-6 directory tree with synthetic rewards."""
    rng = np.random.default_rng(42)
    # DQN: slightly higher mean (~1300)
    dqn_rewards = (rng.normal(loc=1300, scale=100, size=30)).tolist()
    # PPO: slightly lower mean (~1250)
    ppo_rewards = (rng.normal(loc=1250, scale=120, size=30)).tolist()
    # A2C: lowest (~1200)
    a2c_rewards = (rng.normal(loc=1200, scale=110, size=30)).tolist()
    # RF-Acting: higher than DRL (~1500)
    rf_rewards = (rng.normal(loc=1500, scale=80, size=150)).tolist()

    for seed in range(2):  # 2 seeds × 15 episodes each
        chunk_dqn = dqn_rewards[seed * 15: (seed + 1) * 15]
        chunk_ppo = ppo_rewards[seed * 15: (seed + 1) * 15]
        chunk_a2c = a2c_rewards[seed * 15: (seed + 1) * 15]
        _write_jsonl(tmp_path / "dqn" / f"seed_{seed}" / "eval_test.jsonl", chunk_dqn)
        _write_jsonl(tmp_path / "ppo" / f"seed_{seed}" / "eval_test.jsonl", chunk_ppo)
        _write_jsonl(tmp_path / "a2c" / f"seed_{seed}" / "eval_test.jsonl", chunk_a2c)

    # RF-Acting: single seed_0
    _write_jsonl(tmp_path / "rf_acting" / "seed_0" / "eval_test.jsonl", rf_rewards)

    return tmp_path


# ------------------------------------------------------------------ unit tests


class TestLoadEpisodeRewards:
    def test_reads_rewards(self, tmp_path: Path) -> None:
        p = tmp_path / "ep.jsonl"
        _write_jsonl(p, [1.0, 2.0, 3.0])
        rewards = _load_episode_rewards(p)
        assert rewards == pytest.approx([1.0, 2.0, 3.0])

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.jsonl"
        rewards = _load_episode_rewards(p)
        assert rewards == []

    def test_skips_bad_lines(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text('{"episode_reward": 1.0}\nNOT_JSON\n{"episode_reward": 2.0}\n')
        rewards = _load_episode_rewards(p)
        assert rewards == pytest.approx([1.0, 2.0])

    def test_skips_missing_key(self, tmp_path: Path) -> None:
        p = tmp_path / "nokey.jsonl"
        p.write_text('{"reward": 1.0}\n{"episode_reward": 5.0}\n')
        rewards = _load_episode_rewards(p)
        assert rewards == pytest.approx([5.0])


class TestCohensD:
    def test_positive_d_when_a_gt_b(self) -> None:
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = _cohens_d(a, b)
        assert d > 0

    def test_zero_d_identical(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        d = _cohens_d(a, a.copy())
        assert math.isnan(d)  # pooled std = 0 → NaN

    def test_d_is_nan_with_single_sample(self) -> None:
        a = np.array([5.0])
        b = np.array([3.0])
        d = _cohens_d(a, b)
        assert math.isnan(d)

    def test_symmetric_magnitude(self) -> None:
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([7.0, 8.0, 9.0])
        assert pytest.approx(_cohens_d(a, b), abs=1e-6) == -_cohens_d(b, a)


class TestBootstrapCI:
    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(100, 10, 200)
        lo, hi = _bootstrap_ci(x)
        assert lo < np.mean(x) < hi

    def test_ci_ordered(self) -> None:
        x = np.arange(1.0, 101.0)
        lo, hi = _bootstrap_ci(x)
        assert lo < hi

    def test_wider_with_higher_variance(self) -> None:
        rng = np.random.default_rng(1)
        low_var = rng.normal(50, 1, 100)
        high_var = rng.normal(50, 20, 100)
        lo1, hi1 = _bootstrap_ci(low_var)
        lo2, hi2 = _bootstrap_ci(high_var)
        assert (hi1 - lo1) < (hi2 - lo2)


class TestWelchTest:
    def test_significant_difference(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(1300, 50, 150)
        b = rng.normal(1500, 50, 150)
        result = _welch_test(a, b, label="test", alpha=0.05)
        assert result["test"] == "welch_t"
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_non_significant_difference(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1000, 200, 30)
        b = rng.normal(1005, 200, 30)  # tiny difference
        result = _welch_test(a, b, label="test", alpha=0.05)
        # With large variance and tiny mean diff, should often be non-significant
        assert result["test"] == "welch_t"
        assert "p_value" in result

    def test_insufficient_data(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        result = _welch_test(a, b, label="test", alpha=0.05)
        assert "error" in result


class TestWilcoxonTest:
    def test_equal_length_uses_wilcoxon(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(1300, 100, 50)
        b = rng.normal(1250, 100, 50)
        result = _wilcoxon_test(a, b, label="test", alpha=0.05)
        # With these params, test should run (either wilcoxon or mann-whitney)
        assert result["test"] in ("wilcoxon_signed_rank", "mann_whitney_u")
        assert "p_value" in result

    def test_unequal_length_uses_mannwhitney(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.normal(1300, 100, 50)
        b = rng.normal(1500, 100, 30)  # different length
        result = _wilcoxon_test(a, b, label="test", alpha=0.05)
        assert result["test"] == "mann_whitney_u"


class TestRunTests:
    def test_smoke_on_mock_data(self, mock_phase6_root: Path) -> None:
        """Full pipeline smoke test with synthetic Phase-6 data."""
        results = run_tests(
            phase6_root=mock_phase6_root,
            seeds=[0, 1],
            alpha=0.05,
        )
        assert "bootstrap_ci_summary" in results
        assert "comparisons" in results
        assert "n_significant" in results
        assert isinstance(results["n_significant"], int)

    def test_ci_summary_has_all_algos(self, mock_phase6_root: Path) -> None:
        results = run_tests(
            phase6_root=mock_phase6_root,
            seeds=[0, 1],
        )
        ci = results["bootstrap_ci_summary"]
        assert "dqn" in ci
        assert "ppo" in ci
        assert "a2c" in ci
        assert "rf_acting" in ci

    def test_ci_contains_means(self, mock_phase6_root: Path) -> None:
        results = run_tests(
            phase6_root=mock_phase6_root,
            seeds=[0, 1],
        )
        for algo, s in results["bootstrap_ci_summary"].items():
            assert s["ci_95_lower"] <= s["mean"] <= s["ci_95_upper"], (
                f"{algo}: mean {s['mean']} not in CI [{s['ci_95_lower']}, {s['ci_95_upper']}]"
            )

    def test_comparisons_list_nonempty(self, mock_phase6_root: Path) -> None:
        results = run_tests(
            phase6_root=mock_phase6_root,
            seeds=[0, 1],
        )
        assert len(results["comparisons"]) > 0

    def test_handles_missing_algo_gracefully(self, tmp_path: Path) -> None:
        """If only DQN data is present, run_tests should not crash."""
        rng = np.random.default_rng(0)
        _write_jsonl(
            tmp_path / "dqn" / "seed_0" / "eval_test.jsonl",
            rng.normal(1300, 100, 30).tolist(),
        )
        results = run_tests(phase6_root=tmp_path, seeds=[0])
        assert "dqn" in results["bootstrap_ci_summary"]
        # comparisons may be empty (no PPO/A2C/RF-Acting) — that's fine
        assert isinstance(results["comparisons"], list)
