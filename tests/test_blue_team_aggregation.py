"""Tests for src/blue_team/aggregation.py (3.2.2)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.blue_team import aggregation as agg

# --------------------------------------------------------------- helpers


def _make_record(
    *,
    num_timesteps: int,
    episode_reward: float = 0.0,
    episode_length: int = 10,
    compromised: bool = False,
    mttc_steps: int | None = None,
    action_counts: list[int] | None = None,
    per_stage: dict[str, list[int]] | None = None,
    algo: str = "ppo",
    seed: int = 0,
    episode_idx: int = 0,
) -> dict[str, Any]:
    """Build a minimal record matching the EpisodeJSONLCallback schema."""
    return {
        "schema_version": "1.0",
        "run_id": f"{algo}_seed_{seed}",
        "algo": algo,
        "seed": seed,
        "episode_idx": episode_idx,
        "num_timesteps": int(num_timesteps),
        "wallclock_seconds": 0.0,
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "compromised": bool(compromised),
        "mttc_steps": mttc_steps,
        "defender_deescalations": 0,
        "final_stage": 0,
        "final_stage_name": "BENIGN",
        "end_outcome": "ongoing",
        "action_counts": action_counts or [10, 0, 0, 0, 0],
        "action_counts_by_stage": per_stage
        or {
            "0": [10, 0, 0, 0, 0],
            "1": [0, 0, 0, 0, 0],
            "2": [0, 0, 0, 0, 0],
            "3": [0, 0, 0, 0, 0],
            "4": [0, 0, 0, 0, 0],
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------- read_*


class TestReadEpisodesJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "ep.jsonl"
        records = [_make_record(num_timesteps=k * 100) for k in range(5)]
        _write_jsonl(path, records)
        loaded = agg.read_episodes_jsonl(path)
        assert len(loaded) == 5
        assert all(r["schema_version"] == "1.0" for r in loaded)

    def test_rejects_bad_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "ep.jsonl"
        bad = _make_record(num_timesteps=10)
        bad["schema_version"] = "0.9"
        _write_jsonl(path, [bad])
        with pytest.raises(ValueError, match="schema_version"):
            agg.read_episodes_jsonl(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            agg.read_episodes_jsonl(tmp_path / "nope.jsonl")

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "ep.jsonl"
        with path.open("w") as fh:
            fh.write(json.dumps(_make_record(num_timesteps=1)) + "\n")
            fh.write("\n")
            fh.write(json.dumps(_make_record(num_timesteps=2)) + "\n")
        assert len(agg.read_episodes_jsonl(path)) == 2


class TestReadRunsDirectory:
    def test_collects_algo_seed_pairs(self, tmp_path: Path) -> None:
        for algo in ("ppo", "dqn"):
            for seed in (0, 1):
                d = tmp_path / algo / f"seed_{seed}"
                d.mkdir(parents=True)
                _write_jsonl(
                    d / "episodes.jsonl",
                    [_make_record(num_timesteps=100, algo=algo, seed=seed)],
                )
        out = agg.read_runs_directory(tmp_path)
        assert set(out.keys()) == {("ppo", 0), ("ppo", 1), ("dqn", 0), ("dqn", 1)}

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert agg.read_runs_directory(tmp_path / "missing") == {}


# --------------------------------------------------------------- binning


class TestBinByTimesteps:
    def test_groups_by_edge(self) -> None:
        records = [
            _make_record(num_timesteps=50, episode_reward=1.0),
            _make_record(num_timesteps=150, episode_reward=2.0),
            _make_record(num_timesteps=250, episode_reward=4.0),
        ]
        edges = [0, 100, 200, 300]  # 3 buckets
        out = agg.bin_by_timesteps(records, edges, "episode_reward")
        assert out.tolist() == [1.0, 2.0, 4.0]

    def test_empty_bucket_is_nan(self) -> None:
        records = [_make_record(num_timesteps=50, episode_reward=1.0)]
        edges = [0, 100, 200]
        out = agg.bin_by_timesteps(records, edges, "episode_reward")
        assert out[0] == 1.0
        assert math.isnan(out[1])

    def test_rate_aggregator(self) -> None:
        records = [
            _make_record(num_timesteps=50, compromised=True),
            _make_record(num_timesteps=60, compromised=False),
            _make_record(num_timesteps=70, compromised=True),
        ]
        out = agg.bin_by_timesteps(records, [0, 100], "compromised", aggregator="rate")
        assert out[0] == pytest.approx(2.0 / 3.0)

    def test_skips_none_values_for_mttc(self) -> None:
        records = [
            _make_record(num_timesteps=10, mttc_steps=20),
            _make_record(num_timesteps=20, mttc_steps=None),
            _make_record(num_timesteps=30, mttc_steps=40),
        ]
        out = agg.bin_by_timesteps(records, [0, 100], "mttc_steps")
        assert out[0] == pytest.approx(30.0)

    def test_rejects_bad_edges(self) -> None:
        with pytest.raises(ValueError):
            agg.bin_by_timesteps([], [10], "x")
        with pytest.raises(ValueError):
            agg.bin_by_timesteps([], [10, 5], "x")

    def test_bucket_centers_are_midpoints(self) -> None:
        c = agg.bucket_centers([0, 10, 30])
        assert c.tolist() == [5.0, 20.0]


# --------------------------------------------------------------- bootstrap


class TestBootstrapCi:
    def test_constant_signal_has_zero_width(self) -> None:
        lo, mu, hi = agg.bootstrap_ci([5.0] * 10, n_resamples=200, seed=0)
        assert lo == mu == hi == 5.0

    def test_single_value(self) -> None:
        lo, mu, hi = agg.bootstrap_ci([3.14])
        assert lo == mu == hi == 3.14

    def test_empty_returns_nan(self) -> None:
        lo, mu, hi = agg.bootstrap_ci([])
        assert all(math.isnan(v) for v in (lo, mu, hi))

    def test_mean_inside_ci(self) -> None:
        rng = np.random.default_rng(1)
        vals = rng.normal(loc=10.0, scale=2.0, size=50).tolist()
        lo, mu, hi = agg.bootstrap_ci(vals, n_resamples=500, seed=42)
        assert lo <= mu <= hi
        # 95% CI should contain the population mean for n=50.
        assert lo <= 10.0 <= hi

    def test_aggregate_seeds(self) -> None:
        # 3 seeds, 2 buckets each.
        per_seed = [
            np.array([1.0, 4.0]),
            np.array([2.0, 5.0]),
            np.array([3.0, 6.0]),
        ]
        low, mean, high = agg.aggregate_seeds(per_seed, n_resamples=200, seed=0)
        assert mean.tolist() == pytest.approx([2.0, 5.0])
        assert (low <= mean).all() and (mean <= high).all()


# --------------------------------------------------------------- actions


class TestActionAggregation:
    def test_action_counts_by_bin_proportions(self) -> None:
        records = [
            _make_record(num_timesteps=50, action_counts=[5, 0, 0, 0, 0]),
            _make_record(num_timesteps=80, action_counts=[0, 5, 0, 0, 0]),
            _make_record(num_timesteps=150, action_counts=[0, 0, 5, 5, 0]),
        ]
        out = agg.action_counts_by_bin(records, [0, 100, 200])
        # Bucket 0: 5 OBSERVE + 5 LOG -> [0.5, 0.5, 0, 0, 0]
        assert out[0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.0, 0.0])
        # Bucket 1: 5 RESTRICT + 5 BLOCK -> [0, 0, 0.5, 0.5, 0]
        assert out[1].tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5, 0.0])

    def test_per_stage_action_distribution_hand_computed(self) -> None:
        rec = _make_record(
            num_timesteps=100,
            per_stage={
                "0": [10, 0, 0, 0, 0],
                "1": [5, 5, 0, 0, 0],
                "2": [0, 0, 8, 0, 2],
                "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
            },
        )
        out = agg.per_stage_action_distribution([rec])
        assert out[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
        assert out[1].tolist() == [0.5, 0.5, 0.0, 0.0, 0.0]
        assert out[2].tolist() == pytest.approx([0.0, 0.0, 0.8, 0.0, 0.2])
        # Stages with no decisions -> NaN row.
        assert math.isnan(out[3, 0])
        assert math.isnan(out[4, 0])

    def test_per_stage_filters_by_since_timestep(self) -> None:
        early = _make_record(
            num_timesteps=10,
            per_stage={
                "0": [10, 0, 0, 0, 0],
                "1": [0, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0],
                "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
            },
        )
        late = _make_record(
            num_timesteps=1000,
            per_stage={
                "0": [0, 0, 0, 0, 10],
                "1": [0, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0],
                "3": [0, 0, 0, 0, 0],
                "4": [0, 0, 0, 0, 0],
            },
        )
        out = agg.per_stage_action_distribution([early, late], since_timestep=500)
        # Only the late record contributes to stage 0.
        assert out[0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


# --------------------------------------------------------------- summarise


class TestSummariseLastWindow:
    def test_last_10_percent(self) -> None:
        records = [
            _make_record(
                num_timesteps=t,
                episode_reward=float(t),
                compromised=(t > 80),
                mttc_steps=20 if t > 80 else None,
            )
            for t in (10, 20, 50, 80, 90, 100)
        ]
        s = agg.summarise_last_window(records, fraction=0.10)
        # max_ts = 100, cutoff = 90 -> records with ts >= 90 are {90, 100}.
        assert s["n_episodes"] == 2
        assert s["mean_reward"] == pytest.approx(95.0)
        assert s["compromise_rate"] == pytest.approx(1.0)
        assert s["mean_mttc"] == pytest.approx(20.0)

    def test_mitigated_impact_metrics(self) -> None:
        # 4 records all in the last-10% window; 3 compromised, 2 of them
        # mitigated. Expected: compromise_rate=0.75, mit_rate=0.50,
        # mit_among_compromised=2/3.
        records = []
        for t, comp, outcome in [
            (90, True, "impact_mitigated"),
            (95, True, "impact_missed"),
            (97, True, "impact_mitigated"),
            (100, False, "ongoing"),
        ]:
            r = _make_record(num_timesteps=t, compromised=comp, mttc_steps=20 if comp else None)
            r["end_outcome"] = outcome
            records.append(r)
        s = agg.summarise_last_window(records, fraction=0.20)
        assert s["compromise_rate"] == pytest.approx(0.75)
        assert s["mitigated_impact_rate"] == pytest.approx(0.5)
        assert s["mitigated_among_compromised"] == pytest.approx(2.0 / 3.0)

    def test_empty_records(self) -> None:
        s = agg.summarise_last_window([])
        assert s["n_episodes"] == 0
        assert math.isnan(s["mean_reward"])

    def test_rejects_bad_fraction(self) -> None:
        with pytest.raises(ValueError):
            agg.summarise_last_window([_make_record(num_timesteps=1)], fraction=0.0)
        with pytest.raises(ValueError):
            agg.summarise_last_window([_make_record(num_timesteps=1)], fraction=1.5)
