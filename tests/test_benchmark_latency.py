"""Tests for src.benchmark.latency.measure_inference_latency (PLAN §3.3)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.benchmark import latency
from src.benchmark.latency import measure_inference_latency


def _trivial_policy(obs: np.ndarray, info: Dict[str, Any]) -> int:  # noqa: ARG001
    return 0


class TestMeasureInferenceLatency:
    def test_returns_correct_length(self) -> None:
        obs_pool = [np.zeros(10) for _ in range(4)]
        d = measure_inference_latency(
            _trivial_policy, obs_pool, n_warmup=5, n_measure=20,
        )
        assert d.shape == (20,)
        assert d.dtype == np.int64

    def test_durations_positive_with_real_clock(self) -> None:
        # Real-clock smoke test: durations should be non-negative;
        # exact values depend on hardware.
        obs_pool = [np.zeros(10) for _ in range(2)]
        d = measure_inference_latency(
            _trivial_policy, obs_pool, n_warmup=10, n_measure=50,
        )
        assert (d >= 0).all()
        # At least one call took non-zero time on any sane system.
        assert d.sum() > 0

    def test_warmup_samples_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Deterministic clock: each call advances by 1 ns.
        counter = {"t": 0}

        def fake_clock() -> int:
            counter["t"] += 1
            return counter["t"]

        # Track how many times the policy is called. n_warmup + n_measure
        # is the expected count.
        call_count = {"n": 0}

        def counting_policy(obs: np.ndarray, info: Dict[str, Any]) -> int:  # noqa: ARG001
            call_count["n"] += 1
            return 0

        d = measure_inference_latency(
            counting_policy,
            [np.zeros(1)],
            n_warmup=7,
            n_measure=11,
            clock=fake_clock,
        )
        # Warmup does NOT consult the clock; measure does (twice per
        # call). So the clock advanced 2 * n_measure times.
        assert d.shape == (11,)
        # Each measurement is exactly 1 ns under our fake clock
        # (one increment between t0 and t1).
        assert (d == 1).all()
        # And the policy itself was called n_warmup + n_measure times.
        assert call_count["n"] == 7 + 11

    def test_measurement_uses_default_clock_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Replace the module-level _DEFAULT_CLOCK so the no-`clock` path
        # is also covered.
        seq = iter(range(10**6))

        def fake_default() -> int:
            return next(seq)

        monkeypatch.setattr(latency, "_DEFAULT_CLOCK", fake_default)
        d = measure_inference_latency(
            _trivial_policy, [np.zeros(1)], n_warmup=2, n_measure=5,
        )
        assert d.shape == (5,)
        assert (d == 1).all()

    def test_empty_pool_raises(self) -> None:
        with pytest.raises(ValueError, match="obs_pool must contain"):
            measure_inference_latency(_trivial_policy, [])

    def test_negative_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="n_warmup must be >= 0"):
            measure_inference_latency(
                _trivial_policy, [np.zeros(1)], n_warmup=-1,
            )

    def test_zero_measure_raises(self) -> None:
        with pytest.raises(ValueError, match="n_measure must be >= 1"):
            measure_inference_latency(
                _trivial_policy, [np.zeros(1)], n_measure=0,
            )

    def test_info_pool_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="info_pool length"):
            measure_inference_latency(
                _trivial_policy,
                [np.zeros(1), np.zeros(1)],
                info_pool=[{}],
                n_measure=2,
            )

    def test_info_pool_threaded_through(self) -> None:
        # Verify the policy actually receives the info dict it expects.
        seen: list[int] = []

        def reading_policy(obs: np.ndarray, info: Dict[str, Any]) -> int:  # noqa: ARG001
            seen.append(int(info.get("decision_stage", -1)))
            return 0

        obs_pool = [np.zeros(1), np.zeros(1)]
        info_pool = [{"decision_stage": 3}, {"decision_stage": 4}]
        measure_inference_latency(
            reading_policy, obs_pool, info_pool=info_pool,
            n_warmup=0, n_measure=4,
        )
        # 4 measure calls round-robin through 2-entry pool: 0,1,0,1.
        assert seen == [3, 4, 3, 4]
