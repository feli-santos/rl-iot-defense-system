"""Inference-latency micro-benchmark for Phase 6 (PLAN §3.1.3).

The Phase-6 F7 figure compares per-step inference cost across the eight
policies. Real wall-time measurements are noisy on commodity hardware,
so this module follows three principles:

1. **Discard warmup.** ``n_warmup`` calls are run before measurement
   begins so JIT cache, branch predictor, and any lazy ``predict``
   path are warm. The returned array contains only the
   post-warmup samples.

2. **Sample, don't average.** We return a 1-D ``np.ndarray`` of
   per-call durations (in nanoseconds, ``np.int64``). The plot script
   computes median / p95 / p99 / CDF from this array — averaging
   inside the measurement function would discard exactly the
   distributional information F7 visualises.

3. **Test-isolated clock.** ``time.perf_counter_ns`` is read through a
   single function reference (``_DEFAULT_CLOCK``) so tests can
   monkey-patch it with a deterministic counter and assert that the
   warmup / measure split is correct without depending on actual
   wall time.

Public API: :func:`measure_inference_latency`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

# Module-level reference so tests can monkey-patch
# ``src.benchmark.latency._DEFAULT_CLOCK`` for deterministic ns sampling.
_DEFAULT_CLOCK: Callable[[], int] = time.perf_counter_ns


def measure_inference_latency(
    policy_callable: Callable[[np.ndarray, dict[str, Any]], int],
    obs_pool: Sequence[np.ndarray],
    *,
    info_pool: Sequence[dict[str, Any]] | None = None,
    n_warmup: int = 100,
    n_measure: int = 1000,
    clock: Callable[[], int] | None = None,
) -> np.ndarray:
    """Run ``policy_callable`` repeatedly on ``obs_pool`` and return
    per-call wall-time durations in nanoseconds.

    The function indexes ``obs_pool`` (and ``info_pool`` if provided) in
    a deterministic round-robin so different runs over the same
    ``obs_pool`` produce comparable distributions. The total number of
    calls is ``n_warmup + n_measure``; only the trailing ``n_measure``
    durations are returned.

    Args:
        policy_callable: Anything with the
            ``(obs, info) -> int`` signature (a baseline policy or an
            :class:`SB3PolicyAdapter`).
        obs_pool: Pre-computed observations to feed the policy. Must
            have at least one entry. The pool is cycled — for
            ``len(obs_pool) >= n_warmup + n_measure`` every sample is
            unique; for shorter pools the same observations recur (this
            is fine for latency measurement, which only depends on
            input shape and dtype, not on input distribution).
        info_pool: Optional matching pool of ``info`` dicts. If
            ``None``, every call gets ``{}`` — appropriate for SB3,
            random, and constant policies; **not** appropriate for
            :func:`recommended_action_policy` or :class:`RFActingPolicy`
            mid-episode (callers must supply infos with
            ``recommended_action`` set when measuring those).
        n_warmup: Number of pre-measurement calls discarded.
        n_measure: Number of post-warmup calls timed and returned.
        clock: Override for ``time.perf_counter_ns``. Tests pass a
            deterministic counter; production passes ``None``.

    Returns:
        ``np.ndarray`` of shape ``(n_measure,)`` and dtype ``int64``
        carrying per-call durations in nanoseconds.

    Raises:
        ValueError: if ``obs_pool`` is empty, or ``n_warmup`` /
            ``n_measure`` is negative.
    """
    if not obs_pool:
        raise ValueError("obs_pool must contain at least one observation")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    if n_measure < 1:
        raise ValueError(f"n_measure must be >= 1, got {n_measure}")
    clk = clock if clock is not None else _DEFAULT_CLOCK
    pool_n = len(obs_pool)

    # Pre-fetch infos so we don't pay dict-construction cost inside the
    # hot loop (it would bias the measurement upward by tens of ns).
    infos: Sequence[dict[str, Any]]
    if info_pool is not None:
        if len(info_pool) != pool_n:
            raise ValueError(
                f"info_pool length {len(info_pool)} does not match obs_pool length {pool_n}"
            )
        infos = info_pool
    else:
        empty: dict[str, Any] = {}
        infos = [empty] * pool_n

    # ---- warmup ----
    for i in range(n_warmup):
        idx = i % pool_n
        policy_callable(obs_pool[idx], infos[idx])

    # ---- measure ----
    durations = np.empty(n_measure, dtype=np.int64)
    for i in range(n_measure):
        idx = (i + n_warmup) % pool_n
        t0 = clk()
        policy_callable(obs_pool[idx], infos[idx])
        t1 = clk()
        durations[i] = t1 - t0
    return durations


__all__ = ["measure_inference_latency"]
