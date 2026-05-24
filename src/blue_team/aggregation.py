"""Pure-Python aggregation utilities for blue-team figures.

Reading and aggregating ``episodes.jsonl`` and ``eval.jsonl`` is split
out of the plot scripts because

1. The same routines are used by F3 and F4 (and later by ablation's
   benchmark plots), so DRYing the I/O matters.
2. Unit-testing aggregation is easier when it does not depend on a
   matplotlib import path or a ``runs/`` directory layout.

Public API:

- :func:`read_episodes_jsonl(path)` → ``List[Dict]``: one dict per
  episode, validated against the callback schema version.
- :func:`bin_by_timesteps(records, edges, key)`:
    Group records into [edges[i], edges[i+1]) buckets by
    ``num_timesteps`` and return per-bucket means of ``key``. Used to
    produce reward / MTTC / compromise-rate curves.
- :func:`bootstrap_ci(values, n_resamples, alpha, seed)`:
    Percentile bootstrap CI. Returns ``(low, mean, high)``.
- :func:`action_counts_by_bin(records, edges)`:
    Sum ``action_counts`` per timestep-bucket and return marginal
    proportions of each action; produces the F4 stacked-area curve.
- :func:`per_stage_action_distribution(records, since_timestep)`:
    Sum ``action_counts_by_stage`` over records whose
    ``num_timesteps >= since_timestep`` and return per-stage action
    proportions; this is the G5.5 input.

All functions are pure (no I/O beyond the explicit ``read_*``).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


_EXPECTED_SCHEMA = "1.0"


# -------------------------------------------------------------------------- I/O


def read_episodes_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Read an ``episodes.jsonl`` file into a list of dicts.

    Asserts every record carries ``schema_version == "1.0"``; raises
    ``ValueError`` on the first violation. Empty files return an empty
    list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {p}")
    rows: List[Dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            v = rec.get("schema_version")
            if v != _EXPECTED_SCHEMA:
                raise ValueError(
                    f"{p}:{lineno}: unexpected schema_version {v!r}; "
                    f"expected {_EXPECTED_SCHEMA!r}"
                )
            rows.append(rec)
    return rows


def read_runs_directory(
    runs_dir: Union[str, Path],
    *,
    file_name: str = "episodes.jsonl",
) -> Dict[Tuple[str, int], List[Dict]]:
    """Read every ``runs/<algo>/seed_<k>/<file_name>`` under ``runs_dir``.

    Returns a mapping ``(algo, seed) -> records``. Skips any algo or
    seed whose JSONL is missing (with a logged warning).
    """
    base = Path(runs_dir)
    out: Dict[Tuple[str, int], List[Dict]] = {}
    if not base.exists():
        return out
    for algo_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        algo = algo_dir.name
        for seed_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            jsonl = seed_dir / file_name
            if not jsonl.exists():
                logger.warning("missing %s under %s", file_name, seed_dir)
                continue
            out[(algo, seed)] = read_episodes_jsonl(jsonl)
    return out


# ------------------------------------------------------------------ binning


def bin_by_timesteps(
    records: Sequence[Dict],
    edges: Sequence[int],
    key: str,
    *,
    aggregator: str = "mean",
) -> np.ndarray:
    """Group ``records`` into ``[edges[i], edges[i+1])`` buckets by
    ``num_timesteps`` and aggregate ``records[i][key]``.

    Args:
        records: list-of-dicts as returned by :func:`read_episodes_jsonl`.
        edges: monotonic-increasing list of bucket edges (inclusive
            left, exclusive right). Length ``B+1`` for ``B`` buckets.
        key: which scalar key to aggregate.
        aggregator: ``"mean"`` (default) or ``"rate"`` (proportion of
            truthy values, useful for ``compromised``).

    Returns:
        Array of length ``B`` of bucket aggregates. Empty buckets get
        ``np.nan`` (caller decides whether to drop or interpolate).
    """
    if len(edges) < 2:
        raise ValueError("edges must have length >= 2")
    edges_arr = np.asarray(edges, dtype=np.int64)
    if not np.all(np.diff(edges_arr) > 0):
        raise ValueError("edges must be strictly increasing")
    n_buckets = len(edges_arr) - 1
    out = np.full(n_buckets, np.nan, dtype=np.float64)
    if not records:
        return out

    timesteps = np.asarray([r["num_timesteps"] for r in records], dtype=np.int64)
    raw_vals = [r.get(key) for r in records]

    bucket_idx = np.searchsorted(edges_arr, timesteps, side="right") - 1

    for b in range(n_buckets):
        sel = bucket_idx == b
        if not np.any(sel):
            continue
        bucket_vals = [raw_vals[i] for i in np.where(sel)[0]]
        if aggregator == "rate":
            out[b] = float(
                np.mean([1.0 if v else 0.0 for v in bucket_vals])
            )
        elif aggregator == "mean":
            # Skip None values (e.g., mttc_steps when no compromise).
            scalars = [float(v) for v in bucket_vals if v is not None]
            out[b] = float(np.mean(scalars)) if scalars else np.nan
        else:
            raise ValueError(f"unknown aggregator {aggregator!r}")
    return out


def bucket_centers(edges: Sequence[int]) -> np.ndarray:
    """Return the midpoints of consecutive ``edges``."""
    e = np.asarray(edges, dtype=np.float64)
    return 0.5 * (e[:-1] + e[1:])


# ------------------------------------------------------------------ bootstrap


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI on a 1-D array.

    Args:
        values: scalars (e.g., last-10%-mean reward across 5 seeds).
        n_resamples: bootstrap iterations.
        alpha: significance level (returns (1-alpha) CI).
        seed: RNG seed for reproducibility. ``None`` for random.

    Returns:
        ``(low, mean, high)``. If ``values`` is empty returns
        ``(nan, nan, nan)``. If ``len(values) == 1`` returns
        ``(v, v, v)`` (CI is a point).
    """
    arr = np.asarray([v for v in values if v is not None and not _isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(np.mean(arr))
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    n = arr.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(arr[idx].mean())
    low = float(np.percentile(boot_means, 100 * alpha / 2))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (low, mean, high)


def _isnan(v: float) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def aggregate_seeds(
    per_seed_curves: Sequence[np.ndarray],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a list of per-seed bucket-curves and compute per-bucket
    bootstrap CIs across seeds.

    Args:
        per_seed_curves: list of length ``S`` (seeds), each entry
            an array of length ``B`` (buckets). Entries may carry NaNs
            (empty buckets); they are dropped per-bucket before the
            bootstrap.

    Returns:
        ``(low, mean, high)`` arrays each of length ``B``.
    """
    if not per_seed_curves:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    stacked = np.vstack(per_seed_curves)
    n_buckets = stacked.shape[1]
    low = np.full(n_buckets, np.nan)
    mean = np.full(n_buckets, np.nan)
    high = np.full(n_buckets, np.nan)
    for b in range(n_buckets):
        col = stacked[:, b]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        lo, mu, hi = bootstrap_ci(
            finite.tolist(),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
        low[b], mean[b], high[b] = lo, mu, hi
    return low, mean, high


# ------------------------------------------------------------------ actions


def action_counts_by_bin(
    records: Sequence[Dict],
    edges: Sequence[int],
) -> np.ndarray:
    """Sum per-action counts over records bucketed by ``num_timesteps``.

    Returns:
        Array of shape ``(B, 5)`` of *proportions* (rows sum to 1),
        with rows of NaN for empty buckets.
    """
    edges_arr = np.asarray(edges, dtype=np.int64)
    n_buckets = len(edges_arr) - 1
    out = np.full((n_buckets, 5), np.nan, dtype=np.float64)
    if not records:
        return out
    ts = np.asarray([r["num_timesteps"] for r in records], dtype=np.int64)
    bucket_idx = np.searchsorted(edges_arr, ts, side="right") - 1
    counts = np.zeros((n_buckets, 5), dtype=np.int64)
    for r, b in zip(records, bucket_idx):
        if b < 0 or b >= n_buckets:
            continue
        counts[b] += np.asarray(r.get("action_counts", [0] * 5), dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    nz = totals[:, 0] > 0
    if np.any(nz):
        out[nz] = counts[nz] / totals[nz]
    return out


def per_stage_action_distribution(
    records: Sequence[Dict],
    *,
    since_timestep: Optional[int] = None,
) -> np.ndarray:
    """Per-decision-stage action distribution across selected records.

    Args:
        records: list-of-dicts as returned by :func:`read_episodes_jsonl`.
        since_timestep: include only records with
            ``num_timesteps >= since_timestep``. ``None`` uses all.

    Returns:
        Array of shape ``(5, 5)``; ``out[s, a]`` is the fraction of
        decisions at stage ``s`` that chose action ``a``. Rows sum to
        1 (or NaN if no decisions ever happened at that stage).
    """
    counts = np.zeros((5, 5), dtype=np.int64)
    for r in records:
        if since_timestep is not None and r.get("num_timesteps", 0) < since_timestep:
            continue
        per_stage = r.get("action_counts_by_stage", {})
        for s_key, vec in per_stage.items():
            try:
                s = int(s_key)
            except ValueError:
                continue
            if 0 <= s < 5 and len(vec) == 5:
                counts[s] += np.asarray(vec, dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    out = np.full((5, 5), np.nan, dtype=np.float64)
    nz = totals[:, 0] > 0
    if np.any(nz):
        out[nz] = counts[nz] / totals[nz]
    return out


# ------------------------------------------------------------------ summaries


def summarise_last_window(
    records: Sequence[Dict],
    *,
    fraction: float = 0.10,
) -> Dict[str, float]:
    """Compute the headline gate metrics over the last ``fraction`` of
    training timesteps.

    Returns a dict with:

    - ``mean_reward``: arithmetic mean of ``episode_reward``.
    - ``mean_mttc``: arithmetic mean of ``mttc_steps`` over episodes
      where it is not ``None``. ``NaN`` if every episode is
      uncompromised.
    - ``compromise_rate``: fraction of episodes with ``compromised=True``.
    - ``mitigated_impact_rate``: fraction of episodes with
      ``end_outcome == "impact_mitigated"``. Per PLAN §8 D5.4.1 this
      is the gated quantity (G5.4); ``compromise_rate`` is reported
      for completeness but no longer gated.
    - ``mitigated_among_compromised``: fraction of *compromised*
      episodes that ended in ``impact_mitigated``. Useful when
      compromise rate is near 1 (typical with the upper-triangular
      LSTM) — answers "given that IMPACT happened, how often did the
      agent block it?".
    - ``n_episodes``, ``last_window_start``: bookkeeping.

    Empty if ``records`` is empty.
    """
    if not records:
        return {
            "mean_reward": math.nan, "mean_mttc": math.nan,
            "compromise_rate": math.nan, "mitigated_impact_rate": math.nan,
            "mitigated_among_compromised": math.nan,
            "n_episodes": 0, "last_window_start": 0,
        }
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    max_ts = max(r["num_timesteps"] for r in records)
    cutoff = int(max_ts * (1.0 - fraction))
    sel = [r for r in records if r["num_timesteps"] >= cutoff]
    if not sel:
        return {
            "mean_reward": math.nan, "mean_mttc": math.nan,
            "compromise_rate": math.nan, "mitigated_impact_rate": math.nan,
            "mitigated_among_compromised": math.nan,
            "n_episodes": 0, "last_window_start": cutoff,
        }
    rewards = [r["episode_reward"] for r in sel]
    mttc_vals = [r["mttc_steps"] for r in sel if r.get("mttc_steps") is not None]
    compromised = [1.0 if r.get("compromised") else 0.0 for r in sel]
    mitigated = [
        1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0
        for r in sel
    ]
    n_compromised = int(sum(compromised))
    if n_compromised > 0:
        mit_among_comp = float(
            sum(
                1.0
                for r in sel
                if r.get("compromised")
                and r.get("end_outcome") == "impact_mitigated"
            )
            / n_compromised
        )
    else:
        mit_among_comp = math.nan

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_mttc": float(np.mean(mttc_vals)) if mttc_vals else math.nan,
        "compromise_rate": float(np.mean(compromised)),
        "mitigated_impact_rate": float(np.mean(mitigated)),
        "mitigated_among_compromised": mit_among_comp,
        "n_episodes": len(sel),
        "last_window_start": cutoff,
    }


__all__ = [
    "action_counts_by_bin",
    "aggregate_seeds",
    "bin_by_timesteps",
    "bootstrap_ci",
    "bucket_centers",
    "per_stage_action_distribution",
    "read_episodes_jsonl",
    "read_runs_directory",
    "summarise_last_window",
]
