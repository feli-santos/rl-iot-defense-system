"""benchmark F7 — computational overhead (latency CDF + training time).

PLAN §3.1.7, C6. Aligned with IoTWarden Fig. 4(b).

Two-panel figure:

- **Left** — per-step inference latency CDF, one curve per policy.
  X-axis is log-scaled; the budget thresholds from G6.4 (RL ≤ 5 ms,
  RF ≤ 3 ms, rule-based ≤ 1 ms) are drawn as vertical reference
  lines. Source: ``runs/benchmark/<policy>/seed_*/latency.jsonl`` (the
  C3 sidecar — D6.4 — same data the F5 table summarised at p50/p95).

- **Right** — training wallclock per algorithm, **summed over the
  5 seeds** of the blue-team sweep. Source:
  ``runs/blue_team/sweep_manifest.json`` (the per-run ``wallclock_seconds``
  field). Non-RL baselines have zero training time and are
  intentionally absent from the right panel — F7 contrasts the
  "RL training cost" with the "rule-based zero-training" trade-off.

Outputs:
- ``F7_overhead.png``   — two-panel figure (left CDF, right bar).
- ``F7_summary.json``   — per-policy {p50, p95, p99, mean} latency in
                          ms + per-algo total training seconds.
- ``F7_manifest.json``  — SHA-256 hash chain.
- ``F7_caption.md``     — thesis caption (separate, hand-written below).

The platform fingerprint (``platform.platform()``,
``platform.processor()``, Python version) is recorded in
``F7_summary.json`` so a reader can interpret absolute latency
numbers correctly. R6.3 acknowledges 2–3× pessimism vs. server
hardware; the CDF + p99 reporting absorbs that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("scripts.benchmark.plot_overhead")

_ROOT = Path(__file__).resolve().parents[2]


# Latency budgets per policy class (G6.4 / D6.8). Used both for the
# vertical reference lines and for the per-policy pass/fail flag in
# the JSON output.
_RL_POLICIES = {"dqn", "ppo", "a2c"}
_RF_POLICIES = {"rf_acting"}
_RULE_POLICIES = {"random", "always_observe", "always_block", "recommended_action"}

_BUDGET_MS: dict[str, float] = {
    "rl": 5.0,
    "rf": 3.0,
    "rule": 1.0,
}

_DISPLAY: dict[str, str] = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "random": "Random",
    "always_observe": "Always-OBSERVE",
    "always_block": "Always-BLOCK",
    "recommended_action": "Recommended-Action",
    "rf_acting": "RF-Acting",
}

# Plot order (lighter ↔ heavier): rule first, then RL, then RF, so
# the CDF curves layer with the slowest on top.
_CDF_ORDER: list[str] = [
    "always_observe",
    "always_block",
    "recommended_action",
    "random",
    "dqn",
    "a2c",
    "ppo",
    "rf_acting",
]


def _sha256(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _budget_for(policy: str) -> float:
    """Return the G6.4 latency budget (ms) for the given policy class."""
    if policy in _RL_POLICIES:
        return _BUDGET_MS["rl"]
    if policy in _RF_POLICIES:
        return _BUDGET_MS["rf"]
    if policy in _RULE_POLICIES:
        return _BUDGET_MS["rule"]
    # Default to RL budget if unknown — caller will see "no policy
    # class match" in the summary JSON if this is ever exercised.
    return _BUDGET_MS["rl"]


def _gather_latency_ms(runs_root: Path, policy: str) -> np.ndarray:
    """Read all ``latency.jsonl`` rows for one policy, return ms."""
    base = runs_root / policy
    if not base.exists():
        return np.array([], dtype=np.float64)
    durs_ns: list[int] = []
    for sd in sorted(base.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("seed_"):
            continue
        lp = sd / "latency.jsonl"
        if not lp.exists():
            continue
        with lp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                durs_ns.append(int(json.loads(line)["duration_ns"]))
    if not durs_ns:
        return np.array([], dtype=np.float64)
    return np.asarray(durs_ns, dtype=np.float64) / 1e6


def _gather_training_seconds(
    blue_team_runs_root: Path,
) -> dict[str, float]:
    """Sum blue-team ``wallclock_seconds`` per algo from the sweep manifest.

    Returns ``{}`` (rather than raising) when the manifest is missing —
    F7 still draws the latency panel, and the right panel becomes a
    note-only annotation. This is the same robustness pattern the
    HANDOFF describes for runs/blue_team being gitignored on a fresh
    checkout.
    """
    manifest_path = blue_team_runs_root / "sweep_manifest.json"
    if not manifest_path.exists():
        logger.warning("blue-team sweep_manifest.json missing at %s", manifest_path)
        return {}
    sm = json.loads(manifest_path.read_text())
    totals: dict[str, float] = {}
    for run in sm.get("runs", []):
        if not run.get("ok"):
            continue
        algo = run.get("algo", "?")
        totals[algo] = totals.get(algo, 0.0) + float(run.get("wallclock_seconds", 0.0))
    return totals


def _quantiles(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            "p50_ms": math.nan,
            "p95_ms": math.nan,
            "p99_ms": math.nan,
            "mean_ms": math.nan,
            "n_samples": 0,
        }
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "n_samples": int(arr.size),
    }


def _render(
    latencies: dict[str, np.ndarray],
    train_secs: dict[str, float],
    out_path: Path,
) -> None:
    """Two-panel figure: left CDF, right training-time bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_cdf, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        gridspec_kw={"width_ratios": [3, 2]},
    )

    # ---- Left: latency CDF ----
    cmap = plt.get_cmap("tab10")
    colour_map: dict[str, Any] = {}
    plotted: list[str] = []
    for i, pol in enumerate([p for p in _CDF_ORDER if p in latencies]):
        arr = latencies[pol]
        if arr.size == 0:
            continue
        plotted.append(pol)
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, sorted_arr.size + 1) / sorted_arr.size
        # Use solid lines for trained-RL, dashed for everything else
        # so the eye picks out the comparison group quickly.
        ls = "-" if pol in _RL_POLICIES else ("-." if pol in _RF_POLICIES else "--")
        c = cmap(i % 10)
        colour_map[pol] = c
        ax_cdf.plot(
            sorted_arr, cdf, label=_DISPLAY.get(pol, pol), color=c, linestyle=ls, linewidth=1.6
        )
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("Per-step inference latency (ms, log scale)", fontsize=10)
    ax_cdf.set_ylabel("Empirical CDF", fontsize=10)
    ax_cdf.set_title(
        "Inference Latency Distribution per Policy (test_balanced rollouts; CPU, single process)",
        fontsize=10,
    )
    ax_cdf.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_cdf.set_ylim(0.0, 1.02)
    # Vertical reference lines at the G6.4 budgets.
    for label, ms in (
        ("rule ≤ 1 ms", _BUDGET_MS["rule"]),
        ("RF ≤ 3 ms", _BUDGET_MS["rf"]),
        ("RL ≤ 5 ms", _BUDGET_MS["rl"]),
    ):
        ax_cdf.axvline(ms, color="grey", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_cdf.text(ms, 0.04, label, rotation=90, va="bottom", ha="right", fontsize=7, color="grey")
    ax_cdf.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ---- Right: training-time bar ----
    if train_secs:
        algos_ordered = sorted(train_secs.keys(), key=lambda a: train_secs[a])
        hours = [train_secs[a] / 3600.0 for a in algos_ordered]
        labels = [_DISPLAY.get(a, a.upper()) for a in algos_ordered]
        ax_bar.barh(
            labels,
            hours,
            color=[colour_map.get(a, "gray") for a in algos_ordered],
            edgecolor="black",
            linewidth=0.6,
        )
        for y, h in enumerate(hours):
            ax_bar.text(h, y, f" {h:.2f} h", va="center", ha="left", fontsize=9)
        ax_bar.set_xlabel("Total training wallclock, summed over 5 seeds (h)", fontsize=10)
        ax_bar.set_title(
            "blue-team Training Cost per Algorithm (250 K timesteps × 5 seeds, CPU)",
            fontsize=10,
        )
        ax_bar.grid(True, axis="x", linestyle=":", alpha=0.4)
        ax_bar.set_xlim(0, max(hours) * 1.18)
    else:
        ax_bar.set_axis_off()
        ax_bar.text(
            0.5,
            0.5,
            "blue-team sweep_manifest.json not found.\nRun `make phase-5-sweep` to populate.",
            transform=ax_bar.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="dimgrey",
        )

    fig.suptitle("F7 — Computational Overhead (Inference + Training)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1.0, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="benchmark F7 — overhead figure.")
    p.add_argument("--runs-root", default="runs/benchmark")
    p.add_argument("--phase5-runs-root", default="runs/blue_team")
    p.add_argument("--out-dir", default="docs/results/benchmark")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    blue_team_runs_root = Path(args.phase5_runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover policies by directory layout, but keep the canonical
    # F5 ordering for any subset.
    available: list[str] = []
    if runs_root.exists():
        available = [d.name for d in runs_root.iterdir() if d.is_dir()]
    policies = [p for p in _CDF_ORDER if p in available]

    latencies: dict[str, np.ndarray] = {}
    in_hashes: dict[str, str] = {}
    per_policy_summary: dict[str, Any] = {}
    for pol in policies:
        arr = _gather_latency_ms(runs_root, pol)
        latencies[pol] = arr
        q = _quantiles(arr)
        budget = _budget_for(pol)
        per_policy_summary[pol] = {
            **q,
            "budget_ms": budget,
            "g64_pass": bool(np.isfinite(q["p50_ms"]) and q["p50_ms"] <= budget),
            "policy_class": (
                "rl" if pol in _RL_POLICIES else ("rf" if pol in _RF_POLICIES else "rule")
            ),
        }
        # Track input file hashes.
        for sd in sorted((runs_root / pol).iterdir()):
            if not sd.is_dir() or not sd.name.startswith("seed_"):
                continue
            lp = sd / "latency.jsonl"
            sha = _sha256(lp)
            if sha:
                in_hashes[str(lp)] = sha
        logger.info(
            "%s: p50=%s ms p95=%s ms p99=%s ms n=%s budget=%s ms pass=%s",
            pol,
            f"{q['p50_ms']:.3f}" if np.isfinite(q["p50_ms"]) else "—",
            f"{q['p95_ms']:.3f}" if np.isfinite(q["p95_ms"]) else "—",
            f"{q['p99_ms']:.3f}" if np.isfinite(q["p99_ms"]) else "—",
            q["n_samples"],
            budget,
            per_policy_summary[pol]["g64_pass"],
        )

    train_secs = _gather_training_seconds(blue_team_runs_root)
    sweep_manifest_path = blue_team_runs_root / "sweep_manifest.json"

    png_path = out_dir / "F7_overhead.png"
    _render(latencies, train_secs, png_path)

    summary = {
        "schema_version": "1.0",
        "phase": 6,
        "figure": "F7",
        "platform": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "g64_thresholds_ms": _BUDGET_MS,
        "policies": per_policy_summary,
        "blue_team_training_seconds_per_algo": {algo: float(s) for algo, s in train_secs.items()},
        "blue_team_training_hours_per_algo": {
            algo: float(s) / 3600.0 for algo, s in train_secs.items()
        },
    }
    (out_dir / "F7_summary.json").write_text(json.dumps(summary, indent=2))

    eval_manifest_path = runs_root / "eval_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F7",
        "git_sha": _git_sha(),
        "outputs": {
            "png": str(png_path),
            "json": str(out_dir / "F7_summary.json"),
        },
        "inputs": {
            "benchmark_eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
            "blue_team_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            "latency_jsonl_sha256": in_hashes,
        },
    }
    (out_dir / "F7_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("F7 written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
