"""F3 — RL episodic reward / MTTC / compromise-rate curves.

PLAN §3.1.8, D5.8, D5.9. Reads every
``runs/<root>/<algo>/seed_*/episodes.jsonl`` and ``eval.jsonl`` and
renders a 3-panel figure with mean +/- 95 % bootstrap CI bands per
algo. The eval reward is overlaid as a dotted line.

Usage::

    python -m scripts.blue_team.plot_learning_curves \\
        --runs-root runs/blue_team \\
        --out-dir docs/results/05_blue_team \\
        [--n-bins 25] [--bootstrap 1000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.blue_team.aggregation import (  # noqa: E402
    aggregate_seeds,
    bin_by_timesteps,
    bucket_centers,
    read_runs_directory,
    summarise_last_window,
)

logger = logging.getLogger("scripts.blue_team.plot_learning_curves")


# Per-algo line colours, matched in F4.
_ALGO_COLORS = {"dqn": "#d62728", "ppo": "#1f77b4", "a2c": "#2ca02c"}


def _git_sha() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=_ROOT)
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT).decode().strip()
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_curves(
    records: Sequence[dict],
    edges: Sequence[int],
    *,
    metric_keys: Sequence[str],
    aggregators: Sequence[str],
) -> dict[str, np.ndarray]:
    """Bin a single seed's records into per-metric per-bucket curves."""
    out: dict[str, np.ndarray] = {}
    for k, agg in zip(metric_keys, aggregators):
        out[k] = bin_by_timesteps(records, edges, k, aggregator=agg)
    return out


def render(
    runs_root: Path,
    out_dir: Path,
    *,
    n_bins: int = 25,
    bootstrap_n: int = 1000,
    fraction: float = 0.10,
) -> dict[str, Any]:
    """Build F3 + F3_summary.json + manifest.json under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_runs = read_runs_directory(runs_root, file_name="episodes.jsonl")
    eval_runs = read_runs_directory(runs_root, file_name="eval.jsonl")
    if not train_runs:
        raise RuntimeError(f"no training runs found under {runs_root}")

    # Establish the global timestep horizon from the longest seen run.
    max_ts = 0
    for recs in train_runs.values():
        if recs:
            max_ts = max(max_ts, max(r["num_timesteps"] for r in recs))
    if max_ts <= 0:
        raise RuntimeError(f"all training runs under {runs_root} are empty")
    edges = np.linspace(0, max_ts, n_bins + 1, dtype=np.int64).tolist()
    centers = bucket_centers(edges)

    # Collate per-algo per-seed curves.
    # Per D5.10.1 the third panel is "mitigated-impact rate" (a derived
    # boolean from end_outcome), not unconditional compromise rate.
    # We pre-compute the derived field on the records before binning.
    for recs in list(train_runs.values()) + list(eval_runs.values()):
        for r in recs:
            r.setdefault(
                "impact_mitigated",
                r.get("end_outcome") == "impact_mitigated",
            )
    metric_keys = ("episode_reward", "mttc_steps", "impact_mitigated")
    aggregators = ("mean", "mean", "rate")
    panels: dict[str, dict[str, dict[str, np.ndarray]]] = {
        k: {} for k in metric_keys  # algo -> {low, mean, high}
    }
    eval_panels: dict[str, dict[str, dict[str, np.ndarray]]] = {k: {} for k in metric_keys}
    summary: dict[str, Any] = {
        "version": "1.0",
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_root": str(runs_root),
        "n_bins": n_bins,
        "bootstrap_n": bootstrap_n,
        "last_window_fraction": fraction,
        "max_timesteps": int(max_ts),
        "algos": {},
    }

    algos = sorted({a for (a, _) in train_runs})
    for algo in algos:
        # Gather all seeds for this algo.
        seeds_train = sorted(s for (a, s) in train_runs if a == algo)
        seeds_eval = sorted(s for (a, s) in eval_runs if a == algo)
        per_seed_train: dict[str, list[np.ndarray]] = {k: [] for k in metric_keys}
        per_seed_eval: dict[str, list[np.ndarray]] = {k: [] for k in metric_keys}

        last_window_seeds: list[dict[str, Any]] = []
        last_window_seeds_eval: list[dict[str, Any]] = []

        for seed in seeds_train:
            recs = train_runs[(algo, seed)]
            curves = _build_curves(recs, edges, metric_keys=metric_keys, aggregators=aggregators)
            for k in metric_keys:
                per_seed_train[k].append(curves[k])
            last_window_seeds.append(summarise_last_window(recs, fraction=fraction))
        for seed in seeds_eval:
            recs = eval_runs[(algo, seed)]
            curves = _build_curves(recs, edges, metric_keys=metric_keys, aggregators=aggregators)
            for k in metric_keys:
                per_seed_eval[k].append(curves[k])
            last_window_seeds_eval.append(summarise_last_window(recs, fraction=fraction))

        # Aggregate across seeds.
        for k in metric_keys:
            lo, mu, hi = aggregate_seeds(per_seed_train[k], n_resamples=bootstrap_n, seed=0)
            panels[k][algo] = {"low": lo, "mean": mu, "high": hi}
            if per_seed_eval[k]:
                lo_e, mu_e, hi_e = aggregate_seeds(
                    per_seed_eval[k], n_resamples=bootstrap_n, seed=0
                )
                eval_panels[k][algo] = {"low": lo_e, "mean": mu_e, "high": hi_e}

        # Last-window scalars (aggregated across seeds).
        def _agg_lw(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
            vals = [it[key] for it in items if it[key] == it[key]]  # NaN-safe
            if not vals:
                return {"mean": float("nan"), "values": []}
            return {"mean": float(np.mean(vals)), "values": [float(v) for v in vals]}

        _summary_keys = (
            "mean_reward",
            "mean_mttc",
            "compromise_rate",
            "mitigated_impact_rate",
            "mitigated_among_compromised",
        )
        summary["algos"][algo] = {
            "n_seeds_train": len(seeds_train),
            "n_seeds_eval": len(seeds_eval),
            "train_last_window": {k: _agg_lw(last_window_seeds, k) for k in _summary_keys},
            "eval_last_window": {k: _agg_lw(last_window_seeds_eval, k) for k in _summary_keys},
        }

    # ---------- render --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    metric_titles = {
        "episode_reward": ("Mean episodic reward", "reward"),
        "mttc_steps": ("Mean Time-To-Compromise (MTTC)", "steps"),
        "impact_mitigated": ("Mitigated-impact rate (per episode)", "fraction"),
    }
    for ax, k in zip(axes, metric_keys):
        for algo, agg in panels[k].items():
            color = _ALGO_COLORS.get(algo, "k")
            mean = agg["mean"]
            low = agg["low"]
            high = agg["high"]
            valid = np.isfinite(mean)
            ax.plot(centers[valid], mean[valid], color=color, label=f"{algo.upper()} (train)", lw=2)
            ax.fill_between(centers[valid], low[valid], high[valid], color=color, alpha=0.18)
        # Eval overlay (dotted).
        for algo, agg in eval_panels[k].items():
            color = _ALGO_COLORS.get(algo, "k")
            mean = agg["mean"]
            valid = np.isfinite(mean)
            ax.plot(
                centers[valid],
                mean[valid],
                color=color,
                ls=":",
                lw=2,
                label=f"{algo.upper()} (eval)",
            )
        title, ylabel = metric_titles[k]
        ax.set_title(title)
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    n_seeds_summary = sum(
        len(seeds_train)
        for seeds_train in [sorted(s for (a, s) in train_runs if a == algo) for algo in algos]
    )
    fig.suptitle(
        f"F3 — RL learning curves "
        f"(DQN/PPO/A2C, {len(algos)} algos × {n_seeds_summary} seeds total, "
        f"{n_bins} time bins; bands = 95 % bootstrap CI across seeds)",
        y=1.02,
    )
    fig.tight_layout()

    fig_path = out_dir / "F3_learning_curves.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", fig_path)

    # F3_summary.json
    summary_path = out_dir / "F3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", summary_path)

    # manifest.json — hash chain pinned to inputs
    inputs: dict[str, str] = {}
    for (a, s), _recs in {**train_runs}.items():
        p = runs_root / a / f"seed_{s}" / "episodes.jsonl"
        if p.exists():
            inputs[str(p.relative_to(_ROOT) if p.is_absolute() else p)] = _sha256(p)
    for (a, s), _recs in {**eval_runs}.items():
        p = runs_root / a / f"seed_{s}" / "eval.jsonl"
        if p.exists():
            inputs[str(p.relative_to(_ROOT) if p.is_absolute() else p)] = _sha256(p)
    manifest = {
        "version": "1.0",
        "git_sha": summary["git_sha"],
        "timestamp": summary["timestamp"],
        "produced_by": "scripts/blue_team/plot_learning_curves.py",
        "inputs": inputs,
        "outputs": {
            str(fig_path.relative_to(out_dir)): _sha256(fig_path),
            str(summary_path.relative_to(out_dir)): _sha256(summary_path),
        },
    }
    manifest_path = out_dir / "F3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", manifest_path)

    return {
        "fig_path": str(fig_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render F3 learning curves.")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out-dir", default="docs/results/05_blue_team")
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--fraction", type=float, default=0.10)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    render(
        runs_root=Path(args.runs_root),
        out_dir=Path(args.out_dir),
        n_bins=args.n_bins,
        bootstrap_n=args.bootstrap_n,
        fraction=args.fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
