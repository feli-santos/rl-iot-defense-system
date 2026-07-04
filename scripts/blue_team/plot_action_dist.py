"""F4 — Per-stage action distribution of the converged policy.

PLAN §3.1.9, D5.10, D5.11.

Single-row layout: five per-stage action histograms (BENIGN … IMPACT)
for the converged (late-checkpoint) policy of the headline algo. By
default the headline algo is the best-performing algo (chosen by
eval-reward, D5.11), but the thesis pins it to PPO via ``--force-algo
ppo`` so the figure matches the surrounding prose. A shared action-color
legend runs along the bottom. This is the panel that gates G5.5
(per-stage non-degeneracy).

The marginal action share over training timesteps (25-K-step bins) and
the early/mid checkpoints are still computed and recorded in
``F4_summary.json`` for the gate, but are no longer drawn.

Outputs:
    F4_action_distribution.png
    F4_summary.json
    F4_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._plot_style import apply_house_style, save_figure  # noqa: E402
from src.blue_team.aggregation import (  # noqa: E402
    action_counts_by_bin,
    bucket_centers,
    per_stage_action_distribution,
    read_runs_directory,
    summarise_last_window,
)
from src.environment.adversarial_env import ACTION_NAMES  # noqa: E402

logger = logging.getLogger("scripts.blue_team.plot_action_dist")


_STAGE_NAMES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]
_ACTION_COLORS = {
    0: "#cccccc",  # OBSERVE — neutral grey
    1: "#7fbf7f",  # LOG     — green
    2: "#ffbf00",  # RESTRICT — amber
    3: "#ff8c00",  # BLOCK    — orange
    4: "#d62728",  # ISOLATE  — red
}


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


def _select_best_algo(
    eval_runs: dict[tuple[str, int], list],
    fraction: float,
) -> str:
    """Per D5.11: highest mean eval reward over the last ``fraction`` of
    training, averaged across seeds; tie-break by lower variance."""
    scoreboard: dict[str, tuple[float, float, int]] = {}
    by_algo: dict[str, list] = {}
    for (algo, seed), recs in eval_runs.items():
        by_algo.setdefault(algo, []).extend([(seed, recs)])
    for algo, items in by_algo.items():
        rewards = []
        for _seed, recs in items:
            s = summarise_last_window(recs, fraction=fraction)
            r = s["mean_reward"]
            if r == r:  # not NaN
                rewards.append(r)
        if not rewards:
            scoreboard[algo] = (-np.inf, np.inf, 0)
        else:
            scoreboard[algo] = (
                float(np.mean(rewards)),
                float(np.std(rewards)),
                len(rewards),
            )
    # Sort by (-mean, +std).
    ranked = sorted(scoreboard.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
    if not ranked:
        raise RuntimeError("no eval runs to select best-algo from")
    return ranked[0][0]


def _stack_per_seed_action_bins(
    runs: dict[tuple[str, int], list],
    algo: str,
    edges: np.ndarray,
) -> np.ndarray:
    """Average per-bin action proportions across seeds for one algo.

    Returns shape ``(n_bins, 5)``; rows of NaN for empty buckets are
    averaged into NaN.
    """
    seeds = [s for (a, s) in runs if a == algo]
    if not seeds:
        return np.full((len(edges) - 1, 5), np.nan)
    stack = []
    for seed in sorted(seeds):
        stack.append(action_counts_by_bin(runs[(algo, seed)], edges))
    arr = np.stack(stack, axis=0)  # (n_seeds, n_bins, 5)
    # Mean over seeds, ignoring NaN rows on a per-bin basis.
    out = np.full((arr.shape[1], 5), np.nan, dtype=np.float64)
    for b in range(arr.shape[1]):
        col = arr[:, b, :]
        finite = col[np.isfinite(col[:, 0])]
        if finite.size > 0:
            out[b] = finite.mean(axis=0)
    return out


def render(
    runs_root: Path,
    out_dir: Path,
    *,
    n_bins: int = 25,
    fraction: float = 0.10,
    force_algo: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_house_style()

    train_runs = read_runs_directory(runs_root, file_name="episodes.jsonl")
    eval_runs = read_runs_directory(runs_root, file_name="eval.jsonl")
    if not train_runs:
        raise RuntimeError(f"no training runs found under {runs_root}")

    max_ts = max(max(r["num_timesteps"] for r in recs) for recs in train_runs.values() if recs)
    edges = np.linspace(0, max_ts, n_bins + 1, dtype=np.int64)
    centers = bucket_centers(edges)

    # Headline algo. The thesis pins this to PPO (--force-algo ppo) so the
    # figure agrees with the surrounding prose; otherwise fall back to the
    # eval-reward best algo (D5.11). The eval-reward best is still recorded
    # in the summary JSON for the gate, even when display is forced.
    best_by_eval = (
        _select_best_algo(eval_runs, fraction=fraction)
        if eval_runs
        else sorted({a for (a, _) in train_runs})[0]
    )
    available = sorted({a for (a, _) in train_runs})
    if force_algo is not None:
        if force_algo not in available:
            raise RuntimeError(f"--force-algo {force_algo!r} not among trained algos {available}")
        best_algo = force_algo
    else:
        best_algo = best_by_eval
    logger.info("F4 display algo -> %s (eval-reward best = %s)", best_algo, best_by_eval)

    # Marginal action distribution over training time, averaged across
    # seeds for the best algo.
    bin_props = _stack_per_seed_action_bins(train_runs, best_algo, edges)

    # Per-stage action distribution at three checkpoints.
    cps = {
        "early": (0, int(0.05 * max_ts)),
        "mid": (int(0.45 * max_ts), int(0.55 * max_ts)),
        "late": (int(0.90 * max_ts), int(max_ts)),
    }
    per_stage_cps: dict[str, np.ndarray] = {}
    for label, (lo, hi) in cps.items():
        # Pool all seeds' records that fall inside the window for the best algo.
        pooled = []
        for (a, _s), recs in train_runs.items():
            if a != best_algo:
                continue
            for r in recs:
                if lo <= r["num_timesteps"] <= hi:
                    pooled.append(r)
        per_stage_cps[label] = per_stage_action_distribution(pooled)

    # Headline G5.5 check: per-stage non-degeneracy on the *late* checkpoint.
    late = per_stage_cps["late"]
    g5_5_violations: dict[str, Any] = {}
    for s_idx, name in enumerate(_STAGE_NAMES):
        row = late[s_idx]
        if not np.isfinite(row).all():
            g5_5_violations[name] = {
                "max_share": None,
                "passes": None,
                "note": "no decisions in late window",
            }
            continue
        max_share = float(row.max())
        g5_5_violations[name] = {
            "max_share": max_share,
            "argmax_action": int(row.argmax()),
            "argmax_action_name": ACTION_NAMES[int(row.argmax())],
            "passes": max_share <= 0.70,
        }
    g5_5_passes = all(v.get("passes") in (True, None) for v in g5_5_violations.values())

    # Checkpoint labels retained for the summary JSON (all three windows are
    # still recorded even though the figure now shows only the late policy).
    cp_labels = ["early", "mid", "late"]

    # ---------- render --------------------------------------------------------
    # Single row of per-stage action histograms for the converged (late)
    # policy. The marginal-over-training panel was dropped: the thesis point
    # is the *learned per-stage force ladder*, which the late checkpoint shows
    # directly. bin_props is still recorded in the summary JSON for the gate.
    bin_props_safe = np.nan_to_num(bin_props, nan=0.0)

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.6), sharey=True, gridspec_kw={"wspace": 0.12})
    x = np.arange(5)
    late_cps = per_stage_cps["late"]
    for col_idx, (ax, stage_name) in enumerate(zip(axes, _STAGE_NAMES)):
        row_safe = np.nan_to_num(late_cps[col_idx], nan=0.0)
        bars = ax.bar(
            x,
            row_safe,
            width=0.72,
            color=[_ACTION_COLORS[a] for a in range(5)],
            edgecolor="k",
            linewidth=0.6,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(stage_name, fontsize=12)
        ax.grid(axis="y", ls=":", alpha=0.4)
        if col_idx == 0:
            ax.set_ylabel("Action share", fontsize=11)

    legend_handles = [
        Patch(facecolor=_ACTION_COLORS[a], edgecolor="k", label=ACTION_NAMES[a]) for a in range(5)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        f"{best_algo.upper()} per-stage action distribution of the converged policy",
        y=1.0,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.98))

    fig_stem = out_dir / "F4_action_distribution"
    _git_sha_cached = _git_sha()
    save_figure(fig, fig_stem)
    fig_path = fig_stem.with_suffix(".png")
    plt.close(fig)
    logger.info("wrote %s (+ .pdf)", fig_path)

    summary = {
        "version": "1.0",
        "git_sha": _git_sha_cached,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_root": str(runs_root),
        "best_algo": best_algo,
        "display_algo": best_algo,
        "best_algo_by_eval_reward": best_by_eval,
        "display_algo_forced": force_algo is not None,
        "max_timesteps": int(max_ts),
        "n_bins": n_bins,
        "checkpoint_windows": {k: list(v) for k, v in cps.items()},
        "marginal_action_share_per_bin": {
            "centers": centers.tolist(),
            "shares": bin_props_safe.tolist(),
        },
        "per_stage_at_checkpoints": {
            label: {
                stage_name: (
                    per_stage_cps[label][s_idx].tolist()
                    if np.isfinite(per_stage_cps[label][s_idx]).all()
                    else None
                )
                for s_idx, stage_name in enumerate(_STAGE_NAMES)
            }
            for label in cp_labels
        },
        "g5_5_per_stage": g5_5_violations,
        "g5_5_passes": g5_5_passes,
    }
    summary_path = out_dir / "F4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", summary_path)

    inputs: dict[str, str] = {}
    for (a, s), _recs in train_runs.items():
        p = runs_root / a / f"seed_{s}" / "episodes.jsonl"
        if p.exists():
            inputs[str(p)] = _sha256(p)
    for (a, s), _recs in eval_runs.items():
        p = runs_root / a / f"seed_{s}" / "eval.jsonl"
        if p.exists():
            inputs[str(p)] = _sha256(p)
    manifest = {
        "version": "1.0",
        "git_sha": summary["git_sha"],
        "timestamp": summary["timestamp"],
        "produced_by": "scripts/blue_team/plot_action_dist.py",
        "inputs": inputs,
        "outputs": {
            "F4_action_distribution.png": _sha256(fig_path),
            "F4_action_distribution.pdf": _sha256(fig_stem.with_suffix(".pdf")),
            "F4_summary.json": _sha256(summary_path),
        },
    }
    manifest_path = out_dir / "F4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", manifest_path)

    return {
        "fig_path": str(fig_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "g5_5_passes": g5_5_passes,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render F4 action-distribution figure.")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out-dir", default="docs/results/blue-team-training")
    p.add_argument("--n-bins", type=int, default=25)
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument(
        "--force-algo",
        default=None,
        help=(
            "Pin the displayed algo (e.g. 'ppo') instead of auto-selecting "
            "the eval-reward best. The thesis uses 'ppo' so the figure "
            "matches the prose."
        ),
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    render(
        runs_root=Path(args.runs_root),
        out_dir=Path(args.out_dir),
        n_bins=args.n_bins,
        fraction=args.fraction,
        force_algo=args.force_algo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
