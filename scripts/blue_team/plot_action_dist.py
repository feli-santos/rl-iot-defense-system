"""F4 — Action-distribution evolution over training.

PLAN §3.1.9, D5.10, D5.11.

Two-panel layout:
  (a) MAIN — stacked-area chart of marginal action proportions over
      training timesteps for the *headline algo*. By default this is the
      best-performing algo (chosen by eval-reward, D5.11), but the thesis
      pins it to PPO via ``--force-algo ppo`` so the figure matches the
      surrounding prose. 25-K-step bins.
  (b) SUPPLEMENTARY — 3 × 5 small-multiples: rows = checkpoints
      {early=5%, mid=50%, late=100% of training}; cols = decision
      stage. Each small panel is a per-stage action histogram. This
      panel is what gates G5.5 (per-stage non-degeneracy).

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

    # ---------- render --------------------------------------------------------
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        2, 5, height_ratios=(1.2, 1.3), hspace=0.45, wspace=0.2, left=0.07, right=0.98, top=0.92
    )
    # (a) main: stacked area
    ax_main = fig.add_subplot(gs[0, :])
    bin_props_safe = np.nan_to_num(bin_props, nan=0.0)
    ax_main.stackplot(
        centers,
        bin_props_safe.T,
        labels=ACTION_NAMES,
        colors=[_ACTION_COLORS[a] for a in range(5)],
        alpha=0.9,
    )
    ax_main.set_xlim(0, max_ts)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Training timesteps")
    ax_main.set_ylabel("Action share (marginal, mean over seeds)")
    ax_main.set_title(f"(a) {best_algo.upper()} marginal action distribution over training")
    ax_main.legend(loc="upper right", fontsize=8, ncol=5)

    # (b) per-stage histograms at 3 checkpoints
    cp_labels = ["early", "mid", "late"]
    for col_idx, stage_name in enumerate(_STAGE_NAMES):
        ax = fig.add_subplot(gs[1, col_idx])
        x = np.arange(5)
        width = 0.27
        for k, lab in enumerate(cp_labels):
            row = per_stage_cps[lab][col_idx]
            row_safe = np.nan_to_num(row, nan=0.0)
            ax.bar(
                x + (k - 1) * width,
                row_safe,
                width=width,
                color=[_ACTION_COLORS[a] for a in range(5)],
                edgecolor="k",
                linewidth=(0.6 if lab == "late" else 0.0),
                alpha=(1.0 if lab == "late" else 0.55),
                label=f"t={lab}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(stage_name, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("Action share")
        # Mark the recommended action with a star.
        ax.axvline(col_idx, color="k", lw=0.6, ls="--", alpha=0.4)
    fig.suptitle(
        f"{best_algo.upper()} action-distribution evolution over training\n"
        "(top) marginal action share; (bottom) per-stage action histograms "
        "at early / mid / late checkpoints",
        y=0.99,
        fontsize=12,
    )

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
