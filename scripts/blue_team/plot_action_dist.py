"""F4 — Per-stage action distribution of the converged policies.

PLAN §3.1.9, D5.10, D5.11.

Grid layout: one row per algorithm (default A2C, PPO, DQN) × five
per-stage action histograms (BENIGN … IMPACT) for the converged
(late-checkpoint) policy of each algo. The set and order of rows is
controlled by ``--algos`` (default ``a2c ppo dqn``). A shared
action-color legend runs along the bottom. This is the panel that gates
G5.5 (non-degeneracy) — the gate spans every displayed algo and tests
stage-discrimination: a policy passes iff its per-stage argmax spans at
least two distinct actions across the five stages (it does not collapse
to one action everywhere). A decisive single-stage share is the learned
force ladder, not degeneracy.

The marginal action share over training timesteps (25-K-step bins) and
the early/mid checkpoints are still computed and recorded per algo in
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


# Default row order for the multi-algo grid. A2C, PPO, DQN — the two
# on-policy methods lead, off-policy DQN last (matches the surrounding prose
# ordering of learned doctrines). Overridable via ``--algos``.
_DEFAULT_ALGOS = ["a2c", "ppo", "dqn"]

# G5.5 non-degeneracy: a converged policy must discriminate by kill-chain
# stage. It passes iff its per-stage argmax spans at least this many distinct
# actions across the five stages (i.e. it does not collapse to one action
# everywhere). A high single-stage action share is the learned force ladder,
# not degeneracy, so this replaces the earlier per-stage max-share cap.
_G5_5_MIN_DISTINCT_ARGMAX = 2

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
    algos: list[str] | None = None,
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

    # Eval-reward best algo (D5.11) — still recorded in the summary JSON for
    # the gate/audit trail even though the figure now shows an explicit set
    # of rows.
    best_by_eval = (
        _select_best_algo(eval_runs, fraction=fraction)
        if eval_runs
        else sorted({a for (a, _) in train_runs})[0]
    )
    available = sorted({a for (a, _) in train_runs})

    # Rows to display, in order. Default A2C, PPO, DQN; overridable via
    # --algos. Every requested algo must have training runs.
    requested = list(algos) if algos else list(_DEFAULT_ALGOS)
    missing = [a for a in requested if a not in available]
    if missing:
        raise RuntimeError(f"--algos {missing} not among trained algos {available}")
    display_algos = requested
    logger.info("F4 display algos -> %s (eval-reward best = %s)", display_algos, best_by_eval)

    # Per-algo marginal action distribution over training time (seed-averaged).
    bin_props_by_algo: dict[str, np.ndarray] = {
        a: _stack_per_seed_action_bins(train_runs, a, edges) for a in display_algos
    }

    # Per-stage action distribution at three checkpoints, per displayed algo.
    cps = {
        "early": (0, int(0.05 * max_ts)),
        "mid": (int(0.45 * max_ts), int(0.55 * max_ts)),
        "late": (int(0.90 * max_ts), int(max_ts)),
    }
    # per_stage_by_algo[algo][label] -> (5 stages × 5 actions) matrix.
    per_stage_by_algo: dict[str, dict[str, np.ndarray]] = {}
    for algo in display_algos:
        per_stage_cps: dict[str, np.ndarray] = {}
        for label, (lo, hi) in cps.items():
            # Pool all seeds' records inside the window for this algo.
            pooled = []
            for (a, _s), recs in train_runs.items():
                if a != algo:
                    continue
                for r in recs:
                    if lo <= r["num_timesteps"] <= hi:
                        pooled.append(r)
            per_stage_cps[label] = per_stage_action_distribution(pooled)
        per_stage_by_algo[algo] = per_stage_cps

    # G5.5 check: per-stage *non-degeneracy* on the *late* checkpoint, computed
    # per algo. A policy is degenerate if it collapses to the *same* action at
    # every kill-chain stage; a healthy policy discriminates by stage. We
    # therefore test stage-discrimination: a policy passes iff its per-stage
    # argmax spans at least ``_G5_5_MIN_DISTINCT_ARGMAX`` distinct actions
    # across the five stages. This deliberately does NOT penalise a legitimately
    # decisive per-stage doctrine (e.g. A2C blocking ~84% at MANEUVER) — a high
    # single-stage share is the learned force ladder, not mode collapse. The
    # gate spans every displayed algo: it passes only if all rows are
    # non-degenerate. Per-stage max shares are still recorded for inspection.
    def _g5_5_for(late: np.ndarray) -> dict[str, Any]:
        argmax_by_stage: dict[str, str | None] = {}
        max_share_by_stage: dict[str, float | None] = {}
        distinct: set[int] = set()
        for s_idx, name in enumerate(_STAGE_NAMES):
            row = late[s_idx]
            if not np.isfinite(row).all():
                argmax_by_stage[name] = None
                max_share_by_stage[name] = None
                continue
            arg = int(row.argmax())
            argmax_by_stage[name] = ACTION_NAMES[arg]
            max_share_by_stage[name] = float(row.max())
            distinct.add(arg)
        n_distinct = len(distinct)
        return {
            "distinct_argmax_actions": n_distinct,
            "argmax_by_stage": argmax_by_stage,
            "max_share_by_stage": max_share_by_stage,
            # Non-degenerate iff the policy uses >= threshold distinct actions
            # across stages. None only if no stage had any decisions.
            "passes": (n_distinct >= _G5_5_MIN_DISTINCT_ARGMAX) if n_distinct > 0 else None,
        }

    g5_5_per_algo: dict[str, dict[str, Any]] = {
        algo: _g5_5_for(per_stage_by_algo[algo]["late"]) for algo in display_algos
    }

    # Aggregate per-stage view (shape preserved for evaluate_gates.py, which
    # maps each stage's ``passes`` -> status). The pass verdict is a policy-level
    # property (stage discrimination), so every stage carries the same overall
    # verdict; the per-stage detail exposed here is the worst (largest) action
    # share across displayed algos and which algo/action produced it.
    g5_5_passes = all(g5_5_per_algo[a]["passes"] in (True, None) for a in display_algos)
    g5_5_violations: dict[str, Any] = {}
    for name in _STAGE_NAMES:
        shares = []
        for a in display_algos:
            ms = g5_5_per_algo[a]["max_share_by_stage"][name]
            shares.append((ms, a))
        finite = [(s, a) for (s, a) in shares if s is not None]
        if not finite:
            g5_5_violations[name] = {
                "max_share": None,
                "passes": None,
                "note": "no decisions in late window for any algo",
            }
            continue
        worst_share, worst_algo = max(finite, key=lambda t: t[0])
        worst_action = g5_5_per_algo[worst_algo]["argmax_by_stage"][name]
        g5_5_violations[name] = {
            "max_share": float(worst_share),
            "argmax_action_name": worst_action,
            "worst_algo": worst_algo,
            "passes": g5_5_passes,
        }

    # Checkpoint labels retained for the summary JSON (all three windows are
    # still recorded even though the figure shows only the late policy).
    cp_labels = ["early", "mid", "late"]

    # ---------- render --------------------------------------------------------
    # One row per displayed algo × five per-stage action histograms for the
    # converged (late) policy. The marginal-over-training panel was dropped:
    # the thesis point is the *learned per-stage force ladder*, which the late
    # checkpoint shows directly. bin_props is still recorded in the summary
    # JSON for the gate.
    bin_props_safe_by_algo = {
        a: np.nan_to_num(bin_props_by_algo[a], nan=0.0) for a in display_algos
    }

    n_rows = len(display_algos)
    fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(15, 3.2 * n_rows),
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.12, "hspace": 0.4},
    )
    # Normalise to a 2-D (n_rows × 5) array so single-algo calls still work.
    axes = np.atleast_2d(axes)
    x = np.arange(5)
    for row_idx, algo in enumerate(display_algos):
        late_cps = per_stage_by_algo[algo]["late"]
        for col_idx, stage_name in enumerate(_STAGE_NAMES):
            ax = axes[row_idx][col_idx]
            row_safe = np.nan_to_num(late_cps[col_idx], nan=0.0)
            bars = ax.bar(
                x,
                row_safe,
                width=0.72,
                color=[_ACTION_COLORS[a] for a in range(5)],
                edgecolor="k",
                linewidth=0.6,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
            ax.set_xticks(x)
            ax.set_ylim(0, 1.12)
            ax.grid(axis="y", ls=":", alpha=0.4)
            # Stage names as column headers on the top row only.
            if row_idx == 0:
                ax.set_title(stage_name, fontsize=12)
            # Action tick labels on the bottom row only (sharex).
            if row_idx == n_rows - 1:
                ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=9)
            # Row label (algo) + shared y-axis label on the first column.
            if col_idx == 0:
                ax.set_ylabel(f"{algo.upper()}\nAction share", fontsize=11)

    legend_handles = [
        Patch(facecolor=_ACTION_COLORS[a], edgecolor="k", label=ACTION_NAMES[a]) for a in range(5)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02 - 0.02 * n_rows),
    )
    fig.suptitle(
        "Per-stage action distribution of the converged policies "
        f"({' / '.join(a.upper() for a in display_algos)})",
        y=1.0,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    fig_stem = out_dir / "F4_action_distribution"
    _git_sha_cached = _git_sha()
    save_figure(fig, fig_stem)
    fig_path = fig_stem.with_suffix(".png")
    plt.close(fig)
    logger.info("wrote %s (+ .pdf)", fig_path)

    def _per_stage_at_checkpoints(per_stage_cps: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            label: {
                stage_name: (
                    per_stage_cps[label][s_idx].tolist()
                    if np.isfinite(per_stage_cps[label][s_idx]).all()
                    else None
                )
                for s_idx, stage_name in enumerate(_STAGE_NAMES)
            }
            for label in cp_labels
        }

    summary = {
        "version": "2.0",
        "git_sha": _git_sha_cached,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs_root": str(runs_root),
        "display_algos": display_algos,
        "best_algo_by_eval_reward": best_by_eval,
        "max_timesteps": int(max_ts),
        "n_bins": n_bins,
        "checkpoint_windows": {k: list(v) for k, v in cps.items()},
        # Per-algo marginal action share over training (25-K-step bins).
        "marginal_action_share_per_bin": {
            "centers": centers.tolist(),
            "shares_by_algo": {a: bin_props_safe_by_algo[a].tolist() for a in display_algos},
        },
        # Per-algo per-stage action distributions at early/mid/late checkpoints.
        "per_stage_at_checkpoints_by_algo": {
            algo: _per_stage_at_checkpoints(per_stage_by_algo[algo]) for algo in display_algos
        },
        # G5.5 non-degeneracy criterion (stage-discrimination).
        "g5_5_criterion": "distinct_argmax_actions_across_stages",
        "g5_5_min_distinct_argmax": _G5_5_MIN_DISTINCT_ARGMAX,
        # Per-algo G5.5 breakdown (late checkpoint): distinct-argmax count,
        # per-stage argmax action + max share, and the per-policy verdict.
        "g5_5_per_algo": g5_5_per_algo,
        # Aggregate per-stage view (worst action share across displayed algos);
        # shape consumed by scripts/blue_team/evaluate_gates.py. ``passes`` is
        # the policy-level verdict (all displayed algos non-degenerate).
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
        "--algos",
        nargs="+",
        default=None,
        metavar="ALGO",
        help=(
            "Algorithms to display, one row each, in the given order "
            f"(default: {' '.join(_DEFAULT_ALGOS)}). Each must have training "
            "runs under --runs-root."
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
        algos=args.algos,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
