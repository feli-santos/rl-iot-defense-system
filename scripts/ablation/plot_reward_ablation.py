"""Phase-7 F9 — Reward-component ablation plot (PLAN §3.1.4 / D7.1).

Reads ``runs/phase7/reward_sweep/<cell_id>/seed_<k>/eval_test.jsonl``
files (produced by :mod:`scripts.ablation.run_reward_sweep`),
aggregates per cell with the same bootstrap-CI protocol as Phase 6
F5, and renders a multi-panel "what does each reward component
do?" figure with the **Phase-6 oracle ceiling +1624** and **Phase-6
deployable best DQN +1336** as horizontal reference lines.

Outputs:

- ``F9_reward_ablation.png`` — multi-panel scatter/line plot, one
  panel per reward component (5) + one panel for the
  ``impact_is_terminal`` binary axis. Each panel shows mean test
  reward at 0.5× / 1× / 2× of Phase-3 default with 95 % bootstrap
  CIs and the two reference lines.
- ``F9_summary.json`` — per-cell aggregate + per-component slope
  estimate + headline ``best_cell`` (max mean_reward across cells)
  + G7.2 evaluation.
- ``F9_caption.md`` — thesis caption (placeholder; hand-edit).
- ``F9_manifest.json`` — SHA-256 hash chain.

Gate evaluation:

- **G7.2** — pass iff at least one cell's mean test reward exceeds
  the Phase-6 deployable best (DQN +1336) by ≥ 1σ of its bootstrap
  CI. Stretch goal: meet the Phase-6 oracle ceiling (+1624). The
  acceptable failure mode is ``D7.1.1`` — the sweep characterises
  the limit of one-at-a-time Phase-3-style reward shaping; turning
  the gate verdict into a finding rather than a closure (see PLAN
  §8 D7.1.1 placeholder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_reward_ablation")

_ROOT = Path(__file__).resolve().parents[2]


# Reference lines from Phase 6 (audit-AF2 framing).
_PHASE6_ORACLE_CEILING_REWARD = 1624.4
_PHASE6_DEPLOYABLE_BEST_REWARD = 1336.3  # DQN best mean on test_balanced
_DEPLOYABLE_BEST_LABEL = "DQN +1336 (Phase-6 deployable best)"
_ORACLE_CEILING_LABEL = "Rec-Action +1624 (oracle ceiling, AF2)"

_COMPONENT_DISPLAY: Dict[str, str] = {
    "defense_success_bonus":   "defense_success_bonus  (250)",
    "penalty_missed_impact":   "penalty_missed_impact  (150)",
    "reward_proportional":     "reward_proportional      (5)",
    "penalty_disproportionate":"penalty_disproportionate  (5)",
    "reward_benign_passive":   "reward_benign_passive   (10)",
}


def _sha256(path: Path) -> Optional[str]:
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------- aggregation


def _summarise_cell(
    cell_id: str,
    cell_dir: Path,
    cell_config: Dict[str, Any],
    *,
    sha_collector: Dict[str, str],
) -> Dict[str, Any]:
    """Aggregate eval_test.jsonl across all seed_*/ subdirs of one cell."""
    seed_dirs = sorted(
        d for d in cell_dir.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
        and (d / "eval_test.jsonl").exists()
    )
    all_records: List[Dict] = []
    per_seed_means: List[float] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(
                float(np.mean([r["episode_reward"] for r in recs]))
            )
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.resolve().relative_to(_ROOT))] = sha

    if not all_records:
        return {
            "cell_id": cell_id,
            "axis": cell_config.get("axis"),
            "component": cell_config.get("component"),
            "multiplier": cell_config.get("multiplier"),
            "impact_is_terminal": cell_config.get("impact_is_terminal", True),
            "n_seeds": len(seed_dirs),
            "n_episodes": 0,
            "mean_reward": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "compromise_rate": math.nan,
            "mitigated_impact_rate": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    compromised = [1.0 if r.get("compromised") else 0.0 for r in all_records]
    mitigated = [
        1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0
        for r in all_records
    ]
    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means, n_resamples=2000, alpha=0.05, seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards, n_resamples=2000, alpha=0.05, seed=0,
        )

    return {
        "cell_id": cell_id,
        "axis": cell_config.get("axis"),
        "component": cell_config.get("component"),
        "multiplier": cell_config.get("multiplier"),
        "impact_is_terminal": cell_config.get("impact_is_terminal", True),
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "compromise_rate": float(np.mean(compromised)),
        "mitigated_impact_rate": float(np.mean(mitigated)),
    }


# --------------------------------------------------------------- gates


def _evaluate_g72(
    rows: List[Dict[str, Any]],
    deployable_best: float = _PHASE6_DEPLOYABLE_BEST_REWARD,
    oracle_ceiling: float = _PHASE6_ORACLE_CEILING_REWARD,
) -> Dict[str, Any]:
    """G7.2: at least one cell's CI lower bound > Phase-6 DQN +1336."""
    candidates = [
        r for r in rows
        if math.isfinite(r.get("ci_low", math.nan))
        and r.get("axis") != "baseline"
    ]
    if not candidates:
        return {
            "passes": False,
            "reason": "no candidate cells with finite CI",
            "deployable_best_threshold": deployable_best,
        }
    best = max(candidates, key=lambda r: r["mean_reward"])
    passes_deployable = best["ci_low"] > deployable_best
    meets_oracle = best["ci_low"] > oracle_ceiling
    return {
        "passes": bool(passes_deployable),
        "best_cell": best["cell_id"],
        "best_mean_reward": best["mean_reward"],
        "best_ci": [best["ci_low"], best["ci_high"]],
        "deployable_best_threshold": deployable_best,
        "oracle_ceiling": oracle_ceiling,
        "meets_oracle_ceiling_stretch": bool(meets_oracle),
        "delta_to_deployable": best["mean_reward"] - deployable_best,
        "delta_to_oracle": best["mean_reward"] - oracle_ceiling,
        "interpretation": (
            "PASS: at least one reward-component cell beats the "
            "Phase-6 deployable best DQN +1336 by ≥ 1σ. "
            + ("STRETCH MET: cell also exceeds the oracle ceiling +1624 — "
               "the deployable +288 gap is closed.")
            if passes_deployable and meets_oracle else
            ("PASS-WITHOUT-STRETCH: cell beats DQN +1336 but does not reach "
             "the oracle ceiling +1624; the +288 gap is partially closed."
             if passes_deployable else
             "FAIL-WITH-FINDING (D7.1.1): the linear sweep failed to close "
             "the gap, characterising the limit of one-at-a-time Phase-3-"
             "style reward shaping. Closing the gap requires a different "
             "mechanism (curriculum, reward modelling, or attack-aware "
             "exploration), deferred to future work.")
        ),
    }


# --------------------------------------------------------------- render


def _render(
    rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    deployable_best: float = _PHASE6_DEPLOYABLE_BEST_REWARD,
    oracle_ceiling: float = _PHASE6_ORACLE_CEILING_REWARD,
) -> None:
    """Multi-panel figure: one panel per reward component + one for
    impact_is_terminal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group rows by axis component.
    by_component: Dict[str, List[Dict[str, Any]]] = {}
    impact_rows: List[Dict[str, Any]] = []
    baseline: Optional[Dict[str, Any]] = None
    for r in rows:
        if r["axis"] == "baseline":
            baseline = r
        elif r["axis"] == "impact_terminal":
            impact_rows.append(r)
        elif r["axis"] == "reward":
            by_component.setdefault(r["component"], []).append(r)

    component_order = list(_COMPONENT_DISPLAY.keys())
    panels = [c for c in component_order if c in by_component]
    if impact_rows or baseline:
        panels.append("__impact_terminal__")

    n = len(panels)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13.0, 3.6 * n_rows), squeeze=False
    )

    for ax_idx, panel in enumerate(panels):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        if panel == "__impact_terminal__":
            # Bar panel: baseline (True) vs impact_is_terminal=False.
            xs = []
            means = []
            lo = []
            hi = []
            labels = []
            if baseline:
                xs.append(0)
                means.append(baseline["mean_reward"])
                lo.append(max(baseline["mean_reward"] - baseline["ci_low"], 0.0))
                hi.append(max(baseline["ci_high"] - baseline["mean_reward"], 0.0))
                labels.append("True (default)")
            for r in impact_rows:
                xs.append(len(xs))
                means.append(r["mean_reward"])
                lo.append(max(r["mean_reward"] - r["ci_low"], 0.0))
                hi.append(max(r["ci_high"] - r["mean_reward"], 0.0))
                labels.append("False")
            ax.bar(xs, means, yerr=[lo, hi], capsize=4,
                   color=["#9ca3af", "#2563eb"][:len(xs)])
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_title("impact_is_terminal (D7.3)", fontsize=10)
        else:
            # Line panel: 0.5×, 1×, 2× × component.
            comp_rows = by_component[panel]
            # Include the centre baseline at multiplier=1.0.
            if baseline:
                comp_rows = list(comp_rows) + [{
                    **baseline,
                    "component": panel,
                    "multiplier": 1.0,
                }]
            comp_rows.sort(key=lambda r: r["multiplier"])
            xs = [r["multiplier"] for r in comp_rows]
            means = [r["mean_reward"] for r in comp_rows]
            lo = [r["ci_low"] for r in comp_rows]
            hi = [r["ci_high"] for r in comp_rows]
            ax.errorbar(xs, means,
                        yerr=[
                            [m - l for m, l in zip(means, lo)],
                            [h - m for m, h in zip(means, hi)],
                        ],
                        fmt="o-", capsize=4, color="#2563eb",
                        markerfacecolor="#2563eb", linewidth=1.5)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 1.0, 2.0])
            ax.set_xticklabels(["0.5×", "1×", "2×"])
            ax.set_title(_COMPONENT_DISPLAY[panel], fontsize=10)
            ax.set_xlabel("multiplier × Phase-3 default", fontsize=8)

        # Reference lines on every panel.
        ax.axhline(deployable_best, color="#2563eb", linestyle=":",
                   linewidth=0.9, alpha=0.7, label=_DEPLOYABLE_BEST_LABEL)
        ax.axhline(oracle_ceiling, color="#dc2626", linestyle="--",
                   linewidth=0.9, alpha=0.7, label=_ORACLE_CEILING_LABEL)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Shared y-axis scale: include both reference lines + all data.
    all_values = [r["mean_reward"] for r in rows if math.isfinite(r["mean_reward"])]
    all_los = [r["ci_low"] for r in rows if math.isfinite(r["ci_low"])]
    all_his = [r["ci_high"] for r in rows if math.isfinite(r["ci_high"])]
    if all_values and all_los and all_his:
        ymin = min(min(all_los), deployable_best, oracle_ceiling) - 100
        ymax = max(max(all_his), deployable_best, oracle_ceiling) + 100
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r][c].set_ylim(ymin, ymax)

    # Hide unused subplots.
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(
        "F9 — Reward-component ablation (PPO 250K × 5 seeds; sparse one-at-a-time)",
        fontsize=12, y=1.0,
    )
    fig.text(
        0.5, -0.01,
        "PLAN §3.1.4 / D7.1; targets the +288 deployable gap (D6.2.1, audit AF2)",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 F9 — reward-component ablation plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/phase7/reward_sweep")
    p.add_argument("--out-dir", default="docs/results/07_ablation")
    p.add_argument(
        "--phase6-eval-manifest",
        default="runs/phase6/eval_manifest.json",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        logger.error("runs_root not found: %s", runs_root)
        return 1

    sweep_manifest_path = runs_root / "sweep_manifest.json"

    # Discover cells (every subdir with cell_config.json).
    cell_dirs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "cell_config.json").exists()
    )
    if not cell_dirs:
        logger.error("no cell directories under %s", runs_root)
        return 1

    sha_collector: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    for cell_dir in cell_dirs:
        cell_config = json.loads((cell_dir / "cell_config.json").read_text())
        row = _summarise_cell(
            cell_dir.name, cell_dir, cell_config, sha_collector=sha_collector,
        )
        rows.append(row)
        logger.info(
            "F9 cell=%s axis=%s mean=%.1f CI=(%.1f, %.1f) n_ep=%d",
            row["cell_id"], row["axis"],
            row["mean_reward"], row["ci_low"], row["ci_high"],
            row["n_episodes"],
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "F9_reward_ablation.png"
    _render(rows, png_path)

    g72 = _evaluate_g72(rows)
    summary = {
        "schema_version": "1.0",
        "phase": 7,
        "figure": "F9",
        "phase3_defaults": {
            "defense_success_bonus":   250.0,
            "penalty_missed_impact":   150.0,
            "reward_proportional":       5.0,
            "penalty_disproportionate":  5.0,
            "reward_benign_passive":    10.0,
        },
        "phase6_oracle_ceiling": _PHASE6_ORACLE_CEILING_REWARD,
        "phase6_deployable_best": _PHASE6_DEPLOYABLE_BEST_REWARD,
        "rows": rows,
        "gates": {"G7.2": g72},
        "headline": g72.get("interpretation", "?"),
    }
    (out_dir / "F9_summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "schema_version": "1.0",
        "figure": "F9",
        "git_sha": _git_sha(),
        "outputs": {
            "png": str(png_path),
            "json": str(out_dir / "F9_summary.json"),
        },
        "inputs": {
            "phase7_reward_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            "phase6_eval_manifest": {
                "path": str(args.phase6_eval_manifest),
                "sha256": _sha256(Path(args.phase6_eval_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F9_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F9_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F9 — Reward-component ablation.** Mean episodic reward on "
            "`test_balanced` for PPO trained 250K timesteps × 5 seeds at "
            "{0.5×, 1×, 2×} of each Phase-3 reward coefficient (one-at-a-"
            "time sparse grid; D7.1). Reference lines: blue dotted = "
            "Phase-6 deployable best (DQN +1336); red dashed = Phase-6 "
            "oracle ceiling (recommended-action rule, +1624 — *upper bound "
            "on the value of perfect stage detection*, audit AF2). The "
            "rightmost panel sweeps the binary `impact_is_terminal` axis "
            "(D7.3). Error bars are 95 % bootstrap CIs. (PLAN §3.1.4.)\n"
        )

    logger.info(
        "F9 written to %s — G7.2 passes=%s (best=%s, mean=%.1f, "
        "Δ_to_dqn=%+.1f, Δ_to_oracle=%+.1f)",
        out_dir, g72.get("passes"),
        g72.get("best_cell"), g72.get("best_mean_reward", float("nan")),
        g72.get("delta_to_deployable", float("nan")),
        g72.get("delta_to_oracle", float("nan")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
