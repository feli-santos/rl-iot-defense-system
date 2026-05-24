"""ablation F9 — Reward-component ablation plot (PLAN §3.1.4 / D7.1).

Reads ``runs/ablation/reward_sweep/<cell_id>/seed_<k>/eval_test.jsonl``
files (produced by :mod:`scripts.ablation.run_reward_sweep`),
aggregates per cell with the same bootstrap-CI protocol as benchmark
F5, and renders a multi-panel "what does each reward component
do?" figure with the **benchmark oracle ceiling +1624** and **benchmark
deployable best DQN +1336** as horizontal reference lines.

Outputs:

- ``F9_reward_ablation.png`` — multi-panel scatter/line plot, one
  panel per reward component (5) + one panel for the
  ``impact_is_terminal`` binary axis. Each panel shows mean test
  reward at 0.5× / 1× / 2× of environment-design default with 95 % bootstrap
  CIs and the two reference lines.
- ``F9_summary.json`` — per-cell aggregate + per-component slope
  estimate + headline ``best_cell`` (max mean_reward across cells)
  + G7.2 evaluation.
- ``F9_caption.md`` — thesis caption (placeholder; hand-edit).
- ``F9_manifest.json`` — SHA-256 hash chain.

Gate evaluation:

- **G7.2** — pass iff at least one cell's mean test reward exceeds
  the benchmark deployable best (DQN +1336) by ≥ 1σ of its bootstrap
  CI. Stretch goal: meet the benchmark oracle ceiling (+1624). The
  acceptable failure mode is ``D7.1.1`` — the sweep characterises
  the limit of one-at-a-time environment-design-style reward shaping; turning
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
from typing import Any

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_reward_ablation")

_ROOT = Path(__file__).resolve().parents[2]


# Reference lines from benchmark (audit-AF2 framing).
_PHASE6_ORACLE_CEILING_REWARD = 1624.4
_PHASE6_DEPLOYABLE_BEST_REWARD = 1336.3  # DQN best mean on test_balanced
_DEPLOYABLE_BEST_LABEL = "DQN +1336 (benchmark deployable best)"
_ORACLE_CEILING_LABEL = "Rec-Action +1624 (oracle ceiling, AF2)"

_COMPONENT_DISPLAY: dict[str, str] = {
    "defense_success_bonus": "defense_success_bonus  (250)",
    "penalty_missed_impact": "penalty_missed_impact  (150)",
    "reward_proportional": "reward_proportional      (5)",
    "penalty_disproportionate": "penalty_disproportionate  (5)",
    "reward_benign_passive": "reward_benign_passive   (10)",
}


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


# --------------------------------------------------------------- aggregation


def _summarise_cell(
    cell_id: str,
    cell_dir: Path,
    cell_config: dict[str, Any],
    *,
    sha_collector: dict[str, str],
) -> dict[str, Any]:
    """Aggregate eval_test.jsonl across all seed_*/ subdirs of one cell."""
    seed_dirs = sorted(
        d
        for d in cell_dir.iterdir()
        if d.is_dir() and d.name.startswith("seed_") and (d / "eval_test.jsonl").exists()
    )
    all_records: list[dict] = []
    per_seed_means: list[float] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(float(np.mean([r["episode_reward"] for r in recs])))
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
    mitigated = [1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0 for r in all_records]
    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means,
            n_resamples=2000,
            alpha=0.05,
            seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards,
            n_resamples=2000,
            alpha=0.05,
            seed=0,
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


# benchmark DQN (deployable best) security KPI under the environment-design reward
# function. This is the apples-to-apples bar for cells that ALSO use
# the environment-design reward function (impact_is_terminal axis + the
# baseline_defaults centre cell). Reward-coefficient cells use
# a DIFFERENT reward function and are NOT directly comparable on raw
# reward — they are evaluated on the security KPI only.
_PHASE6_DEPLOYABLE_BEST_MITIGATED = 0.153  # DQN mitigated_impact_rate on test_balanced


def _evaluate_g72(
    rows: list[dict[str, Any]],
    deployable_best: float = _PHASE6_DEPLOYABLE_BEST_REWARD,
    oracle_ceiling: float = _PHASE6_ORACLE_CEILING_REWARD,
    deployable_best_mitigated: float = _PHASE6_DEPLOYABLE_BEST_MITIGATED,
) -> dict[str, Any]:
    """G7.2 — corrected: compare cells fairly on a metric that is
    commensurable across cells with **different reward functions**.

    The audit-D7.1.1 framing (2026-05-01): cells that scale a reward
    *coefficient* by 0.5×/2× are **not directly comparable to
    benchmark raw reward** because the reward scale itself moved. The
    only benchmark-comparable raw-reward cells in this 12-cell sparse
    grid are (a) ``baseline_defaults`` (environment-design reward fn,
    one extra seed pool) and (b) the two ``impact_is_terminal``
    cells (env semantics change, reward fn unchanged).

    The honest pass criterion therefore splits into two strands:

    1. **Reward-comparable strand** (raw-reward gate): for cells
       under the unchanged environment-design reward fn, does any cell's
       CI_low exceed benchmark DQN +1336?
    2. **Security-KPI strand** (mitigated_impact_rate gate): for
       *all* cells (incl. coefficient-scaled), does any cell beat
       benchmark DQN's mitigated_impact_rate (0.153) by ≥ 1.5×? This
       is the metric that survives reward-function changes.

    G7.2 PASSES iff strand-1 holds (the original-shape gate). If
    strand-1 fails but strand-2 holds, that is **D7.1.1
    PASS-WITH-FINDING**: reward shaping cannot move the
    apples-to-apples reward number, but the env-semantics flip
    (`impact_is_terminal=False`) or coefficient scaling does
    improve real security on the security KPI — useful diagnostic,
    narrowed thesis claim.
    """
    finite = [r for r in rows if math.isfinite(r.get("ci_low", math.nan))]
    if not finite:
        return {
            "passes": False,
            "reason": "no candidate cells with finite CI",
            "deployable_best_threshold": deployable_best,
        }

    # Strand 1: reward-comparable cells only (environment-design reward fn).
    # axis="reward" cells scale a coefficient ⇒ NOT comparable.
    # axis="baseline" + axis="impact_terminal" preserve the reward fn.
    reward_comparable = [r for r in finite if r.get("axis") in ("baseline", "impact_terminal")]
    best_rc = max(reward_comparable, key=lambda r: r["mean_reward"]) if reward_comparable else None
    passes_strand1 = bool(best_rc and best_rc["ci_low"] > deployable_best)
    meets_oracle_strand1 = bool(best_rc and best_rc["ci_low"] > oracle_ceiling)

    # Strand 2: security KPI across ALL cells (incl. coefficient-scaled).
    sec_candidates = [
        r
        for r in finite
        if math.isfinite(r.get("mitigated_impact_rate", math.nan)) and r.get("axis") != "baseline"
    ]
    best_sec = (
        max(sec_candidates, key=lambda r: r["mitigated_impact_rate"]) if sec_candidates else None
    )
    sec_threshold = deployable_best_mitigated * 1.5
    passes_strand2 = bool(best_sec and best_sec["mitigated_impact_rate"] >= sec_threshold)

    # The *raw-reward winner* (any axis) is reported for transparency
    # but is NOT the headline in the corrected logic — the
    # 2026-05-01 audit (Finding #1) showed that the +2926 cell is a
    # mix of real improvement on the security KPI (mit_rate 0.547 vs
    # 0.153 baseline) and reward-coefficient scaling (×2 the bonus).
    raw_winner = max(finite, key=lambda r: r["mean_reward"])

    if passes_strand1 and meets_oracle_strand1:
        interp = (
            f"PASS: at least one reward-comparable cell "
            f"(`{best_rc['cell_id']}`) beats the benchmark deployable "
            f"best DQN +{deployable_best:.0f} by ≥ 1σ on RAW REWARD "
            f"(commensurable to benchmark). STRETCH MET: cell also "
            f"exceeds the oracle ceiling +{oracle_ceiling:.0f} — "
            f"the deployable +288 gap is closed."
        )
    elif passes_strand1:
        interp = (
            f"PASS-WITHOUT-STRETCH: reward-comparable cell "
            f"(`{best_rc['cell_id']}`) beats DQN +{deployable_best:.0f} "
            f"on RAW REWARD but does not reach the oracle ceiling "
            f"+{oracle_ceiling:.0f}; the +288 gap is partially closed."
        )
    elif passes_strand2:
        interp = (
            f"FAIL-WITH-FINDING (D7.1.1, activated 2026-05-01): no "
            f"reward-comparable cell (environment-design reward fn preserved) "
            f"beats DQN +{deployable_best:.0f} on raw reward by ≥ 1σ. "
            f"BUT: the security-KPI strand passes — cell "
            f"`{best_sec['cell_id']}` improves mitigated_impact_rate "
            f"to {best_sec['mitigated_impact_rate']:.3f} "
            f"(vs DQN baseline {deployable_best_mitigated:.3f}, "
            f"≥ 1.5× threshold {sec_threshold:.3f}). The "
            f"one-at-a-time linear sweep characterised the limit of "
            f"environment-design-style reward shaping at the apples-to-apples "
            f"reward level, but env-semantics + coefficient scaling "
            f"do move the real-security needle. Closing the +288 "
            f"reward gap under fixed reward semantics requires a "
            f"different mechanism (curriculum, reward modelling, or "
            f"attack-aware exploration), deferred to future work."
        )
    else:
        interp = (
            "FAIL-WITH-FINDING (D7.1.1): the linear sweep failed to "
            "close the gap on either strand — neither raw reward "
            "(reward-comparable cells) nor security KPI "
            "(mitigated_impact_rate) beats benchmark DQN by the "
            "≥ 1σ / ≥ 1.5× threshold. Characterises the limit of "
            "one-at-a-time environment-design-style reward shaping. Closing "
            "the gap requires a different mechanism (curriculum, "
            "reward modelling, or attack-aware exploration), "
            "deferred to future work."
        )

    return {
        "passes": bool(passes_strand1),
        # Reward-comparable strand (the canonical G7.2 gate).
        "best_reward_comparable_cell": best_rc["cell_id"] if best_rc else None,
        "best_reward_comparable_mean": (best_rc["mean_reward"] if best_rc else None),
        "best_reward_comparable_ci": ([best_rc["ci_low"], best_rc["ci_high"]] if best_rc else None),
        "best_reward_comparable_mitigated": (
            best_rc.get("mitigated_impact_rate") if best_rc else None
        ),
        # Security-KPI strand (D7.1.1 fallback metric).
        "best_security_kpi_cell": (best_sec["cell_id"] if best_sec else None),
        "best_security_kpi_mitigated": (best_sec["mitigated_impact_rate"] if best_sec else None),
        "best_security_kpi_reward": (best_sec["mean_reward"] if best_sec else None),
        "security_kpi_strand_passes": passes_strand2,
        "security_kpi_threshold": sec_threshold,
        # Raw-reward winner (for transparency, NOT the headline).
        "raw_reward_winner_cell": raw_winner["cell_id"],
        "raw_reward_winner_mean": raw_winner["mean_reward"],
        "raw_reward_winner_note": (
            "raw-reward winner across ALL cells; NOT directly "
            "comparable to benchmark if axis='reward' because reward-"
            "coefficient cells use a different reward function. See "
            "best_reward_comparable_* for the apples-to-apples row."
        ),
        # benchmark baselines.
        "deployable_best_threshold": deployable_best,
        "deployable_best_mitigated": deployable_best_mitigated,
        "oracle_ceiling": oracle_ceiling,
        "meets_oracle_ceiling_stretch": meets_oracle_strand1,
        # Legacy fields preserved (close_phase7 reads these). They
        # now point at the reward-comparable strand to match the
        # canonical gate.
        "best_cell": best_rc["cell_id"] if best_rc else raw_winner["cell_id"],
        "best_mean_reward": (best_rc["mean_reward"] if best_rc else raw_winner["mean_reward"]),
        "best_ci": (
            [best_rc["ci_low"], best_rc["ci_high"]]
            if best_rc
            else [raw_winner["ci_low"], raw_winner["ci_high"]]
        ),
        "delta_to_deployable": (
            best_rc["mean_reward"] - deployable_best
            if best_rc
            else raw_winner["mean_reward"] - deployable_best
        ),
        "delta_to_oracle": (
            best_rc["mean_reward"] - oracle_ceiling
            if best_rc
            else raw_winner["mean_reward"] - oracle_ceiling
        ),
        "interpretation": interp,
    }


# --------------------------------------------------------------- render


def _render(
    rows: list[dict[str, Any]],
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
    by_component: dict[str, list[dict[str, Any]]] = {}
    impact_rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 3.6 * n_rows), squeeze=False)

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
            ax.bar(xs, means, yerr=[lo, hi], capsize=4, color=["#9ca3af", "#2563eb"][: len(xs)])
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_title("impact_is_terminal (D7.3)", fontsize=10)
        else:
            # Line panel: 0.5×, 1×, 2× × component.
            comp_rows = by_component[panel]
            # Include the centre baseline at multiplier=1.0.
            if baseline:
                comp_rows = list(comp_rows) + [
                    {
                        **baseline,
                        "component": panel,
                        "multiplier": 1.0,
                    }
                ]
            comp_rows.sort(key=lambda r: r["multiplier"])
            xs = [r["multiplier"] for r in comp_rows]
            means = [r["mean_reward"] for r in comp_rows]
            lo = [r["ci_low"] for r in comp_rows]
            hi = [r["ci_high"] for r in comp_rows]
            ax.errorbar(
                xs,
                means,
                yerr=[
                    [m - lo_val for m, lo_val in zip(means, lo)],
                    [h - m for m, h in zip(means, hi)],
                ],
                fmt="o-",
                capsize=4,
                color="#2563eb",
                markerfacecolor="#2563eb",
                linewidth=1.5,
            )
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 1.0, 2.0])
            ax.set_xticklabels(["0.5×", "1×", "2×"])
            ax.set_title(_COMPONENT_DISPLAY[panel], fontsize=10)
            ax.set_xlabel("multiplier × environment-design default", fontsize=8)

        # Reference lines on every panel.
        ax.axhline(
            deployable_best,
            color="#2563eb",
            linestyle=":",
            linewidth=0.9,
            alpha=0.7,
            label=_DEPLOYABLE_BEST_LABEL,
        )
        ax.axhline(
            oracle_ceiling,
            color="#dc2626",
            linestyle="--",
            linewidth=0.9,
            alpha=0.7,
            label=_ORACLE_CEILING_LABEL,
        )
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
        fontsize=12,
        y=1.0,
    )
    fig.text(
        0.5,
        -0.01,
        "PLAN §3.1.4 / D7.1; targets the +288 deployable gap (D6.2.1, audit AF2)",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F9 — reward-component ablation plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/ablation/reward_sweep")
    p.add_argument("--out-dir", default="docs/results/07_ablation")
    p.add_argument(
        "--phase6-eval-manifest",
        default="runs/benchmark/eval_manifest.json",
    )
    # Step-8 F2 (07_HANDOFF.md §5): explicit upstream-manifest SHA pins.
    p.add_argument(
        "--phase5-sweep-manifest",
        default="runs/blue_team/sweep_manifest.json",
        help="blue-team sweep_manifest.json (warm-start trained checkpoints).",
    )
    p.add_argument(
        "--phase1-splits-manifest",
        default="docs/results/01_dataset/manifest.json",
        help="dataset-prep splits manifest.json (post-3cd2fb9; SHA 1e99d596...).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
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
        d for d in runs_root.iterdir() if d.is_dir() and (d / "cell_config.json").exists()
    )
    if not cell_dirs:
        logger.error("no cell directories under %s", runs_root)
        return 1

    sha_collector: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        cell_config = json.loads((cell_dir / "cell_config.json").read_text())
        row = _summarise_cell(
            cell_dir.name,
            cell_dir,
            cell_config,
            sha_collector=sha_collector,
        )
        rows.append(row)
        logger.info(
            "F9 cell=%s axis=%s mean=%.1f CI=(%.1f, %.1f) n_ep=%d",
            row["cell_id"],
            row["axis"],
            row["mean_reward"],
            row["ci_low"],
            row["ci_high"],
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
            "defense_success_bonus": 250.0,
            "penalty_missed_impact": 150.0,
            "reward_proportional": 5.0,
            "penalty_disproportionate": 5.0,
            "reward_benign_passive": 10.0,
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
            "ablation_reward_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            "benchmark_eval_manifest": {
                "path": str(args.benchmark_eval_manifest),
                "sha256": _sha256(Path(args.benchmark_eval_manifest)),
            },
            # Step-8 F2: explicit upstream-manifest SHA pins so the F9
            # hash chain is self-contained (no transitive lookups).
            "blue_team_sweep_manifest": {
                "path": str(args.blue_team_sweep_manifest),
                "sha256": _sha256(Path(args.blue_team_sweep_manifest)),
            },
            "phase1_splits_manifest": {
                "path": str(args.phase1_splits_manifest),
                "sha256": _sha256(Path(args.phase1_splits_manifest)),
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
            "{0.5×, 1×, 2×} of each environment-design reward coefficient (one-at-a-"
            "time sparse grid; D7.1). Reference lines: blue dotted = "
            "benchmark deployable best (DQN +1336); red dashed = benchmark "
            "oracle ceiling (recommended-action rule, +1624 — *upper bound "
            "on the value of perfect stage detection*, audit AF2). The "
            "rightmost panel sweeps the binary `impact_is_terminal` axis "
            "(D7.3). Error bars are 95 % bootstrap CIs. (PLAN §3.1.4.)\n"
        )

    logger.info(
        "F9 written to %s — G7.2 passes=%s (best=%s, mean=%.1f, Δ_to_dqn=%+.1f, Δ_to_oracle=%+.1f)",
        out_dir,
        g72.get("passes"),
        g72.get("best_cell"),
        g72.get("best_mean_reward", float("nan")),
        g72.get("delta_to_deployable", float("nan")),
        g72.get("delta_to_oracle", float("nan")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
