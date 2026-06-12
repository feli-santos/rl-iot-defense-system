"""ablation F12 — Security-vs-availability Pareto frontier (PLAN §3.1.6 / D7.5).

Plotter-only (D7.5): F12 is *derived* from F9 + F10 outputs, not a
separate sweep. Reads:

  runs/ablation/reward_sweep/<cell_id>/seed_<k>/eval_test.jsonl
  runs/ablation/aggressiveness/{ppo,rule}_p<p>/seed_<k>/eval_test.jsonl
  runs/benchmark/<policy>/seed_<k>/eval_test.jsonl  (benchmark anchors)

For each (cell, policy) point computes:

  security_gain     = 1 − compromise_rate
                      (mean over episodes of NOT info["compromised"])
  availability_cost = (BLOCK + ISOLATE share of all decisions)
                      (sum over stages of action_counts_by_stage[stage][3..4]
                       / total decisions)

Plots a 2-D scatter on (availability_cost, security_gain) with the
Pareto frontier highlighted.

Outputs:
- ``F12_pareto.png``
- ``F12_summary.json`` — every point + Pareto frontier set + G7.4
                          evaluation
- ``F12_caption.md`` (placeholder)
- ``F12_manifest.json`` (SHA chain over all input JSONLs)

Gate evaluation:

- **G7.4** — pass iff the Pareto frontier has ≥ 3 distinct dominant
  points (no single point dominates all of {security, availability}).
  This validates the "operating-point chooser" thesis contribution.
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

from src.blue_team.aggregation import read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_pareto")

_ROOT = Path(__file__).resolve().parents[2]


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


def _summarise_records(records: list[dict]) -> tuple[float, float, int]:
    """Return (security_gain, availability_cost, n_episodes) for a
    list of EpisodeRecord JSONL rows.

    Definitions (PLAN §3.1.6):
      security_gain     = 1 − compromise_rate
      availability_cost = (BLOCK + ISOLATE) decisions / total decisions
    """
    if not records:
        return math.nan, math.nan, 0

    compromised = [1.0 if r.get("compromised") else 0.0 for r in records]
    security_gain = 1.0 - float(np.mean(compromised))

    total_decisions = 0
    high_action_decisions = 0
    for r in records:
        # action_counts_by_stage is {stage_str: [counts_per_action]}.
        for _stage_key, counts in (r.get("action_counts_by_stage") or {}).items():
            if not counts or len(counts) < 5:
                continue
            total_decisions += sum(counts)
            high_action_decisions += counts[3] + counts[4]
    availability_cost = (
        float(high_action_decisions) / float(total_decisions) if total_decisions > 0 else math.nan
    )
    return security_gain, availability_cost, len(records)


def _summarise_seed_dirs(
    seed_dirs: list[Path],
    *,
    sha_collector: dict[str, str],
) -> tuple[float, float, int]:
    """Aggregate eval_test.jsonl across seed dirs and summarise."""
    all_records: list[dict] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        if not jsonl.exists():
            continue
        all_records.extend(read_episodes_jsonl(jsonl))
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.resolve().relative_to(_ROOT))] = sha
    return _summarise_records(all_records)


def _collect_phase6_points(
    benchmark_root: Path,
    sha_collector: dict[str, str],
) -> list[dict[str, Any]]:
    """benchmark anchor points: 8 baseline policies on test_balanced."""
    points: list[dict[str, Any]] = []
    if not benchmark_root.exists():
        logger.warning("phase6 root missing: %s — skipping anchors", benchmark_root)
        return points
    for policy_dir in sorted(benchmark_root.iterdir()):
        if not policy_dir.is_dir():
            continue
        seed_dirs = sorted(
            d for d in policy_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
        )
        if not seed_dirs:
            continue
        sec, avail, n_ep = _summarise_seed_dirs(
            seed_dirs,
            sha_collector=sha_collector,
        )
        points.append(
            {
                "source": "phase6",
                "policy": policy_dir.name,
                "label": policy_dir.name,
                "security_gain": sec,
                "availability_cost": avail,
                "n_episodes": n_ep,
            }
        )
    return points


def _collect_f9_points(
    f9_root: Path,
    sha_collector: dict[str, str],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if not f9_root.exists():
        logger.warning("F9 root missing: %s — skipping", f9_root)
        return points
    for cell_dir in sorted(f9_root.iterdir()):
        if not cell_dir.is_dir() or not (cell_dir / "cell_config.json").exists():
            continue
        cell_config = json.loads((cell_dir / "cell_config.json").read_text())
        seed_dirs = sorted(
            d for d in cell_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
        )
        sec, avail, n_ep = _summarise_seed_dirs(
            seed_dirs,
            sha_collector=sha_collector,
        )
        points.append(
            {
                "source": "f9_reward_sweep",
                "cell_id": cell_dir.name,
                "axis": cell_config.get("axis"),
                "component": cell_config.get("component"),
                "multiplier": cell_config.get("multiplier"),
                "label": f"f9:{cell_dir.name}",
                "security_gain": sec,
                "availability_cost": avail,
                "n_episodes": n_ep,
            }
        )
    return points


def _collect_f10_points(
    f10_root: Path,
    sha_collector: dict[str, str],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if not f10_root.exists():
        logger.warning("F10 root missing: %s — skipping", f10_root)
        return points
    for kind_dir in sorted(f10_root.iterdir()):
        # name = "ppo_p<X>" or "rule_p<X>".
        if not kind_dir.is_dir() or "_p" not in kind_dir.name:
            continue
        kind, p_slug = kind_dir.name.rsplit("_p", 1)
        try:
            p_value = float(p_slug.replace("p", "."))
        except ValueError:
            continue
        seed_dirs = sorted(
            d for d in kind_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
        )
        sec, avail, n_ep = _summarise_seed_dirs(
            seed_dirs,
            sha_collector=sha_collector,
        )
        points.append(
            {
                "source": "f10_aggressiveness",
                "kind": kind,
                "p_down": p_value,
                "label": f"f10:{kind}_p{p_value:.1f}",
                "security_gain": sec,
                "availability_cost": avail,
                "n_episodes": n_ep,
            }
        )
    return points


# --------------------------------------------------------------- pareto


def _pareto_frontier(points: list[dict[str, Any]]) -> list[int]:
    """Return indices into ``points`` that are on the Pareto frontier.

    Direction: higher security_gain is better; LOWER availability_cost
    is better. Point i is on the frontier iff no other point j has
    (security_gain_j > security_gain_i AND availability_cost_j ≤
    availability_cost_i) OR (security_gain_j ≥ security_gain_i AND
    availability_cost_j < availability_cost_i).
    """
    frontier: list[int] = []
    for i, pi in enumerate(points):
        si, ai = pi["security_gain"], pi["availability_cost"]
        if not (math.isfinite(si) and math.isfinite(ai)):
            continue
        dominated = False
        for j, pj in enumerate(points):
            if i == j:
                continue
            sj, aj = pj["security_gain"], pj["availability_cost"]
            if not (math.isfinite(sj) and math.isfinite(aj)):
                continue
            if (sj > si and aj <= ai) or (sj >= si and aj < ai):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def _evaluate_g74(points: list[dict[str, Any]], frontier: list[int]) -> dict[str, Any]:
    """G7.4: ≥ 3 distinct dominant Pareto points."""
    # Dedupe nearly-identical points to avoid over-counting numerical
    # ties (within 1 % on each axis).
    distinct: list[int] = []
    for idx in frontier:
        p = points[idx]
        is_distinct = True
        for d in distinct:
            other = points[d]
            if (
                abs(p["security_gain"] - other["security_gain"]) < 0.01
                and abs(p["availability_cost"] - other["availability_cost"]) < 0.01
            ):
                is_distinct = False
                break
        if is_distinct:
            distinct.append(idx)
    return {
        "passes": len(distinct) >= 3,
        "n_distinct_frontier_points": len(distinct),
        "frontier_indices": list(frontier),
        "frontier_distinct_indices": distinct,
        "interpretation": (
            f"PASS: Pareto frontier has {len(distinct)} distinct dominant "
            "points — non-trivial trade-off surface; operating-point choice "
            "is a real defender contribution."
            if len(distinct) >= 3
            else f"FAIL-WITH-FINDING (R7.3): only {len(distinct)} distinct "
            "dominant point(s) on the frontier. Under the tug-of-war dynamics the "
            "stage-aware proportional oracle (recommended_action) attains perfect "
            "security (security_gain=1.0) at near-zero availability cost, strictly "
            "dominating always_block (which also prevents 100% but at unit "
            "availability cost) and every interior learned policy. The "
            "security-availability trade-off therefore collapses to a single "
            "dominant operating point *given perfect stage perception*; the "
            "interior placement of the RL agents quantifies the cost of partial "
            "observability (POMDP), not a genuine multi-point frontier."
        ),
    }


# --------------------------------------------------------------- render


def _render(
    points: list[dict[str, Any]],
    frontier: list[int],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # Group by source for colour/marker.
    colours = {
        "phase6": "#9ca3af",
        "f9_reward_sweep": "#2563eb",
        "f10_aggressiveness": "#16a34a",
    }
    markers = {
        "phase6": "s",
        "f9_reward_sweep": "o",
        "f10_aggressiveness": "^",
    }
    label_done = set()
    for i, p in enumerate(points):
        if not (math.isfinite(p["security_gain"]) and math.isfinite(p["availability_cost"])):
            continue
        src = p["source"]
        label = src if src not in label_done else None
        label_done.add(src)
        ax.scatter(
            p["availability_cost"],
            p["security_gain"],
            c=colours.get(src, "#000"),
            marker=markers.get(src, "o"),
            s=80 if i in frontier else 40,
            edgecolors="black" if i in frontier else "none",
            linewidth=1.0 if i in frontier else 0,
            alpha=0.85,
            label=label,
            zorder=3,
        )

    # Frontier polyline (sorted by availability_cost ascending).
    front_pts = sorted(
        [
            points[i]
            for i in frontier
            if math.isfinite(points[i]["security_gain"])
            and math.isfinite(points[i]["availability_cost"])
        ],
        key=lambda p: p["availability_cost"],
    )
    if front_pts:
        ax.plot(
            [p["availability_cost"] for p in front_pts],
            [p["security_gain"] for p in front_pts],
            "--",
            color="#dc2626",
            linewidth=1.4,
            alpha=0.85,
            zorder=2,
            label="Pareto frontier",
        )

    ax.set_xlabel("Availability cost (BLOCK + ISOLATE share of decisions)", fontsize=10)
    ax.set_ylabel("Security gain (1 − compromise rate)", fontsize=10)
    ax.set_title(
        "F12 — Security vs. availability Pareto "
        "(F9 reward sweep + F10 aggressiveness sweep + benchmark anchors)",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F12 — security-vs-availability Pareto plot. "
        "Plotter-only (D7.5); reads F9 + F10 + benchmark outputs.",
    )
    p.add_argument("--phase6-runs", default="runs/benchmark")
    p.add_argument("--phase7-f9-runs", default="runs/ablation/reward_sweep")
    p.add_argument("--phase7-f10-runs", default="runs/ablation/aggressiveness")
    p.add_argument("--out-dir", default="docs/results/ablation")
    # Step-8 F2 (07_HANDOFF.md §5): explicit upstream-manifest SHA pins.
    p.add_argument(
        "--phase5-sweep-manifest",
        default="runs/blue_team/sweep_manifest.json",
        help="blue-team sweep_manifest.json (warm-start trained checkpoints).",
    )
    p.add_argument(
        "--phase1-splits-manifest",
        default="docs/results/dataset/manifest.json",
        help="dataset-prep splits manifest.json (post-3cd2fb9; SHA 1e99d596...).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    sha_collector: dict[str, str] = {}
    points: list[dict[str, Any]] = []
    points += _collect_phase6_points(Path(args.phase6_runs), sha_collector)
    points += _collect_f9_points(Path(args.phase7_f9_runs), sha_collector)
    points += _collect_f10_points(Path(args.phase7_f10_runs), sha_collector)
    logger.info("F12: collected %d points (phase6 + F9 + F10)", len(points))

    if not points:
        logger.error(
            "F12: no points collected — run phase-6, phase-7-reward, phase-7-aggressiveness first."
        )
        return 1

    frontier = _pareto_frontier(points)
    g74 = _evaluate_g74(points, frontier)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "F12_pareto.png"
    _render(points, frontier, png_path)

    summary = {
        "schema_version": "1.0",
        "phase": 7,
        "figure": "F12",
        "n_points_total": len(points),
        "n_frontier_points": len(frontier),
        "points": points,
        "frontier_indices": frontier,
        "gates": {"G7.4": g74},
        "headline": g74.get("interpretation", "?"),
    }
    (out_dir / "F12_summary.json").write_text(json.dumps(summary, indent=2))

    manifest = {
        "schema_version": "1.0",
        "figure": "F12",
        "git_sha": _git_sha(),
        "outputs": {
            "png": str(png_path),
            "json": str(out_dir / "F12_summary.json"),
        },
        "inputs": {
            "benchmark_eval_manifest": {
                "path": str(Path(args.phase6_runs) / "eval_manifest.json"),
                "sha256": _sha256(Path(args.phase6_runs) / "eval_manifest.json"),
            },
            "phase7_f9_sweep_manifest": {
                "path": str(Path(args.phase7_f9_runs) / "sweep_manifest.json"),
                "sha256": _sha256(Path(args.phase7_f9_runs) / "sweep_manifest.json"),
            },
            "phase7_f10_sweep_manifest": {
                "path": str(Path(args.phase7_f10_runs) / "sweep_manifest.json"),
                "sha256": _sha256(Path(args.phase7_f10_runs) / "sweep_manifest.json"),
            },
            # Step-8 F2: explicit upstream-manifest SHA pins so the F12
            # hash chain is self-contained (no transitive lookups).
            "blue_team_sweep_manifest": {
                "path": str(args.phase5_sweep_manifest),
                "sha256": _sha256(Path(args.phase5_sweep_manifest)),
            },
            "phase1_splits_manifest": {
                "path": str(args.phase1_splits_manifest),
                "sha256": _sha256(Path(args.phase1_splits_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F12_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F12_caption.md"
    caption_path.write_text(
        "**F12 — Security vs. availability Pareto.** Each point is one "
        "(reward_config, p_down) cell from F9 + F10, "
        "plus the eight benchmark anchor policies, with x = "
        "availability cost (BLOCK + ISOLATE share of decisions) and "
        "y = security gain (1 − compromise rate). The dashed red curve "
        "highlights the Pareto frontier. Squares = benchmark anchors; "
        "circles = F9 reward-sweep cells; triangles = F10 "
        "environment-difficulty (p_down) cells. Larger black-edged markers "
        "are on the frontier. (PLAN §3.1.6 / D7.5; G7.4 evaluator.)\n"
    )

    logger.info(
        "F12 written to %s — G7.4 passes=%s (%d distinct frontier points)",
        out_dir,
        g74.get("passes"),
        g74.get("n_distinct_frontier_points"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
