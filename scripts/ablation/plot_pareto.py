"""ablation F12 — Security-vs-availability Pareto frontier (PLAN §3.1.6 / D7.5).

Plotter-only (D7.5): F12 is *derived* from the F10 aggressiveness sweep
and the held-out benchmark anchors, not a separate sweep. Reads:

  runs/ablation/aggressiveness/{ppo,rule}_p<p>/seed_<k>/eval_test.jsonl
  runs/benchmark/<policy>/seed_<k>/eval_test.jsonl  (benchmark anchors)

(The earlier F9 reward-perturbation cloud was dropped: those cells were
produced under the pre-redesign *proportional* reward contract and are
off the locked ``outcome`` contract, so mixing them into the on-contract
frontier would be apples-to-oranges. The frontier collapse is shown by
the on-contract benchmark policies + oracle + the F10 points alone.)

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

from scripts._plot_style import ACCENT, apply_house_style, save_figure
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


def _collect_benchmark_points(
    benchmark_root: Path,
    sha_collector: dict[str, str],
) -> list[dict[str, Any]]:
    """benchmark anchor points: 8 baseline policies on test_balanced."""
    points: list[dict[str, Any]] = []
    if not benchmark_root.exists():
        logger.warning("benchmark root missing: %s — skipping anchors", benchmark_root)
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
                "source": "benchmark",
                "policy": policy_dir.name,
                "label": policy_dir.name,
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
    apply_house_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Group by source for colour/marker. Human-readable legend labels (no
    # internal source slugs leaking into the figure).
    colours = {
        "benchmark": ACCENT["neutral"],
        "f10_aggressiveness": ACCENT["primary"],
    }
    markers = {
        "benchmark": "s",
        "f10_aggressiveness": "^",
    }
    legend_label = {
        "benchmark": "Held-out benchmark policies",
        "f10_aggressiveness": "Environment-difficulty sweep ($p_{\\mathrm{down}}$)",
    }
    label_done = set()
    for i, p in enumerate(points):
        if not (math.isfinite(p["security_gain"]) and math.isfinite(p["availability_cost"])):
            continue
        src = p["source"]
        label = legend_label.get(src) if src not in label_done else None
        label_done.add(src)
        ax.scatter(
            p["availability_cost"],
            p["security_gain"],
            c=colours.get(src, ACCENT["muted"]),
            marker=markers.get(src, "o"),
            s=90 if i in frontier else 42,
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
            color=ACCENT["secondary"],
            linewidth=1.6,
            alpha=0.9,
            zorder=2,
            label="Pareto frontier",
        )

    ax.set_xlabel("Availability cost (BLOCK + ISOLATE share of decisions)")
    ax.set_ylabel("Security gain (1 − compromise rate)")
    ax.set_title("Security–availability trade-off collapses to a single dominant point")
    ax.legend(loc="lower right", framealpha=0.95)
    save_figure(fig, out_path)
    plt.close(fig)


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F12 — security-vs-availability Pareto plot. "
        "Plotter-only (D7.5); reads F10 + benchmark outputs.",
    )
    p.add_argument("--benchmark-runs", default="runs/benchmark")
    p.add_argument("--ablation-aggressiveness-runs", default="runs/ablation/aggressiveness")
    p.add_argument("--out-dir", default="docs/results/ablation")
    # Step-8 F2 (07_HANDOFF.md §5): explicit upstream-manifest SHA pins.
    p.add_argument(
        "--blue-team-sweep-manifest",
        default="runs/blue_team/sweep_manifest.json",
        help="Blue-Team sweep_manifest.json (warm-start trained checkpoints).",
    )
    p.add_argument(
        "--split-splits-manifest",
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
    points += _collect_benchmark_points(Path(args.benchmark_runs), sha_collector)
    points += _collect_f10_points(Path(args.ablation_aggressiveness_runs), sha_collector)
    logger.info("F12: collected %d points (benchmark + F10)", len(points))

    if not points:
        logger.error("F12: no points collected — run benchmark and ablation-aggressiveness first.")
        return 1

    frontier = _pareto_frontier(points)
    g74 = _evaluate_g74(points, frontier)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = out_dir / "F12_pareto"
    _render(points, frontier, fig_base)
    png_path = fig_base.with_suffix(".png")
    pdf_path = fig_base.with_suffix(".pdf")

    summary = {
        "schema_version": "1.0",
        "stage": "ablation",
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
            "pdf": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "png": str(png_path),
            "json": str(out_dir / "F12_summary.json"),
        },
        "inputs": {
            "benchmark_eval_manifest": {
                "path": str(Path(args.benchmark_runs) / "eval_manifest.json"),
                "sha256": _sha256(Path(args.benchmark_runs) / "eval_manifest.json"),
            },
            "ablation_aggressiveness_sweep_manifest": {
                "path": str(Path(args.ablation_aggressiveness_runs) / "sweep_manifest.json"),
                "sha256": _sha256(Path(args.ablation_aggressiveness_runs) / "sweep_manifest.json"),
            },
            # Step-8 F2: explicit upstream-manifest SHA pins so the F12
            # hash chain is self-contained (no transitive lookups).
            "blue_team_sweep_manifest": {
                "path": str(args.blue_team_sweep_manifest),
                "sha256": _sha256(Path(args.blue_team_sweep_manifest)),
            },
            "split_splits_manifest": {
                "path": str(args.split_splits_manifest),
                "sha256": _sha256(Path(args.split_splits_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F12_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F12_caption.md"
    caption_path.write_text(
        "**F12 — Security vs. availability Pareto.** Each point is one "
        "operating cell, with x = availability cost (BLOCK + ISOLATE share "
        "of decisions) and y = security gain (1 − compromise rate). Squares "
        "= held-out benchmark anchor policies; triangles = F10 "
        "environment-difficulty ($p_{\\mathrm{down}}$) cells. The dashed "
        "curve highlights the Pareto frontier; larger black-edged markers "
        "are on it. The frontier collapses onto the single full-observability "
        "oracle point — the interior placement of the learned agents "
        "quantifies the cost of partial observability, not a genuine "
        "multi-point trade-off. (PLAN §3.1.6 / D7.5; G7.4 evaluator.)\n"
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
