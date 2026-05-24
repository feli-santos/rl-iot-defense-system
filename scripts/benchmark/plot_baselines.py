"""benchmark F8 — RL vs non-RL baselines bar chart (PLAN §3.1.8, C7).

Horizontal bar chart of per-policy ``mean_reward`` with 95 % bootstrap
CIs, sorted descending. A horizontal reference line marks the
recommended-action floor (the IoTWarden hand-crafted rule baseline).

Reads the F5 summary JSON directly so F8's numbers are
**guaranteed identical** to the F5 table — there is one source of
per-policy aggregation in benchmark, and that is
:mod:`scripts.benchmark.build_summary_table`. F8 simply re-renders
the same numbers in a different visual idiom.

Outputs:
- ``F8_baselines.png``  — horizontal bar chart with CI whiskers.
- ``F8_summary.json``   — sorted per-policy {mean, ci_low, ci_high} +
                          G6.5 separation analysis (does the trained-RL
                          CI overlap any non-RL CI?).
- ``F8_manifest.json``  — SHA-256 hash chain.
- ``F8_caption.md``     — thesis caption (separate, hand-written).

G6.5 evaluation (PLAN §3.4): the gate passes if the trained-RL row's
95 % bootstrap CI does **not overlap** any of {random, always_observe,
always_block, rf_acting, recommended_action}'s CIs. Per D6.2.1 the
*direction* of the non-overlap is no longer required to be "RL above
the rule baseline" — F5 already showed the rule baseline strictly
dominates RL on test_balanced. F8 records the overlap analysis
faithfully and lets RESULTS interpret it.
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

logger = logging.getLogger("scripts.benchmark.plot_baselines")

_ROOT = Path(__file__).resolve().parents[2]


# Same display ordering as F5; F8 sorts by mean_reward at render time.
_DISPLAY: dict[str, str] = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "random": "Random",
    "always_observe": "Always-OBSERVE",
    "always_block": "Always-BLOCK",
    "recommended_action": "Recommended-Action (rule)",
    "rf_acting": "RF-Acting (supervised + rules)",
}

_RL_POLICIES = {"dqn", "ppo", "a2c"}
_NON_RL_BASELINES = {
    "random",
    "always_observe",
    "always_block",
    "recommended_action",
    "rf_acting",
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


def _ci_overlap(
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    """Return True if two intervals (a_low, a_high) and (b_low, b_high)
    overlap (closed intervals)."""
    a_low, a_high = a
    b_low, b_high = b
    return not (a_high < b_low or b_high < a_low)


def _evaluate_g65(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-RL-policy: does its CI overlap any non-RL baseline CI?

    Returns a dict with one key per RL policy mapping to:
      ``{"overlaps_with": [...], "g65_pass": bool}``

    The gate passes when no overlap exists with any non-RL baseline.
    The *direction* of separation (RL > rule, or rule > RL) is not
    encoded in G6.5 — the test_balanced revision (D6.2.1) makes the
    rule baseline the dominant policy, so this gate now records "no
    overlap" in either direction.
    """
    by_pol = {r["policy"]: r for r in rows}
    out: dict[str, Any] = {}
    for rl in [p for p in ("dqn", "ppo", "a2c") if p in by_pol]:
        rl_ci = (by_pol[rl]["mean_reward_ci_low"], by_pol[rl]["mean_reward_ci_high"])
        overlaps: list[str] = []
        for base in [b for b in _NON_RL_BASELINES if b in by_pol]:
            base_ci = (
                by_pol[base]["mean_reward_ci_low"],
                by_pol[base]["mean_reward_ci_high"],
            )
            if any(not math.isfinite(v) for v in (*rl_ci, *base_ci)):
                continue
            if _ci_overlap(rl_ci, base_ci):
                overlaps.append(base)
        out[rl] = {
            "overlaps_with": overlaps,
            "g65_pass": len(overlaps) == 0,
        }
    return out


def _render(
    rows: list[dict[str, Any]],
    rec_floor: float | None,
    out_path: Path,
) -> None:
    """Horizontal bar chart, sorted by mean_reward descending."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort ascending so the top row visually has the highest reward.
    sorted_rows = sorted(rows, key=lambda r: r["mean_reward"])
    labels = [_DISPLAY.get(r["policy"], r["policy"]) for r in sorted_rows]
    means = [r["mean_reward"] for r in sorted_rows]
    lo_err = [max(r["mean_reward"] - r["mean_reward_ci_low"], 0.0) for r in sorted_rows]
    hi_err = [max(r["mean_reward_ci_high"] - r["mean_reward"], 0.0) for r in sorted_rows]
    yerr = [lo_err, hi_err]
    colours = ["#2563eb" if r["policy"] in _RL_POLICIES else "#9ca3af" for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    bars = ax.barh(
        labels,
        means,
        xerr=yerr,
        color=colours,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"ecolor": "black", "capsize": 4, "lw": 1.0},
    )
    for bar, m, lo, hi in zip(
        bars,
        means,
        [r["mean_reward_ci_low"] for r in sorted_rows],
        [r["mean_reward_ci_high"] for r in sorted_rows],
    ):
        # Place annotation past the right error whisker so it never
        # overlaps the bar visually.
        x = m + (max(0.0, hi - m)) + 30.0
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{m:.0f}  ({lo:.0f}, {hi:.0f})",
            va="center",
            ha="left",
            fontsize=8,
        )

    if rec_floor is not None and math.isfinite(rec_floor):
        ax.axvline(
            rec_floor,
            color="#dc2626",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
            label=f"Recommended-Action floor ({rec_floor:.0f})",
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    ax.set_xlabel("Mean episodic reward on test_balanced (95 % bootstrap CI)", fontsize=10)
    ax.set_title(
        "F8 — RL vs Non-RL Baselines (n=150 deterministic episodes / policy)",
        fontsize=11,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    # Headroom for the long annotations.
    ax.set_xlim(min(means) - 200, max([r["mean_reward_ci_high"] for r in sorted_rows]) + 600)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="benchmark F8 — baselines bar chart.")
    p.add_argument("--runs-root", default="runs/benchmark")
    p.add_argument("--out-dir", default="docs/results/06_benchmark")
    p.add_argument(
        "--f5-summary",
        default="docs/results/06_benchmark/F5_summary.json",
        help="Path to F5_summary.json (the source of per-policy means + CIs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    f5_path = Path(args.f5_summary)
    if not f5_path.exists():
        logger.error(
            "F5_summary.json not found at %s — run "
            "`python -m scripts.benchmark.build_summary_table` first.",
            f5_path,
        )
        return 1
    f5 = json.loads(f5_path.read_text())
    rows: list[dict[str, Any]] = f5["rows"]
    if not rows:
        logger.error("F5 rows are empty; cannot render F8")
        return 1

    rec_row = next(
        (r for r in rows if r["policy"] == "recommended_action"),
        None,
    )
    rec_floor = float(rec_row["mean_reward"]) if rec_row is not None else None

    g65 = _evaluate_g65(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "F8_baselines.png"
    _render(rows, rec_floor, png_path)

    # Sort copy for the JSON output (mirror the visual ordering).
    sorted_rows = sorted(rows, key=lambda r: -r["mean_reward"])
    summary = {
        "schema_version": "1.0",
        "phase": 6,
        "figure": "F8",
        "recommended_action_floor": rec_floor,
        "best_policy": sorted_rows[0]["policy"] if sorted_rows else None,
        "rows_sorted_desc": [
            {
                "policy": r["policy"],
                "mean_reward": r["mean_reward"],
                "ci_low": r["mean_reward_ci_low"],
                "ci_high": r["mean_reward_ci_high"],
                "n_episodes": r["n_episodes"],
                "is_rl": r["policy"] in _RL_POLICIES,
            }
            for r in sorted_rows
        ],
        "g65": g65,
    }
    (out_dir / "F8_summary.json").write_text(json.dumps(summary, indent=2))

    eval_manifest_path = Path(args.runs_root) / "eval_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F8",
        "git_sha": _git_sha(),
        "outputs": {
            "png": str(png_path),
            "json": str(out_dir / "F8_summary.json"),
        },
        "inputs": {
            "f5_summary": {
                "path": str(f5_path),
                "sha256": _sha256(f5_path),
            },
            "benchmark_eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
        },
    }
    (out_dir / "F8_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "F8 written to %s (best=%s, rec_floor=%.1f)",
        out_dir,
        sorted_rows[0]["policy"] if sorted_rows else "—",
        rec_floor if rec_floor is not None else float("nan"),
    )
    for rl, info in g65.items():
        logger.info(
            "G6.5 %s: pass=%s (overlaps with %s)",
            rl,
            info["g65_pass"],
            info["overlaps_with"] or "[]",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
