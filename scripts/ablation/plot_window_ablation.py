"""sensitivity-sweep FA_window — plot window_size ablation (C22).

Reads per-seed eval.jsonl files from the window ablation sweep directories
and produces a bar chart ``tex/figs/FA_window_ablation.png`` plus a JSON
summary.

Usage::

    python -m scripts.ablation.plot_window_ablation \\
        --sweep-root runs/ablation_window \\
        --out-dir tex/figs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


def _bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Return (low, high) 95% bootstrap CI for the mean."""
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _read_final_rewards(run_dir: Path, fraction: float = 0.1) -> list[float]:
    """Read last ``fraction`` of eval episodes from eval.jsonl."""
    eval_path = run_dir / "eval.jsonl"
    if not eval_path.exists():
        logger.warning("eval.jsonl not found in %s", run_dir)
        return []
    lines = [json.loads(ln) for ln in eval_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    cutoff = max(1, int(len(lines) * (1 - fraction)))
    return [row["episode_reward"] for row in lines[cutoff:] if "episode_reward" in row]


def build_summary(sweep_root: str) -> dict[str, Any]:
    """Load sweep manifest and compute per-window_size mean/CI."""
    root = Path(sweep_root)
    manifest_path = root / "sweep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    window_sizes = sorted({r["window_size"] for r in manifest["runs"]})

    rows = []
    for w in window_sizes:
        all_rewards: list[float] = []
        seed_rows = [r for r in manifest["runs"] if r["window_size"] == w]
        for run in seed_rows:
            if not run["ok"]:
                continue
            run_dir = Path(run["out_dir"])
            # Handle symlinks to blue_team_primary or reused_from.txt
            reused_txt = run_dir / "reused_from.txt"
            if not (run_dir / "eval.jsonl").exists() and reused_txt.exists():
                run_dir = Path(reused_txt.read_text().strip())
            all_rewards.extend(_read_final_rewards(run_dir))

        if not all_rewards:
            logger.warning("No rewards for window_size=%d", w)
            rows.append(
                {
                    "window_size": w,
                    "n_episodes": 0,
                    "mean_reward": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "obs_dim": w * 29 * 2,
                }
            )
            continue

        mean_r = float(np.mean(all_rewards))
        ci_low, ci_high = _bootstrap_ci(all_rewards)
        rows.append(
            {
                "window_size": w,
                "obs_dim": w * 29 * 2,  # w × features × 2 (with deltas)
                "n_episodes": len(all_rewards),
                "mean_reward": round(mean_r, 3),
                "ci_low": round(ci_low, 3),
                "ci_high": round(ci_high, 3),
            }
        )
        logger.info(
            "w=%d  obs_dim=%d  n=%d  mean=%.1f  95%%CI=[%.1f, %.1f]",
            w,
            w * 29 * 2,
            len(all_rewards),
            mean_r,
            ci_low,
            ci_high,
        )

    return {
        "schema_version": "1.0",
        "figure": "FA_window_ablation",
        "sweep_root": str(sweep_root),
        "rows": rows,
    }


def plot(summary: dict[str, Any], out_dir: str) -> Path:
    """Render bar chart. Returns path to saved PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    windows = [r["window_size"] for r in rows]
    means = [r["mean_reward"] for r in rows]
    errs_low = [r["mean_reward"] - r["ci_low"] for r in rows]
    errs_high = [r["ci_high"] - r["mean_reward"] for r in rows]

    # Colour the primary (w=5) bar differently
    colors = ["#4878CF" if w != 5 else "#F5A623" for w in windows]

    x = np.arange(len(windows))
    width = 0.55

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        x,
        means,
        width=width,
        yerr=[errs_low, errs_high],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )

    # Annotate means
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{mean_val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"w={w}" for w in windows])
    ax.set_xlabel("Observation Window Size (rows)", fontsize=11)
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.set_title(
        "Window-Length Ablation (PPO, 3 seeds × 30 ep, last 10%)\nOrange = primary contract (w=5)",
        fontsize=9,
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()

    out_path = Path(out_dir) / "FA_window_ablation.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("FA_window_ablation: wrote %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="sensitivity-sweep FA_window — plot window_size ablation (C22).",
    )
    p.add_argument("--sweep-root", default="runs/ablation_window")
    p.add_argument("--out-dir", default="tex/figs/")
    p.add_argument("--fraction", type=float, default=0.1)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    summary = build_summary(args.sweep_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "FA_window_ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", json_path)

    plot(summary, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
