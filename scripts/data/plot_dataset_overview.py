"""
Phase-1 figure F0 — Dataset overview.

Reads the processed CICIoT2023 snapshot and the splits manifest, and produces:

- ``docs/results/01_dataset/F0_class_distribution.png``
  Bar chart: 34 CICIoT2023 attack classes (sorted by Kill Chain stage),
  showing the post-rebalance counts.
- ``docs/results/01_dataset/F0_stage_distribution.png``
  Bar chart: 5 Kill Chain stages, with split-by-split overlay
  (train / val / test).

A side-effect of this script is also a small JSON
``docs/results/01_dataset/F0_summary.json`` with the numbers used in the
captions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.label_mapper import STAGE_NAMES, AbstractStateLabelMapper  # noqa: E402

LOG = logging.getLogger("plot_dataset_overview")

# Stage colors chosen to be color-blind safe (Okabe-Ito).
STAGE_COLORS = {
    0: "#0072B2",  # blue
    1: "#009E73",  # green
    2: "#F0E442",  # yellow
    3: "#E69F00",  # orange
    4: "#D55E00",  # red
}


def _load(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    string_labels = np.asarray(np.load(processed_dir / "labels.npy", allow_pickle=False))
    if string_labels.ndim != 1:
        string_labels = string_labels.ravel()

    mapper = AbstractStateLabelMapper()
    stage_ids = np.asarray([mapper.get_stage_id(str(lbl)) for lbl in string_labels])
    return string_labels, stage_ids


def _plot_class_distribution(
    string_labels: np.ndarray, stage_ids: np.ndarray, out_path: Path
) -> dict:
    import matplotlib.pyplot as plt

    counts = Counter(string_labels.tolist())
    # Sort labels by (stage, count desc) so the chart reads left-to-right
    # along the kill-chain.
    mapper = AbstractStateLabelMapper()
    items = sorted(
        counts.items(),
        key=lambda kv: (mapper.get_stage_id(kv[0]), -kv[1]),
    )
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [STAGE_COLORS[mapper.get_stage_id(lbl)] for lbl in labels]

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_ylabel("Rows in snapshot (log scale)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title("CICIoT2023 — class distribution after rebalancing (n = 442 237)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Stage legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[s], label=f"{s} {STAGE_NAMES[s]}")
        for s in sorted(STAGE_COLORS)
    ]
    ax.legend(handles=handles, loc="upper right", title="Kill Chain stage", fontsize=8)

    # Annotate the smallest classes (≤ 5 000 rows) for the
    # under-representation caveat.
    for bar, lbl, val in zip(bars, labels, values):
        if val <= 5_000:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 1.05,
                f"{val:,}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)
    return dict(items)


def _plot_stage_distribution(
    stage_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    out_path: Path,
) -> dict:
    import matplotlib.pyplot as plt

    stages = list(range(5))
    series: dict[str, list[int]] = {}
    for split_name, idx in splits.items():
        c = Counter(int(x) for x in stage_ids[idx].tolist())
        series[split_name] = [c.get(s, 0) for s in stages]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    width = 0.25
    offsets = {"train": -width, "val": 0.0, "test": width}
    hatches = {"train": "", "val": "//", "test": "xx"}
    for split_name, vals in series.items():
        if split_name not in offsets:
            continue
        xs = np.arange(len(stages)) + offsets[split_name]
        bars = ax.bar(
            xs, vals, width=width, label=split_name,
            edgecolor="black", linewidth=0.4,
            color=[STAGE_COLORS[s] for s in stages],
            hatch=hatches[split_name],
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels([f"{s}\n{STAGE_NAMES[s]}" for s in stages])
    ax.set_ylabel("Rows")
    ax.set_title("Kill-Chain stage distribution per split (seed=42)")
    ax.legend(title="Split", loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out_path)
    return series


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed/ciciot2023"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/results/01_dataset"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )

    splits_dir = args.processed_dir / "splits"
    if not splits_dir.exists():
        raise FileNotFoundError(
            f"{splits_dir} not found — run scripts/data/build_split_indices.py first."
        )

    string_labels, stage_ids = _load(args.processed_dir)
    splits = {
        name: np.load(splits_dir / f"{name}.idx.npy")
        for name in ("train", "val", "test")
    }

    class_counts = _plot_class_distribution(
        string_labels, stage_ids, args.out_dir / "F0_class_distribution.png"
    )
    stage_per_split = _plot_stage_distribution(
        stage_ids, splits, args.out_dir / "F0_stage_distribution.png"
    )

    summary = {
        "figure_id": "F0",
        "n_total": int(string_labels.size),
        "n_classes": len(class_counts),
        "n_stages": 5,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "stage_per_split": {
            split: [int(x) for x in vals] for split, vals in stage_per_split.items()
        },
    }
    summary_path = args.out_dir / "F0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOG.info("Wrote %s", summary_path)

    print(json.dumps({k: v for k, v in summary.items() if k != "class_counts"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
