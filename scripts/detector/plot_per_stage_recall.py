"""F11 / Fig. 4.3 — standalone re-plotter for the stage-detector figure.

Renders the per-stage recall bar chart (production MLP ``StageDetector`` vs the
supervised ``RandomForest`` baseline) plus the detector confusion matrix
**directly from the committed ``F11_summary.json``** — no detector retraining
required. This keeps the thesis figure in sync with the canonical detector
numbers (RF macro-F1 = 0.918, StageDetector macro-F1 = 0.835 on
``test_balanced``) and emits a vector ``per_stage_recall.pdf`` that matches the
thesis ``\\includegraphics`` target.

Usage::

    python -m scripts.detector.plot_per_stage_recall \\
        --summary docs/results/stage-detector/F11_summary.json \\
        --out-dir docs/results/stage-detector
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scripts._plot_style import ACCENT, apply_house_style, save_figure, sha256_file

logger = logging.getLogger("scripts.detector.plot_per_stage_recall")

_ROOT = Path(__file__).resolve().parents[2]

# Canonical kill-chain stage order (matches tests/conftest.py).
STAGE_NAMES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]


def _recall_vector(per_stage_recall: dict[str, float]) -> np.ndarray:
    """Order a {stage_name: recall} dict into the canonical stage order."""
    return np.array([float(per_stage_recall[s]) for s in STAGE_NAMES], dtype=np.float64)


def _render(summary: dict[str, Any], out_path: Path) -> None:
    apply_house_style()
    import matplotlib.pyplot as plt

    models = summary["models"]
    sd = models["StageDetector"]["test_balanced"]
    rf = models["RandomForest"]["test_balanced"]
    n_stages = len(STAGE_NAMES)

    fig, (ax_bar, ax_cm) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # ---- Left: per-stage recall, grouped bars (RF emphasised as the winner).
    width = 0.38
    x = np.arange(n_stages)
    series = [
        ("RandomForest", _recall_vector(rf["per_stage_recall"]), ACCENT["secondary"]),
        ("StageDetector (MLP)", _recall_vector(sd["per_stage_recall"]), ACCENT["primary"]),
    ]
    for i, (name, rec, colour) in enumerate(series):
        ax_bar.bar(
            x + (i - 0.5) * width,
            rec,
            width=width,
            label=name,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )

    best_name, best_f1 = max(
        (("RandomForest", rf["macro_f1"]), ("StageDetector (MLP)", sd["macro_f1"])),
        key=lambda kv: kv[1],
    )
    ax_bar.axhline(
        float(best_f1),
        ls="--",
        color=ACCENT["neutral"],
        alpha=0.8,
        label=f"best macro-F1 = {best_f1:.3f} ({best_name})",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(STAGE_NAMES, rotation=15)
    ax_bar.set_ylabel("Recall")
    ax_bar.set_title("Per-stage recall on test_balanced")
    ax_bar.set_ylim(0, 1.02)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(loc="lower left", fontsize=8)

    # ---- Right: StageDetector confusion matrix (row-normalised %).
    cm = np.asarray(sd["confusion_matrix"], dtype=np.float64)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm / row_sums

    im = ax_cm.imshow(cm_norm, cmap="Greens", vmin=0.0, vmax=1.0)
    ax_cm.set_xticks(range(n_stages))
    ax_cm.set_yticks(range(n_stages))
    ax_cm.set_xticklabels(STAGE_NAMES, rotation=15)
    ax_cm.set_yticklabels(STAGE_NAMES)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("StageDetector confusion (test_balanced)")
    for i in range(n_stages):
        for j in range(n_stages):
            ax_cm.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:4.1f}%",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="row-normalised")

    fig.suptitle(
        "Stage detection: detector head vs supervised baselines",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-render the F11 / Fig. 4.3 stage-detector figure from "
        "the committed F11_summary.json (no retraining).",
    )
    p.add_argument(
        "--summary",
        default="docs/results/stage-detector/F11_summary.json",
        help="Path to the canonical F11_summary.json.",
    )
    p.add_argument(
        "--out-dir",
        default="docs/results/stage-detector",
        help="Directory to write per_stage_recall.{pdf,png} + manifest.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    summary_path = Path(args.summary)
    if not summary_path.exists():
        logger.error("summary not found: %s — run `make detector` first.", summary_path)
        return 1
    summary = json.loads(summary_path.read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = out_dir / "per_stage_recall"
    _render(summary, fig_base)
    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")
    logger.info("wrote %s and %s", pdf_path, png_path)

    # Manifest: this figure is a pure re-render of F11_summary.json.
    manifest = {
        "version": "1.0",
        "figure": "per_stage_recall",
        "produced_by": "scripts/detector/plot_per_stage_recall.py",
        "inputs": {
            "F11_summary.json": sha256_file(summary_path),
        },
        "outputs": {
            "per_stage_recall.pdf": sha256_file(pdf_path),
            "per_stage_recall.png": sha256_file(png_path),
        },
    }
    (out_dir / "per_stage_recall_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
