"""F11 / Fig. 4.3 — tuned RandomForest stage-detection figure.

Renders the stage-detection figure for the **tuned RandomForest** that
underlies the deployable RF-Acting baseline — the only supervised detector
that feeds a thesis result. (The lightweight production MLP is no longer
reported: it fed no RL result, since the trained agents observe only the
290-dim windowed feature vector with ``include_stage_pred=False``.)

Two panels, both on the held-out ``test_balanced`` split:
  * per-class F1 (bar chart) with the macro-F1 reference line, showing the RF
    is strong across *all* five stages (worst class ~0.87), not merely riding
    the trivially-separable IMPACT class;
  * the row-normalised confusion matrix, exposing the honest RECON<->ACCESS
    overlap as the only material confusion.

Sourced from ``docs/results/stage-detector/tuned_rf_stage_detection.json``
(produced by re-scoring ``artifacts/detector/random_forest.joblib`` on
``test_balanced`` with raw features / no scaler, matching the benchmark
``RFActingPolicy``). No detector retraining required.

Usage::

    python -m scripts.detector.plot_tuned_rf_detection \\
        --summary docs/results/stage-detector/tuned_rf_stage_detection.json \\
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

logger = logging.getLogger("scripts.detector.plot_tuned_rf_detection")

# Canonical kill-chain stage order (matches tests/conftest.py).
STAGE_NAMES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]


def _ordered(vals: dict[str, float]) -> np.ndarray:
    """Order a {stage_name: value} dict into the canonical stage order."""
    return np.array([float(vals[s]) for s in STAGE_NAMES], dtype=np.float64)


def _render_per_class_f1(summary: dict[str, Any], out_path: Path) -> None:
    """Per-class F1 bar chart for the tuned RF with a macro-F1 reference line."""
    apply_house_style()
    import matplotlib.pyplot as plt

    per_class = _ordered(summary["per_class_f1"])
    macro = float(summary["macro_f1"])
    n_stages = len(STAGE_NAMES)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.6))
    x = np.arange(n_stages)
    bars = ax.bar(
        x,
        per_class,
        width=0.62,
        color=ACCENT["secondary"],
        edgecolor="white",
        linewidth=0.6,
        label="tuned RandomForest",
    )
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)

    ax.axhline(
        macro,
        ls="--",
        color=ACCENT["neutral"],
        alpha=0.85,
        label=f"macro-F1 = {macro:.3f}",
    )

    plateau = summary.get("tuning_plateau")
    if plateau:
        ax.annotate(
            "grid plateau: val macro-F1 "
            f"{plateau['mean']:.3f}$\\pm${plateau['sd']:.3f}\n"
            f"over all 54 configs (span {plateau['spread']:.3f})",
            xy=(0.985, 0.045),
            xycoords="axes fraction",
            fontsize=8.5,
            color=ACCENT["neutral"],
            ha="right",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES, rotation=15)
    ax.set_ylabel("F1")
    ax.set_title("Tuned RandomForest per-class F1 on test_balanced")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _render_confusion(summary: dict[str, Any], out_path: Path) -> None:
    """Tuned-RF row-normalised confusion matrix on test_balanced."""
    apply_house_style()
    import matplotlib.pyplot as plt

    n_stages = len(STAGE_NAMES)
    cm = np.asarray(summary["confusion_matrix"], dtype=np.float64)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 6.4))
    im = ax.imshow(cm_norm, cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_stages))
    ax.set_yticks(range(n_stages))
    ax.set_xticklabels(STAGE_NAMES, rotation=15)
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Tuned RandomForest confusion (test_balanced)")
    for i in range(n_stages):
        for j in range(n_stages):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j] * 100:4.1f}%",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render the tuned-RF stage-detection figure "
        "(per-class F1 + confusion) from tuned_rf_stage_detection.json.",
    )
    p.add_argument(
        "--summary",
        default="docs/results/stage-detector/tuned_rf_stage_detection.json",
        help="Path to the tuned-RF detection summary JSON.",
    )
    p.add_argument(
        "--out-dir",
        default="docs/results/stage-detector",
        help="Directory to write tuned_rf_per_class_f1.{pdf,png} + confusion + manifest.",
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
        logger.error("summary not found: %s", summary_path)
        return 1
    summary = json.loads(summary_path.read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f1_base = out_dir / "tuned_rf_per_class_f1"
    cm_base = out_dir / "tuned_rf_confusion"
    _render_per_class_f1(summary, f1_base)
    _render_confusion(summary, cm_base)

    f1_pdf = f1_base.with_suffix(".pdf")
    f1_png = f1_base.with_suffix(".png")
    cm_pdf = cm_base.with_suffix(".pdf")
    cm_png = cm_base.with_suffix(".png")
    logger.info("wrote %s, %s, %s, %s", f1_pdf, f1_png, cm_pdf, cm_png)

    manifest = {
        "version": "1.0",
        "figure": "tuned_rf_stage_detection",
        "produced_by": "scripts/detector/plot_tuned_rf_detection.py",
        "inputs": {
            "tuned_rf_stage_detection.json": sha256_file(summary_path),
        },
        "outputs": {
            "tuned_rf_per_class_f1.pdf": sha256_file(f1_pdf),
            "tuned_rf_per_class_f1.png": sha256_file(f1_png),
            "tuned_rf_confusion.pdf": sha256_file(cm_pdf),
            "tuned_rf_confusion.png": sha256_file(cm_png),
        },
    }
    (out_dir / "tuned_rf_stage_detection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
