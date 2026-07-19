"""Generate the two per-stage feature-signature figures for the dataset chapter.

Two independent figures illustrating the CICIoT2023 kill-chain projection:
  (a) ``tex/figs/dataset_raw_traffic_a.pdf`` — PCA 2-D scatter of the 29-dim
      feature space, coloured by kill-chain stage, showing the distributional
      overlap that makes adjacent-stage observation aliasing (rate $\alpha$) a
      genuine partial-observability stressor.
  (b) ``tex/figs/dataset_raw_traffic_b.pdf`` — mean-feature heatmap per stage
      (5 rows x 29 columns, z-normalised), showing the per-stage traffic
      signatures that the supervised detector and the RL agent must discriminate.

Uses the shared journal-quality house style (``scripts/_plot_style.py``).
Run:  PYTHONPATH=. .venv/bin/python scripts/thesis/plot_dataset_raw_traffic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _plot_style import ACCENT, apply_house_style, save_figure

# --------------------------------------------------------------------- paths
DATA_DIR = Path("data/processed/ciciot2023")
OUT_PATH_A = Path("tex/figs/dataset_raw_traffic_a.pdf")
OUT_PATH_B = Path("tex/figs/dataset_raw_traffic_b.pdf")

FEATURE_NAMES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Duration",
    "Rate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "urg_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "TCP",
    "UDP",
    "ICMP",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Tot size",
    "IAT",
    "Covariance",
    "Variance",
]

STAGE_NAMES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]

# Colour-blind-safe 5-stage palette (greyscale-distinguishable by lightness).
STAGE_COLORS = [
    "#2166ac",  # blue   — BENIGN
    "#4d9221",  # green  — RECON
    "#e08214",  # amber  — ACCESS
    "#b2182b",  # red    — MANEUVER
    "#4d4d4d",  # grey   — IMPACT
]

# Subsample for the scatter (442k points is too dense for a PDF vector plot).
SCATTER_SUBSAMPLE = 12000


def main() -> None:
    apply_house_style()

    features = np.load(DATA_DIR / "features.npy")
    stages = np.load(DATA_DIR / "stages.npy").ravel()
    n_rows, n_features = features.shape
    assert n_features == 29, f"expected 29 features, got {n_features}"
    assert stages.max() <= 4, f"unexpected stage index {stages.max()}"

    rng = np.random.default_rng(seed=42)

    # --- Panel (a): PCA 2-D scatter -----------------------------------------
    # Subsample per stage so rare stages are visible alongside IMPACT (43.9%).
    idx_scatter = []
    for s in range(5):
        idx_s = np.where(stages == s)[0]
        take = min(len(idx_s), SCATTER_SUBSAMPLE // 5)
        idx_scatter.append(rng.choice(idx_s, size=take, replace=False))
    idx_scatter = np.concatenate(idx_scatter)
    rng.shuffle(idx_scatter)

    x_scatter = features[idx_scatter]
    y_scatter = stages[idx_scatter]

    # Standardise then PCA → 2-D (z-normalised so no single high-magnitude
    # feature like flow_duration dominates the projection).
    x_std = StandardScaler().fit_transform(x_scatter)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(x_std)

    # --- Panel (b): mean-feature heatmap -----------------------------------
    # Per-stage mean of each feature, then z-normalise across stages so the
    # heatmap shows *relative* signature differences regardless of magnitude.
    means = np.zeros((5, n_features))
    for s in range(5):
        means[s] = features[stages == s].mean(axis=0)
    # z-normalise per feature (column) across stages
    means_norm = (means - means.mean(axis=0, keepdims=True)) / (
        means.std(axis=0, keepdims=True) + 1e-12
    )

    # --- Figure (a): PCA scatter (own standalone figure) --------------------
    fig_a, ax_pca = plt.subplots(figsize=(7.2, 5.4))
    for s in range(5):
        mask = y_scatter == s
        ax_pca.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            c=STAGE_COLORS[s],
            s=12,
            alpha=0.45,
            edgecolors="none",
            label=f"{STAGE_NAMES[s]} (n={int((stages == s).sum()):,})",
            rasterized=True,
        )
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)")
    ax_pca.legend(loc="best", fontsize=9, markerscale=1.6, framealpha=0.85)
    ax_pca.axhline(0, color=ACCENT["muted"], lw=0.5, ls="-", alpha=0.4)
    ax_pca.axvline(0, color=ACCENT["muted"], lw=0.5, ls="-", alpha=0.4)
    fig_a.tight_layout()
    save_figure(fig_a, OUT_PATH_A)
    plt.close(fig_a)

    # --- Figure (b): mean-feature heatmap (own standalone figure) -----------
    fig_b, ax_heat = plt.subplots(figsize=(11.0, 3.8))
    im = ax_heat.imshow(
        means_norm,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(n_features))
    ax_heat.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7.5, ha="center")
    ax_heat.set_yticks(range(5))
    ax_heat.set_yticklabels(STAGE_NAMES, fontsize=10)
    ax_heat.set_xlabel("CICIoT2023 feature")
    cbar = fig_b.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.set_label("z-score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig_b.tight_layout()
    save_figure(fig_b, OUT_PATH_B)
    plt.close(fig_b)

    print(
        f"[dataset_raw_traffic] wrote {OUT_PATH_A} and {OUT_PATH_B}  "
        f"({n_rows:,} rows, {n_features} features)"
    )


if __name__ == "__main__":
    main()
