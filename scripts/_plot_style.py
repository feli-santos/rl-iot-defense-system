"""Shared journal-quality plotting style for all thesis figures.

This module centralises the look-and-feel that was previously copy-pasted
(and drifted) across every ``scripts/**/plot_*.py``. Import it once at the
top of a plotter, call :func:`apply_house_style` before building figures,
and use :func:`save_figure` to emit a vector PDF + raster PNG pair with
consistent DPI and tight bounding boxes.

The palette is colour-blind-safe (green/red/grey are distinguishable under
deuteranopia and in greyscale by virtue of differing lightness + marker
shape). It is lifted from the cleanest existing figure, the aliasing
alpha-curve (Fig. 4.6), so the whole thesis reads as one visual family.

Design goals (MSc-thesis / journal grade):
  * Vector PDF is the primary artefact (LaTeX ``\\includegraphics``); PNG is
    a convenience preview at the same logical size.
  * One palette, one font stack, one grid treatment everywhere.
  * Policy colours/markers/labels are defined ONCE so every figure that
    shows PPO/DQN/A2C/RF/Oracle agrees.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------- palette

# Canonical per-policy display order, labels, and styling. RL agents (the
# contribution) lead; the supervised baseline and the full-observability
# oracle follow. PPO and RF-Acting are the headline contrast, so they get
# the heaviest lines + saturated colours; DQN/A2C are lighter greens.
POLICY_ORDER: tuple[str, ...] = (
    "ppo",
    "dqn",
    "a2c",
    "rf_acting",
    "recommended_action",
)

POLICY_LABEL: dict[str, str] = {
    "ppo": "PPO (windowed RL)",
    "dqn": "DQN (windowed RL)",
    "a2c": "A2C (windowed RL)",
    "rf_acting": "RF-Acting (supervised)",
    "recommended_action": "Oracle (full obs.)",
    "rule": "Oracle (full obs.)",
}

POLICY_STYLE: dict[str, dict[str, Any]] = {
    "ppo": {"color": "#1b7837", "marker": "o", "lw": 2.4, "zorder": 5},
    "dqn": {"color": "#7fbf7b", "marker": "s", "lw": 1.6, "zorder": 3},
    "a2c": {"color": "#b8e186", "marker": "^", "lw": 1.6, "zorder": 3},
    "rf_acting": {"color": "#b2182b", "marker": "D", "lw": 2.4, "zorder": 5},
    "recommended_action": {
        "color": "#4d4d4d",
        "marker": "x",
        "lw": 1.4,
        "ls": "--",
        "zorder": 2,
    },
    "rule": {
        "color": "#4d4d4d",
        "marker": "x",
        "lw": 1.4,
        "ls": "--",
        "zorder": 2,
    },
}

# Single-series accent colours for non-policy figures (e.g. evasion sweep,
# detector recall). Use these instead of ad-hoc hex literals so hue choices
# stay consistent with the policy palette.
ACCENT = {
    "primary": "#1b7837",  # green — matches PPO / the contribution
    "secondary": "#b2182b",  # red — matches RF / the contrast
    "neutral": "#4d4d4d",  # grey — oracle / reference lines
    "muted": "#999999",  # light grey — zero-lines, gridlines
    "amber": "#e08214",  # categorical accent (kept colour-blind-safe)
    "blue": "#2166ac",  # categorical accent
}

# Reference DPI for raster previews. Vector PDF ignores this.
RASTER_DPI = 200


def apply_house_style() -> None:
    """Install the shared rcParams. Call once before building any figure.

    Idempotent — safe to call from every plotter's ``main``.
    """
    plt.rcParams.update(
        {
            # Fonts: a serif stack pairs with abnTeX2/LaTeX body text and
            # keeps maths consistent with the thesis. mathtext uses the same.
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            # Lines / markers.
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "lines.markeredgewidth": 1.2,
            # Axes / grid: light, unobtrusive, journal-clean.
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.color": ACCENT["muted"],
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Legend.
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.fancybox": False,
            # Savefig defaults (PDF primary): tight + small pad.
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": RASTER_DPI,
            "pdf.fonttype": 42,  # embed TrueType (editable, no Type-3 warnings)
            "ps.fonttype": 42,
        }
    )


def policy_style(policy: str) -> dict[str, Any]:
    """Return the styling dict for a policy (empty dict if unknown)."""
    return dict(POLICY_STYLE.get(policy, {}))


def policy_label(policy: str) -> str:
    """Return the display label for a policy (falls back to the key)."""
    return POLICY_LABEL.get(policy, policy)


def save_figure(fig: plt.Figure, out_path: Path | str) -> None:
    """Save *fig* as both a vector PDF (primary) and a raster PNG preview.

    *out_path* may carry any suffix; both ``.pdf`` and ``.png`` siblings are
    written. The PDF is the artefact LaTeX includes; the PNG is for quick
    previews and the results index.
    """
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=RASTER_DPI)


def sha256_file(path: Path | str) -> str | None:
    """SHA-256 of a file (streamed), or ``None`` if it does not exist.

    Replaces the ``_sha256`` helper duplicated across every plotter so the
    reproducibility hash-chain is computed identically everywhere.
    """
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
