"""Observation-tensor schematic (Fig.: defender observation construction).

Draws the flat 290-dimensional observation the windowed blue-team agent sees at
each step: a stack of the ``w = 5`` most recent per-flow rows, each row holding
the 29 retained CICIoT2023 features together with their first-order temporal
deltas, i.e. ``290 = 5 x 29 x 2``. The 29 feature columns are grouped and
colour-banded by the five thematic groups pinned in
``docs/results/dataset/feature_provenance.json`` so the reader can see what the
vector actually contains rather than an opaque "290-dim" label.

This figure is a *schematic* driven entirely by the committed provenance JSON
(the observation basis), NOT by the gitignored raw feature arrays, so it
regenerates identically on a fresh checkout.

Run: PYTHONPATH=. .venv/bin/python scripts/thesis/plot_obs_tensor_schematic.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _plot_style import ACCENT, apply_house_style, save_figure, sha256_file  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROVENANCE = REPO / "docs" / "results" / "dataset" / "feature_provenance.json"
OUT_DIR = REPO / "tex" / "figs"
MANIFEST = REPO / "docs" / "results" / "dataset" / "feature_basis_manifest.json"

WINDOW_SIZE = 5  # w

# Group display order (top-to-bottom in the feature axis) + a colour-blind-safe
# band colour per group, reusing the house categorical accents.
GROUP_ORDER = [
    ("flow_timing", "Flow timing \\& rate", ACCENT["blue"]),
    ("header_and_size", "Header \\& size", ACCENT["primary"]),
    ("tcp_flags", "TCP flags", ACCENT["amber"]),
    ("protocol_indicators", "Protocol indicators", ACCENT["secondary"]),
    ("distribution_moments", "Distribution moments", ACCENT["neutral"]),
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - git optional
        return "unknown"


def main() -> None:
    prov = json.loads(PROVENANCE.read_text())
    groups = prov["thematic_groups"]
    n_features = prov["selection"]["final_feature_count"]
    assert n_features == sum(
        len(groups[g]) for g, _, _ in GROUP_ORDER
    ), "thematic_groups must partition the 29 retained features"

    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    # Geometry: x-axis = window frames t-4..t (5 columns), each split into a
    # value half and a delta half; y-axis = the 29 features, banded by group.
    frame_w = 1.0
    gap = 0.18
    half = frame_w / 2.0

    # Build the per-feature -> group colour map, ordered by group.
    row_colors: list[str] = []
    group_spans: list[tuple[str, int, int, str]] = []  # label, y0, y1, colour
    y = 0
    for key, label, colour in GROUP_ORDER:
        members = groups[key]
        group_spans.append((label, y, y + len(members), colour))
        for _ in members:
            row_colors.append(colour)
        y += len(members)
    total_rows = y

    # Draw the 5 frames.
    for f in range(WINDOW_SIZE):
        x0 = f * (frame_w + gap)
        for r in range(total_rows):
            yr = total_rows - 1 - r  # feature 0 at top
            # value cell
            ax.add_patch(
                mpatches.Rectangle(
                    (x0, yr),
                    half,
                    1.0,
                    facecolor=row_colors[r],
                    edgecolor="white",
                    linewidth=0.15,
                    alpha=0.85,
                )
            )
            # delta cell (hatched, lighter) — the first-order temporal delta
            ax.add_patch(
                mpatches.Rectangle(
                    (x0 + half, yr),
                    half,
                    1.0,
                    facecolor=row_colors[r],
                    edgecolor="white",
                    linewidth=0.15,
                    alpha=0.35,
                    hatch="////",
                )
            )
        # frame outline + label
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 0),
                frame_w,
                total_rows,
                fill=False,
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        lbl = "$t$" if f == WINDOW_SIZE - 1 else f"$t-{WINDOW_SIZE - 1 - f}$"
        ax.text(
            x0 + half,
            total_rows + 0.7,
            lbl,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Group brackets + labels on the left of the first frame.
    for label, y0, y1, colour in group_spans:
        yc = total_rows - (y0 + y1) / 2.0
        ax.text(
            -0.55,
            yc,
            label,
            ha="right",
            va="center",
            fontsize=9,
            color="#222222",
        )
        ax.plot(
            [-0.12, -0.12],
            [total_rows - y1 + 0.06, total_rows - y0 - 0.06],
            color=colour,
            linewidth=3.0,
            solid_capstyle="butt",
        )

    total_w = WINDOW_SIZE * frame_w + (WINDOW_SIZE - 1) * gap
    # "value | delta" mini-legend under a frame.
    ax.text(
        total_w + 0.35,
        total_rows * 0.62,
        "each cell pair:",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.add_patch(
        mpatches.Rectangle(
            (total_w + 0.35, total_rows * 0.48),
            0.32,
            total_rows * 0.06,
            facecolor=ACCENT["neutral"],
            edgecolor="white",
            alpha=0.85,
        )
    )
    ax.text(
        total_w + 0.72,
        total_rows * 0.51,
        "value $x$",
        ha="left",
        va="center",
        fontsize=8.5,
    )
    ax.add_patch(
        mpatches.Rectangle(
            (total_w + 0.35, total_rows * 0.36),
            0.32,
            total_rows * 0.06,
            facecolor=ACCENT["neutral"],
            edgecolor="white",
            alpha=0.35,
            hatch="////",
        )
    )
    ax.text(
        total_w + 0.72,
        total_rows * 0.39,
        r"delta $\Delta x$",
        ha="left",
        va="center",
        fontsize=8.5,
    )

    ax.set_title(
        f"Windowed observation: ${WINDOW_SIZE}$ frames "
        f"$\\times$ ${n_features}$ features $\\times$ 2 "
        f"(value, $\\Delta$) $= {WINDOW_SIZE * n_features * 2}$",
        fontsize=11.5,
    )
    ax.set_xlim(-2.2, total_w + 2.0)
    ax.set_ylim(-0.4, total_rows + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "obs_tensor_schematic"
    save_figure(fig, out_base)
    plt.close(fig)

    _write_manifest(
        outputs={
            "obs_tensor_schematic.pdf": sha256_file(out_base.with_suffix(".pdf")),
            "obs_tensor_schematic.png": sha256_file(out_base.with_suffix(".png")),
        }
    )
    print(f"wrote {out_base.with_suffix('.pdf')}")


def _write_manifest(outputs: dict[str, str | None]) -> None:
    """Emit / refresh the shared feature-basis figure manifest (hash chain)."""
    manifest = {
        "figure_ids": [],
        "figures": {},
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "docs/results/dataset/feature_provenance.json": sha256_file(PROVENANCE),
        },
        "outputs": {},
        "stage": "dataset",
        "produced_by": [],
        "tier": "context",
        "title": "Feature-basis schematics (observation tensor + selection funnel)",
        "_note": (
            "Two schematic figures derived purely from the committed "
            "feature_provenance.json (the 29-column observation basis); they do "
            "NOT read the gitignored raw feature arrays, so they regenerate "
            "identically on a fresh clone. This manifest is written/merged by "
            "both scripts/thesis/plot_obs_tensor_schematic.py and "
            "scripts/thesis/plot_feature_selection_funnel.py."
        ),
    }
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text())
            manifest["git_sha"] = _git_sha()
            manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
            manifest.setdefault("inputs", {})["docs/results/dataset/feature_provenance.json"] = (
                sha256_file(PROVENANCE)
            )
        except json.JSONDecodeError:
            pass

    fig_id = "FobsTensor"
    manifest.setdefault("figure_ids", [])
    if fig_id not in manifest["figure_ids"]:
        manifest["figure_ids"].append(fig_id)
    manifest.setdefault("figures", {})[fig_id] = {
        "title": "Windowed observation tensor (5 x 29 x 2 = 290)",
        "output": "obs_tensor_schematic.pdf",
        "thesis_chapter": "Ch. 3",
        "thesis_section": "§3.2",
    }
    manifest.setdefault("produced_by", [])
    pb = "scripts/thesis/plot_obs_tensor_schematic.py"
    if pb not in manifest["produced_by"]:
        manifest["produced_by"].append(pb)
    manifest.setdefault("outputs", {}).update(dict(outputs))

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
