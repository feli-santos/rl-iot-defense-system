"""Feature-selection funnel (Fig.: 46 -> 29 leakage-safe reduction).

Visualises the three-stage, train-split-only reduction from the 46 raw
CICIoT2023 numeric columns to the 29 retained observation features, exactly as
recorded in ``docs/results/dataset/feature_provenance.json``:

    46  --(zero-variance removal)-->  42
        --(low-variance, var < 0.01)-->  35
        --(high-correlation, |Pearson| > 0.95)-->  29

Each stage lists the features it drops; the 17 dropped columns are named so the
reduction is fully auditable. Driven entirely by the committed provenance JSON
(not the gitignored raw arrays), so it regenerates identically on a fresh
checkout.

Run: PYTHONPATH=. .venv/bin/python scripts/thesis/plot_feature_selection_funnel.py
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


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - git optional
        return "unknown"


def _wrap(names: list[str], per_line: int = 4) -> str:
    """Comma-join feature names, wrapping every *per_line* into a new line."""
    lines = []
    for i in range(0, len(names), per_line):
        lines.append(", ".join(names[i : i + per_line]))
    return ",\n".join(lines)


def main() -> None:
    prov = json.loads(PROVENANCE.read_text())
    sel = prov["selection"]
    dropped = prov["dropped"]

    n0 = sel["original_feature_count"]  # 46
    n_final = sel["final_feature_count"]  # 29
    var_thr = sel["variance_threshold"]  # 0.01
    corr_thr = sel["correlation_threshold"]  # 0.95

    d_zero = dropped["zero_variance"]
    d_low = dropped["low_variance"]
    d_high = dropped["high_correlation"]

    # Surviving counts after each stage.
    n1 = n0 - len(d_zero)  # 42
    n2 = n1 - len(d_low)  # 35
    n3 = n2 - len(d_high)  # 29
    assert n3 == n_final, f"funnel arithmetic mismatch: {n3} != {n_final}"

    # Funnel bars: (count, colour). Width proportional to count.
    stages = [
        (f"Raw numeric columns\n({n0})", n0, ACCENT["muted"]),
        (
            f"After zero-variance removal\n({n1})",
            n1,
            ACCENT["blue"],
        ),
        (
            f"After low-variance filter\n(var $\\geq$ {var_thr}) ({n2})",
            n2,
            ACCENT["amber"],
        ),
        (
            f"Retained observation basis\n($|$Pearson$| \\leq$ {corr_thr}) ({n3})",
            n3,
            ACCENT["primary"],
        ),
    ]
    drops = [
        (len(d_zero), "zero variance", d_zero),
        (len(d_low), f"variance $<$ {var_thr}", d_low),
        (len(d_high), f"$|$Pearson$| >$ {corr_thr}", d_high),
    ]

    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    bar_h = 0.62
    y_step = 1.0
    max_c = float(n0)
    x_center = 0.0

    for i, (label, count, colour) in enumerate(stages):
        y = -(i * y_step)
        w = count / max_c
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x_center - w / 2, y - bar_h / 2),
                w,
                bar_h,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=colour,
                edgecolor="#333333",
                linewidth=0.9,
                alpha=0.9,
                mutation_aspect=0.5,
            )
        )
        ax.text(
            x_center,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
            color="white" if colour != ACCENT["muted"] else "#222222",
            fontweight="bold",
        )

    # Drop annotations on the right, between consecutive bars.
    for i, (k, rule, names) in enumerate(drops):
        y = -(i * y_step) - y_step / 2.0
        # connector arrow down the centre
        ax.annotate(
            "",
            xy=(x_center, -(i + 1) * y_step + bar_h / 2),
            xytext=(x_center, -i * y_step - bar_h / 2),
            arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.4},
        )
        txt = f"$-{k}$  ({rule})\n" + _wrap(names, per_line=4)
        ax.text(
            0.62,
            y,
            txt,
            ha="left",
            va="center",
            fontsize=7.6,
            color=ACCENT["secondary"],
            family="monospace",
        )

    ax.set_title(
        f"Leakage-safe feature reduction: {n0} $\\rightarrow$ {n3} (train split only)",
        fontsize=11.5,
    )
    ax.set_xlim(-0.7, 2.1)
    ax.set_ylim(-len(stages) * y_step + 0.4, 0.7)
    ax.axis("off")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUT_DIR / "feature_selection_funnel"
    save_figure(fig, out_base)
    plt.close(fig)

    _write_manifest(
        outputs={
            "feature_selection_funnel.pdf": sha256_file(out_base.with_suffix(".pdf")),
            "feature_selection_funnel.png": sha256_file(out_base.with_suffix(".png")),
        }
    )
    print(f"wrote {out_base.with_suffix('.pdf')}")


def _write_manifest(outputs: dict[str, str | None]) -> None:
    """Merge into the shared feature-basis figure manifest (hash chain)."""
    manifest: dict = {
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

    fig_id = "FselectFunnel"
    manifest.setdefault("figure_ids", [])
    if fig_id not in manifest["figure_ids"]:
        manifest["figure_ids"].append(fig_id)
    manifest.setdefault("figures", {})[fig_id] = {
        "title": "Feature-selection funnel (46 -> 29)",
        "output": "feature_selection_funnel.pdf",
        "thesis_chapter": "Ch. 3",
        "thesis_section": "§3.2",
    }
    manifest.setdefault("produced_by", [])
    pb = "scripts/thesis/plot_feature_selection_funnel.py"
    if pb not in manifest["produced_by"]:
        manifest["produced_by"].append(pb)
    manifest.setdefault("outputs", {}).update(dict(outputs))

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
