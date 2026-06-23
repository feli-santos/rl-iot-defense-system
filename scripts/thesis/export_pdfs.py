#!/usr/bin/env python3
"""Export F-named figure PNGs to same-named PDFs for thesis staging.

This script is a *fallback* bridge for figures whose plotter emits a ``.png``
but no vector ``.pdf``. The redesigned plotters under ``scripts/`` write a true
vector ``.pdf`` next to their ``.png`` (via ``scripts._plot_style.save_figure``);
those vector PDFs are journal-quality and MUST NOT be clobbered by a raster
wrap. Therefore this script only raster-wraps a PNG when **no sibling PDF
exists at all** — a genuine vector PDF, once present, is always preserved.

Only ``F*``-prefixed PNGs are considered; legacy-named PNGs are ignored so the
PDF set is the canonical F-named one.

Usage
-----
    python scripts/thesis/export_pdfs.py
    python scripts/thesis/export_pdfs.py --check   # exit 1 if any PDF missing
    make export-figure-pdfs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "docs/results"


def _find_pngs() -> list[Path]:
    """Return committed F-named figure PNGs under docs/results/, sorted."""
    return sorted(RESULTS_DIR.glob("*/F*.png"))


def _needs_wrap(png: Path, pdf: Path) -> bool:
    """Only wrap when there is no sibling PDF at all.

    A sibling PDF is treated as a plotter-emitted vector figure and is always
    preserved (never overwritten by a raster wrap), regardless of mtime.
    """
    return not pdf.exists()


def _png_to_pdf(png: Path, pdf: Path) -> None:
    """Embed a PNG raster into a same-aspect PDF page at native resolution."""
    img = mpimg.imread(str(png))
    height, width = img.shape[0], img.shape[1]
    dpi = 200.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(img, interpolation="none", aspect="auto")
    ax.axis("off")
    fig.savefig(str(pdf), dpi=dpi, format="pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if any F-named PDF is missing or older than its PNG.",
    )
    args = parser.parse_args(argv)

    pngs = _find_pngs()
    if not pngs:
        print("export_pdfs: no docs/results/*/F*.png figures found.")
        return 0

    missing: list[Path] = []
    written: list[Path] = []
    for png in pngs:
        pdf = png.with_suffix(".pdf")
        if args.check:
            if _needs_wrap(png, pdf):
                missing.append(pdf)
            continue
        if _needs_wrap(png, pdf):
            _png_to_pdf(png, pdf)
            written.append(pdf)

    if args.check:
        if missing:
            print(f"export_pdfs --check: {len(missing)} missing PDF(s):")
            for p in missing:
                print(f"  MISSING: {p.relative_to(REPO_ROOT)}")
            return 1
        print(f"export_pdfs --check: all {len(pngs)} F-named PNGs have a sibling PDF.")
        return 0

    print(
        f"export_pdfs: raster-wrapped {len(written)} PNG(s) lacking a PDF; "
        f"{len(pngs) - len(written)} already had a (vector) PDF "
        f"({len(pngs)} F-named PNGs total)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
