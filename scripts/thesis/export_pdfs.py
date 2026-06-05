#!/usr/bin/env python3
"""Export committed F-named figure PNGs to same-named PDFs for thesis staging.

The figure plotters under ``scripts/`` all emit ``.png`` rasters into
``docs/results/<area>/``. The thesis (``tex/``) consumes ``.pdf`` figures via
``make sync-figures``. This script bridges the two by wrapping each committed
``docs/results/**/F*.png`` into a same-named ``F*.pdf`` — a single raster image
embedded at native resolution on a correctly-sized PDF page (aspect and DPI
preserved). No plotter is modified.

Only ``F*``-prefixed PNGs are exported; legacy-named PNGs (kept transiently for
back-compat) are ignored so the PDF set is the canonical F-named one.

Usage
-----
    python scripts/thesis/export_pdfs.py
    python scripts/thesis/export_pdfs.py --check   # exit 1 if any PDF missing/stale
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


def _is_stale(png: Path, pdf: Path) -> bool:
    return (not pdf.exists()) or (png.stat().st_mtime > pdf.stat().st_mtime)


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

    stale: list[Path] = []
    written: list[Path] = []
    for png in pngs:
        pdf = png.with_suffix(".pdf")
        if args.check:
            if _is_stale(png, pdf):
                stale.append(pdf)
            continue
        if _is_stale(png, pdf):
            _png_to_pdf(png, pdf)
            written.append(pdf)

    if args.check:
        if stale:
            print(f"export_pdfs --check: {len(stale)} stale/missing PDF(s):")
            for p in stale:
                print(f"  STALE: {p.relative_to(REPO_ROOT)}")
            return 1
        print(f"export_pdfs --check: all {len(pngs)} F-named PDFs up-to-date.")
        return 0

    print(
        f"export_pdfs: wrote {len(written)} PDF(s); "
        f"{len(pngs) - len(written)} already fresh ({len(pngs)} F-named PNGs total)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
