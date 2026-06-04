#!/usr/bin/env python3
"""Anti-staleness freshness gate.

Checks that every *derived* artifact (tex/generated/*.tex, G*_scoreboard.json,
RESULTS_INDEX.md) was generated *after* its canonical JSON source was last
modified.  Exits non-zero if any derived file is stale, making it suitable as
a CI gate (``make verify-fresh``).

Usage
-----
    python scripts/thesis/verify_fresh.py          # exit 0 = all fresh
    python scripts/thesis/verify_fresh.py --fix    # also runs render-tables + gen-results-index
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry: (derived_artifact, [canonical_source, ...])
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DERIVED = [
    (
        REPO_ROOT / "tex/generated/numbers.tex",
        [
            REPO_ROOT / "docs/results/06_benchmark/F5_summary.json",
            REPO_ROOT / "docs/results/06_benchmark/benign_fpr.json",
            REPO_ROOT / "docs/results/07_ablation/F9_summary.json",
        ],
    ),
    (
        REPO_ROOT / "tex/generated/tables.tex",
        [
            REPO_ROOT / "docs/results/06_benchmark/F5_summary.json",
            REPO_ROOT / "docs/results/06_benchmark/F7_summary.json",
        ],
    ),
    (
        REPO_ROOT / "docs/results/06_benchmark/G6_scoreboard.json",
        [REPO_ROOT / "docs/results/06_benchmark/F5_summary.json"],
    ),
    (
        REPO_ROOT / "docs/results/07_ablation/G7_scoreboard.json",
        [
            REPO_ROOT / "docs/results/07_ablation/F9_summary.json",
            REPO_ROOT / "docs/results/07_ablation/F15_summary.json",
        ],
    ),
    (
        REPO_ROOT / "docs/RESULTS_INDEX.md",
        [
            REPO_ROOT / "docs/results/06_benchmark/F5_summary.json",
            REPO_ROOT / "docs/results/07_ablation/F9_summary.json",
            REPO_ROOT / "docs/results/07_ablation/F15_summary.json",
            REPO_ROOT / "docs/results/06_benchmark/benign_fpr.json",
        ],
    ),
]


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def check() -> list[str]:
    """Return list of stale-derived-artifact descriptions (empty = all fresh)."""
    stale: list[str] = []
    for derived, sources in DERIVED:
        if not derived.exists():
            stale.append(f"MISSING  {_rel(derived)}")
            continue
        d_mtime = _mtime(derived)
        for src in sources:
            if not src.exists():
                continue  # source not present on this checkout (data-gitignored)
            if src.stat().st_mtime > d_mtime:
                stale.append(
                    f"STALE    {_rel(derived)}\n"
                    f"         source newer: {_rel(src)}"
                )
                break  # one stale source is enough to flag the derived artifact
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="If stale artifacts exist, re-run make render-tables and gen-results-index.",
    )
    args = parser.parse_args(argv)

    stale = check()

    if not stale:
        print("verify-fresh: all derived artifacts are up-to-date.")
        return 0

    print("verify-fresh: STALE artifacts detected:")
    for s in stale:
        print(f"  {s}")

    if args.fix:
        print("\nRunning make render-tables gen-results-index ...")
        ret = subprocess.run(
            ["make", "render-tables", "gen-results-index"],
            cwd=REPO_ROOT,
        ).returncode
        if ret != 0:
            print("ERROR: make render-tables gen-results-index failed.")
            return ret
        # Re-check
        stale2 = check()
        if not stale2:
            print("verify-fresh: all artifacts refreshed successfully.")
            return 0
        print("verify-fresh: still stale after fix attempt:")
        for s in stale2:
            print(f"  {s}")
        return 1

    print(
        "\nRun `make render-tables gen-results-index` to regenerate, "
        "or `make verify-fresh-fix` to auto-fix."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
