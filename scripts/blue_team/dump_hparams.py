"""T1 — Per-algo hyperparameters table.

PLAN §3.1.10. Reads each ``run_manifest.json`` under ``runs/blue_team/`` and
emits a markdown table + machine-readable JSON listing the hyperparameters
that produced the figures.

Usage::

    python -m scripts.blue_team.dump_hparams \\
        --runs-root runs/blue_team \\
        --out-dir docs/results/05_blue_team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("scripts.blue_team.dump_hparams")


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1 or v == 0:
            return f"{v:g}"
        return f"{v:.0e}" if v < 1e-2 else f"{v:.3f}"
    return str(v)


def _table(rows: list[dict[str, Any]]) -> str:
    """Build a markdown table from a list of {algo: ..., **hparams}."""
    if not rows:
        return "_No runs found._\n"
    keys: list[str] = ["algo", "total_timesteps"]
    for r in rows:
        for k in r["algo_hparams"]:
            if k not in keys:
                keys.append(k)
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, sep]
    for r in rows:
        cells = [r["algo"], str(r["total_timesteps"])]
        for k in keys[2:]:
            cells.append(_format_value(r["algo_hparams"].get(k, "")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render(runs_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(runs_root.rglob("run_manifest.json")):
        d = json.loads(manifest_path.read_text())
        algo = d.get("algo")
        if algo is None:
            continue
        # All seeds for the same algo should carry identical hparams (D5.4).
        # We keep the first one seen as canonical and assert equality on
        # subsequent ones.
        if algo not in runs:
            runs[algo] = {
                "algo": algo,
                "total_timesteps": d["total_timesteps"],
                "algo_hparams": d["algo_hparams"],
                "seeds": [d["seed"]],
                "examples": [str(manifest_path)],
            }
        else:
            entry = runs[algo]
            entry["seeds"].append(d["seed"])
            if entry["algo_hparams"] != d["algo_hparams"]:
                logger.warning(
                    "hparams divergence for algo=%s between %s and %s",
                    algo,
                    entry["examples"][0],
                    manifest_path,
                )
            entry["examples"].append(str(manifest_path))

    rows = sorted(runs.values(), key=lambda r: r["algo"])
    md_path = out_dir / "T1_hparams.md"
    md = (
        "# T1 — blue-team Per-Algorithm Hyperparameters\n\n"
        f"Generated from {sum(len(r['seeds']) for r in rows)} runs across "
        f"{len(rows)} algorithms (seeds: "
        f"{sorted({s for r in rows for s in r['seeds']})}).\n\n"
        "All values are PLAN §8 D5.4 defaults. Phase 8 may revisit\n"
        "hyperparameters; blue-team reports them as a frozen reference.\n\n"
    )
    md += _table(rows)
    md_path.write_text(md)
    logger.info("wrote %s", md_path)

    json_path = out_dir / "T1_hparams.json"
    json_path.write_text(json.dumps(rows, indent=2))
    logger.info("wrote %s", json_path)
    return {"md_path": str(md_path), "json_path": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render T1 hyperparameters table.")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out-dir", default="docs/results/05_blue_team")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    render(Path(args.runs_root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
