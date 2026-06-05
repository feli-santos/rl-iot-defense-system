#!/usr/bin/env python3
"""Auto-generate docs/RESULTS_INDEX.md from canonical result JSONs.

Lists every figure/table with its source JSON, git SHA, generation
timestamp, and key headline numbers so there is a single, machine-readable
registry of what was produced and when.

Usage
-----
    python scripts/thesis/gen_results_index.py
    make gen-results-index
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "docs/results"
OUT = REPO_ROOT / "docs/RESULTS_INDEX.md"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load(p: Path) -> dict | None:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _fmt_ts(ts: str | None) -> str:
    return ts if ts else "—"


def _git_sha(manifest: dict | None) -> str:
    if manifest is None:
        return "—"
    return manifest.get("git_sha", "—")[:10]


def _mtime_str(p: Path) -> str:
    if not p.exists():
        return "—"
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_benchmark() -> str:
    f5 = _load(RESULTS_DIR / "benchmark/main_results.json")
    fpr = _load(RESULTS_DIR / "benchmark/benign_fpr.json")
    man = _load(RESULTS_DIR / "benchmark/main_results_manifest.json")

    lines = ["## 06 Held-Out Benchmark\n"]
    lines.append(
        f"**Source:** `docs/results/benchmark/main_results.json`  \n"
        f"**git SHA:** {_git_sha(man)}  \n"
        f"**File mtime:** {_mtime_str(RESULTS_DIR / 'benchmark/main_results.json')}  \n"
    )

    if f5:
        rows = f5.get("rows", [])
        # Build canonical table
        lines.append("\n| Policy | Mean Reward | CI 95% | n_eps | n_seeds | MIT-rate | FPR |\n")
        lines.append("|--------|-------------|--------|-------|---------|----------|-----|\n")
        policies_fpr: dict = {}
        if fpr:
            policies_fpr = fpr.get("policies", fpr)
        for r in rows:
            pol = r["policy"]
            fpr_val = "—"
            if pol in policies_fpr:
                d = policies_fpr[pol]
                fpr_val = f"{d['benign_fpr']:.2%}" if isinstance(d, dict) else f"{d:.2%}"
            lines.append(
                f"| {pol} | {r['mean_reward']:+.1f} | "
                f"[{r['mean_reward_ci_low']:+.1f}, {r['mean_reward_ci_high']:+.1f}] | "
                f"{r['n_episodes']} | {r.get('n_seeds','—')} | "
                f"{r.get('mitigated_impact_rate',0):.3f} | {fpr_val} |\n"
            )
    return "".join(lines)


def _section_ablation() -> str:
    f9 = _load(RESULTS_DIR / "ablation/reward_ablation.json")
    f15 = _load(RESULTS_DIR / "ablation/ood_robustness.json")
    man9 = _load(RESULTS_DIR / "ablation/reward_ablation_manifest.json")
    man15 = _load(RESULTS_DIR / "ablation/ood_robustness_manifest.json")

    lines = ["## 07 Ablation & OOD Robustness\n"]
    lines.append(
        f"**F9 source:** `docs/results/ablation/reward_ablation.json`  \n"
        f"**F9 git SHA:** {_git_sha(man9)}  \n"
        f"**F9 mtime:** {_mtime_str(RESULTS_DIR / 'ablation/reward_ablation.json')}  \n\n"
        f"**F15 source:** `docs/results/ablation/ood_robustness.json`  \n"
        f"**F15 git SHA:** {_git_sha(man15)}  \n"
        f"**F15 mtime:** {_mtime_str(RESULTS_DIR / 'ablation/ood_robustness.json')}  \n"
    )

    if f9:
        rows = f9.get("rows", [])
        if rows:
            lines.append("\n### Reward-coefficient ablation (F9)\n")
            lines.append("| cell_id | impact_is_terminal | mean_reward | CI | MIT-rate |\n")
            lines.append("|---------|-------------------|-------------|-----|----------|\n")
            for r in rows:
                lines.append(
                    f"| {r['cell_id']} | {r['impact_is_terminal']} | "
                    f"{r['mean_reward']:+.1f} | "
                    f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | "
                    f"{r.get('mitigated_impact_rate',0):.3f} |\n"
                )

    if f15:
        ood_rows = f15.get("rows", [])
        if ood_rows:
            lines.append("\n### OOD robustness (F15)\n")
            lines.append("| class | policy | mean_reward | CI |\n")
            lines.append("|-------|--------|-------------|----|\n")
            for r in ood_rows:
                lines.append(
                    f"| {r.get('ood_class','—')} | {r.get('policy','—')} | "
                    f"{r['mean_reward']:+.1f} | "
                    f"[{r.get('ci_low',r['mean_reward']):+.1f}, "
                    f"{r.get('ci_high',r['mean_reward']):+.1f}] |\n"
                )

    return "".join(lines)


def _section_detector() -> str:
    f11 = _load(RESULTS_DIR / "stage-detector/detector_summary.json")
    man = _load(RESULTS_DIR / "stage-detector/manifest.json")

    lines = ["## 04 Stage Detector\n"]
    lines.append(
        f"**Source:** `docs/results/stage-detector/detector_summary.json`  \n"
        f"**git SHA:** {_git_sha(man)}  \n"
        f"**mtime:** {_mtime_str(RESULTS_DIR / 'stage-detector/detector_summary.json')}  \n"
    )
    if f11:
        for k in ("macro_f1", "accuracy", "weighted_f1"):
            if k in f11:
                lines.append(f"- **{k}:** {f11[k]}\n")
    return "".join(lines)


def _section_red_team() -> str:
    man = _load(RESULTS_DIR / "red-team-model/manifest.json")

    lines = ["## 02 Red-Team LSTM\n"]
    lines.append(
        f"**Source:** `docs/results/red-team-model/red_team_gates.json`  \n"
        f"**git SHA:** {_git_sha(man)}  \n"
        f"**mtime:** {_mtime_str(RESULTS_DIR / 'red-team-model/red_team_gates.json')}  \n"
    )
    return "".join(lines)


def _section_derived() -> str:
    lines = ["## Derived Artifacts (tex/generated/)\n\n"]
    lines.append(
        "> These files are auto-generated by `make render-tables`. "
        "Run `make verify-fresh` to confirm they match the canonical JSONs.\n\n"
    )
    for name in ("numbers.tex", "tables.tex"):
        p = REPO_ROOT / "tex/generated" / name
        lines.append(f"- `tex/generated/{name}` — last written: {_mtime_str(p)}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> int:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sections = [
        f"# Results Index\n\n"
        f"> Auto-generated by `make gen-results-index` on {now}.  \n"
        f"> **Do not hand-edit.** Re-run `make gen-results-index` after any data update.\n\n"
        f"> Run `make verify-fresh` to confirm all derived artifacts are up-to-date.\n\n"
        "---\n\n",
        _section_benchmark(),
        "\n---\n\n",
        _section_ablation(),
        "\n---\n\n",
        _section_detector(),
        "\n---\n\n",
        _section_red_team(),
        "\n---\n\n",
        _section_derived(),
    ]

    content = "".join(sections)
    OUT.write_text(content)
    print(f"Wrote {OUT} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
