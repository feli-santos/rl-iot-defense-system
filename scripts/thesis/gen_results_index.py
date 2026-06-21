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
    falpha = _load(RESULTS_DIR / "ablation/Falpha_summary.json")
    man = _load(RESULTS_DIR / "ablation/Falpha_manifest.json")
    src = RESULTS_DIR / "ablation/Falpha_summary.json"

    lines = ["## 06 Held-Out Benchmark\n"]
    lines.append(
        f"**Source:** `docs/results/ablation/Falpha_summary.json` (alpha=0 point)  \n"
        f"**git SHA:** {_git_sha(man)}  \n"
        f"**File mtime:** {_mtime_str(src)}  \n"
    )

    if falpha:
        per_alpha = falpha.get("per_alpha", {})
        cell = per_alpha.get("0.0", {})
        if cell:
            lines.append(
                "\n| Policy | Mean Reward | CI 95% | n_eps | n_seeds |\n"
                "|--------|-------------|--------|-------|---------|\n"
            )
            for pol in ("ppo", "dqn", "a2c", "rf_acting", "recommended_action"):
                r = cell.get(pol)
                if not r:
                    continue
                lines.append(
                    f"| {pol} | {r['mean']:+.1f} | "
                    f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] | "
                    f"{r.get('n', '—')} | {r.get('n_seeds', '—')} |\n"
                )
    return "".join(lines)


def _section_ablation() -> str:
    falpha = _load(RESULTS_DIR / "ablation/Falpha_summary.json")
    fcoupling = _load(RESULTS_DIR / "ablation/Fcoupling_summary.json")
    f15 = _load(RESULTS_DIR / "ablation/F15_summary.json")
    man_alpha = _load(RESULTS_DIR / "ablation/Falpha_manifest.json")
    man_coup = _load(RESULTS_DIR / "ablation/Fcoupling_manifest.json")
    man15 = _load(RESULTS_DIR / "ablation/F15_manifest.json")

    lines = ["## 07 Ablation & OOD Robustness\n"]
    lines.append(
        f"**Falpha source:** `docs/results/ablation/Falpha_summary.json`  \n"
        f"**Falpha git SHA:** {_git_sha(man_alpha)}  \n"
        f"**Falpha mtime:** {_mtime_str(RESULTS_DIR / 'ablation/Falpha_summary.json')}  \n\n"
        f"**Fcoupling source:** `docs/results/ablation/Fcoupling_summary.json`  \n"
        f"**Fcoupling git SHA:** {_git_sha(man_coup)}  \n"
        f"**Fcoupling mtime:** {_mtime_str(RESULTS_DIR / 'ablation/Fcoupling_summary.json')}  \n\n"
        f"**F15 source:** `docs/results/ablation/F15_summary.json`  \n"
        f"**F15 git SHA:** {_git_sha(man15)}  \n"
        f"**F15 mtime:** {_mtime_str(RESULTS_DIR / 'ablation/F15_summary.json')}  \n"
    )

    if falpha:
        per_alpha = falpha.get("per_alpha", {})
        if per_alpha:
            lines.append("\n### Observation-aliasing alpha-curve (Falpha)\n")
            lines.append("| alpha | PPO [CI] | DQN | A2C | RF-Acting [CI] | Oracle |\n")
            lines.append("|-------|----------|-----|-----|----------------|--------|\n")
            for akey in ("0.0", "0.2", "0.4", "0.6"):
                c = per_alpha.get(akey, {})
                if not c:
                    continue
                ppo = c.get("ppo", {})
                dqn = c.get("dqn", {})
                a2c = c.get("a2c", {})
                rf = c.get("rf_acting", {})
                orc = c.get("recommended_action", {})
                lines.append(
                    f"| {akey} | {ppo.get('mean', 0):+.1f} "
                    f"[{ppo.get('ci_low', 0):+.1f}, {ppo.get('ci_high', 0):+.1f}] | "
                    f"{dqn.get('mean', 0):+.1f} | {a2c.get('mean', 0):+.1f} | "
                    f"{rf.get('mean', 0):+.1f} "
                    f"[{rf.get('ci_low', 0):+.1f}, {rf.get('ci_high', 0):+.1f}] | "
                    f"{orc.get('mean', 0):+.1f} |\n"
                )

    if fcoupling:
        per_mode = fcoupling.get("per_mode", {})
        if per_mode:
            lines.append("\n### Reward-coupling ablation (Fcoupling)\n")
            lines.append("| mode | best RL | best RL reward | RF-Acting [CI] | RF-minus-RL gap |\n")
            lines.append("|------|---------|----------------|----------------|-----------------|\n")
            for mode in ("coupled", "outcome"):
                m = per_mode.get(mode, {})
                if not m:
                    continue
                lines.append(
                    f"| {mode.capitalize()} | {m.get('best_algo', '—').upper()} | "
                    f"{m.get('best_rl_reward', 0):+.1f} | "
                    f"{m.get('rf_acting_reward', 0):+.1f} "
                    f"[{m.get('rf_acting_ci_low', 0):+.1f}, {m.get('rf_acting_ci_high', 0):+.1f}] | "
                    f"{m.get('rf_minus_rl_gap', 0):+.1f} |\n"
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

    # F15b: recall-vs-advantage figure (PNG only; no summary/manifest JSON shipped).
    f15b_png = RESULTS_DIR / "ablation/F15b_recall_vs_advantage.png"
    if f15b_png.exists():
        lines.append("\n### Recall vs. advantage (F15b)\n")
        lines.append(
            f"**Figure:** `docs/results/ablation/F15b_recall_vs_advantage.png`  \n"
            f"**File mtime:** {_mtime_str(f15b_png)}  \n"
        )

    return "".join(lines)


def _section_detector() -> str:
    f11 = _load(RESULTS_DIR / "stage-detector/F11_summary.json")
    man = _load(RESULTS_DIR / "stage-detector/manifest.json")

    lines = ["## 04 Stage Detector\n"]
    lines.append(
        f"**Source:** `docs/results/stage-detector/F11_summary.json`  \n"
        f"**git SHA:** {_git_sha(man)}  \n"
        f"**mtime:** {_mtime_str(RESULTS_DIR / 'stage-detector/F11_summary.json')}  \n"
    )
    if f11:
        for k in ("macro_f1", "accuracy", "weighted_f1"):
            if k in f11:
                lines.append(f"- **{k}:** {f11[k]}\n")
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
        _section_derived(),
    ]

    content = "".join(sections)
    OUT.write_text(content)
    print(f"Wrote {OUT} ({len(content)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
