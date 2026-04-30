"""Phase-7 F15 — OOD-class robustness plotter (audit-AF1, headline).

Reads ``runs/phase7/ood/<class>/<policy>/seed_<k>/eval_test.jsonl``
files (produced by :mod:`scripts.ablation.run_ood_eval`), aggregates
per (class, policy) with the same bootstrap-CI protocol Phase-6 used
in :mod:`scripts.benchmark.build_summary_table`, and emits:

- ``F15_ood_robustness.png`` — 4-class × 8-policy grouped horizontal
  bar chart with 95 % bootstrap CI whiskers, one panel per OOD class.
  Same visual idiom as Phase-6 F8.
- ``F15_summary.json`` — per (class, policy) row + a ``headline`` block
  reporting whether trained RL beat RF-Acting on each class (G7.9
  evaluator).
- ``F15_caption.md`` — thesis caption (separate, hand-written; this
  script writes a placeholder).
- ``F15_manifest.json`` — SHA-256 hash chain over every input JSONL,
  the upstream Phase-7 ``eval_manifest.json``, the Phase-6
  ``eval_manifest.json``, and the Phase-5 ``sweep_manifest.json``.

Gate evaluation:

- **G7.8** (audit-AF1) — pass iff every (class, policy) cell produced
  a non-empty eval_test.jsonl with a finite mean_reward (no NaNs in
  the 4 × 8 result matrix).
- **G7.9** (audit-AF1, headline) — pass iff on `VulnerabilityScan`,
  the *best* trained RL mean_reward exceeds RF-Acting mean_reward by
  at least 1σ of the per-policy bootstrap CI (i.e., the lower bound
  of the trained-RL CI exceeds the upper bound of the RF-Acting CI).
  Acceptable failure mode: turns into a finding (D7.9.1) — the
  thesis claim narrows from "RL closes the OOD gap" to "RL is
  *robust to* (not *better at*) the OOD class".

Phase-4 reminder: Phase 4 RESULTS §3.2 reported the supervised RF
stage detector has **0.001 recall on `VulnerabilityScan`**. F15
quantifies how much of that blind spot the trained RL policy
recovers by acting on raw features rather than the detector's
classification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_ood_robustness")

_ROOT = Path(__file__).resolve().parents[2]


# Display ordering — same as Phase-6 F8, plus rule baseline as the
# oracle ceiling marker.
_POLICY_ORDER: List[str] = [
    "recommended_action",  # oracle ceiling
    "rf_acting",
    "dqn",
    "ppo",
    "a2c",
    "always_block",
    "random",
    "always_observe",
]
_DISPLAY: Dict[str, str] = {
    "recommended_action": "Rec-Action (oracle)",
    "rf_acting":          "RF-Acting",
    "dqn":                "DQN",
    "ppo":                "PPO",
    "a2c":                "A2C",
    "always_block":       "Always-BLOCK",
    "random":             "Random",
    "always_observe":     "Always-OBSERVE",
}
_RL_ALGOS = {"dqn", "ppo", "a2c"}
_OOD_CLASSES_DEFAULT: List[str] = [
    "DDoS-HTTP_Flood", "Mirai-udpplain", "VulnerabilityScan", "XSS",
]


def _sha256(path: Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------- aggregation


def _discover_seed_dirs(
    runs_root: Path, ood_class: str, policy: str
) -> List[Path]:
    """Find every ``seed_<k>`` dir under
    ``<runs_root>/<ood_class>/<policy>/`` that contains
    ``eval_test.jsonl``."""
    base = runs_root / ood_class / policy
    if not base.exists():
        return []
    return sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
        and (d / "eval_test.jsonl").exists()
    )


def _summarise_cell(
    ood_class: str,
    policy: str,
    seed_dirs: List[Path],
    *,
    sha_collector: Dict[str, str],
) -> Dict[str, Any]:
    """Compute one (ood_class, policy) row.

    Same bootstrap protocol as Phase-6 F5: when n_seeds ≥ 3, bootstrap
    across per-seed means; else bootstrap across all episodes.
    """
    all_records: List[Dict] = []
    per_seed_means: List[float] = []
    input_paths: List[Path] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(
                float(np.mean([r["episode_reward"] for r in recs]))
            )
        input_paths.append(jsonl)
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.relative_to(_ROOT))] = sha

    if not all_records:
        return {
            "ood_class": ood_class,
            "policy": policy,
            "n_seeds": len(seed_dirs),
            "n_episodes": 0,
            "mean_reward": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "compromise_rate": math.nan,
            "mitigated_impact_rate": math.nan,
            "mean_episode_length": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    compromised = [1.0 if r.get("compromised") else 0.0 for r in all_records]
    mitigated = [
        1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0
        for r in all_records
    ]
    lengths = [r["episode_length"] for r in all_records]

    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means, n_resamples=2000, alpha=0.05, seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards, n_resamples=2000, alpha=0.05, seed=0,
        )

    return {
        "ood_class": ood_class,
        "policy": policy,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "compromise_rate": float(np.mean(compromised)),
        "mitigated_impact_rate": float(np.mean(mitigated)),
        "mean_episode_length": float(np.mean(lengths)),
    }


# --------------------------------------------------------------- gates


def _evaluate_g79(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """G7.9 (audit-AF1, headline): on ``VulnerabilityScan``, does the
    best trained RL mean_reward beat RF-Acting by ≥ 1σ of bootstrap CI?

    Pass criterion (PLAN §3.4): the lower bound of the trained-RL CI
    exceeds the upper bound of the RF-Acting CI on
    ``VulnerabilityScan``. This is a strict CI-non-overlap test in the
    direction "RL above RF" — neutral overlap or RL-below-RF flips
    the gate to PASS-WITH-FINDING / FAIL-WITH-FINDING (the latter
    activates the pre-emptive D7.9.1 placeholder in PLAN §8).
    """
    target_class = "VulnerabilityScan"
    by_pol = {
        r["policy"]: r for r in rows
        if r["ood_class"] == target_class and r["policy"] in (
            "rf_acting", "dqn", "ppo", "a2c",
        )
    }
    if "rf_acting" not in by_pol:
        return {
            "passes": False,
            "reason": "rf_acting row missing for VulnerabilityScan",
        }
    rf = by_pol["rf_acting"]
    rf_ci_high = rf["ci_high"]

    rl_rows = {a: by_pol[a] for a in ("dqn", "ppo", "a2c") if a in by_pol}
    if not rl_rows:
        return {"passes": False, "reason": "no trained RL rows present"}

    best_algo = max(rl_rows, key=lambda a: rl_rows[a]["mean_reward"])
    best = rl_rows[best_algo]

    # CI-non-overlap test in the direction "RL > RF".
    passes = bool(
        math.isfinite(best["ci_low"])
        and math.isfinite(rf_ci_high)
        and best["ci_low"] > rf_ci_high
    )

    return {
        "ood_class": target_class,
        "passes": passes,
        "best_rl_algo": best_algo,
        "best_rl_mean_reward": best["mean_reward"],
        "best_rl_ci": [best["ci_low"], best["ci_high"]],
        "rf_acting_mean_reward": rf["mean_reward"],
        "rf_acting_ci": [rf["ci_low"], rf["ci_high"]],
        "delta_mean": best["mean_reward"] - rf["mean_reward"],
        "phase4_rf_recall_on_vulnerability_scan": 0.001,
        "interpretation": (
            "PASS: trained RL recovers some of the supervised RF "
            "blind spot on VulnerabilityScan."
            if passes
            else "FAIL-WITH-FINDING: trained RL does NOT beat RF-Acting "
                 "on VulnerabilityScan; the thesis claim narrows from "
                 "'RL closes the OOD gap' to 'RL is robust to (not "
                 "better at) the OOD class'. See PLAN §8 D7.9.1."
        ),
    }


def _evaluate_g78(
    rows: List[Dict[str, Any]],
    expected_classes: List[str],
    expected_policies: List[str],
) -> Dict[str, Any]:
    """G7.8: 4 × 8 matrix is complete with no NaN means."""
    have: Dict[Tuple[str, str], Dict[str, Any]] = {
        (r["ood_class"], r["policy"]): r for r in rows
    }
    missing: List[Tuple[str, str]] = []
    nan_cells: List[Tuple[str, str]] = []
    for c in expected_classes:
        for p in expected_policies:
            if (c, p) not in have:
                missing.append((c, p))
                continue
            if not math.isfinite(have[(c, p)]["mean_reward"]):
                nan_cells.append((c, p))
    return {
        "passes": not missing and not nan_cells,
        "expected_classes": expected_classes,
        "expected_policies": expected_policies,
        "n_cells_expected": len(expected_classes) * len(expected_policies),
        "n_cells_present": len(rows),
        "missing_cells": [list(c) for c in missing],
        "nan_cells": [list(c) for c in nan_cells],
    }


# --------------------------------------------------------------- render


def _render(
    rows: List[Dict[str, Any]],
    ood_classes: List[str],
    out_path: Path,
) -> None:
    """4-panel grouped bar chart (one panel per OOD class)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_class: Dict[str, Dict[str, Dict[str, Any]]] = {c: {} for c in ood_classes}
    for r in rows:
        by_class.setdefault(r["ood_class"], {})[r["policy"]] = r

    n = len(ood_classes)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13.5, 4.5 * n_rows), squeeze=False
    )

    for ax_idx, ood_class in enumerate(ood_classes):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        cell = by_class.get(ood_class, {})
        # Sort the displayed policies by mean_reward asc so the highest
        # bar is at the top of each panel (mirrors F8 idiom).
        present = [p for p in _POLICY_ORDER if p in cell]
        present.sort(key=lambda p: cell[p]["mean_reward"])

        labels = [_DISPLAY.get(p, p) for p in present]
        means = [cell[p]["mean_reward"] for p in present]
        lo_err = [
            max(cell[p]["mean_reward"] - cell[p]["ci_low"], 0.0)
            for p in present
        ]
        hi_err = [
            max(cell[p]["ci_high"] - cell[p]["mean_reward"], 0.0)
            for p in present
        ]
        colours = [
            "#dc2626" if p == "recommended_action"  # oracle marker
            else "#2563eb" if p in _RL_ALGOS         # RL blue
            else "#9ca3af"                            # baselines grey
            for p in present
        ]

        bars = ax.barh(
            labels, means, xerr=[lo_err, hi_err],
            color=colours, edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=0.8),
        )
        for bar, m in zip(bars, means):
            x = m + 30.0
            ax.text(
                x, bar.get_y() + bar.get_height() / 2,
                f"{m:.0f}", va="center", ha="left", fontsize=7,
            )

        ax.set_title(ood_class, fontsize=10)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        if means:
            xmin = min(m - lo for m, lo in zip(means, lo_err))
            xmax = max(m + hi for m, hi in zip(means, hi_err))
            ax.set_xlim(xmin - 80, xmax + 250)

    # Hide any unused subplot.
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(
        "F15 — OOD-class robustness (mean episodic reward, 95 % bootstrap CI)",
        fontsize=12, y=1.0,
    )
    fig.text(
        0.5, -0.01,
        "audit-AF1 · trained-RL recovery of supervised-detector OOD blind spots · "
        "Phase-4 RF recall on VulnerabilityScan = 0.001",
        ha="center", fontsize=8, style="italic",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 F15 — OOD-class robustness plot + summary "
                    "(audit-AF1, headline G7.9 evaluator).",
    )
    p.add_argument(
        "--runs-root", default="runs/phase7/ood",
        help="Where run_ood_eval.py wrote its outputs.",
    )
    p.add_argument(
        "--out-dir", default="docs/results/07_ablation",
        help="Where to write F15_*.{png,json,md}.",
    )
    p.add_argument(
        "--ood-classes", nargs="+", default=_OOD_CLASSES_DEFAULT,
    )
    p.add_argument(
        "--policies", nargs="+", default=list(_POLICY_ORDER),
    )
    p.add_argument(
        "--phase6-eval-manifest",
        default="runs/phase6/eval_manifest.json",
    )
    p.add_argument(
        "--phase5-sweep-manifest",
        default="runs/phase5/sweep_manifest.json",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        logger.error("runs_root not found: %s — run "
                     "`python -m scripts.ablation.run_ood_eval` first.",
                     runs_root)
        return 1

    sha_collector: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    for ood_class in args.ood_classes:
        for policy in args.policies:
            seed_dirs = _discover_seed_dirs(runs_root, ood_class, policy)
            row = _summarise_cell(
                ood_class, policy, seed_dirs, sha_collector=sha_collector,
            )
            rows.append(row)
            logger.info(
                "F15 cell: ood=%s policy=%s n_seeds=%d n_ep=%d mean=%.1f CI=(%.1f, %.1f)",
                ood_class, policy, row["n_seeds"], row["n_episodes"],
                row["mean_reward"], row["ci_low"], row["ci_high"],
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render the figure.
    png_path = out_dir / "F15_ood_robustness.png"
    _render(rows, list(args.ood_classes), png_path)

    # Evaluate gates.
    g78 = _evaluate_g78(rows, list(args.ood_classes), list(args.policies))
    g79 = _evaluate_g79(rows)

    summary = {
        "schema_version": "1.0",
        "phase": 7,
        "figure": "F15",
        "audit_finding": "AF1 — promote OOD-class robustness to Tier-1 "
                          "deliverable (2026-04-30 mentor audit).",
        "ood_classes": list(args.ood_classes),
        "policies": list(args.policies),
        "rows": rows,
        "gates": {
            "G7.8": g78,
            "G7.9": g79,
        },
        "headline": (
            f"G7.9: {g79.get('interpretation', '?')}"
            if g78.get("passes") else
            "G7.8 FAIL — F15 result matrix incomplete; G7.9 not evaluated."
        ),
    }
    (out_dir / "F15_summary.json").write_text(json.dumps(summary, indent=2))

    # Manifest with hash chain (D7.7).
    eval_manifest_path = runs_root / "eval_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F15",
        "git_sha": _git_sha(),
        "audit_finding": "AF1",
        "outputs": {
            "png":  str(png_path),
            "json": str(out_dir / "F15_summary.json"),
        },
        "inputs": {
            "phase7_ood_eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
            "phase6_eval_manifest": {
                "path": str(args.phase6_eval_manifest),
                "sha256": _sha256(Path(args.phase6_eval_manifest)),
            },
            "phase5_sweep_manifest": {
                "path": str(args.phase5_sweep_manifest),
                "sha256": _sha256(Path(args.phase5_sweep_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F15_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Caption placeholder (hand-edit before publication).
    caption_path = out_dir / "F15_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F15 — OOD-class robustness.** Mean episodic reward of every "
            "Phase-6 policy under each held-out attack class "
            "(DDoS-HTTP_Flood, Mirai-udpplain, VulnerabilityScan, XSS), "
            "with the env's `RealizationEngine.allowed_indices` restricted "
            "to that class's row indices. The supervised RF baseline collapses "
            "on `VulnerabilityScan` (Phase-4 F11 recall = 0.001); F15 "
            "quantifies how much of that blind spot trained RL recovers by "
            "acting on raw features rather than the detector's "
            "classification. Error bars are 95 % bootstrap CIs. "
            "(Audit AF1, 2026-04-30; PLAN §3.1.3 / D7.6.)\n"
        )

    logger.info(
        "F15 written to %s — G7.8=%s, G7.9=%s",
        out_dir, g78["passes"], g79.get("passes"),
    )
    if g79.get("passes") is False:
        logger.warning("G7.9 PASS=False — see F15_summary.json#headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
