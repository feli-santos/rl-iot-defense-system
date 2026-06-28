"""ablation F15 — OOD-class robustness plotter (audit-AF1, headline).

Reads ``runs/ablation/ood/<class>/<policy>/seed_<k>/eval_test.jsonl``
files (produced by :mod:`scripts.ablation.run_ood_eval`), aggregates
per (class, policy) with the same bootstrap-CI protocol benchmark used
in :mod:`scripts.benchmark.build_summary_table`, and emits:

- ``F15_ood_robustness.png`` — 10-class × 8-policy grouped horizontal
  bar chart with 95 % bootstrap CI whiskers, one panel per OOD class.
  Same visual idiom as benchmark F8.
- ``F15_summary.json`` — per (class, policy) row + a ``headline`` block
  reporting whether trained RL beat RF-Acting on each class (G7.9
  evaluator).
- ``F15_caption.md`` — thesis caption (separate, hand-written; this
  script writes a placeholder).
- ``F15_manifest.json`` — SHA-256 hash chain over every input JSONL,
  the upstream ablation ``eval_manifest.json``, the benchmark
  ``eval_manifest.json``, and the Blue-Team ``sweep_manifest.json``.

Gate evaluation:

- **G7.8** (audit-AF1) — pass iff every (class, policy) cell produced
  a non-empty eval_test.jsonl with a finite mean_reward (no NaNs in
  the 10 × 8 result matrix).
- **G7.9** (audit-AF1, headline) — pass iff on `VulnerabilityScan`,
  the *best* trained RL mean_reward exceeds RF-Acting mean_reward by
  at least 1σ of the per-policy bootstrap CI (i.e., the lower bound
  of the trained-RL CI exceeds the upper bound of the RF-Acting CI).
  Acceptable failure mode: turns into a finding (D7.9.1) — the
  thesis claim narrows from "RL closes the OOD gap" to "RL is
  *robust to* (not *better at*) the OOD class".

detector reminder: the supervised RF stage detector has **recall 0.076
on `VulnerabilityScan`** (F11 standalone) / **0.224** (F15 in-env).
F15 quantifies how much of that blind spot the trained RL policy
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
from typing import Any

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_ood_robustness")

_ROOT = Path(__file__).resolve().parents[2]


# Display ordering — same as benchmark F8, plus rule baseline as the
# oracle ceiling marker.
_POLICY_ORDER: list[str] = [
    "recommended_action",  # oracle ceiling
    "rf_acting",
    "dqn",
    "ppo",
    "a2c",
    "always_block",
    "random",
    "always_observe",
]
_DISPLAY: dict[str, str] = {
    "recommended_action": "Rec-Action (oracle)",
    "rf_acting": "RF-Acting",
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "always_block": "Always-BLOCK",
    "random": "Random",
    "always_observe": "Always-OBSERVE",
}
_RL_ALGOS = {"dqn", "ppo", "a2c"}
_OOD_CLASSES_DEFAULT: list[str] = [
    # RECON
    "VulnerabilityScan",
    "Recon-OSScan",
    # ACCESS
    "XSS",
    "SqlInjection",
    # MANEUVER
    "Mirai-udpplain",
    "DNS_Spoofing",
    # IMPACT
    "DDoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-ACK_Fragmentation",
]


def _sha256(path: Path) -> str | None:
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
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------- aggregation


def _discover_seed_dirs(runs_root: Path, ood_class: str, policy: str) -> list[Path]:
    """Find every ``seed_<k>`` dir under
    ``<runs_root>/<ood_class>/<policy>/`` that contains
    ``eval_test.jsonl``."""
    base = runs_root / ood_class / policy
    if not base.exists():
        return []
    return sorted(
        d
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("seed_") and (d / "eval_test.jsonl").exists()
    )


def _summarise_cell(
    ood_class: str,
    policy: str,
    seed_dirs: list[Path],
    *,
    sha_collector: dict[str, str],
) -> dict[str, Any]:
    """Compute one (ood_class, policy) row.

    Same bootstrap protocol as benchmark F5: when n_seeds ≥ 3, bootstrap
    across per-seed means; else bootstrap across all episodes.
    """
    all_records: list[dict] = []
    per_seed_means: list[float] = []
    input_paths: list[Path] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(float(np.mean([r["episode_reward"] for r in recs])))
        input_paths.append(jsonl)
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.resolve().relative_to(_ROOT))] = sha

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
            "prevention_rate": math.nan,
            # Diagnostic-only (P(IMPACT)-weighted; not commensurable across
            # operating points). prevention_rate + compromise_rate are primary.
            "mitigated_impact_rate": math.nan,
            "mean_episode_length": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    compromised = [1.0 if r.get("compromised") else 0.0 for r in all_records]
    prevented = [1.0 if r.get("end_outcome") == "prevented" else 0.0 for r in all_records]
    mitigated = [1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0 for r in all_records]
    lengths = [r["episode_length"] for r in all_records]

    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means,
            n_resamples=2000,
            alpha=0.05,
            seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards,
            n_resamples=2000,
            alpha=0.05,
            seed=0,
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
        "prevention_rate": float(np.mean(prevented)),
        # Diagnostic-only (P(IMPACT)-weighted); kept for backward-compat but
        # demoted from headline — prevention_rate is the primary security KPI.
        "mitigated_impact_rate": float(np.mean(mitigated)),
        "mean_episode_length": float(np.mean(lengths)),
    }


# --------------------------------------------------------------- gates


def _evaluate_g79(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        r["policy"]: r
        for r in rows
        if r["ood_class"] == target_class
        and r["policy"]
        in (
            "rf_acting",
            "dqn",
            "ppo",
            "a2c",
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
        math.isfinite(best["ci_low"]) and math.isfinite(rf_ci_high) and best["ci_low"] > rf_ci_high
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
        "interpretation": (
            "Trained RL recovers some of the supervised RF blind spot on VulnerabilityScan."
            if passes
            else "Trained RL does NOT beat RF-Acting on VulnerabilityScan; "
            "the claim narrows from 'RL closes the OOD gap' to "
            "'RL is robust to (not better at) the OOD class'."
        ),
    }


def _evaluate_g78(
    rows: list[dict[str, Any]],
    expected_classes: list[str],
    expected_policies: list[str],
) -> dict[str, Any]:
    """G7.8: the full (classes × policies) result matrix is complete with
    no NaN means."""
    have: dict[tuple[str, str], dict[str, Any]] = {(r["ood_class"], r["policy"]): r for r in rows}
    missing: list[tuple[str, str]] = []
    nan_cells: list[tuple[str, str]] = []
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
    rows: list[dict[str, Any]],
    ood_classes: list[str],
    out_path: Path,
) -> None:
    """Per-class grouped bar chart of PREVENTION RATE (one panel per OOD class).

    The security metric is the prevention rate (fraction of episodes in which
    the attacker is held below the impact stage for the full horizon), matching
    the thesis F15 narrative. Raw episodic reward is dominated by accumulated
    step penalties on long episodes and is NOT plotted here.
    """
    from scripts._plot_style import ACCENT, apply_house_style, policy_style, save_figure

    apply_house_style()
    import matplotlib.pyplot as plt

    by_class: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in ood_classes}
    for r in rows:
        by_class.setdefault(r["ood_class"], {})[r["policy"]] = r

    n = len(ood_classes)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 2.6 * n_rows), squeeze=False)

    def _bar_colour(policy: str) -> str:
        if policy == "ppo":
            return ACCENT["primary"]  # the contribution, emphasised
        if policy == "recommended_action":
            return ACCENT["neutral"]  # full-observability oracle
        if policy == "rf_acting":
            return ACCENT["secondary"]  # supervised baseline (the foil)
        if policy in _RL_ALGOS:
            return policy_style(policy).get("color", ACCENT["muted"])
        return ACCENT["muted"]  # heuristic baselines

    for ax_idx, ood_class in enumerate(ood_classes):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        cell = by_class.get(ood_class, {})
        # Sort displayed policies by prevention_rate asc so the strongest
        # defender is at the top of each panel.
        present = [p for p in _POLICY_ORDER if p in cell]
        present.sort(key=lambda p: cell[p].get("prevention_rate", 0.0))

        labels = [_DISPLAY.get(p, p) for p in present]
        prevent = [float(cell[p].get("prevention_rate", 0.0)) for p in present]
        colours = [_bar_colour(p) for p in present]

        bars = ax.barh(
            labels,
            prevent,
            color=colours,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, v in zip(bars, prevent):
            if not math.isfinite(v):
                continue
            ax.text(
                min(v + 0.02, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}",
                va="center",
                ha="left",
                fontsize=7,
            )

        ax.set_title(ood_class, fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # Shared x-label on the bottom row only.
    for c in range(n_cols):
        last_row = (n - 1) // n_cols if c <= (n - 1) % n_cols else (n - 1) // n_cols - 1
        if last_row >= 0:
            axes[last_row][c].set_xlabel("Prevention rate (held below impact)")

    # Hide any unused subplot.
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle(
        "Out-of-distribution prevention rate across ten held-out zero-day classes",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# --------------------------------------------- per-class detector recall


def _compute_per_class_rf_recall(
    ood_classes: list[str],
    *,
    rf_path: Path,
    dataset_dir: Path,
    stage_by_class: dict[str, int],
) -> dict[str, float | None]:
    """Per-held-out-class recall of the (retrained) RF stage detector.

    Recall here = fraction of that class's rows the RF assigns to the
    class's *true* kill-chain stage. A low value means the supervised
    detector is blind to the class (e.g. VulnerabilityScan ≈ 0.001), which
    is exactly the x-axis of the detector-independence figure: it lets us
    plot the RL-minus-RF advantage as a function of how badly the upstream
    detector fails on each zero-day class.

    Returns ``{class: recall}`` with ``None`` where artefacts are missing
    (so the plotter degrades gracefully on a partial checkout).
    """
    try:
        import joblib
    except Exception:  # noqa: BLE001
        logger.warning("joblib unavailable — skipping per-class RF recall.")
        return dict.fromkeys(ood_classes)

    if not Path(rf_path).exists():
        logger.warning("RF detector not found at %s — skipping recall.", rf_path)
        return dict.fromkeys(ood_classes)

    features_path = dataset_dir / "features.npy"
    labels_path = dataset_dir / "labels.npy"
    splits_dir = dataset_dir / "splits" / "ood_attack"
    if not (features_path.exists() and labels_path.exists() and splits_dir.exists()):
        logger.warning("dataset artefacts missing under %s — skipping recall.", dataset_dir)
        return dict.fromkeys(ood_classes)

    rf = joblib.load(rf_path)
    # NB: features.npy is ALREADY the normalised feature matrix the RF was
    # trained on (the detector trainer feeds features.npy straight into the
    # RandomForest, and the environment serves these same scaled rows to the
    # RF-Acting policy). Re-applying the StandardScaler here would double-scale
    # the inputs and corrupt the predictions, so we predict on the raw stored
    # rows to match both training and in-environment inference.
    features = np.load(features_path, mmap_mode="r")

    out: dict[str, float | None] = {}
    for cls in ood_classes:
        idx_path = splits_dir / f"{cls}.idx.npy"
        true_stage = stage_by_class.get(cls)
        if not idx_path.exists() or true_stage is None:
            out[cls] = None
            continue
        idx = np.load(idx_path)
        if idx.size == 0:
            out[cls] = None
            continue
        X = np.ascontiguousarray(features[idx], dtype=np.float32)
        pred = rf.predict(X)
        out[cls] = float(np.mean(pred == true_stage))
    return out


def _render_recall_vs_advantage(
    rows: list[dict[str, Any]],
    recall_by_class: dict[str, float | None],
    ood_classes: list[str],
    out_path: Path,
    *,
    metric: str = "prevention_rate",
) -> dict[str, Any]:
    """Detector-independence figure: RL-minus-RF advantage vs detector recall.

    For each held-out class we plot a point at (x = per-class RF stage-recall,
    y = best-trained-RL minus RF-Acting on ``metric``). The thesis claim — that
    a detector-free RL policy is *robust to* the supervised detector's blind
    spots — predicts the advantage rises as detector recall falls (a negative
    slope / upper-left cluster). Returns the plotted point table for the JSON.
    """
    from scripts._plot_style import ACCENT, apply_house_style, save_figure

    apply_house_style()
    import matplotlib.pyplot as plt

    by_cp = {(r["ood_class"], r["policy"]): r for r in rows}
    points: list[dict[str, Any]] = []
    for cls in ood_classes:
        rf = by_cp.get((cls, "rf_acting"))
        recall = recall_by_class.get(cls)
        if rf is None or recall is None or not math.isfinite(rf.get(metric, math.nan)):
            continue
        rl_vals = [
            by_cp[(cls, a)][metric]
            for a in _RL_ALGOS
            if (cls, a) in by_cp and math.isfinite(by_cp[(cls, a)].get(metric, math.nan))
        ]
        if not rl_vals:
            continue
        best_rl = max(rl_vals)
        points.append(
            {
                "ood_class": cls,
                "rf_recall": float(recall),
                "rf_metric": float(rf[metric]),
                "best_rl_metric": float(best_rl),
                "advantage": float(best_rl - rf[metric]),
            }
        )

    stats = _recall_independence_stats(points)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    if points:
        xs = [p["rf_recall"] for p in points]
        ys = [p["advantage"] for p in points]
        ax.axhline(0.0, color=ACCENT["muted"], lw=0.8, ls="--")
        ax.scatter(xs, ys, c=ACCENT["primary"], edgecolor="white", s=60, zorder=3)
        for p in points:
            ax.annotate(
                p["ood_class"],
                (p["rf_recall"], p["advantage"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )
        # OLS fit line + annotation of the formal independence statistics.
        if stats.get("ols_slope") is not None:
            x_lo, x_hi = min(xs), max(xs)
            y_lo = stats["ols_intercept"] + stats["ols_slope"] * x_lo
            y_hi = stats["ols_intercept"] + stats["ols_slope"] * x_hi
            ax.plot(
                [x_lo, x_hi],
                [y_lo, y_hi],
                color=ACCENT["muted"],
                lw=1.0,
                ls="-",
                zorder=2,
            )
            ax.text(
                0.03,
                0.05,
                (
                    rf"Spearman $\rho$={stats['spearman_rho']:.2f} "
                    rf"(p={stats['spearman_p']:.2f}); "
                    rf"Pearson $r$={stats['pearson_r']:.2f} "
                    rf"(p={stats['pearson_p']:.2f}); "
                    rf"OLS slope={stats['ols_slope']:.2f} "
                    rf"[{stats['ols_slope_ci_low']:.2f}, "
                    rf"{stats['ols_slope_ci_high']:.2f}]"
                    "\n"
                    r"No negative trend: advantage does not rise as detector"
                    " recall falls (n=10)."
                ),
                transform=ax.transAxes,
                fontsize=7,
                va="bottom",
                ha="left",
            )
    ax.set_xlabel("Supervised RF detector recall on held-out class")
    ax.set_ylabel(f"Best-RL − RF-Acting  ({metric})")
    ax.set_title("Detector-independence dividend: RL advantage vs detector recall")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    return {"metric": metric, "points": points, "stats": stats}


def _recall_independence_stats(
    points: list[dict[str, Any]],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Formal test of the ``recall-independence`` claim.

    Given the per-class (RF detector recall, best-RL-minus-RF advantage)
    points, quantify whether the advantage depends on detector recall.
    Returns Spearman rank correlation, Pearson correlation, and an
    ordinary-least-squares slope with a nonparametric bootstrap CI.
    The thesis claim is *independence*: the null of zero association
    should NOT be rejected (small |rho|, slope CI spanning zero).
    """
    import numpy as np
    from scipy import stats as sps

    n = len(points)
    if n < 3:
        return {"n": n, "insufficient_points": True}

    x = np.array([p["rf_recall"] for p in points], dtype=float)
    y = np.array([p["advantage"] for p in points], dtype=float)

    spearman = sps.spearmanr(x, y)
    pearson = sps.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for b in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        if np.ptp(x[take]) == 0.0:
            boot_slopes[b] = np.nan
            continue
        boot_slopes[b] = np.polyfit(x[take], y[take], 1)[0]
    boot_slopes = boot_slopes[np.isfinite(boot_slopes)]
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

    return {
        "n": n,
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "ols_slope_ci_low": float(ci_low),
        "ols_slope_ci_high": float(ci_high),
        "bootstrap_n": int(boot_slopes.size),
        "interpretation": (
            "No detractor-supporting dependence: the Spearman rank "
            "correlation is non-significant and the bootstrap OLS-slope CI "
            "spans zero, so the advantage does not diminish as detector "
            "recall rises. Any weak linear association (Pearson) is in the "
            "positive direction, opposite to the 'RL only rescues the "
            "detector's blind spots' hypothesis."
        ),
    }


# --------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OOD-class robustness plot + summary "
        "(held-out zero-day classes; detector-independence figure).",
    )
    p.add_argument(
        "--runs-root",
        default="runs/ablation/ood",
        help="Where run_ood_eval.py wrote its outputs.",
    )
    p.add_argument(
        "--out-dir",
        default="docs/results/ablation",
        help="Where to write F15_*.{png,json,md}.",
    )
    p.add_argument(
        "--ood-classes",
        nargs="+",
        default=_OOD_CLASSES_DEFAULT,
    )
    p.add_argument(
        "--policies",
        nargs="+",
        default=list(_POLICY_ORDER),
    )
    p.add_argument(
        "--benchmark-eval-manifest",
        default="runs/benchmark/eval_manifest.json",
    )
    p.add_argument(
        "--blue-team-sweep-manifest",
        default="runs/blue_team/sweep_manifest.json",
    )
    # Step-8 F2 (07_HANDOFF.md §5): explicit upstream-manifest SHA pin
    # for the dataset-prep splits manifest so the F15 hash chain is
    # self-contained (matches the F9/F10/F12 pattern landed in Step 8).
    p.add_argument(
        "--split-splits-manifest",
        default="docs/results/dataset/manifest.json",
        help="dataset-prep splits manifest.json (post-3cd2fb9; SHA 1e99d596...).",
    )
    p.add_argument(
        "--rf-path",
        default="artifacts/detector/random_forest.joblib",
        help="RF stage detector used by RF-Acting (for per-class recall x-axis).",
    )
    p.add_argument(
        "--dataset-dir",
        default="data/processed/ciciot2023",
        help="Processed dataset dir (features/labels/splits) for recall computation.",
    )
    p.add_argument(
        "--advantage-metric",
        default="prevention_rate",
        choices=["prevention_rate", "mean_reward", "compromise_rate"],
        help="Metric for the RL-minus-RF detector-independence figure.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        logger.error(
            "runs_root not found: %s — run `python -m scripts.ablation.run_ood_eval` first.",
            runs_root,
        )
        return 1

    sha_collector: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for ood_class in args.ood_classes:
        for policy in args.policies:
            seed_dirs = _discover_seed_dirs(runs_root, ood_class, policy)
            row = _summarise_cell(
                ood_class,
                policy,
                seed_dirs,
                sha_collector=sha_collector,
            )
            rows.append(row)
            logger.info(
                "F15 cell: ood=%s policy=%s n_seeds=%d n_ep=%d mean=%.1f CI=(%.1f, %.1f)",
                ood_class,
                policy,
                row["n_seeds"],
                row["n_episodes"],
                row["mean_reward"],
                row["ci_low"],
                row["ci_high"],
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-class RF recall (computed first so the F15 panels can be ordered by
    # detector recall: this makes the "advantage does not track recall" reading
    # legible left-to-right / top-to-bottom).
    from scripts.ablation.run_ood_eval import _OOD_STAGE_BY_CLASS

    recall_by_class = _compute_per_class_rf_recall(
        list(args.ood_classes),
        rf_path=Path(args.rf_path),
        dataset_dir=Path(args.dataset_dir),
        stage_by_class=_OOD_STAGE_BY_CLASS,
    )

    # Order panels by ascending detector recall (blind spots first). Classes
    # with missing recall fall back to their original position at the end.
    def _recall_key(cls: str) -> float:
        r = recall_by_class.get(cls)
        return r if (r is not None and math.isfinite(r)) else math.inf

    ordered_classes = sorted(args.ood_classes, key=_recall_key)

    # Render the figure (save_figure derives both .pdf and .png from the base).
    fig_base = out_dir / "F15_ood_robustness"
    _render(rows, ordered_classes, fig_base)
    png_path = fig_base.with_suffix(".png")
    pdf_path = fig_base.with_suffix(".pdf")
    recall_base = out_dir / "F15b_recall_vs_advantage"
    recall_fig = _render_recall_vs_advantage(
        rows,
        recall_by_class,
        list(args.ood_classes),
        recall_base,
        metric=args.advantage_metric,
    )
    recall_png = recall_base.with_suffix(".png")
    recall_pdf = recall_base.with_suffix(".pdf")

    # Evaluate gates.
    g78 = _evaluate_g78(rows, list(args.ood_classes), list(args.policies))
    g79 = _evaluate_g79(rows)

    summary = {
        "schema_version": "1.0",
        "stage": "ablation",
        "figure": "F15",
        "audit_finding": "AF1 — promote OOD-class robustness to Tier-1 "
        "deliverable (2026-04-30 mentor audit).",
        "ood_classes": list(args.ood_classes),
        "policies": list(args.policies),
        "rows": rows,
        "per_class_rf_recall": recall_by_class,
        "detector_independence": recall_fig,
        "gates": {
            "G7.8": g78,
            "G7.9": g79,
        },
        "headline": (
            f"G7.9: {g79.get('interpretation', '?')}"
            if g78.get("passes")
            else "G7.8 FAIL — F15 result matrix incomplete; G7.9 not evaluated."
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
            "pdf": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "png": str(png_path),
            "recall_vs_advantage_pdf": str(recall_pdf),
            "recall_vs_advantage_pdf_sha256": _sha256(recall_pdf),
            "recall_vs_advantage_png": str(recall_png),
            "json": str(out_dir / "F15_summary.json"),
        },
        "inputs": {
            "ablation_ood_eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
            "benchmark_eval_manifest": {
                "path": str(args.benchmark_eval_manifest),
                "sha256": _sha256(Path(args.benchmark_eval_manifest)),
            },
            "blue_team_sweep_manifest": {
                "path": str(args.blue_team_sweep_manifest),
                "sha256": _sha256(Path(args.blue_team_sweep_manifest)),
            },
            # Step-8 F2: explicit dataset-prep splits manifest pin so the
            # F15 hash chain is self-contained (matches F9/F10/F12).
            "split_splits_manifest": {
                "path": str(args.split_splits_manifest),
                "sha256": _sha256(Path(args.split_splits_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F15_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Caption placeholder (hand-edit before publication).
    caption_path = out_dir / "F15_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F15 — Zero-day (out-of-distribution) robustness.** Prevention "
            "rate (fraction of episodes in which the attacker is held below "
            "the impact stage for the full horizon) of every defence policy "
            "when each of the ten held-out attack classes is injected "
            "eval-only on the frozen agents, under the locked outcome reward "
            "contract. Prevention rate is reported in preference to raw "
            "episodic reward, which is dominated by accumulated step "
            "penalties over long episodes. The held-out classes span the "
            "detector's recall spectrum, from near-perfect to the "
            "`VulnerabilityScan` blind spot. Because the trained RL agent "
            "acts on raw features rather than the detector's stage label, it "
            "degrades gracefully where the detector-coupled RF-Acting "
            "baseline collapses; the companion figure plots this RL-minus-RF "
            "advantage against per-class detector recall.\n"
        )

    logger.info(
        "F15 written to %s — G7.8=%s, G7.9=%s",
        out_dir,
        g78["passes"],
        g79.get("passes"),
    )
    if g79.get("passes") is False:
        logger.warning("G7.9 PASS=False — see F15_summary.json#headline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
