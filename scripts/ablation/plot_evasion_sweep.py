"""ablation F17 — Evasion-reactive sensitivity plot (prevention pivot).

Reads ``runs/ablation/evasion/ppo_e<e>/seed_<k>/eval_test.jsonl`` (produced
by ``scripts.ablation.run_evasion_sweep``) and renders how a PPO defender —
*trained and evaluated against an evasive attacker* — fares as the evasion
coupling strengthens, all under the primary ``impact_is_terminal=False``
contract:

  x-axis:  evasion_prob ∈ {0.0, 0.25, 0.5, 0.75}
  y-axis:  mean episodic reward on test_balanced (95 % bootstrap CI)
  curve:   trained PPO (across N seeds, on-contract outcome reward)

``evasion_prob`` models an *evasive* attacker (adversarial_env.py
"evasion-before-commit"): when the defender has recently applied force
(BLOCK/ISOLATE) and the attacker is at a pre-trigger stage (RECON/ACCESS),
with probability ``evasion_prob`` the attacker STALLS in place instead of
progressing. At ``evasion_prob=0`` this reduces to the standard Markov
attacker (so the e=0 cell is the within-sweep reference).

Outputs:
- ``F17_evasion_sweep.png``
- ``F17_summary.json`` — per-e {mean_reward, ci, compromise_rate, n_*};
                          + G7.10 evaluation
- ``F17_caption.md`` (placeholder)
- ``F17_manifest.json`` (SHA chain)

Gate evaluation:

- **G7.10** — pass iff the trained PPO defender stays ROBUST to a stronger
  evasive attacker: mean test reward at the highest evasion level is within
  a tolerance band of the reward at evasion=0 (the highest-evasion low-CI is
  not below the e=0 low-CI by more than ``--robust-tol`` of the e=0 mean).
  Otherwise FAIL-WITH-FINDING (D7.10.1): evasion materially degrades the
  trained defender — a documented robustness limit, not a hard failure.
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

from scripts._plot_style import ACCENT, apply_house_style, save_figure
from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_evasion_sweep")

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_EVASION_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75]


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


def _e_slug(e: float) -> str:
    """Filesystem-safe slug for an evasion value, e.g. 0.25 -> '0p25'.

    Must match ``run_evasion_sweep._e_slug`` (2-decimal form).
    """
    return f"{e:.2f}".replace(".", "p")


def _summarise(
    runs_root: Path,
    e: float,
    *,
    sha_collector: dict[str, str],
) -> dict[str, Any]:
    """Aggregate one evasion-level cell across all seeds."""
    base = runs_root / f"ppo_e{_e_slug(e)}"
    seed_dirs = (
        sorted(
            d
            for d in base.iterdir()
            if base.exists()
            and d.is_dir()
            and d.name.startswith("seed_")
            and (d / "eval_test.jsonl").exists()
        )
        if base.exists()
        else []
    )
    all_records: list[dict] = []
    per_seed_means: list[float] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(float(np.mean([r["episode_reward"] for r in recs])))
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.resolve().relative_to(_ROOT))] = sha

    if not all_records:
        return {
            "evasion_prob": e,
            "n_seeds": len(seed_dirs),
            "n_episodes": 0,
            "mean_reward": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "compromise_rate": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    # Bootstrap over per-seed means when we have enough seeds (matches F10),
    # else fall back to the episode-level pool.
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

    compromised = [bool(r.get("compromised")) for r in all_records]
    return {
        "evasion_prob": e,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "compromise_rate": float(np.mean(compromised)),
    }


def _evaluate_g710(
    rows: list[dict[str, Any]],
    *,
    robust_tol: float,
) -> dict[str, Any]:
    """G7.10: PPO stays robust as the evasive attacker strengthens.

    Pass iff the highest-evasion cell's reward is within a tolerance band of
    the evasion=0 reference: the highest-evasion low-CI is not below the e=0
    low-CI by more than ``robust_tol`` × |e=0 mean|.
    """
    by_e = {r["evasion_prob"]: r for r in rows}
    es = sorted(by_e.keys())
    if not es:
        return {"passes": False, "reason": "no evasion cells", "interpretation": "FAIL"}

    e_ref = by_e[es[0]]
    e_max = by_e[es[-1]]
    finite = math.isfinite(e_ref.get("mean_reward", math.nan)) and math.isfinite(
        e_max.get("mean_reward", math.nan)
    )
    if not finite:
        return {
            "passes": False,
            "reference_evasion": es[0],
            "max_evasion": es[-1],
            "reason": "NaN reference or max-evasion cell",
            "interpretation": (
                "FAIL-WITH-FINDING (D7.10.1): missing/NaN cell prevented the "
                "robustness comparison."
            ),
        }

    tol_abs = robust_tol * abs(e_ref["mean_reward"])
    degradation = e_ref["ci_low"] - e_max["ci_low"]
    passes = bool(degradation <= tol_abs)
    return {
        "passes": passes,
        "reference_evasion": es[0],
        "max_evasion": es[-1],
        "reference_mean_reward": e_ref["mean_reward"],
        "reference_ci": [e_ref["ci_low"], e_ref["ci_high"]],
        "max_evasion_mean_reward": e_max["mean_reward"],
        "max_evasion_ci": [e_max["ci_low"], e_max["ci_high"]],
        "robust_tol": robust_tol,
        "tolerance_abs": tol_abs,
        "ci_low_degradation": degradation,
        "reference_compromise_rate": e_ref.get("compromise_rate"),
        "max_evasion_compromise_rate": e_max.get("compromise_rate"),
        "interpretation": (
            f"PASS: the PPO defender stays robust to an evasive attacker — "
            f"mean test reward at evasion={es[-1]} "
            f"({e_max['mean_reward']:.1f}) is within {robust_tol:.0%} of the "
            f"evasion=0 reference ({e_ref['mean_reward']:.1f}); the "
            f"evasion-before-commit coupling does not collapse the learned "
            f"defense."
            if passes
            else (
                f"FAIL-WITH-FINDING (D7.10.1): an evasive attacker materially "
                f"degrades the trained defender — mean test reward falls from "
                f"{e_ref['mean_reward']:.1f} (evasion=0) to "
                f"{e_max['mean_reward']:.1f} (evasion={es[-1]}), a CI-low drop "
                f"of {degradation:.1f} exceeding the {robust_tol:.0%} band "
                f"({tol_abs:.1f}). This is a documented robustness limit of "
                f"the prevention spine under attacker evasion, not a hard "
                f"failure; flag in prose (threats-to-validity)."
            )
        ),
    }


def _render(rows: list[dict[str, Any]], out_path: Path) -> None:
    apply_house_style()
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["evasion_prob"])
    n_seeds = max((int(r.get("n_seeds", 0) or 0) for r in rows), default=0)
    seed_lbl = f"PPO ({n_seeds} seeds)" if n_seeds else "PPO"

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    xs = [r["evasion_prob"] for r in rows]
    means = [r["mean_reward"] for r in rows]
    lo = [r["ci_low"] for r in rows]
    hi = [r["ci_high"] for r in rows]
    ax.plot(
        xs,
        means,
        marker="o",
        color=ACCENT["primary"],
        label=seed_lbl,
        linewidth=2.4,
        zorder=5,
    )
    ax.fill_between(xs, lo, hi, alpha=0.18, color=ACCENT["primary"], zorder=1)

    ax.axhline(0.0, color=ACCENT["muted"], lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("Attacker evasion probability (stall-on-force at RECON/ACCESS)")
    ax.set_ylabel("Mean episodic reward on test_balanced")
    ax.set_title("Defender robustness to an evasive attacker")
    ax.legend(loc="best", framealpha=0.9)
    if xs:
        ax.set_xticks(xs)

    save_figure(fig, out_path)
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F17 — evasion-reactive sweep plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/ablation/evasion")
    p.add_argument("--out-dir", default="docs/results/ablation")
    p.add_argument(
        "--evasion-values",
        nargs="+",
        type=float,
        default=_DEFAULT_EVASION_VALUES,
    )
    p.add_argument(
        "--robust-tol",
        type=float,
        default=0.25,
        help="G7.10 robustness tolerance as a fraction of the evasion=0 mean.",
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
        logger.error("runs_root not found: %s", runs_root)
        return 1

    sha_collector: dict[str, str] = {}
    rows = [_summarise(runs_root, e, sha_collector=sha_collector) for e in args.evasion_values]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = out_dir / "F17_evasion_sweep"
    _render(rows, fig_base)
    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")

    g710 = _evaluate_g710(rows, robust_tol=args.robust_tol)
    summary = {
        "schema_version": "1.0",
        "stage": "ablation",
        "figure": "F17",
        "evasion_values": list(args.evasion_values),
        "rows": rows,
        "gates": {"G7.10": g710},
        "headline": g710.get("interpretation", "?"),
    }
    (out_dir / "F17_summary.json").write_text(json.dumps(summary, indent=2))

    sweep_manifest_path = runs_root / "sweep_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F17",
        "git_sha": _git_sha(),
        "outputs": {
            "pdf": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "png": str(png_path),
            "json": str(out_dir / "F17_summary.json"),
        },
        "inputs": {
            "ablation_evasion_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F17_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F17_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F17 — Defender robustness to an evasive attacker.** Mean "
            "episodic reward on `test_balanced` for PPO (green) trained and "
            "evaluated against an *evasive* attacker "
            "as a function of `evasion_prob` (the probability the attacker "
            "stalls in place at RECON/ACCESS when the defender has recently "
            "applied force). Shaded band: "
            "95 % bootstrap CI. The `evasion_prob=0` cell is the standard "
            "Markov-attacker reference. (PLAN §3.1.6; D7.10.)\n"
        )

    logger.info(
        "F17 written to %s — G7.10 passes=%s",
        out_dir,
        g710.get("passes"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
