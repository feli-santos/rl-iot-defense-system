"""ablation F17 — Evasion-reactive sensitivity plot (prevention pivot).

Reads ``runs/ablation/evasion/<algo>_e<e>/seed_<k>/eval_test.jsonl`` (produced
by ``scripts.ablation.run_evasion_sweep``) and renders how the fixed det-5M
alpha-04 defenders (PPO, A2C, DQN) — *evaluated against an increasingly
evasive attacker* — fare as the evasion coupling strengthens, all under the
primary ``impact_is_terminal=False`` contract:

  x-axis:  evasion_prob ∈ {0.0, 0.25, 0.5, 0.75}
  y-axis:  mean episodic reward on test_balanced (95 % bootstrap CI)
  curves:  fixed PPO / A2C / DQN (across N seeds each, on-contract outcome
           reward) overlaid for a cross-algorithm robustness comparison

``evasion_prob`` models an *evasive* attacker via evasive persistence
(adversarial_env.py "post-detection hardening"): once the attacker senses
force (BLOCK/ISOLATE) at a pre-commit stage (RECON/ACCESS) it hardens, and
on the next proportional (correctly-forced) step it RESISTS eviction with
probability ``evasion_prob`` — holding its ground rather than being
de-escalated (correct force still holds the line; it is never a loss for the
defender, the attacker just is not removed that turn). At ``evasion_prob=0``
this reduces to the standard tug-of-war attacker (so the e=0 cell is the
within-sweep reference).

Outputs:
- ``F17_evasion_sweep.png``
- ``F17_summary.json`` — per-e {mean_reward, ci, compromise_rate, n_*};
                          + G7.10 evaluation
- ``F17_caption.md`` (placeholder)
- ``F17_manifest.json`` (SHA chain)

Gate evaluation:

- **G7.10** — evaluated per-algorithm (ppo/a2c/dqn). For each algo, pass iff
  the trained defender stays ROBUST to a stronger evasive attacker: mean test
  reward at the highest evasion level is within a tolerance band of the reward
  at evasion=0 (the highest-evasion low-CI is not below the e=0 low-CI by more
  than ``--robust-tol`` of the e=0 mean). Otherwise FAIL-WITH-FINDING
  (D7.10.1): evasion materially degrades that defender — a documented
  robustness limit, not a hard failure. PPO retains the headline pass
  semantics; A2C/DQN are reported alongside as cross-algorithm findings.
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

from scripts._plot_style import ACCENT, apply_house_style, policy_label, policy_style, save_figure
from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_evasion_sweep")

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_EVASION_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75]
_DEFAULT_ALGOS: list[str] = ["ppo", "a2c", "dqn"]


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
    algo: str,
    e: float,
    *,
    sha_collector: dict[str, str],
) -> dict[str, Any]:
    """Aggregate one (algo, evasion-level) cell across all seeds."""
    base = runs_root / f"{algo}_e{_e_slug(e)}"
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
            "algo": algo,
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
        "algo": algo,
        "evasion_prob": e,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "compromise_rate": float(np.mean(compromised)),
    }


def _evaluate_g710_algo(
    algo: str,
    rows: list[dict[str, Any]],
    *,
    robust_tol: float,
) -> dict[str, Any]:
    """G7.10 for a single algorithm: stays robust as evasion strengthens.

    Pass iff the highest-evasion cell's reward is within a tolerance band of
    the evasion=0 reference: the highest-evasion low-CI is not below the e=0
    low-CI by more than ``robust_tol`` × |e=0 mean|.
    """
    label = algo.upper()
    by_e = {r["evasion_prob"]: r for r in rows}
    es = sorted(by_e.keys())
    if not es:
        return {
            "algo": algo,
            "passes": False,
            "reason": "no evasion cells",
            "interpretation": "FAIL",
        }

    e_ref = by_e[es[0]]
    e_max = by_e[es[-1]]
    finite = math.isfinite(e_ref.get("mean_reward", math.nan)) and math.isfinite(
        e_max.get("mean_reward", math.nan)
    )
    if not finite:
        return {
            "algo": algo,
            "passes": False,
            "reference_evasion": es[0],
            "max_evasion": es[-1],
            "reason": "NaN reference or max-evasion cell",
            "interpretation": (
                f"FAIL-WITH-FINDING (D7.10.1): missing/NaN {label} cell "
                f"prevented the robustness comparison."
            ),
        }

    tol_abs = robust_tol * abs(e_ref["mean_reward"])
    degradation = e_ref["ci_low"] - e_max["ci_low"]
    passes = bool(degradation <= tol_abs)
    return {
        "algo": algo,
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
            f"PASS: the {label} defender stays robust to an evasive attacker — "
            f"mean test reward at evasion={es[-1]} "
            f"({e_max['mean_reward']:.1f}) is within {robust_tol:.0%} of the "
            f"evasion=0 reference ({e_ref['mean_reward']:.1f}); evasive "
            f"persistence (post-detection hardening) degrades but does not "
            f"collapse the learned defense."
            if passes
            else (
                f"FAIL-WITH-FINDING (D7.10.1): an evasive attacker materially "
                f"degrades the trained {label} defender — mean test reward "
                f"falls from {e_ref['mean_reward']:.1f} (evasion=0) to "
                f"{e_max['mean_reward']:.1f} (evasion={es[-1]}), a CI-low drop "
                f"of {degradation:.1f} exceeding the {robust_tol:.0%} band "
                f"({tol_abs:.1f}). This is a documented robustness limit of "
                f"the prevention spine under attacker evasion, not a hard "
                f"failure; flag in prose (threats-to-validity)."
            )
        ),
    }


def _evaluate_g710(
    algo_rows_by_algo: dict[str, list[dict[str, Any]]],
    *,
    robust_tol: float,
) -> dict[str, Any]:
    """G7.10 across algorithms; PPO cell carries the headline pass semantics.

    Returns a dict with a top-level ``passes`` (mirroring the PPO cell for
    back-compat with older readers / close_ablation.py), a ``per_algo`` map of
    per-algorithm evaluations, and flat mirrors of the PPO cell's key fields.
    """
    per_algo = {
        algo: _evaluate_g710_algo(algo, rows, robust_tol=robust_tol)
        for algo, rows in algo_rows_by_algo.items()
    }
    ppo_cell = per_algo.get("ppo", {})
    out: dict[str, Any] = {
        "passes": bool(ppo_cell.get("passes", False)),
        "headline_algo": "ppo",
        "robust_tol": robust_tol,
        "per_algo": per_algo,
    }
    # Flat back-compat mirrors of the PPO cell (older readers / render_tables).
    for key in (
        "reference_evasion",
        "max_evasion",
        "reference_mean_reward",
        "reference_ci",
        "max_evasion_mean_reward",
        "max_evasion_ci",
        "tolerance_abs",
        "ci_low_degradation",
        "reference_compromise_rate",
        "max_evasion_compromise_rate",
        "interpretation",
    ):
        if key in ppo_cell:
            out[key] = ppo_cell[key]
    return out


def _render(
    algo_rows_by_algo: dict[str, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    apply_house_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    all_xs: list[float] = []
    for algo in _DEFAULT_ALGOS:
        rows = algo_rows_by_algo.get(algo)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["evasion_prob"])
        n_seeds = max((int(r.get("n_seeds", 0) or 0) for r in rows), default=0)
        style = policy_style(algo)
        label = policy_label(algo) + (f", {n_seeds} seeds" if n_seeds else "")

        xs = [r["evasion_prob"] for r in rows]
        means = [r["mean_reward"] for r in rows]
        lo = [r["ci_low"] for r in rows]
        hi = [r["ci_high"] for r in rows]
        all_xs = xs or all_xs
        ax.plot(
            xs,
            means,
            marker=style.get("marker", "o"),
            color=style["color"],
            label=label,
            linewidth=style.get("lw", 2.0),
            zorder=style.get("zorder", 3),
        )
        ax.fill_between(
            xs,
            lo,
            hi,
            alpha=0.15,
            color=style["color"],
            zorder=1,
        )

    ax.axhline(0.0, color=ACCENT["muted"], lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("Attacker evasion probability (stall-on-force at RECON/ACCESS)")
    ax.set_ylabel("Mean episodic reward on test_balanced")
    ax.set_title("Defender robustness to an evasive attacker")
    ax.legend(loc="best", framealpha=0.9)
    if all_xs:
        ax.set_xticks(all_xs)

    save_figure(fig, out_path)
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F17 — evasion-reactive sweep plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/ablation/evasion")
    p.add_argument("--out-dir", default="docs/results/ablation")
    p.add_argument(
        "--algos",
        nargs="+",
        choices=("ppo", "a2c", "dqn"),
        default=list(_DEFAULT_ALGOS),
        help="Algorithms to overlay (must have matching <algo>_e<e> run dirs).",
    )
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
    algo_rows_by_algo: dict[str, list[dict[str, Any]]] = {
        algo: [
            _summarise(runs_root, algo, e, sha_collector=sha_collector) for e in args.evasion_values
        ]
        for algo in args.algos
    }
    # Back-compat: top-level ``rows`` mirrors the PPO curve for older readers
    # (render_tables.py macros, close_ablation.py).
    rows = algo_rows_by_algo.get("ppo", [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = out_dir / "F17_evasion_sweep"
    _render(algo_rows_by_algo, fig_base)
    pdf_path = fig_base.with_suffix(".pdf")
    png_path = fig_base.with_suffix(".png")

    g710 = _evaluate_g710(algo_rows_by_algo, robust_tol=args.robust_tol)
    summary = {
        "schema_version": "1.0",
        "stage": "ablation",
        "figure": "F17",
        "algos": list(args.algos),
        "evasion_values": list(args.evasion_values),
        "rows": rows,
        "algo_rows": algo_rows_by_algo,
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
            "**F17 — Defender robustness to an evasive-persistence attacker.** "
            "Mean episodic reward on `test_balanced` for the fixed "
            "deterministic-5M α=0.4 defenders (PPO, A2C, DQN; N seeds each), "
            "evaluated (not retrained) against an attacker exhibiting *evasive "
            "persistence* (post-detection hardening) as a function of "
            "`evasion_prob` (the probability that, after sensing defensive "
            "force at RECON/ACCESS, the attacker resists the next eviction "
            "attempt). Shaded bands: 95 % bootstrap CI. The `evasion_prob=0` "
            "cell is the standard Markov-attacker reference. (PLAN §3.1.6; "
            "D7.10.)\n"
        )

    per_algo_pass = {algo: cell.get("passes") for algo, cell in g710.get("per_algo", {}).items()}
    logger.info(
        "F17 written to %s — G7.10 passes=%s per_algo=%s",
        out_dir,
        g710.get("passes"),
        per_algo_pass,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
