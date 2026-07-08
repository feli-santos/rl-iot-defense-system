"""ablation F10 — Environment-difficulty sensitivity plot (PLAN §3.1.5).

Reads ``runs/ablation/aggressiveness/{ppo,a2c,dqn,rule}_p<p>/seed_<k>/eval_test.jsonl``
and renders the cross-algorithm sensitivity curve over the live tug-of-war
de-escalation success probability ``p_down`` (conceptually aligned
with IoTWarden Fig. 6, Bhattacharjee et al., 2023):

  x-axis:  p_down ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
           (lower = harsher environment: a correct defender action is
            less likely to push the attacker back down the kill chain)
  y-axis:  mean episodic reward on test_balanced (locked outcome contract)
  curves:  fixed det-5M PPO / A2C / DQN defenders (across seeds, each
           re-evaluated — not retrained — under shifted p_down)
           + recommended-action oracle rule

Outputs:
- ``F10_aggressiveness.png``
- ``F10_summary.json`` — per-algo per-p {mean, ci}; + per-algo G7.3
- ``F10_caption.md`` (placeholder)
- ``F10_manifest.json`` (SHA chain)

Gate evaluation:

- **G7.3** — evaluated per algorithm (ppo/a2c/dqn). For each RL algorithm,
  pass iff its mean test reward at p=0.0 < at p=0.6 by ≥ 1σ of bootstrap CI
  (i.e., the high-end CI at p=0.0 is below the low-end CI at p=0.6); AND the
  rule curve is monotone non-decreasing (to within CI noise) in p. PPO
  retains its original pass semantics as the headline defender; A2C/DQN are
  reported alongside as findings.
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

logger = logging.getLogger("scripts.ablation.plot_aggressiveness")

_ROOT = Path(__file__).resolve().parents[2]

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


def _p_slug(p: float) -> str:
    return f"{p:.1f}".replace(".", "p")


def _summarise(
    runs_root: Path,
    kind: str,
    p: float,
    *,
    sha_collector: dict[str, str],
) -> dict[str, Any]:
    """Aggregate one (kind, p) cell across all seeds."""
    base = runs_root / f"{kind}_p{_p_slug(p)}"
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
            "kind": kind,
            "p": p,
            "n_seeds": len(seed_dirs),
            "n_episodes": 0,
            "mean_reward": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
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
        "kind": kind,
        "p": p,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _rule_monotone_non_decreasing(rule_rows: list[dict[str, Any]]) -> bool:
    """Rule monotone non-decreasing in p (within CI noise: each successive
    cell's mean is >= prev cell's mean - prev CI half)."""
    by_p_rule = {r["p"]: r for r in rule_rows}
    sorted_ps = sorted(by_p_rule.keys())
    for i in range(1, len(sorted_ps)):
        prev = by_p_rule[sorted_ps[i - 1]]
        curr = by_p_rule[sorted_ps[i]]
        if not (math.isfinite(prev["mean_reward"]) and math.isfinite(curr["mean_reward"])):
            continue
        prev_band = (prev["ci_high"] - prev["ci_low"]) / 2.0
        if curr["mean_reward"] < prev["mean_reward"] - prev_band:
            return False
    return True


def _evaluate_g73_algo(
    algo: str,
    algo_rows: list[dict[str, Any]],
    rule_monotone: bool,
) -> dict[str, Any]:
    """G7.3 for one RL algorithm: algo p=0.0 reward < p=0.6 reward by ≥ 1σ;
    rule monotone (shared across algos)."""
    label = algo.upper()
    by_p = {r["p"]: r for r in algo_rows}

    p0 = by_p.get(0.0)
    p06 = by_p.get(0.6)
    if (
        p0 is None
        or p06 is None
        or not (math.isfinite(p0["ci_high"]) and math.isfinite(p06["ci_low"]))
    ):
        strict_lt = False
        reason = "missing or NaN p=0.0 or p=0.6 cell"
    else:
        strict_lt = p0["ci_high"] < p06["ci_low"]
        reason = (
            f"p=0.0 CI=({p0['ci_low']:.1f}, {p0['ci_high']:.1f}); "
            f"p=0.6 CI=({p06['ci_low']:.1f}, {p06['ci_high']:.1f})"
        )

    passes = bool(strict_lt and rule_monotone)
    return {
        "algo": algo,
        "passes": passes,
        "algo_p0_lt_p06_strict": bool(strict_lt),
        "algo_reason": reason,
        "rule_monotone_non_decreasing": rule_monotone,
        "interpretation": (
            f"PASS: {label} benefits from a more lenient environment "
            "(higher p_down ⇒ higher reward) by ≥ 1σ between p_down=0.0 and "
            "p_down=0.6, and the rule curve is monotone non-decreasing in "
            "p_down — the value function shifts with environment difficulty "
            "as expected (conceptually aligned with IoTWarden Fig. 6)."
            if passes
            else f"FAIL-WITH-FINDING ({label}): see algo_reason / rule_monotone "
            "fields. The expected qualitative shape (more lenient environment ⇒ "
            "higher RL reward) was NOT replicated; PLAN §6 R7.3 covers "
            "the reframe."
        ),
    }


def _evaluate_g73(
    algo_rows_by_algo: dict[str, list[dict[str, Any]]],
    rule_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """G7.3 per algorithm. PPO retains the headline pass semantics; A2C/DQN
    reported alongside as findings. Top-level ``passes`` mirrors PPO."""
    rule_monotone = _rule_monotone_non_decreasing(rule_rows)
    per_algo = {
        algo: _evaluate_g73_algo(algo, rows, rule_monotone)
        for algo, rows in algo_rows_by_algo.items()
    }
    ppo_res = per_algo.get("ppo", {})
    return {
        "passes": bool(ppo_res.get("passes", False)),
        "headline_algo": "ppo",
        "rule_monotone_non_decreasing": rule_monotone,
        "per_algo": per_algo,
        # Back-compat mirrors of the PPO cell (close_ablation.py / older readers).
        "ppo_p0_lt_p06_strict": ppo_res.get("algo_p0_lt_p06_strict", False),
        "ppo_reason": ppo_res.get("algo_reason", "missing ppo rows"),
        "interpretation": ppo_res.get("interpretation", "?"),
    }


def _render(
    algo_rows_by_algo: dict[str, list[dict[str, Any]]],
    rule_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    from scripts._plot_style import (
        ACCENT,
        apply_house_style,
        policy_label,
        policy_style,
        save_figure,
    )

    apply_house_style()
    import matplotlib.pyplot as plt

    rule_rows = sorted(rule_rows, key=lambda r: r["p"])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    def _plot(rows, *, color, label, lw, zorder, ls="-", marker="o"):
        xs = [r["p"] for r in rows]
        means = [r["mean_reward"] for r in rows]
        lo = [r["ci_low"] for r in rows]
        hi = [r["ci_high"] for r in rows]
        ax.plot(
            xs,
            means,
            marker=marker,
            color=color,
            label=label,
            linewidth=lw,
            linestyle=ls,
            zorder=zorder,
        )
        ax.fill_between(xs, lo, hi, alpha=0.15, color=color, zorder=zorder - 2)

    # One coloured curve per RL algorithm, using the canonical shared
    # per-policy palette (scripts/_plot_style.py) so PPO/A2C/DQN colours and
    # markers agree with every other thesis figure. Oracle rule follows.
    for algo in _DEFAULT_ALGOS:
        rows = sorted(algo_rows_by_algo.get(algo, []), key=lambda r: r["p"])
        if not rows:
            continue
        style = policy_style(algo)
        n_seeds = max((int(r.get("n_seeds", 0)) for r in rows), default=0)
        base_lbl = policy_label(algo)
        lbl = f"{base_lbl}, {n_seeds} seeds" if n_seeds else base_lbl
        _plot(
            rows,
            color=style.get("color", ACCENT["primary"]),
            label=lbl,
            lw=style.get("lw", 2.0),
            zorder=style.get("zorder", 4),
            ls=style.get("ls", "-"),
            marker=style.get("marker", "o"),
        )

    if rule_rows:
        rule_style = policy_style("rule")
        _plot(
            rule_rows,
            color=rule_style.get("color", ACCENT["neutral"]),
            label="Oracle (recommended-action, full obs.)",
            lw=rule_style.get("lw", 1.6),
            zorder=rule_style.get("zorder", 2),
            ls=rule_style.get("ls", "--"),
            marker=rule_style.get("marker", "x"),
        )

    ax.axhline(0.0, color=ACCENT["muted"], lw=0.8, ls=":")
    ax.set_xlabel(r"De-escalation success $p_{\mathrm{down}}$ (lower = harsher environment)")
    ax.set_ylabel("Mean episodic reward on test_balanced")
    ax.set_title("Defender sensitivity to environment difficulty")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F10 — attack-aggressiveness plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/ablation/aggressiveness")
    p.add_argument("--out-dir", default="docs/results/ablation")
    p.add_argument(
        "--algos",
        nargs="+",
        choices=("ppo", "a2c", "dqn"),
        default=list(_DEFAULT_ALGOS),
        help="RL algorithms to overlay (default: ppo a2c dqn).",
    )
    p.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    # Step-8 F2 (07_HANDOFF.md §5): explicit upstream-manifest SHA pins.
    p.add_argument(
        "--blue-team-sweep-manifest",
        default="runs/redesign_5M_det/alpha_04/sweep_manifest.json",
        help="Blue-Team sweep_manifest.json (warm-start trained checkpoints).",
    )
    p.add_argument(
        "--benchmark-eval-manifest",
        default="runs/benchmark/eval_manifest.json",
        help="benchmark eval_manifest.json (oracle-rule reference rolls).",
    )
    p.add_argument(
        "--split-splits-manifest",
        default="docs/results/dataset/manifest.json",
        help="dataset-prep splits manifest.json (post-3cd2fb9; SHA 1e99d596...).",
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
        algo: [_summarise(runs_root, algo, p, sha_collector=sha_collector) for p in args.p_values]
        for algo in args.algos
    }
    rule_rows = [
        _summarise(runs_root, "rule", p, sha_collector=sha_collector) for p in args.p_values
    ]
    # Back-compat: keep the standalone ``ppo_rows`` key that render_tables.py
    # and older readers consume.
    ppo_rows = algo_rows_by_algo.get("ppo", [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_base = out_dir / "F10_aggressiveness"
    _render(algo_rows_by_algo, rule_rows, fig_base)
    png_path = fig_base.with_suffix(".png")
    pdf_path = fig_base.with_suffix(".pdf")

    g73 = _evaluate_g73(algo_rows_by_algo, rule_rows)
    summary = {
        "schema_version": "1.0",
        "stage": "ablation",
        "figure": "F10",
        "algos": list(args.algos),
        "p_values": list(args.p_values),
        "algo_rows": algo_rows_by_algo,
        "ppo_rows": ppo_rows,
        "rule_rows": rule_rows,
        "gates": {"G7.3": g73},
        "headline": g73.get("interpretation", "?"),
    }
    (out_dir / "F10_summary.json").write_text(json.dumps(summary, indent=2))

    sweep_manifest_path = runs_root / "sweep_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F10",
        "git_sha": _git_sha(),
        "outputs": {
            "pdf": str(pdf_path),
            "pdf_sha256": _sha256(pdf_path),
            "png": str(png_path),
            "json": str(out_dir / "F10_summary.json"),
        },
        "inputs": {
            "ablation_aggressiveness_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            # Step-8 F2: explicit upstream-manifest SHA pins so the
            # F10 hash chain is self-contained (no transitive lookups).
            "blue_team_sweep_manifest": {
                "path": str(args.blue_team_sweep_manifest),
                "sha256": _sha256(Path(args.blue_team_sweep_manifest)),
            },
            "benchmark_eval_manifest": {
                "path": str(args.benchmark_eval_manifest),
                "sha256": _sha256(Path(args.benchmark_eval_manifest)),
            },
            "split_splits_manifest": {
                "path": str(args.split_splits_manifest),
                "sha256": _sha256(Path(args.split_splits_manifest)),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F10_manifest.json").write_text(json.dumps(manifest, indent=2))

    n_seeds_ppo = max((int(r.get("n_seeds", 0)) for r in ppo_rows), default=0)
    algos_present = [a for a in args.algos if algo_rows_by_algo.get(a)]
    algo_names = ", ".join(a.upper() for a in algos_present) or "PPO"
    caption_path = out_dir / "F10_caption.md"
    caption_path.write_text(
        "**F10 — Sensitivity to environment difficulty.** Mean episodic "
        "reward on `test_balanced` under the locked outcome reward contract "
        f"for the fixed deterministic-5M α=0.4 defenders ({algo_names}; "
        f"{n_seeds_ppo} seeds each), re-evaluated (not retrained) under each "
        "shifted environment, and the recommended-action oracle (grey, full "
        "observability) as a function of the tug-of-war de-escalation success "
        "probability `p_down` (lower = harsher environment, where a correct "
        "defender action is less likely to push the attacker back down the "
        "kill chain). Shaded bands: 95 % bootstrap CIs. (PLAN §3.1.5; D7.2.)\n"
    )

    logger.info(
        "F10 written to %s — G7.3 passes(ppo)=%s; per-algo=%s",
        out_dir,
        g73.get("passes"),
        {a: g73.get("per_algo", {}).get(a, {}).get("passes") for a in args.algos},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
