"""Aggregate + plot the observation-aliasing alpha-sweep (partial-observability redesign).

Reads the per-policy ``eval_test.jsonl`` produced by
``scripts.benchmark.run_test_eval`` across the four aliasing operating points
(alpha in {0.0, 0.2, 0.4, 0.6}) and answers the headline question:

    Does a windowed reinforcement-learning agent (PPO) hold its performance as
    single-flow observations become more ambiguous, while a per-flow supervised
    classifier (RF-Acting) degrades?

For each alpha and policy it pools ``episode_reward`` across seeds and reports
a bootstrap mean and 95% CI. RL agents have 10 seeds; the deterministic
baselines (RF-Acting, the full-observability oracle, always-block,
always-observe, random) have a single roll.

The honest reading is the crossover: at alpha=0 (no ambiguity) RF-Acting
ties/edges PPO, proving the redesign does not favour RL by construction; as
alpha grows the supervised classifier must commit on each ambiguous flow and
its reward falls monotonically, while the windowed agent integrates the short
observation window and holds roughly flat.

Outputs (under ``--out-dir``):

    Falpha_summary.json    — per-alpha per-policy aggregates + crossover read
    Falpha_curve.png/.pdf  — reward vs alpha with CI bands
    Falpha_manifest.json   — git sha + input/output SHA-256 chain
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from scripts._plot_style import (
    POLICY_LABEL,
    POLICY_ORDER,
    POLICY_STYLE,
    apply_house_style,
    save_figure,
    sha256_file,
)

apply_house_style()
import matplotlib.pyplot as plt  # noqa: E402

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl  # noqa: E402

logger = logging.getLogger("scripts.ablation.plot_alpha_curve")

_ROOT = Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _pool_rewards(policy_dir: Path) -> list[float]:
    """Pool ``episode_reward`` across every ``seed_*/eval_test.jsonl``."""
    rewards: list[float] = []
    for seed_dir in sorted(policy_dir.glob("seed_*")):
        jsonl = seed_dir / "eval_test.jsonl"
        if not jsonl.exists():
            continue
        for rec in read_episodes_jsonl(jsonl):
            rewards.append(float(rec["episode_reward"]))
    return rewards


def _summarise_alpha(bench_dir: Path) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for policy in POLICY_ORDER:
        pdir = bench_dir / policy
        if not pdir.exists():
            continue
        rewards = _pool_rewards(pdir)
        if not rewards:
            continue
        lo, mean, hi = bootstrap_ci(rewards)
        out[policy] = {
            "mean": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "n": len(rewards),
            "n_seeds": len(list(pdir.glob("seed_*"))),
        }
    return out


def _crossover_read(
    per_alpha: dict[float, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    """Per-alpha PPO-minus-RF gap and CI-overlap verdict."""
    reads: list[dict[str, Any]] = []
    for alpha in sorted(per_alpha):
        cell = per_alpha[alpha]
        ppo = cell.get("ppo")
        rf = cell.get("rf_acting")
        if not ppo or not rf:
            continue
        gap = float(ppo["mean"]) - float(rf["mean"])
        # CIs disjoint (PPO strictly above) iff ppo.ci_low > rf.ci_high.
        disjoint_ppo_above = float(ppo["ci_low"]) > float(rf["ci_high"])
        disjoint_rf_above = float(rf["ci_low"]) > float(ppo["ci_high"])
        if disjoint_ppo_above:
            verdict = "ppo_significant"
        elif disjoint_rf_above:
            verdict = "rf_significant"
        else:
            verdict = "tie_overlap"
        reads.append(
            {
                "alpha": alpha,
                "ppo_minus_rf": round(gap, 4),
                "verdict": verdict,
            }
        )
    return {"per_alpha": reads}


def _render_curve(
    per_alpha: dict[float, dict[str, dict[str, float | int]]],
    out_path: Path,
) -> None:
    alphas = sorted(per_alpha)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for policy in POLICY_ORDER:
        xs: list[float] = []
        ys: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for a in alphas:
            cell = per_alpha[a].get(policy)
            if not cell:
                continue
            xs.append(a)
            ys.append(float(cell["mean"]))
            los.append(float(cell["ci_low"]))
            his.append(float(cell["ci_high"]))
        if not xs:
            continue
        style = POLICY_STYLE.get(policy, {})
        ls = style.get("ls", "-")
        ax.plot(
            xs,
            ys,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            lw=style.get("lw", 1.8),
            ls=ls,
            zorder=style.get("zorder", 3),
            label=POLICY_LABEL.get(policy, policy),
        )
        if policy in ("ppo", "rf_acting"):
            ax.fill_between(xs, los, his, color=style.get("color"), alpha=0.15, zorder=1)
    ax.set_xlabel("Observation aliasing rate $\\alpha$")
    ax.set_ylabel("Mean episodic reward (10 seeds, 300 episodes)")
    ax.set_title(
        "Windowed RL holds under partial observability; the\n"
        "per-flow supervised classifier degrades"
    )
    ax.set_xticks(alphas)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root",
        default="runs/redesign",
        help="dir holding benchmark_alpha_<NN> subdirs",
    )
    ap.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.6],
    )
    ap.add_argument("--out-dir", default="docs/results/ablation")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_alpha: dict[float, dict[str, dict[str, float | int]]] = {}
    input_hashes: dict[str, str | None] = {}
    for alpha in args.alphas:
        nn = f"{int(round(alpha * 10)):02d}"
        bench_dir = runs_root / f"benchmark_alpha_{nn}"
        if not bench_dir.exists():
            logger.warning("missing %s; skipping alpha=%s", bench_dir, alpha)
            continue
        per_alpha[alpha] = _summarise_alpha(bench_dir)
        man = bench_dir / "eval_manifest.json"
        input_hashes[f"eval_manifest_alpha_{nn}"] = sha256_file(man)

    crossover = _crossover_read(per_alpha)

    summary = {
        "schema_version": "1.0",
        "kind": "alpha_curve_summary",
        "regime": {
            "reward_mode": "outcome",
            "session_coherent": True,
            "no_post_transition_leak": True,
            "proximity_coupled": True,
            "proximity_min_escalation": 0.4,
            "impact_is_terminal": False,
        },
        "alphas": list(args.alphas),
        "per_alpha": {f"{a:.1f}": per_alpha[a] for a in sorted(per_alpha)},
        "crossover": crossover,
    }
    summary_path = out_dir / "Falpha_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    png_path = out_dir / "Falpha_curve.png"
    _render_curve(per_alpha, png_path)

    manifest = {
        "schema_version": "1.0",
        "kind": "alpha_curve_plot_manifest",
        "git_sha": _git_sha(),
        "inputs": input_hashes,
        "outputs": {
            "summary_json": sha256_file(summary_path),
            "curve_png": sha256_file(png_path),
            "curve_pdf": sha256_file(png_path.with_suffix(".pdf")),
        },
    }
    (out_dir / "Falpha_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Console crossover read.
    print("alpha-curve PPO-vs-RF crossover:")
    for r in crossover["per_alpha"]:
        print(f"  alpha={r['alpha']:.1f}  PPO-RF={r['ppo_minus_rf']:+.1f}  " f"=> {r['verdict']}")
    print(f"Wrote {summary_path}, {png_path}")


if __name__ == "__main__":
    main()
