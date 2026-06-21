"""Aggregate + plot the coupled-vs-decoupled reward ablation (WS2).

Reads the per-cell ``eval_test.jsonl`` produced by
``scripts.ablation.run_reward_coupling`` and answers one question directly:

    Does the strongest deployable supervised baseline (RF-Acting) beat the
    best RL agent ONLY because the coupled reward shapes every step by the
    hidden stage's recommended action?

For each reward mode it computes:

* ``best_rl_reward``  — the best of the per-algorithm mean test rewards,
  each algorithm pooled across seeds, with a bootstrap CI.
* ``rf_acting_reward``— RF-Acting's mean test reward (single deterministic
  roll; frozen classifier).
* ``rf_minus_rl_gap`` — ``rf_acting_reward - best_rl_reward``. A large
  POSITIVE gap means the supervised baseline dominates RL.

The headline read is the change in that gap between ``coupled`` and
``outcome``: a positive coupled gap that shrinks (or reverses) under the
outcome contract is the evidence that RF-Acting's dominance is an artefact
of the privileged-reward design, not of RL being the wrong tool.

Outputs (under ``--out-dir``):

    Fcoupling_summary.json     — per-mode aggregates + the gap deltas
    Fcoupling_reward_gap.png   — grouped bar: best-RL vs RF-Acting per mode
    Fcoupling_manifest.json    — git sha + input/output SHA-256 chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_reward_coupling")

_ROOT = Path(__file__).resolve().parents[2]
_RL_ALGOS: tuple[str, ...] = ("dqn", "ppo", "a2c")


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
    try:  # noqa: SIM105
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _episode_rewards(jsonl_path: Path) -> list[float]:
    """Return per-episode rewards from a cell's eval_test.jsonl (or [])."""
    if not Path(jsonl_path).exists():
        return []
    try:
        records = read_episodes_jsonl(jsonl_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", jsonl_path, exc)
        return []
    return [float(r["episode_reward"]) for r in records if "episode_reward" in r]


def summarise_mode(
    out_root: Path, mode: str, *, algos: tuple[str, ...] = _RL_ALGOS, seeds: list[int]
) -> dict[str, Any]:
    """Aggregate one reward mode into best-RL vs RF-Acting reward + gap.

    ``best_rl_reward`` is the maximum over algorithms of each algorithm's
    mean test reward (pooled across seeds). The bootstrap CI for the winning
    algorithm is reported. RF-Acting is a single deterministic roll.
    """
    mode_root = out_root / mode

    # Per-algorithm pooled rewards across seeds.
    per_algo: dict[str, dict[str, Any]] = {}
    for algo in algos:
        pooled: list[float] = []
        for seed in seeds:
            pooled.extend(
                _episode_rewards(mode_root / algo / f"seed_{seed}" / "eval_test.jsonl")
            )
        if pooled:
            lo, mean, hi = bootstrap_ci(pooled)
            per_algo[algo] = {
                "mean_reward": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_episodes": len(pooled),
            }
        else:
            per_algo[algo] = {
                "mean_reward": None,
                "ci_low": None,
                "ci_high": None,
                "n_episodes": 0,
            }

    # Best RL = algorithm with the highest mean reward (ignoring empty cells).
    ranked = [
        (a, s["mean_reward"]) for a, s in per_algo.items() if s["mean_reward"] is not None
    ]
    best_algo = max(ranked, key=lambda t: t[1])[0] if ranked else None
    best_rl_reward = per_algo[best_algo]["mean_reward"] if best_algo else None

    # RF-Acting (frozen classifier, single roll).
    rf_rewards = _episode_rewards(mode_root / "rf_acting" / "eval_test.jsonl")
    if rf_rewards:
        rf_lo, rf_mean, rf_hi = bootstrap_ci(rf_rewards)
    else:
        rf_mean = rf_lo = rf_hi = None

    gap = (
        rf_mean - best_rl_reward
        if (rf_mean is not None and best_rl_reward is not None)
        else None
    )

    return {
        "mode": mode,
        "per_algo": per_algo,
        "best_algo": best_algo,
        "best_rl_reward": best_rl_reward,
        "rf_acting_reward": rf_mean,
        "rf_acting_ci_low": rf_lo,
        "rf_acting_ci_high": rf_hi,
        "rf_minus_rl_gap": gap,
    }


def _render_gap_figure(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = summary["modes"]
    best_rl = [summary["per_mode"][m]["best_rl_reward"] for m in modes]
    rf = [summary["per_mode"][m]["rf_acting_reward"] for m in modes]

    x = np.arange(len(modes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    b1 = ax.bar(
        x - width / 2,
        [v if v is not None else 0.0 for v in best_rl],
        width,
        label="Best RL agent",
        color="#2c7fb8",
    )
    b2 = ax.bar(
        x + width / 2,
        [v if v is not None else 0.0 for v in rf],
        width,
        label="RF-Acting (supervised)",
        color="#d95f0e",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Coupled\n(stage-shaped reward)", "Outcome\n(decoupled reward)"][: len(modes)]
        if set(modes) <= {"coupled", "outcome"}
        else modes
    )
    ax.set_ylabel("Mean test reward")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Reward design controls RF-Acting's apparent advantage")
    ax.legend(loc="best", fontsize=9)

    for m, xi in zip(modes, x):
        gap = summary["per_mode"][m]["rf_minus_rl_gap"]
        if gap is None:
            continue
        top = max(
            summary["per_mode"][m]["best_rl_reward"] or 0.0,
            summary["per_mode"][m]["rf_acting_reward"] or 0.0,
        )
        ax.annotate(
            f"RF−RL gap\n{gap:+.1f}",
            xy=(xi, top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate + plot the coupled-vs-decoupled reward ablation."
    )
    p.add_argument("--out-root", default="runs/ablation/reward_coupling")
    p.add_argument("--modes", nargs="+", default=["coupled", "outcome"])
    p.add_argument("--algos", nargs="+", default=list(_RL_ALGOS))
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    p.add_argument("--out-dir", default="docs/results/ablation")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_mode = {
        mode: summarise_mode(
            out_root, mode, algos=tuple(args.algos), seeds=list(args.seeds)
        )
        for mode in args.modes
    }
    summary: dict[str, Any] = {
        "kind": "reward_coupling_summary",
        "modes": list(args.modes),
        "algos": list(args.algos),
        "per_mode": per_mode,
    }
    # The headline delta: how much the RF-minus-RL gap changes from coupled
    # to outcome. A large positive coupled gap that collapses under outcome
    # is the evidence the supervised dominance is a reward-design artefact.
    if {"coupled", "outcome"} <= set(args.modes):
        g_coupled = per_mode["coupled"]["rf_minus_rl_gap"]
        g_outcome = per_mode["outcome"]["rf_minus_rl_gap"]
        if g_coupled is not None and g_outcome is not None:
            summary["gap_coupled"] = g_coupled
            summary["gap_outcome"] = g_outcome
            summary["gap_reduction"] = g_coupled - g_outcome

    summary_path = out_dir / "Fcoupling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    fig_path = out_dir / "Fcoupling_reward_gap.png"
    try:
        _render_gap_figure(summary, fig_path)
        fig_ok = fig_path.exists()
    except Exception as exc:  # noqa: BLE001
        logger.warning("figure render skipped: %s", exc)
        fig_ok = False

    manifest = {
        "schema_version": "1.0",
        "kind": "reward_coupling_plot_manifest",
        "git_sha": _git_sha(),
        "inputs": {
            "sweep_manifest": _sha256(out_root / "sweep_manifest.json"),
        },
        "outputs": {
            "summary_json": _sha256(summary_path),
            "reward_gap_png": _sha256(fig_path) if fig_ok else None,
        },
    }
    (out_dir / "Fcoupling_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "coupling summary -> %s (gap_reduction=%s)",
        summary_path,
        summary.get("gap_reduction"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
