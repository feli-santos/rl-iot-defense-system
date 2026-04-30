"""Phase-7 F10 — Attack-aggressiveness sensitivity plot (PLAN §3.1.5).

Reads ``runs/phase7/aggressiveness/{ppo,rule}_p<p>/seed_<k>/eval_test.jsonl``
and renders the two-line sensitivity curve aligned with IoTWarden
Fig. 6 (Bhattacharjee et al., 2023):

  x-axis:  p_defender_deescalation ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  y-axis:  mean episodic reward on test_balanced
  curves:  trained PPO (across 5 seeds) + recommended-action oracle rule

Outputs:
- ``F10_aggressiveness.png``
- ``F10_summary.json`` — per-p {ppo_mean, ppo_ci, rule_mean, rule_ci};
                          + G7.3 evaluation
- ``F10_caption.md`` (placeholder)
- ``F10_manifest.json`` (SHA chain)

Gate evaluation:

- **G7.3** — pass iff PPO mean test reward at p=0.0 < at p=0.6 by
  ≥ 1σ of bootstrap CI (i.e., the high-end CI at p=0.0 is below
  the low-end CI at p=0.6); AND the rule curve is monotone non-
  decreasing (to within CI noise) in p.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.ablation.plot_aggressiveness")

_ROOT = Path(__file__).resolve().parents[2]


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


def _p_slug(p: float) -> str:
    return f"{p:.1f}".replace(".", "p")


def _summarise(
    runs_root: Path,
    kind: str,
    p: float,
    *,
    sha_collector: Dict[str, str],
) -> Dict[str, Any]:
    """Aggregate one (kind, p) cell across all seeds."""
    base = runs_root / f"{kind}_p{_p_slug(p)}"
    seed_dirs = sorted(
        d for d in base.iterdir()
        if base.exists() and d.is_dir() and d.name.startswith("seed_")
        and (d / "eval_test.jsonl").exists()
    ) if base.exists() else []
    all_records: List[Dict] = []
    per_seed_means: List[float] = []
    for sd in seed_dirs:
        jsonl = sd / "eval_test.jsonl"
        recs = read_episodes_jsonl(jsonl)
        all_records.extend(recs)
        if recs:
            per_seed_means.append(
                float(np.mean([r["episode_reward"] for r in recs]))
            )
        sha = _sha256(jsonl)
        if sha is not None:
            sha_collector[str(jsonl.relative_to(_ROOT))] = sha

    if not all_records:
        return {
            "kind": kind, "p": p,
            "n_seeds": len(seed_dirs), "n_episodes": 0,
            "mean_reward": math.nan,
            "ci_low": math.nan, "ci_high": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means, n_resamples=2000, alpha=0.05, seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards, n_resamples=2000, alpha=0.05, seed=0,
        )

    return {
        "kind": kind, "p": p,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _evaluate_g73(
    ppo_rows: List[Dict[str, Any]],
    rule_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """G7.3: PPO p=0.0 reward < p=0.6 reward by ≥ 1σ; rule monotone."""
    by_p_ppo = {r["p"]: r for r in ppo_rows}
    by_p_rule = {r["p"]: r for r in rule_rows}

    p0 = by_p_ppo.get(0.0)
    p06 = by_p_ppo.get(0.6)
    if p0 is None or p06 is None or not (
        math.isfinite(p0["ci_high"]) and math.isfinite(p06["ci_low"])
    ):
        ppo_strict_lt = False
        ppo_reason = "missing or NaN p=0.0 or p=0.6 cell"
    else:
        ppo_strict_lt = p0["ci_high"] < p06["ci_low"]
        ppo_reason = (
            f"p=0.0 CI=({p0['ci_low']:.1f}, {p0['ci_high']:.1f}); "
            f"p=0.6 CI=({p06['ci_low']:.1f}, {p06['ci_high']:.1f})"
        )

    # Rule monotone non-decreasing in p (within CI noise: each
    # successive cell's mean is >= prev cell's mean - prev CI half).
    rule_monotone = True
    sorted_ps = sorted(by_p_rule.keys())
    for i in range(1, len(sorted_ps)):
        prev = by_p_rule[sorted_ps[i - 1]]
        curr = by_p_rule[sorted_ps[i]]
        if not (math.isfinite(prev["mean_reward"]) and math.isfinite(curr["mean_reward"])):
            continue
        prev_band = (prev["ci_high"] - prev["ci_low"]) / 2.0
        if curr["mean_reward"] < prev["mean_reward"] - prev_band:
            rule_monotone = False
            break

    passes = bool(ppo_strict_lt and rule_monotone)
    return {
        "passes": passes,
        "ppo_p0_lt_p06_strict": bool(ppo_strict_lt),
        "ppo_reason": ppo_reason,
        "rule_monotone_non_decreasing": rule_monotone,
        "interpretation": (
            "PASS: PPO benefits from a more lenient defender (p↑ ⇒ reward↑) "
            "by ≥ 1σ between p=0.0 and p=0.6, and the rule curve is "
            "monotone non-decreasing in p — replicates the IoTWarden Fig. 6 "
            "qualitative shape on CICIoT2023."
            if passes else
            "FAIL-WITH-FINDING: see ppo_reason / rule_monotone fields. "
            "The expected qualitative shape (more lenient defender ⇒ "
            "higher RL reward) was NOT replicated; PLAN §6 R7.3 covers "
            "the reframe."
        ),
    }


def _render(
    ppo_rows: List[Dict[str, Any]],
    rule_rows: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ppo_rows = sorted(ppo_rows, key=lambda r: r["p"])
    rule_rows = sorted(rule_rows, key=lambda r: r["p"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    def _plot(rows, color, label):
        xs = [r["p"] for r in rows]
        means = [r["mean_reward"] for r in rows]
        lo = [r["ci_low"] for r in rows]
        hi = [r["ci_high"] for r in rows]
        ax.plot(xs, means, "o-", color=color, label=label, linewidth=1.8)
        ax.fill_between(xs, lo, hi, alpha=0.18, color=color)

    if ppo_rows:
        _plot(ppo_rows, "#2563eb", "PPO (5 seeds, 250K timesteps)")
    if rule_rows:
        _plot(rule_rows, "#dc2626",
              "Recommended-Action oracle rule (1 seed × 150 ep)")

    ax.set_xlabel("p_defender_deescalation (defender's de-escalation success rate)",
                  fontsize=10)
    ax.set_ylabel("Mean episodic reward on test_balanced (95 % bootstrap CI)",
                  fontsize=10)
    ax.set_title(
        "F10 — Sensitivity to attack aggressiveness "
        "(IoTWarden Fig. 6 re-implementation)",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 F10 — attack-aggressiveness plot + summary.",
    )
    p.add_argument("--runs-root", default="runs/phase7/aggressiveness")
    p.add_argument("--out-dir", default="docs/results/07_ablation")
    p.add_argument(
        "--p-values", nargs="+", type=float,
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
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
        logger.error("runs_root not found: %s", runs_root)
        return 1

    sha_collector: Dict[str, str] = {}
    ppo_rows = [
        _summarise(runs_root, "ppo", p, sha_collector=sha_collector)
        for p in args.p_values
    ]
    rule_rows = [
        _summarise(runs_root, "rule", p, sha_collector=sha_collector)
        for p in args.p_values
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "F10_aggressiveness.png"
    _render(ppo_rows, rule_rows, png_path)

    g73 = _evaluate_g73(ppo_rows, rule_rows)
    summary = {
        "schema_version": "1.0",
        "phase": 7,
        "figure": "F10",
        "p_values": list(args.p_values),
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
            "png":  str(png_path),
            "json": str(out_dir / "F10_summary.json"),
        },
        "inputs": {
            "phase7_aggressiveness_sweep_manifest": {
                "path": str(sweep_manifest_path),
                "sha256": _sha256(sweep_manifest_path),
            },
            "eval_jsonls_sha256": sha_collector,
        },
    }
    (out_dir / "F10_manifest.json").write_text(json.dumps(manifest, indent=2))

    caption_path = out_dir / "F10_caption.md"
    if not caption_path.exists():
        caption_path.write_text(
            "**F10 — Sensitivity to attack aggressiveness.** Mean episodic "
            "reward on `test_balanced` for trained PPO (blue, 5 seeds × 250K "
            "timesteps) and the recommended-action oracle rule (red, 1 "
            "seed × 150 episodes) as a function of "
            "`p_defender_deescalation`. Shaded bands: 95 % bootstrap CIs. "
            "Aligned with IoTWarden Fig. 6. (PLAN §3.1.5; D7.2.)\n"
        )

    logger.info(
        "F10 written to %s — G7.3 passes=%s",
        out_dir, g73.get("passes"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
