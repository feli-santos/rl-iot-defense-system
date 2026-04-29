"""Phase-5 exit-gate evaluator.

Reads ``runs/phase5/`` + ``F3_summary.json`` + ``F4_summary.json`` and
emits a per-gate scoreboard ``docs/results/05_blue_team/G5_scoreboard.json``
with PASS/FAIL/PASS-WITH-FINDING + headline numbers.

This is the final step before RESULTS.md / CHANGELOG.

Usage::

    python -m scripts.blue_team.evaluate_gates \\
        --runs-root runs/phase5 \\
        --out-dir docs/results/05_blue_team
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.blue_team.aggregation import (  # noqa: E402
    read_runs_directory,
    summarise_last_window,
)

logger = logging.getLogger("scripts.blue_team.evaluate_gates")


# Default lifecycle constants from PLAN D5.3.
_MIN_EPISODE_LENGTH = 20


def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def _evaluate_g5_2_g5_3_g5_4(
    eval_runs: Dict[tuple, list],
    fraction: float,
) -> Dict[str, Any]:
    """Aggregate per-algo eval-summary across seeds for G5.2, G5.3, G5.4."""
    by_algo: Dict[str, list] = {}
    for (algo, seed), recs in eval_runs.items():
        by_algo.setdefault(algo, []).append((seed, recs))

    out: Dict[str, Any] = {}
    for algo, items in by_algo.items():
        per_seed = []
        for seed, recs in items:
            s = summarise_last_window(recs, fraction=fraction)
            s["seed"] = seed
            per_seed.append(s)
        # Mean across seeds (NaN-safe).
        def _mean(key: str) -> float:
            vals = [s[key] for s in per_seed if not _isnan(s.get(key))]
            return float(sum(vals) / len(vals)) if vals else float("nan")
        out[algo] = {
            "mean_reward": _mean("mean_reward"),
            "mean_mttc": _mean("mean_mttc"),
            "compromise_rate": _mean("compromise_rate"),
            "mitigated_impact_rate": _mean("mitigated_impact_rate"),
            "mitigated_among_compromised": _mean("mitigated_among_compromised"),
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
        }
    return out


def _select_best_algo(per_algo: Dict[str, Any]) -> str:
    """Pick the algo with the highest mean reward; tie-break by lowest std."""
    if not per_algo:
        raise RuntimeError("no algos evaluated")
    ranked = sorted(
        per_algo.items(),
        key=lambda kv: (-kv[1]["mean_reward"], -kv[1].get("mean_mttc", 0.0)),
    )
    return ranked[0][0]


def evaluate(runs_root: Path, out_dir: Path, fraction: float = 0.10) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_runs = read_runs_directory(runs_root, file_name="episodes.jsonl")
    eval_runs = read_runs_directory(runs_root, file_name="eval.jsonl")
    if not eval_runs:
        raise RuntimeError(f"no eval runs found under {runs_root}")

    per_algo = _evaluate_g5_2_g5_3_g5_4(eval_runs, fraction=fraction)
    best_algo = _select_best_algo(per_algo)

    # G5.2 — at least one algo's eval reward > 0
    g5_2_passes = any(v["mean_reward"] > 0 for v in per_algo.values())
    g5_2_observed = max(
        (v["mean_reward"] for v in per_algo.values() if not _isnan(v["mean_reward"])),
        default=float("nan"),
    )

    # G5.3 — best algo's mean MTTC >= min_episode_length - 1 (see D5.4.1).
    g5_3_observed = per_algo[best_algo]["mean_mttc"]
    g5_3_passes = (
        not _isnan(g5_3_observed)
        and g5_3_observed >= (_MIN_EPISODE_LENGTH - 1)
    )

    # G5.4 — best algo's mitigated-impact rate >= 0.5 (D5.4.1).
    g5_4_observed = per_algo[best_algo]["mitigated_impact_rate"]
    g5_4_passes = (
        not _isnan(g5_4_observed) and g5_4_observed >= 0.5
    )

    # G5.5 — pulled from F4_summary.json (per-stage non-degeneracy).
    f4_summary_path = out_dir / "F4_summary.json"
    if f4_summary_path.exists():
        f4 = json.loads(f4_summary_path.read_text())
        g5_5_passes = bool(f4.get("g5_5_passes", False))
        g5_5_per_stage = f4.get("g5_5_per_stage", {})
    else:
        logger.warning("F4_summary.json not found; G5.5 marked as UNKNOWN")
        g5_5_passes = None
        g5_5_per_stage = None

    scoreboard = {
        "version": "1.0",
        "runs_root": str(runs_root),
        "fraction": fraction,
        "best_algo": best_algo,
        "per_algo_summary": per_algo,
        "gates": {
            "G5.1": {
                "description": "full pytest suite green",
                "evaluated": False,
                "note": "evaluated separately by `pytest -q`",
            },
            "G5.2": {
                "description": "at least one algo eval reward > 0 over last 10% × 5 seeds",
                "threshold": 0.0,
                "observed": g5_2_observed,
                "passes": bool(g5_2_passes),
            },
            "G5.3": {
                "description": (
                    f"best algo mean MTTC >= {_MIN_EPISODE_LENGTH - 1} "
                    "(D5.4.1)"
                ),
                "threshold": _MIN_EPISODE_LENGTH - 1,
                "observed": g5_3_observed,
                "best_algo": best_algo,
                "passes": bool(g5_3_passes),
            },
            "G5.4": {
                "description": "best algo mitigated-impact rate >= 0.5 (D5.4.1)",
                "threshold": 0.5,
                "observed": g5_4_observed,
                "best_algo": best_algo,
                "passes": bool(g5_4_passes),
            },
            "G5.5": {
                "description": "no per-stage action share > 70% at late checkpoint",
                "threshold": 0.70,
                "passes": g5_5_passes,
                "per_stage": g5_5_per_stage,
            },
            "G5.6": {
                "description": "no regression on Phase-3 frozen tests",
                "evaluated": False,
                "note": "evaluated separately by `pytest -q tests/test_phase3_env_gates.py tests/test_adversarial_env.py`",
            },
            "G5.7": {
                "description": "F3/F4/T1 manifests exist & hash-pin inputs",
                "passes": all([
                    (out_dir / "F3_manifest.json").exists(),
                    (out_dir / "F4_manifest.json").exists(),
                    (out_dir / "T1_hparams.json").exists(),
                ]),
            },
        },
    }
    score_path = out_dir / "G5_scoreboard.json"
    score_path.write_text(json.dumps(scoreboard, indent=2))
    logger.info("wrote %s", score_path)

    # Pretty print to stdout.
    print("=== Phase-5 gate scoreboard ===")
    for gid, g in scoreboard["gates"].items():
        passes = g.get("passes")
        if passes is True:
            mark = "PASS"
        elif passes is False:
            mark = "FAIL"
        else:
            mark = "----"
        observed = g.get("observed")
        obs_str = (
            f"  observed={observed:.3f}" if isinstance(observed, (int, float))
            and not _isnan(observed) else ""
        )
        print(f"  {gid:5} [{mark}] {g['description']}{obs_str}")
    print(f"\nbest algo (by eval reward): {best_algo}")
    print("\nper-algo eval summary (last %.0f%%):" % (fraction * 100))
    for algo, v in per_algo.items():
        print(
            f"  {algo:4} reward={v['mean_reward']:+8.1f}  "
            f"MTTC={v['mean_mttc']:.2f}  "
            f"comp%={v['compromise_rate']:.2f}  "
            f"mit_imp={v['mitigated_impact_rate']:.2f}  "
            f"mit|comp={v['mitigated_among_compromised']:.2f}"
        )
    return scoreboard


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase-5 gate evaluator.")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out-dir", default="docs/results/05_blue_team")
    p.add_argument("--fraction", type=float, default=0.10)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    evaluate(Path(args.runs_root), Path(args.out_dir), fraction=args.fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
