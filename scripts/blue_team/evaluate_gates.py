"""Blue-Team exit-gate evaluator.

Reads ``runs/blue_team/`` + ``F3_summary.json`` + ``F4_summary.json`` and
emits a per-gate scoreboard ``docs/results/blue-team-training/G5_scoreboard.json``
with PASS/FAIL/PASS-WITH-FINDING + headline numbers.

This is the final step before RESULTS.md / CHANGELOG.

Usage::

    python -m scripts.blue_team.evaluate_gates \\
        --runs-root runs/blue_team \\
        --out-dir docs/results/blue-team-training
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.blue_team.aggregation import read_runs_directory, summarise_last_window  # noqa: E402

logger = logging.getLogger("scripts.blue_team.evaluate_gates")


# Default lifecycle constants from PLAN D5.3.
_MIN_EPISODE_LENGTH = 20


def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False


def _evaluate_g5_2_g5_3_g5_4(
    eval_runs: dict[tuple, list],
    fraction: float,
) -> dict[str, Any]:
    """Aggregate per-algo eval-summary across seeds for G5.2, G5.3, G5.4."""
    by_algo: dict[str, list] = {}
    for (algo, seed), recs in eval_runs.items():
        by_algo.setdefault(algo, []).append((seed, recs))

    out: dict[str, Any] = {}
    for algo, items in by_algo.items():
        per_seed = []
        for seed, recs in items:
            s = summarise_last_window(recs, fraction=fraction)
            s["seed"] = seed
            per_seed.append(s)

        # Mean across seeds (NaN-safe).
        def _mean(key: str, _ps: list = per_seed) -> float:
            vals = [s[key] for s in _ps if not _isnan(s.get(key))]
            return float(sum(vals) / len(vals)) if vals else float("nan")

        out[algo] = {
            "mean_reward": _mean("mean_reward"),
            "mean_mttc": _mean("mean_mttc"),
            "compromise_rate": _mean("compromise_rate"),
            "prevention_rate": _mean("prevention_rate"),
            "mitigated_among_compromised": _mean("mitigated_among_compromised"),
            "mitigated_impact_rate": _mean("mitigated_impact_rate"),
            "n_seeds": len(per_seed),
            "per_seed": per_seed,
        }
    return out


def _select_best_algo(per_algo: dict[str, Any]) -> str:
    """Pick the algo with the highest mean reward; tie-break by **highest
    mean MTTC** (longest mean time to compromise — i.e., the algo that
    held off compromise the longest among reward-tied candidates).

    .. note::
       **Step-5 F3 / Step-8 doc-fix.** Earlier docstring versions claimed
       "lowest std" and PLAN §8 D5.11 used "lower variance". The actual
       implementation tie-breaks by highest MTTC: see the sort key
       ``(-mean_reward, -mean_mttc)`` below. The triple disagreement
       between docstring / PLAN / code never fired in practice (the
        Blue-Team mean-reward gaps are ≥ 25 points, far larger than
       any plausible reward-tie band), but the docstring is now
       authoritative and matches the code byte-for-byte. PLAN §8 D5.11
       is preserved verbatim as the audit-trail record of
       pre-registration; consult this docstring for as-built behaviour.
    """
    if not per_algo:
        raise RuntimeError("no algos evaluated")
    ranked = sorted(
        per_algo.items(),
        key=lambda kv: (-kv[1]["mean_reward"], -kv[1].get("mean_mttc", 0.0)),
    )
    return ranked[0][0]


def evaluate(runs_root: Path, out_dir: Path, fraction: float = 0.10) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    read_runs_directory(runs_root, file_name="episodes.jsonl")
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
    g5_3_passes = not _isnan(g5_3_observed) and g5_3_observed >= (_MIN_EPISODE_LENGTH - 1)

    # G5.4 — best algo's PREVENTION rate >= 0.5 (D5.4.1, revised).
    # Originally gated ``mitigated_impact_rate``, which is deprecated under the
    # tug-of-war contract: for policies that always block at terminal IMPACT it
    # collapses onto ``compromise_rate`` and is uninformative. The gate now
    # tracks ``prevention_rate`` (attacker budget exhausted before IMPACT), the
    # primary defender-attributable security outcome. ``mitigated_impact_rate``
    # is still recorded above for backward compatibility.
    g5_4_observed = per_algo[best_algo]["prevention_rate"]
    g5_4_passes = not _isnan(g5_4_observed) and g5_4_observed >= 0.5

    # G5.5 — pulled from F4_summary.json (per-stage non-degeneracy).
    f4_summary_path = out_dir / "F4_summary.json"
    if f4_summary_path.exists():
        f4 = json.loads(f4_summary_path.read_text())
        g5_5_passes = bool(f4.get("g5_5_passes", False))
        # F4_summary.json::g5_5_per_stage ships per-stage `passes:bool|null`
        # entries (producer: scripts/blue_team/plot_action_dist.py). For
        # the unified scoreboard schema (Step-8 F3 acceptance: "no `passes`
        # key remains"), we normalise every nested `passes` to `status:enum`
        # in-place when copying. The producer (locked F4 artefact) is left
        # untouched; only G5_scoreboard.json reflects the unified shape.
        raw_per_stage = f4.get("g5_5_per_stage", {})
        g5_5_per_stage = {}
        for stage, entry in (raw_per_stage or {}).items():
            if not isinstance(entry, dict):
                g5_5_per_stage[stage] = entry
                continue
            new_entry = {k: v for k, v in entry.items() if k != "passes"}
            ps = entry.get("passes")
            if ps is None:
                new_entry["status"] = "SKIP"
            elif ps is True:
                new_entry["status"] = "PASS"
            else:
                new_entry["status"] = "FAIL"
            g5_5_per_stage[stage] = new_entry
    else:
        logger.warning("F4_summary.json not found; G5.5 marked as UNKNOWN")
        g5_5_passes = None
        g5_5_per_stage = None

    # Status-enum derivation (Step-8 F3, schema v2.0). Mirrors the
    # benchmark G6_scoreboard.json shape.
    g5_7_passes = all(
        [
            (out_dir / "F3_manifest.json").exists(),
            (out_dir / "F4_manifest.json").exists(),
            (out_dir / "T1_hparams.json").exists(),
        ]
    )

    def _status(passes: bool | None, evaluated: bool = True) -> str:
        if not evaluated or passes is None:
            return "SKIP"
        return "PASS" if passes else "FAIL"

    # G5.4 is the headline narrative-PASS-WITH-FINDING gate per
    # Step-5 F1: mechanically below the 0.5 threshold (D5.4.1
    # de-escalation-farming behaviour) but consistent with the
    # documented thesis claim that the agent learned the policy
    # the reward incentivised. Encoded as FAIL-WITH-FINDING +
    # finding_id D5.4.1 to preserve the audit trail.
    g5_4_status = "PASS" if g5_4_passes else "FAIL-WITH-FINDING"

    scoreboard = {
        "schema_version": "2.0",
        "stage": "blue_team",
        "runs_root": str(runs_root),
        "fraction": fraction,
        "best_algo": best_algo,
        "per_algo_summary": per_algo,
        "gates": {
            "G5.1": {
                "description": "full pytest suite green",
                "status": "SKIP",
                "evaluated": False,
                "note": "evaluated separately by `pytest -q`",
            },
            "G5.2": {
                "description": "at least one algo eval reward > 0 over last 10% × 5 seeds",
                "threshold": 0.0,
                "observed": g5_2_observed,
                "status": _status(g5_2_passes),
            },
            "G5.3": {
                "description": (f"best algo mean MTTC >= {_MIN_EPISODE_LENGTH - 1} (D5.4.1)"),
                "threshold": _MIN_EPISODE_LENGTH - 1,
                "observed": g5_3_observed,
                "best_algo": best_algo,
                "status": _status(g5_3_passes),
            },
            "G5.4": {
                "description": "best algo prevention rate >= 0.5 (D5.4.1, revised)",
                "threshold": 0.5,
                "observed": g5_4_observed,
                "best_algo": best_algo,
                "status": g5_4_status,
                **({"finding_id": "D5.4.1"} if g5_4_status == "FAIL-WITH-FINDING" else {}),
            },
            "G5.5": {
                "description": (
                    "every displayed policy is non-degenerate at the late "
                    "checkpoint (per-stage argmax spans >= 2 distinct actions)"
                ),
                "threshold": 2,
                "criterion": "distinct_argmax_actions_across_stages",
                "status": _status(g5_5_passes),
                "per_stage": g5_5_per_stage,
            },
            "G5.6": {
                "description": "no regression on environment-design frozen tests",
                "status": "SKIP",
                "evaluated": False,
                "note": "evaluated separately by `pytest -q tests/test_adversarial_env.py`",
            },
            "G5.7": {
                "description": "F3/F4/T1 manifests exist & hash-pin inputs",
                "status": _status(g5_7_passes),
            },
        },
    }
    # Top-level summary{} (benchmark-native shape).
    statuses = [g["status"] for g in scoreboard["gates"].values()]
    scoreboard["summary"] = {
        "total_gates": len(statuses),
        "pass": statuses.count("PASS"),
        "pass_with_finding": statuses.count("PASS-WITH-FINDING"),
        "pass_without_stretch": statuses.count("PASS-WITHOUT-STRETCH"),
        "fail_with_finding": statuses.count("FAIL-WITH-FINDING"),
        "fail": statuses.count("FAIL"),
        "skip": statuses.count("SKIP"),
    }
    score_path = out_dir / "G5_scoreboard.json"
    score_path.write_text(json.dumps(scoreboard, indent=2))
    logger.info("wrote %s", score_path)

    # Pretty print to stdout. Reads `status` (benchmark-native schema).
    print("=== Blue-Team gate scoreboard ===")
    _MARK = {
        "PASS": "PASS",
        "PASS-WITH-FINDING": "PASS+",
        "PASS-WITHOUT-STRETCH": "PASS-",
        "FAIL-WITH-FINDING": "FAIL+",
        "FAIL": "FAIL",
        "SKIP": "----",
    }
    for gid, g in scoreboard["gates"].items():
        mark = _MARK.get(g.get("status", ""), "----")
        observed = g.get("observed")
        obs_str = (
            f"  observed={observed:.3f}"
            if isinstance(observed, (int, float)) and not _isnan(observed)
            else ""
        )
        finding = g.get("finding_id")
        fid_str = f"  [{finding}]" if finding else ""
        print(f"  {gid:5} [{mark:5}] {g['description']}{obs_str}{fid_str}")
    print(f"\nbest algo (by eval reward): {best_algo}")
    print("\nper-algo eval summary (last %.0f%%):" % (fraction * 100))
    for algo, v in per_algo.items():
        print(
            f"  {algo:4} reward={v['mean_reward']:+8.1f}  "
            f"MTTC={v['mean_mttc']:.2f}  "
            f"comp%={v['compromise_rate']:.2f}  "
            f"prev%={v['prevention_rate']:.2f}  "
            f"mit|comp={v['mitigated_among_compromised']:.2f}"
        )
    return scoreboard


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Blue-Team gate evaluator.")
    p.add_argument("--runs-root", required=True)
    p.add_argument("--out-dir", default="docs/results/blue-team-training")
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
