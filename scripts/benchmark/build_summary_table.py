"""benchmark F5 — final security metrics table (PLAN §3.1.5, C4).

Reads ``runs/benchmark/<policy>/seed_<k>/eval_test.jsonl`` (and the matching
``latency.jsonl`` sidecars) for every policy, aggregates per-policy
metrics across seeds, and writes:

- ``docs/results/06_benchmark/F5_summary.json`` — machine-readable.
- ``docs/results/06_benchmark/F5_summary.md``   — Markdown table for
  the thesis chapter.
- ``docs/results/06_benchmark/F5_summary.csv``  — same data, flat.
- ``docs/results/06_benchmark/F5_table.png``    — rendered table figure.
- ``docs/results/06_benchmark/F5_manifest.json`` — SHA-256 hash chain
  over every input JSONL + the upstream benchmark eval manifest + the
  git SHA at production time (G6.7 / D6.9).

Columns reported per row (per PLAN §3.1):

- ``mean_reward``                — episodic reward, averaged over n=150 ep.
- ``mean_mttc``                  — mean time-to-compromise (steps).
- ``compromise_rate``            — fraction of episodes that reached IMPACT.
- ``mitigated_impact_rate``      — fraction of episodes ending in
                                   ``end_outcome == "impact_mitigated"``.
- ``mean_episode_length``        — average wall length in env steps.
- ``mean_inference_latency_ms``  — median per-step inference time
                                   from the sidecar latency.jsonl.
- ``p95_inference_latency_ms``   — 95th percentile for the same.

The script also computes 95 % bootstrap CIs on ``mean_reward``
(sampled across seeds for non-deterministic policies, or across
episodes for deterministic ones) and stamps them into the JSON
output for F8 to consume directly.

Per D6.10: "best algo" is the row with max ``mean_reward``; ties broken
by lower ``p95_inference_latency_ms``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.blue_team.aggregation import bootstrap_ci, read_episodes_jsonl

logger = logging.getLogger("scripts.benchmark.build_summary_table")

_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- helpers


# Display name + canonical row ordering (top-to-bottom in the table).
# Trained-RL first (the thesis's headline rows), then non-RL baselines
# in increasing aggression.
_POLICY_ORDER: List[str] = [
    "dqn", "ppo", "a2c",
    "random", "always_observe", "always_block",
    "recommended_action", "rf_acting",
]

_DISPLAY_NAMES: Dict[str, str] = {
    "dqn": "DQN",
    "ppo": "PPO",
    "a2c": "A2C",
    "random": "Random",
    "always_observe": "Always-OBSERVE",
    "always_block": "Always-BLOCK",
    "recommended_action": "Recommended-Action (rule)",
    "rf_acting": "RF-Acting (supervised + rules)",
}


def _sha256(path: Path) -> Optional[str]:
    """SHA-256 of file content (1 MiB chunks); ``None`` if absent."""
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
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_latency_ms(path: Path) -> np.ndarray:
    """Read ``latency.jsonl`` and return per-step durations in **milliseconds**.

    Returns an empty array when the file is missing or empty so callers
    can NaN-out the latency columns rather than crashing the whole table.
    """
    if not path.exists():
        return np.array([], dtype=np.float64)
    durs_ns: List[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            durs_ns.append(int(row["duration_ns"]))
    if not durs_ns:
        return np.array([], dtype=np.float64)
    return np.asarray(durs_ns, dtype=np.float64) / 1e6


# ---------------------------------------------------------- per-policy ----


def _discover_seed_dirs(runs_root: Path, policy: str) -> List[Path]:
    """Return ``[seed_0, seed_1, ...]`` directories for one policy.

    Order is preserved by integer-seed sort so cross-policy tables line
    up. Missing policies yield an empty list (caller decides whether
    to skip or warn).
    """
    base = runs_root / policy
    if not base.exists():
        return []
    out: List[Tuple[int, Path]] = []
    for child in base.iterdir():
        if not child.is_dir() or not child.name.startswith("seed_"):
            continue
        try:
            seed = int(child.name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((seed, child))
    return [p for _, p in sorted(out)]


def _summarise_policy(
    policy: str,
    seed_dirs: List[Path],
) -> Dict[str, Any]:
    """Compute the F5 row for one policy.

    The mean-reward bootstrap CI is computed differently for
    non-deterministic vs deterministic baselines:

    - **non-deterministic** (DQN/PPO/A2C/random): one mean-per-seed,
      bootstrap across 5 seeds. This matches blue-team's seed-level
      uncertainty narrative.
    - **deterministic** (always-X / recommended / RF-acting): one
      mean per *episode*; bootstrap across all 150 episodes. Otherwise
      the CI would be a single point and look misleadingly tight.
    """
    all_records: List[Dict] = []
    per_seed_means: List[float] = []
    latency_chunks: List[np.ndarray] = []
    for sd in seed_dirs:
        recs = read_episodes_jsonl(sd / "eval_test.jsonl")
        all_records.extend(recs)
        if recs:
            per_seed_means.append(float(np.mean([r["episode_reward"] for r in recs])))
        latency_chunks.append(_load_latency_ms(sd / "latency.jsonl"))

    if not all_records:
        return {
            "policy": policy,
            "n_seeds": len(seed_dirs),
            "n_episodes": 0,
            "mean_reward": math.nan,
            "mean_reward_ci_low": math.nan,
            "mean_reward_ci_high": math.nan,
            "mean_mttc": math.nan,
            "compromise_rate": math.nan,
            "mitigated_impact_rate": math.nan,
            "mean_episode_length": math.nan,
            "mean_inference_latency_ms": math.nan,
            "p50_inference_latency_ms": math.nan,
            "p95_inference_latency_ms": math.nan,
            "p99_inference_latency_ms": math.nan,
        }

    rewards = [r["episode_reward"] for r in all_records]
    mttc_vals = [r["mttc_steps"] for r in all_records if r.get("mttc_steps") is not None]
    compromised = [1.0 if r.get("compromised") else 0.0 for r in all_records]
    mitigated = [
        1.0 if r.get("end_outcome") == "impact_mitigated" else 0.0
        for r in all_records
    ]
    lengths = [r["episode_length"] for r in all_records]

    # Bootstrap CI: choose granularity to keep the math meaningful.
    if len(per_seed_means) >= 3:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            per_seed_means, n_resamples=2000, alpha=0.05, seed=0,
        )
    else:
        ci_low, _ci_mean, ci_high = bootstrap_ci(
            rewards, n_resamples=2000, alpha=0.05, seed=0,
        )

    # Latency: concatenate seed-level chunks; compute robust quantiles.
    all_lat = np.concatenate(latency_chunks) if latency_chunks else np.array([])
    if all_lat.size:
        p50 = float(np.percentile(all_lat, 50))
        p95 = float(np.percentile(all_lat, 95))
        p99 = float(np.percentile(all_lat, 99))
        mean_lat = float(np.mean(all_lat))
    else:
        p50 = p95 = p99 = mean_lat = math.nan

    return {
        "policy": policy,
        "n_seeds": len(seed_dirs),
        "n_episodes": len(all_records),
        "mean_reward": float(np.mean(rewards)),
        "mean_reward_ci_low": float(ci_low),
        "mean_reward_ci_high": float(ci_high),
        "mean_mttc": float(np.mean(mttc_vals)) if mttc_vals else math.nan,
        "compromise_rate": float(np.mean(compromised)),
        "mitigated_impact_rate": float(np.mean(mitigated)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_inference_latency_ms": mean_lat,
        "p50_inference_latency_ms": p50,
        "p95_inference_latency_ms": p95,
        "p99_inference_latency_ms": p99,
    }


# ---------------------------------------------------------- best-algo


def _best_row(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Return the policy name with max mean_reward (ties: lower p95 latency).

    Mirrors D6.10. Returns ``None`` when every row's mean_reward is NaN
    (e.g., empty sweep).
    """
    candidates = [
        r for r in rows
        if r.get("mean_reward") is not None
        and not math.isnan(r.get("mean_reward", math.nan))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (-r["mean_reward"], r.get("p95_inference_latency_ms", math.inf)),
    )
    return candidates[0]["policy"]


# ------------------------------------------------------------ rendering


def _render_markdown(rows: List[Dict[str, Any]], best: Optional[str]) -> str:
    """Render F5 as a Markdown table matching the thesis-paper format."""
    headers = [
        "Policy", "n", "Mean reward (95 % CI)", "MTTC",
        "Compromise %", "Mitigated %", "Ep. length",
        "Latency p50 (ms)", "Latency p95 (ms)",
    ]
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    out = ["| " + " | ".join(headers) + " |", sep]
    for r in rows:
        is_best = (r["policy"] == best)
        name = _DISPLAY_NAMES.get(r["policy"], r["policy"])
        if is_best:
            name = f"**{name}**"
        ci = f"{r['mean_reward']:.1f} ({r['mean_reward_ci_low']:.1f}, {r['mean_reward_ci_high']:.1f})"
        cells = [
            name,
            f"{r['n_episodes']}",
            ci,
            f"{r['mean_mttc']:.2f}" if not math.isnan(r['mean_mttc']) else "—",
            f"{100*r['compromise_rate']:.1f}",
            f"{100*r['mitigated_impact_rate']:.1f}",
            f"{r['mean_episode_length']:.1f}",
            f"{r['p50_inference_latency_ms']:.3f}" if not math.isnan(r['p50_inference_latency_ms']) else "—",
            f"{r['p95_inference_latency_ms']:.3f}" if not math.isnan(r['p95_inference_latency_ms']) else "—",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _render_png(
    rows: List[Dict[str, Any]],
    best: Optional[str],
    out_path: Path,
) -> None:
    """Render the F5 table as a PNG via matplotlib (no external deps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headers = [
        "Policy", "n", "Mean reward (95 % CI)", "MTTC",
        "Comp %", "Mit %", "Ep. len.", "p50 ms", "p95 ms",
    ]
    cell_text: List[List[str]] = []
    for r in rows:
        name = _DISPLAY_NAMES.get(r["policy"], r["policy"])
        if r["policy"] == best:
            name += " ★"
        ci = (
            f"{r['mean_reward']:.0f} "
            f"({r['mean_reward_ci_low']:.0f}, {r['mean_reward_ci_high']:.0f})"
        )
        cell_text.append([
            name,
            f"{r['n_episodes']}",
            ci,
            f"{r['mean_mttc']:.2f}" if not math.isnan(r['mean_mttc']) else "—",
            f"{100*r['compromise_rate']:.1f}",
            f"{100*r['mitigated_impact_rate']:.1f}",
            f"{r['mean_episode_length']:.1f}",
            f"{r['p50_inference_latency_ms']:.3f}" if not math.isnan(r['p50_inference_latency_ms']) else "—",
            f"{r['p95_inference_latency_ms']:.3f}" if not math.isnan(r['p95_inference_latency_ms']) else "—",
        ])
    n = len(cell_text)
    fig, ax = plt.subplots(figsize=(13, 0.6 + 0.45 * (n + 1)))
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    # Highlight header.
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    # Highlight best row.
    if best is not None:
        for i, r in enumerate(rows, start=1):
            if r["policy"] == best:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor("#fef3c7")
    # Section divider between trained-RL and baselines.
    n_rl = sum(1 for r in rows if r["policy"] in ("dqn", "ppo", "a2c"))
    if n_rl > 0:
        for j in range(len(headers)):
            cell = table[(n_rl, j)]
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
    fig.suptitle(
        "F5 — Final Security Metrics on `test_balanced` "
        "(★ = best by mean reward, tie-break p95 latency)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- main


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """CSV with one row per policy, all metric columns."""
    if not rows:
        out_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="benchmark F5 — security metrics table.")
    p.add_argument("--runs-root", default="runs/benchmark")
    p.add_argument("--out-dir", default="docs/results/06_benchmark")
    p.add_argument(
        "--policies", nargs="+", default=_POLICY_ORDER,
        help="Subset / ordering of policies to include.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    input_jsonl_hashes: Dict[str, str] = {}
    for pol in args.policies:
        seed_dirs = _discover_seed_dirs(runs_root, pol)
        if not seed_dirs:
            logger.warning("no seed dirs for policy %r under %s; skipping", pol, runs_root)
            continue
        row = _summarise_policy(pol, seed_dirs)
        rows.append(row)
        for sd in seed_dirs:
            jp = sd / "eval_test.jsonl"
            sha = _sha256(jp)
            if sha:
                input_jsonl_hashes[str(jp)] = sha
            lp = sd / "latency.jsonl"
            sha2 = _sha256(lp)
            if sha2:
                input_jsonl_hashes[str(lp)] = sha2

    best = _best_row(rows)

    summary_doc = {
        "schema_version": "1.0",
        "phase": 6,
        "figure": "F5",
        "best_policy": best,
        "rows": rows,
    }
    (out_dir / "F5_summary.json").write_text(json.dumps(summary_doc, indent=2))
    (out_dir / "F5_summary.md").write_text(_render_markdown(rows, best))
    _write_csv(rows, out_dir / "F5_summary.csv")
    _render_png(rows, best, out_dir / "F5_table.png")

    eval_manifest_path = runs_root / "eval_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "figure": "F5",
        "git_sha": _git_sha(),
        "outputs": {
            "json": str(out_dir / "F5_summary.json"),
            "md":   str(out_dir / "F5_summary.md"),
            "csv":  str(out_dir / "F5_summary.csv"),
            "png":  str(out_dir / "F5_table.png"),
        },
        "inputs": {
            "eval_manifest": {
                "path": str(eval_manifest_path),
                "sha256": _sha256(eval_manifest_path),
            },
            "eval_jsonl_sha256": input_jsonl_hashes,
        },
        "best_policy": best,
    }
    (out_dir / "F5_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "F5 built: %d rows, best=%s — wrote %s",
        len(rows), best, out_dir,
    )
    # Echo the best row to stdout for convenience.
    if best is not None:
        for r in rows:
            if r["policy"] == best:
                logger.info(
                    "best policy: %s mean_reward=%.2f (95%% CI %.2f, %.2f) "
                    "p50_lat=%.3f ms p95_lat=%.3f ms",
                    r["policy"], r["mean_reward"],
                    r["mean_reward_ci_low"], r["mean_reward_ci_high"],
                    r["p50_inference_latency_ms"], r["p95_inference_latency_ms"],
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
