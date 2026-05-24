"""benchmark statistical significance tests (thesis review issue C4).

Loads per-seed episodic reward arrays from benchmark JSONL outputs and runs:

1. Paired Wilcoxon signed-rank test (scipy) across seeds for key comparisons.
2. Welch's t-test for independent-samples comparisons.
3. Effect sizes (Cohen's d).

Key comparisons (per consolidated review C4):
  a) DQN vs PPO on test_balanced (unpaired — different env seeds)
  b) DQN vs A2C on test_balanced
  c) Best DRL (DQN) vs RF-Acting
  d) impact_is_terminal=True vs False (reads F9 ablation data if present)

Outputs:
  - ``results/06_benchmark/statistical_tests.json``
  - Console summary table

Usage::

    python -m scripts.benchmark.run_statistical_tests \\
        [--phase6-root runs/benchmark] \\
        [--out-path results/06_benchmark/statistical_tests.json] \\
        [--alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers


def _load_episode_rewards(
    jsonl_path: Path,
) -> list[float]:
    """Load episodic rewards from a benchmark JSONL file."""
    rewards: list[float] = []
    if not jsonl_path.exists():
        logger.warning("JSONL not found: %s", jsonl_path)
        return rewards
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rewards.append(float(rec["episode_reward"]))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("skipping bad line: %s", exc)
    return rewards


def _collect_algo_rewards(
    benchmark_root: Path,
    algo: str,
    seeds: list[int],
) -> dict[int, list[float]]:
    """Collect per-seed reward arrays for a given algorithm."""
    per_seed: dict[int, list[float]] = {}
    for seed in seeds:
        jsonl = benchmark_root / algo / f"seed_{seed}" / "eval_test.jsonl"
        rewards = _load_episode_rewards(jsonl)
        if rewards:
            per_seed[seed] = rewards
        else:
            logger.warning("no data for %s seed %d", algo, seed)
    return per_seed


def _flatten(per_seed: dict[int, list[float]]) -> np.ndarray:
    """Flatten all seeds into a single reward array."""
    all_rewards: list[float] = []
    for rewards in per_seed.values():
        all_rewards.extend(rewards)
    return np.array(all_rewards, dtype=float)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = math.sqrt(
        ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1)))
        / (len(a) + len(b) - 2)
    )
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    label: str,
    alpha: float,
) -> dict[str, Any]:
    """Run Wilcoxon signed-rank test if samples are equal length; else Mann-Whitney U."""
    try:
        from scipy import stats  # type: ignore[import]
    except ImportError:
        return {
            "test": "unavailable",
            "error": "scipy not installed; run: pip install scipy",
        }

    d = _cohens_d(a, b)

    if len(a) == len(b) and len(a) >= 2:
        # Paired test (same seeds rolled on same env)
        try:
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
            return {
                "label": label,
                "test": "wilcoxon_signed_rank",
                "n": len(a),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < alpha),
                "alpha": alpha,
                "cohens_d": d,
            }
        except Exception as exc:  # noqa: BLE001 — e.g., all-zero differences
            logger.debug("Wilcoxon failed (%s); falling back to Mann-Whitney: %s", label, exc)

    # Independent / unequal-length: Mann-Whitney U
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "label": label,
        "test": "mann_whitney_u",
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "cohens_d": d,
    }


def _welch_test(
    a: np.ndarray,
    b: np.ndarray,
    label: str,
    alpha: float,
) -> dict[str, Any]:
    """Run Welch's t-test (independent samples, unequal variance)."""
    try:
        from scipy import stats  # type: ignore[import]
    except ImportError:
        return {
            "test": "unavailable",
            "error": "scipy not installed",
        }
    if len(a) < 2 or len(b) < 2:
        return {"label": label, "test": "welch_t", "error": "insufficient data"}
    stat, p = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)
    return {
        "label": label,
        "test": "welch_t",
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "cohens_d": d,
    }


def _bootstrap_ci(
    x: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for the mean of x."""
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    )
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


# ------------------------------------------------------------------ main logic


def run_tests(
    benchmark_root: Path,
    seeds: list[int],
    alpha: float = 0.05,
    ablation_path: Path | None = None,
) -> dict[str, Any]:
    """Run all statistical tests and return the results dict."""

    comparisons: list[dict[str, Any]] = []

    # Load per-algo rewards
    algo_rewards: dict[str, np.ndarray] = {}
    for algo in ["dqn", "ppo", "a2c"]:
        per_seed = _collect_algo_rewards(benchmark_root, algo, seeds)
        if per_seed:
            algo_rewards[algo] = _flatten(per_seed)
            lo, hi = _bootstrap_ci(algo_rewards[algo])
            logger.info(
                "%s: n=%d  mean=%.1f  95%%CI=[%.1f, %.1f]",
                algo.upper(),
                len(algo_rewards[algo]),
                float(np.mean(algo_rewards[algo])),
                lo,
                hi,
            )
        else:
            logger.warning("No data for %s — skipping", algo)

    # Load RF-Acting rewards (single seed=0, 150 episodes)
    rf_jsonl = benchmark_root / "rf_acting" / "seed_0" / "eval_test.jsonl"
    rf_rewards_list = _load_episode_rewards(rf_jsonl)
    if rf_rewards_list:
        algo_rewards["rf_acting"] = np.array(rf_rewards_list, dtype=float)
        lo, hi = _bootstrap_ci(algo_rewards["rf_acting"])
        logger.info(
            "RF-Acting: n=%d  mean=%.1f  95%%CI=[%.1f, %.1f]",
            len(algo_rewards["rf_acting"]),
            float(np.mean(algo_rewards["rf_acting"])),
            lo,
            hi,
        )

    # Comparison a: DQN vs PPO
    if "dqn" in algo_rewards and "ppo" in algo_rewards:
        comparisons.append(
            _welch_test(
                algo_rewards["dqn"],
                algo_rewards["ppo"],
                label="DQN vs PPO (test_balanced)",
                alpha=alpha,
            )
        )
        comparisons.append(
            _wilcoxon_test(
                algo_rewards["dqn"],
                algo_rewards["ppo"],
                label="DQN vs PPO (test_balanced)",
                alpha=alpha,
            )
        )

    # Comparison b: DQN vs A2C
    if "dqn" in algo_rewards and "a2c" in algo_rewards:
        comparisons.append(
            _welch_test(
                algo_rewards["dqn"],
                algo_rewards["a2c"],
                label="DQN vs A2C (test_balanced)",
                alpha=alpha,
            )
        )

    # Comparison c: Best DRL vs RF-Acting
    if "dqn" in algo_rewards and "rf_acting" in algo_rewards:
        comparisons.append(
            _welch_test(
                algo_rewards["dqn"],
                algo_rewards["rf_acting"],
                label="DQN vs RF-Acting (test_balanced)",
                alpha=alpha,
            )
        )
        comparisons.append(
            _wilcoxon_test(
                algo_rewards["dqn"],
                algo_rewards["rf_acting"],
                label="DQN vs RF-Acting (test_balanced)",
                alpha=alpha,
            )
        )

    # Comparison d: impact_is_terminal=True vs False (from F9 ablation data)
    if ablation_path is not None and ablation_path.exists():
        try:
            abl_data = json.loads(ablation_path.read_text())
            true_rewards = np.array(abl_data.get("terminal_true_rewards", []), dtype=float)
            false_rewards = np.array(abl_data.get("terminal_false_rewards", []), dtype=float)
            if len(true_rewards) >= 2 and len(false_rewards) >= 2:
                comparisons.append(
                    _welch_test(
                        false_rewards,
                        true_rewards,
                        label="impact_is_terminal=False vs True (F9 ablation)",
                        alpha=alpha,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load F9 ablation data: %s", exc)

    # Summary per-algorithm bootstrap CIs
    ci_summary: dict[str, Any] = {}
    for algo, rewards in algo_rewards.items():
        lo, hi = _bootstrap_ci(rewards)
        ci_summary[algo] = {
            "n": len(rewards),
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards, ddof=1)),
            "ci_95_lower": lo,
            "ci_95_upper": hi,
            "ci_overlaps_with": {},
        }
    # Check pairwise CI overlap
    algos = list(ci_summary.keys())
    for i, a in enumerate(algos):
        for b in algos[i + 1 :]:
            lo_a, hi_a = ci_summary[a]["ci_95_lower"], ci_summary[a]["ci_95_upper"]
            lo_b, hi_b = ci_summary[b]["ci_95_lower"], ci_summary[b]["ci_95_upper"]
            overlaps = not (hi_a < lo_b or hi_b < lo_a)
            ci_summary[a]["ci_overlaps_with"][b] = overlaps
            ci_summary[b]["ci_overlaps_with"][a] = overlaps

    return {
        "alpha": alpha,
        "seeds": seeds,
        "bootstrap_ci_summary": ci_summary,
        "comparisons": comparisons,
        "n_significant": sum(1 for c in comparisons if c.get("significant") is True),
    }


# ------------------------------------------------------------------ CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="benchmark statistical significance tests (C4).",
    )
    p.add_argument("--phase6-root", default="runs/benchmark")
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds to include in the DRL comparisons.",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--out-path",
        default="results/06_benchmark/statistical_tests.json",
    )
    p.add_argument(
        "--ablation-path",
        default=None,
        help="Optional path to F9 ablation JSON containing "
        "'terminal_true_rewards' and 'terminal_false_rewards' arrays.",
    )
    p.add_argument("--verbose", type=int, default=1)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    benchmark_root = Path(args.benchmark_root)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ablation_path = Path(args.ablation_path) if args.ablation_path else None

    results = run_tests(
        benchmark_root=benchmark_root,
        seeds=args.seeds,
        alpha=args.alpha,
        ablation_path=ablation_path,
    )

    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Statistical tests written to %s", out_path)
    logger.info(
        "Significant comparisons: %d / %d",
        results["n_significant"],
        len(results["comparisons"]),
    )

    # Print summary table
    print("\n=== Statistical Test Summary ===")
    print(f"{'Algorithm':<15} {'N':>6} {'Mean':>10} {'95%CI':>24}")
    print("-" * 60)
    for algo, s in results["bootstrap_ci_summary"].items():
        ci_str = f"[{s['ci_95_lower']:.1f}, {s['ci_95_upper']:.1f}]"
        print(f"{algo:<15} {s['n']:>6} {s['mean']:>10.1f} {ci_str:>24}")
    print("\n=== Pairwise Tests ===")
    for c in results["comparisons"]:
        if "p_value" in c:
            sig = "✓" if c.get("significant") else "✗"
            print(
                f"{sig} [{c['test']}] {c['label']}: "
                f"p={c['p_value']:.4f}, d={c.get('cohens_d', float('nan')):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
