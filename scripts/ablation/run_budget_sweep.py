"""Calibration probe for the finite attacker budget (Phase C-cal HARD GATE).

Sweeps ``attacker_budget`` over a grid against a set of fixed baseline policies
and reports, per (budget, policy) cell:

- ``compromise_rate``      = mean(info["compromised"])
- ``prevention_rate``      = mean(info["outcome"] == "prevented")
- ``prevention_post_grace``= mean(prevented AND terminal step >= min_episode_length)

The post-grace conditioning matters: the grace-period clamp downgrades any IMPACT
before ``min_episode_length`` to MANEUVER, so a budget exhausted inside the grace
window is *not* defender-attributable. Only post-grace preventions count as the
defender having genuinely starved the attacker.

GOAL: decide whether ``compromise_rate`` tracks defender policy quality (a sigmoid
of compromise_rate vs budget where stronger policies shift the curve). This is a
HARD GATE run *before* the expensive Phase-D re-train.

Usage (probe / smoke first, then full):

    .venv/bin/python -m scripts.ablation.run_budget_sweep --n-episodes 30 --smoke
    .venv/bin/python -m scripts.ablation.run_budget_sweep --n-episodes 200 \
        --out docs/review/budget_calibration.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.benchmark.baseline_policies import (
    always_block,
    always_observe,
    random_policy,
    recommended_action_policy,
)
from src.blue_team.env_factory import _build_env
from src.blue_team.run_config import EnvConfigSerializable

DATASET_PATH = "data/processed/ciciot2023"
SPLITS_MANIFEST = "data/processed/ciciot2023/splits/manifest.json"

BUDGET_GRID: list[Optional[int]] = [8, 12, 20, 30, 40, 50, 60, 80, None]
MIN_EPISODE_LENGTH = 20  # grace floor; preventions before this are not attributable


def _make_policies(seed: int) -> dict[str, Callable[[np.ndarray, dict], int]]:
    rng = np.random.default_rng(seed)
    return {
        "random": lambda obs, info: random_policy(obs, info, rng=rng),
        "always_observe": always_observe,
        "always_block": always_block,
        "recommended_action": recommended_action_policy,
    }


def _roll_episode(env, policy: Callable[[np.ndarray, dict], int], seed: int) -> dict:
    """Roll a single seeded episode and return its terminal telemetry."""
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        action = policy(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
    return {
        "compromised": bool(info.get("compromised", False)),
        "outcome": info.get("outcome", ""),
        "attacker_exhausted": bool(info.get("attacker_exhausted", False)),
        "steps": steps,
    }


def _run_cell(
    budget: Optional[int],
    policy: Callable[[np.ndarray, dict], int],
    generator_path: Path,
    n_episodes: int,
    base_seed: int,
) -> dict:
    spec = EnvConfigSerializable(
        split="test_balanced",
        exclude_ood=True,
        impact_is_terminal=False,  # primary training contract
        reward_mode="outcome",  # re-posed contract; sweep KPIs are dynamics-invariant
        attacker_budget=budget,
    )
    env = _build_env(
        spec=spec,
        generator_path=generator_path,
        dataset_path=DATASET_PATH,
        splits_manifest=SPLITS_MANIFEST,
        seed=base_seed,
    )
    compromised = 0
    prevented = 0
    prevented_post_grace = 0
    for ep in range(n_episodes):
        # Re-seed each episode for reproducibility / decorrelation.
        rec = _roll_episode(env, policy, seed=base_seed + ep)
        if rec["compromised"]:
            compromised += 1
        if rec["outcome"] == "prevented":
            prevented += 1
            if rec["steps"] >= MIN_EPISODE_LENGTH:
                prevented_post_grace += 1
    return {
        "compromise_rate": compromised / n_episodes,
        "prevention_rate": prevented / n_episodes,
        "prevention_post_grace": prevented_post_grace / n_episodes,
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick probe: fewer episodes, prints timing per cell.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/review/budget_calibration.json",
    )
    parser.add_argument(
        "--generator-path",
        type=str,
        default="artifacts/generator/red_team",
        help="Ignored under the Markov attacker; kept for env construction.",
    )
    args = parser.parse_args()

    generator_path = Path(args.generator_path)
    generator_path.mkdir(parents=True, exist_ok=True)

    policies = _make_policies(args.base_seed)
    results: dict[str, dict] = {}
    t0 = time.time()

    for budget in BUDGET_GRID:
        bkey = "None" if budget is None else str(budget)
        results[bkey] = {}
        for pname, policy in policies.items():
            cell_t0 = time.time()
            cell = _run_cell(
                budget=budget,
                policy=policy,
                generator_path=generator_path,
                n_episodes=args.n_episodes,
                base_seed=args.base_seed,
            )
            results[bkey][pname] = cell
            dt = time.time() - cell_t0
            print(
                f"budget={bkey:>4} policy={pname:<19} "
                f"compromise={cell['compromise_rate']:.3f} "
                f"prevent={cell['prevention_rate']:.3f} "
                f"prevent_pg={cell['prevention_post_grace']:.3f} "
                f"({dt:.1f}s)"
            )

    elapsed = time.time() - t0
    payload = {
        "budget_grid": [b if b is not None else None for b in BUDGET_GRID],
        "min_episode_length": MIN_EPISODE_LENGTH,
        "n_episodes": args.n_episodes,
        "base_seed": args.base_seed,
        "dataset_path": DATASET_PATH,
        "split": "test_balanced",
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }

    # Console summary table: compromise_rate, rows=budget, cols=policy.
    pnames = list(policies.keys())
    print("\ncompromise_rate (rows=budget, cols=policy):")
    print("budget  " + "  ".join(f"{p:>19}" for p in pnames))
    for budget in BUDGET_GRID:
        bkey = "None" if budget is None else str(budget)
        cells = "  ".join(f"{results[bkey][p]['compromise_rate']:>19.3f}" for p in pnames)
        print(f"{bkey:>6}  {cells}")

    if not args.smoke:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {out_path} ({elapsed:.1f}s total)")
    else:
        print(f"\n[smoke] not writing JSON ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
