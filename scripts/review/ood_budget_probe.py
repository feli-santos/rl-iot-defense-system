"""Diagnostic: OOD prevention/compromise vs attacker budget.

Quantifies exactly where the finite intrusion budget MASKS the
detector-independence effect. For a genuinely-blind OOD class
(VulnerabilityScan, RF recall 0.076) and a sighted class
(DoS-SYN_Flood, recall 0.998), sweep the attacker budget across a
range including unbounded and record prevention/compromise for:

  rf_acting (detector-coupled), dqn (detector-free RL), always_block,
  recommended_action (full-obs oracle).

NOT a thesis artefact — a review-time investigation. Output:
docs/review/ood_budget_probe.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.ablation.run_ood_eval import _build_ood_env
from src.benchmark.baseline_policies import (
    RFActingPolicy,
    SB3PolicyAdapter,
    always_block,
    always_observe,
)

_ROOT = Path(__file__).resolve().parents[2]


def _roll(policy, env, n_episodes: int) -> dict:
    prevented = reached_impact = mitigated = 0
    other = 0
    ep_rewards: list[float] = []
    for _ in range(n_episodes):
        obs = env.reset()
        info = {}
        done = False
        last_info = {}
        ep_reward = 0.0
        while not done:
            a = policy(obs[0], info)
            obs, r, dones, infos = env.step(np.array([a]))
            ep_reward += float(r[0])
            done = bool(dones[0])
            last_info = infos[0]
        ep_rewards.append(ep_reward)
        outcome = last_info.get("outcome")
        if outcome == "prevented":
            prevented += 1
        elif outcome == "impact_mitigated":
            reached_impact += 1
            mitigated += 1
        elif outcome in ("compromised", "impact_unmitigated", "impact_missed"):
            reached_impact += 1
        else:
            other += 1
    n = float(n_episodes)
    arr = np.asarray(ep_rewards, dtype=float)
    return {
        "prevention_rate": prevented / n,
        # any episode that reached IMPACT (whether or not blocked on the
        # terminal turn) -- the security-relevant "attack got through" event
        "reached_impact_rate": reached_impact / n,
        "mitigated_rate": mitigated / n,
        "other_rate": other / n,
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
    }


def _load_rl(algo: str, seed: int):
    from stable_baselines3 import A2C, DQN, PPO

    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    run_root = _ROOT / f"runs/blue_team/{algo}/seed_{seed}"
    ckpt = run_root / "best_model.zip"
    if not ckpt.exists():
        ckpt = run_root / "model.zip"
    return SB3PolicyAdapter(cls.load(str(ckpt)), deterministic=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["VulnerabilityScan", "DoS-SYN_Flood", "DNS_Spoofing"],
    )
    ap.add_argument(
        "--budgets", nargs="+", default=["20", "30", "40", "None"]
    )
    ap.add_argument(
        "--budget-cost-model",
        default="hybrid",
        choices=["hybrid", "targeted"],
        help="Attacker-budget drain model under test. 'targeted' drains only "
        "under correctly-aimed proportional force.",
    )
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator-path", default="artifacts/generator/red_team")
    ap.add_argument("--dataset-path", default="data/processed/ciciot2023")
    ap.add_argument(
        "--rf-path", default="artifacts/detector/random_forest.joblib"
    )
    ap.add_argument("--out", default="docs/review/ood_budget_probe.json")
    args = ap.parse_args()

    splits_manifest = str(
        Path(args.dataset_path) / "splits" / "manifest.json"
    )
    import joblib

    rf = joblib.load(args.rf_path)

    rl_by_algo = {a: _load_rl(a, args.seed) for a in ("dqn", "ppo", "a2c")}

    results: dict = {}
    t0 = time.time()
    for cls in args.classes:
        results[cls] = {}
        for b_raw in args.budgets:
            budget = None if b_raw == "None" else int(b_raw)
            ns = SimpleNamespace(
                generator_path=args.generator_path,
                dataset_path=args.dataset_path,
                splits_manifest=splits_manifest,
                attacker_budget=budget,
                reward_mode="outcome",
                budget_cost_model=args.budget_cost_model,
            )
            cell = {}
            # probe obs to size RFActingPolicy
            env0 = _build_ood_env(ns, cls, seed=args.seed)
            obs_dim = env0.observation_space.shape[0]
            window = 5
            per_row = obs_dim // window
            num_features = per_row // 2  # include_deltas
            env0.close()

            policies = {
                "rf_acting": RFActingPolicy(
                    rf,
                    num_features=num_features,
                    window_size=window,
                    include_deltas=True,
                ),
                "dqn": rl_by_algo["dqn"],
                "ppo": rl_by_algo["ppo"],
                "a2c": rl_by_algo["a2c"],
                "always_block": always_block,
                "always_observe": always_observe,
            }
            for pname, pol in policies.items():
                env = _build_ood_env(ns, cls, seed=args.seed)
                cell[pname] = _roll(pol, env, args.n_episodes)
                env.close()
            results[cls][b_raw] = cell
            best_rl = max(
                ("dqn", "ppo", "a2c"),
                key=lambda a: cell[a]["mean_reward"],
            )
            print(
                f"{cls:22s} B={b_raw:>4}  "
                + "  ".join(
                    f"{p}=prev{cell[p]['prevention_rate']:.2f}/"
                    f"imp{cell[p]['reached_impact_rate']:.2f}/"
                    f"rew{cell[p]['mean_reward']:+.0f}"
                    for p in ("rf_acting", best_rl, "always_block")
                )
                + f"  [bestRL={best_rl}]"
            )

    payload = {
        "kind": "ood_budget_probe",
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "budget_cost_model": args.budget_cost_model,
        "classes": args.classes,
        "budgets": args.budgets,
        "elapsed_seconds": time.time() - t0,
        "legend": "values are prevention_rate / compromise_rate",
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
