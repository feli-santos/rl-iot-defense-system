"""Step-6 SMOKE GATE: one-seed alpha-sweep on the partial-observability redesign.

Trains RecurrentPPO + feedforward RL (dqn/ppo/a2c) for a *reduced* budget on
the redesigned environment (observation aliasing + session-coherent sampling +
no post-transition leakage + proximity-coupled defense tolerance, no budget,
outcome reward) across an aliasing-rate grid, then rolls every agent plus the
detector-coupled RF-Acting baseline and the full-observability oracle.

PASS CRITERIA (the gate before the full 10-seed sweep):
  (i)   at alpha=0 RF-Acting wins  -> we did NOT rob RF;
  (ii)  at alpha~=0.4 RecurrentPPO >= RF-Acting eval reward;
  (iii) no policy collapse (action distribution not all-OBSERVE / all-ISOLATE).

NOT a thesis artefact. Output: docs/review/redesign_smoke.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.algorithms.adversarial_algorithm import (
    AdversarialAlgorithm,
    AdversarialAlgorithmConfig,
)
from src.benchmark.baseline_policies import (
    _RECOMMENDED_BY_STAGE,
    RFActingPolicy,
    SB3PolicyAdapter,
    always_block,
    always_observe,
    recommended_action_policy,
)
from src.blue_team.env_factory import make_eval_env, make_train_env
from src.blue_team.run_config import EnvConfigSerializable

_ROOT = Path(__file__).resolve().parents[2]

_RL_HPARAMS = {
    "recurrent_ppo": {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "dqn": {
        "learning_rate": 1e-3,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

_ACTION_NAMES = ["OBSERVE", "LOG", "RESTRICT", "BLOCK", "ISOLATE"]


def _redesign_spec(alpha: float, split: str) -> EnvConfigSerializable:
    """The partial-observability redesign operating regime."""
    return EnvConfigSerializable(
        split=split,
        exclude_ood=True,
        impact_is_terminal=False,
        reward_mode="outcome",
        aliasing_rate=alpha,
        session_coherent=True,
        no_post_transition_leak=True,
        proximity_coupled=True,
    )


def _train_rl(algo, alpha, *, timesteps, seed, data, splits, recurrent_n_steps=None):
    spec = _redesign_spec(alpha, split="train")
    train_env = make_train_env(
        spec=spec,
        dataset_path=data,
        splits_manifest=splits,
        seed=seed,
    )
    hp = dict(_RL_HPARAMS[algo])
    if algo == "recurrent_ppo" and recurrent_n_steps is not None:
        hp["n_steps"] = int(recurrent_n_steps)
    cfg = AdversarialAlgorithmConfig(
        algorithm_type=algo,
        total_timesteps=timesteps,
        verbose=0,
        **dict(hp.items()),
    )
    model = AdversarialAlgorithm(cfg).create_model(train_env)
    model.set_random_seed(seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    train_env.close()
    return SB3PolicyAdapter(model, deterministic=True)


def _roll(policy, env, n_episodes):
    """Roll a policy, threading decision_step so recurrent adapters reset
    their LSTM state at each episode boundary (mirrors eval_runner)."""
    prevented = reached_impact = mitigated = 0
    ep_rewards = []
    action_counter: Counter = Counter()
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        last_info = {}
        ep_reward = 0.0
        step_idx = 0
        # decision-stage seed for the full-obs oracle (BENIGN at episode start;
        # thereafter taken from the previous step's true attack_stage). Mirrors
        # eval_runner._info_seed so recommended_action_policy can act.
        decision_stage = 0
        while not done:
            info = {
                "decision_step": step_idx,
                "attack_stage": decision_stage,
                "recommended_action": _RECOMMENDED_BY_STAGE[decision_stage],
            }
            a = int(policy(obs[0], info))
            action_counter[a] += 1
            obs, r, dones, infos = env.step(np.array([a]))
            ep_reward += float(r[0])
            done = bool(dones[0])
            last_info = infos[0]
            if not done:
                decision_stage = int(last_info.get("attack_stage", 0))
            step_idx += 1
        ep_rewards.append(ep_reward)
        outcome = last_info.get("outcome")
        if outcome == "prevented":
            prevented += 1
        elif outcome == "impact_mitigated":
            reached_impact += 1
            mitigated += 1
        elif outcome in ("compromised", "impact_unmitigated", "impact_missed"):
            reached_impact += 1
    n = float(n_episodes)
    arr = np.asarray(ep_rewards, dtype=float)
    total_actions = sum(action_counter.values()) or 1
    action_dist = {_ACTION_NAMES[a]: action_counter.get(a, 0) / total_actions for a in range(5)}
    top = max(action_dist.values())
    return {
        "prevention_rate": prevented / n,
        "reached_impact_rate": reached_impact / n,
        "mitigated_rate": mitigated / n,
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "action_dist": action_dist,
        "collapsed": bool(top >= 0.95),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", default=["0.0", "0.2", "0.4", "0.6"])
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--n-episodes", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=["recurrent_ppo", "dqn", "ppo", "a2c"])
    ap.add_argument(
        "--recurrent-n-steps",
        type=int,
        default=None,
        help="Override n_steps for recurrent_ppo (longer BPTT).",
    )
    ap.add_argument("--dataset-path", default="data/processed/ciciot2023")
    ap.add_argument("--rf-path", default="artifacts/detector/random_forest.joblib")
    ap.add_argument("--out", default="docs/review/redesign_smoke.json")
    args = ap.parse_args()

    data = args.dataset_path
    splits = str(Path(data) / "splits" / "manifest.json")
    rf = joblib.load(args.rf_path)

    results = {}
    t0 = time.time()
    for a_raw in args.alphas:
        alpha = float(a_raw)
        print(f"\n===== alpha={alpha} =====", flush=True)
        cell = {}

        # --- train RL agents on the redesigned env ---
        rl_adapters = {}
        for algo in args.algos:
            ts = time.time()
            rl_adapters[algo] = _train_rl(
                algo,
                alpha,
                timesteps=args.timesteps,
                seed=args.seed,
                data=data,
                splits=splits,
                recurrent_n_steps=args.recurrent_n_steps,
            )
            print(f"  trained {algo} in {time.time()-ts:.0f}s", flush=True)

        # --- build eval env (val split) + size RF-Acting ---
        eval_spec = _redesign_spec(alpha, split="val_balanced")
        probe = make_eval_env(
            spec=eval_spec,
            dataset_path=data,
            splits_manifest=splits,
            seed=args.seed + 9999,
        )
        obs_dim = probe.observation_space.shape[0]
        per_row = obs_dim // 5
        num_features = per_row // 2
        probe.close()

        policies = {
            "rf_acting": RFActingPolicy(
                rf,
                num_features=num_features,
                window_size=5,
                include_deltas=True,
            ),
            "recommended_action": recommended_action_policy,
            "always_block": always_block,
            "always_observe": always_observe,
        }
        policies.update(rl_adapters)

        for pname, pol in policies.items():
            env = make_eval_env(
                spec=eval_spec,
                dataset_path=data,
                splits_manifest=splits,
                seed=args.seed + 9999,
            )
            cell[pname] = _roll(pol, env, args.n_episodes)
            env.close()

        results[a_raw] = cell
        rl_present = [a for a in args.algos if a in cell]
        best_rl = max(rl_present, key=lambda a: cell[a]["mean_reward"])
        rf_rew = cell["rf_acting"]["mean_reward"]
        print(
            f"  alpha={alpha}: rf_acting={rf_rew:+.1f}  "
            + "  ".join(
                f"{a}={cell[a]['mean_reward']:+.1f}"
                f"{'*COLLAPSE' if cell[a]['collapsed'] else ''}"
                for a in rl_present
            )
            + f"  oracle={cell['recommended_action']['mean_reward']:+.1f}"
            + f"  | best_rl={best_rl} "
            f"(gap_vs_rf={cell[best_rl]['mean_reward']-rf_rew:+.1f})",
            flush=True,
        )

    payload = {
        "kind": "redesign_smoke_gate",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "n_episodes": args.n_episodes,
        "alphas": args.alphas,
        "algos": args.algos,
        "regime": {
            "session_coherent": True,
            "no_post_transition_leak": True,
            "proximity_coupled": True,
            "reward_mode": "outcome",
            "impact_is_terminal": False,
        },
        "elapsed_seconds": time.time() - t0,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out} ({payload['elapsed_seconds']:.0f}s total)")


if __name__ == "__main__":
    main()
