"""ablation F15 — OOD-class robustness evaluator (audit-AF1).

Evaluates every benchmark policy (rule oracle, RF-Acting, trained
DQN/PPO/A2C, random, always-OBSERVE, always-BLOCK) on each of the
four held-out OOD attack classes (DDoS-HTTP_Flood, Mirai-udpplain,
VulnerabilityScan, XSS) by restricting the env's
``RealizationEngine.allowed_indices`` to that class's row indices.

This is the audit-AF1-promoted Tier-1 deliverable that supplies the
detector-to-thesis-claim payoff: detector RESULTS §3.2 reported the
supervised RF stage detector has 0.001 recall on ``VulnerabilityScan``;
F15 quantifies how much of that blind spot the trained RL policy
recovers.

**No retraining**. F15 reuses the frozen blue-team trained checkpoints
(D7.6) and the benchmark ``eval_runner`` harness unchanged. Only the
RealizationEngine constraint changes per outer loop.

Usage::

    python -m scripts.ablation.run_ood_eval \\
        [--ood-classes DDoS-HTTP_Flood Mirai-udpplain VulnerabilityScan XSS] \\
        [--policies rule rf_acting dqn ppo a2c random always_observe always_block] \\
        [--n-episodes 30] [--seeds 0 1 2 3 4] \\
        [--phase5-runs runs/blue_team] [--out-root runs/ablation/ood] \\
        [--rf-path artifacts/detector/random_forest.joblib] \\
        [--smoke]

Per PLAN §3.1.3 / D7.6 the default sweep is:

- 4 OOD classes ×
  (3 RL algos × 5 seeds × 30 ep
   + 1 random × 5 seeds × 30 ep
   + 4 deterministic baselines × 1 seed × 150 ep)
  = 4 × (450 + 150 + 600) = 4 × 1 200 = **4 800 episodes**.

Expected wallclock < 1 hour CPU on Apple silicon (no model load
repeated; episodes short post-environment-design lifecycle fix).

Output layout::

    runs/ablation/ood/
        eval_manifest.json                 — top-level F15 manifest
        <ood_class>/<policy>/seed_<k>/
            eval_test.jsonl                — schema-v1.0 EpisodeRecord
            (no latency.jsonl — F15 inherits benchmark F7's latency claim)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.benchmark.baseline_policies import (
    RFActingPolicy,
    SB3PolicyAdapter,
    always_block,
    always_observe,
    random_policy,
    recommended_action_policy,
)
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import _build_env, make_eval_env
from src.blue_team.run_config import EnvConfigSerializable
from src.environment.adversarial_env import AdversarialIoTEnv
from src.utils.realization_engine import RealizationEngine
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger("scripts.ablation.run_ood_eval")

_ROOT = Path(__file__).resolve().parents[2]


# Each OOD class lives at exactly one stage in the CICIoT2023 label
# taxonomy (verified empirically). Pre-cached so we don't re-scan
# state_indices on every cell. If the OOD splits are ever rebuilt to
# include other stages, ``_resolve_ood_stage`` falls back to a runtime
# scan of state_indices.json.
_OOD_STAGE_BY_CLASS: Dict[str, int] = {
    "DDoS-HTTP_Flood":   4,  # IMPACT
    "Mirai-udpplain":    3,  # MANEUVER
    "VulnerabilityScan": 1,  # RECON  ← detector F11 0.001-recall blind spot
    "XSS":               2,  # ACCESS
}


# --------------------------------------------------------------------- helpers


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


def _load_sb3_model(algo: str, model_path: Path, env: Any) -> Any:
    a = algo.lower()
    if a == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path, env=env, device="cpu")
    if a == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path, env=env, device="cpu")
    if a == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(model_path, env=env, device="cpu")
    raise ValueError(f"unknown algo {algo!r}; expected dqn / ppo / a2c")


def _ood_eval_env_spec() -> EnvConfigSerializable:
    """environment-design-frozen reward config (D7.4) on the *train* split.

    Why train (not test_balanced)? F15's hybrid realiser pattern
    (see _build_ood_env) keeps the *non-OOD-stage* feature pool from
    the training-distribution rows, so the agent sees normal-looking
    features at every non-attack-stage step and OOD-class features
    only at the attack-arrival step. Using train as the base pool is
    the most generous setup for trained RL: any failure to react is
    truly an OOD-feature failure, not a distribution-shift on
    benign / non-attack features.
    """
    return EnvConfigSerializable(
        split="train",
        exclude_ood=True,  # base pool is in-distribution; OOD overlay is added below
    )


def _resolve_ood_stage(
    dataset_path: Path,
    ood_indices: np.ndarray,
) -> int:
    """Return the unique stage id where every OOD-class row lives.

    Reads ``state_indices.json`` and intersects with ``ood_indices``;
    raises if rows span multiple stages or no stages.
    """
    state_indices_path = dataset_path / "state_indices.json"
    raw = json.loads(state_indices_path.read_text())
    state_indices = {int(k): set(int(i) for i in v) for k, v in raw.items()}
    ood_set = set(int(i) for i in ood_indices.tolist())
    found_stages = [s for s, idx_set in state_indices.items() if ood_set & idx_set]
    if not found_stages:
        raise ValueError(
            f"OOD-class rows have no overlap with any stage in state_indices.json"
        )
    if len(found_stages) > 1:
        # Pick the stage with the most overlap; warn the user.
        overlaps = {
            s: len(ood_set & state_indices[s]) for s in found_stages
        }
        chosen = max(overlaps, key=overlaps.get)
        logger.warning(
            "OOD-class rows span multiple stages %s (counts %s); "
            "using stage %d (largest overlap) as the F15 attack stage.",
            found_stages, overlaps, chosen,
        )
        return chosen
    return found_stages[0]


def _build_ood_env(
    args: argparse.Namespace,
    ood_class: str,
    seed: Optional[int] = None,
) -> Any:
    """Build a fresh eval env with a *hybrid* OOD realiser.

    F15 design (revised after smoke surfaced empty-non-OOD-stage bug):
    each OOD class lives at exactly one stage (e.g.,
    ``VulnerabilityScan`` is RECON-only, ``DDoS-HTTP_Flood`` is
    IMPACT-only — verified empirically against
    ``state_indices.json``). The realiser is therefore built as a
    hybrid:

      * stages other than the OOD class's stage  → train-split rows
        (normal in-distribution features)
      * the OOD class's stage                    → OOD-class rows
        (the held-out distribution-shift)

    The agent sees normal-looking features at every step UNTIL the
    env transitions to the OOD class's stage, at which point the
    realiser samples from the held-out attack class. This is the
    semantically-correct F15 eval: it isolates the agent's response
    to OOD features at the attack-arrival moment.

    The base RealizationEngine is built on the train split with
    ``exclude_ood=True`` (so it has the full 5-stage in-distribution
    pool), then the OOD class's stage gets its ``_state_indices``
    entry surgically replaced with the OOD-class rows.
    """
    spec = _ood_eval_env_spec()

    # 1. Build the env on the train pool (all 5 stages populated).
    env = _build_env(
        spec=spec,
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )

    # 2. Surgically replace the OOD-stage's row pool with OOD-class rows.
    splits_manifest_path = Path(args.splits_manifest)
    manifest_dir = splits_manifest_path.parent.parent
    ood_idx_path = manifest_dir / "splits" / "ood_attack" / f"{ood_class}.idx.npy"
    if not ood_idx_path.exists():
        raise FileNotFoundError(f"OOD index file not found: {ood_idx_path}")
    ood_indices = np.load(ood_idx_path)

    # Resolve which stage this OOD class belongs to. Use the cached
    # mapping when available; otherwise scan state_indices.json.
    if ood_class in _OOD_STAGE_BY_CLASS:
        ood_stage = _OOD_STAGE_BY_CLASS[ood_class]
    else:
        ood_stage = _resolve_ood_stage(Path(args.dataset_path), ood_indices)

    # Replace the OOD stage's index pool. We mutate the engine's
    # internal _state_indices directly (the public surface doesn't
    # expose a setter; this is a deliberate F15-only intervention).
    new_indices = [int(i) for i in ood_indices.tolist()]
    env._realization_engine._state_indices[ood_stage] = new_indices  # type: ignore[attr-defined]
    env._realization_engine._stage_counts[ood_stage] = len(new_indices)  # type: ignore[attr-defined]
    env._realization_engine._total_samples = sum(
        env._realization_engine._stage_counts.values()  # type: ignore[attr-defined]
    )
    logger.debug(
        "F15 hybrid realiser: ood_class=%s ood_stage=%d (replaced %d train-pool rows "
        "with %d OOD-class rows; other stages keep train-pool rows)",
        ood_class, ood_stage,
        len(new_indices),
        len(new_indices),
    )

    # 3. Wrap in Monitor + DummyVecEnv (mirror env_factory.make_eval_env).
    return DummyVecEnv([lambda: Monitor(env, filename=None)])


# ---------------------------------------------------------------- argparse


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F15 — OOD-class robustness eval (audit-AF1). "
                    "Rolls every benchmark policy on each held-out OOD attack class.",
    )
    p.add_argument(
        "--ood-classes", nargs="+",
        default=["DDoS-HTTP_Flood", "Mirai-udpplain", "VulnerabilityScan", "XSS"],
        help="Held-out attack classes to evaluate (one outer loop each).",
    )
    p.add_argument(
        "--policies", nargs="+",
        default=[
            "recommended_action", "rf_acting",
            "dqn", "ppo", "a2c",
            "random", "always_observe", "always_block",
        ],
        help="Subset of the 8 benchmark policies to evaluate.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Seeds for non-deterministic policies (RL + random). "
             "Deterministic baselines always use seed=0.",
    )
    p.add_argument(
        "--n-episodes", type=int, default=30,
        help="Episodes per (ood_class, RL_algo, seed) and "
             "per (ood_class, random, seed) cell.",
    )
    p.add_argument(
        "--n-deterministic-episodes", type=int, default=150,
        help="Episodes per (ood_class, deterministic_baseline) cell "
             "(single seed=0).",
    )
    p.add_argument("--phase5-runs", default="runs/blue_team",
                   help="Where the trained blue-team model.zip files live.")
    p.add_argument("--out-root", default="runs/ablation/ood")
    p.add_argument("--generator-path", default="artifacts/generator/red_team")
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument(
        "--rf-path", default="artifacts/detector/random_forest.joblib",
    )
    p.add_argument(
        "--phase6-eval-manifest",
        default="runs/benchmark/eval_manifest.json",
        help="Upstream benchmark eval manifest, hash-pinned in the F15 manifest "
             "for the SHA-256 reproducibility chain (D7.7).",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 1 OOD class × 2 policies × 1 seed × 2 ep.")
    p.add_argument("--verbose", type=int, default=1)
    return p


# ---------------------------------------------------------------- per-cell


_RL_ALGOS = {"dqn", "ppo", "a2c"}
_DETERMINISTIC_BASELINES = {
    "always_observe", "always_block", "recommended_action", "rf_acting",
}


def _roll_rl(
    args: argparse.Namespace,
    ood_class: str,
    algo: str,
    seed: int,
) -> Dict[str, Any]:
    """One (ood_class, RL_algo, seed) cell."""
    model_path = Path(args.blue_team_runs) / algo / f"seed_{seed}" / "model.zip"
    out_dir = Path(args.out_root) / ood_class / algo / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    run_id = f"f15_{ood_class}_{algo}_seed_{seed}"

    if not model_path.exists():
        msg = f"missing blue-team checkpoint at {model_path}"
        logger.error(msg)
        return {
            "kind": "trained", "ood_class": ood_class, "algo": algo, "seed": seed,
            "run_id": run_id, "ok": False, "error": msg,
            "model_path": str(model_path), "model_sha256": None,
        }

    n_ep = 2 if args.smoke else args.n_episodes
    env = _build_ood_env(args, ood_class, seed=seed)
    try:
        model = _load_sb3_model(algo, model_path, env)
        policy = SB3PolicyAdapter(model, deterministic=True)
        t0 = time.time()
        stats = run_policy(
            policy, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name=algo,
            latency_path=None,  # F15 does not re-measure latency (D7.4)
            seed=seed,
        )
        wallclock = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "kind": "trained",
        "ood_class": ood_class,
        "algo": algo,
        "seed": seed,
        "run_id": run_id,
        "ok": True,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "wallclock_seconds": wallclock,
        **stats,
    }


def _roll_random(
    args: argparse.Namespace,
    ood_class: str,
    seed: int,
) -> Dict[str, Any]:
    """One (ood_class, random, seed) cell."""
    out_dir = Path(args.out_root) / ood_class / "random" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    run_id = f"f15_{ood_class}_random_seed_{seed}"

    n_ep = 2 if args.smoke else args.n_episodes
    rng = np.random.default_rng(seed)

    def _seeded_random(obs: np.ndarray, info: Dict[str, Any]) -> int:
        return random_policy(obs, info, rng=rng)

    env = _build_ood_env(args, ood_class, seed=seed)
    try:
        t0 = time.time()
        stats = run_policy(
            _seeded_random, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name="random",
            latency_path=None,
            seed=seed,
        )
        wallclock = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "kind": "baseline",
        "ood_class": ood_class,
        "policy": "random",
        "seed": seed,
        "run_id": run_id,
        "ok": True,
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "wallclock_seconds": wallclock,
        **stats,
    }


def _roll_deterministic(
    args: argparse.Namespace,
    ood_class: str,
    policy_name: str,
) -> Dict[str, Any]:
    """One (ood_class, deterministic_baseline) cell. Single seed=0, n=150."""
    out_dir = Path(args.out_root) / ood_class / policy_name / "seed_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    run_id = f"f15_{ood_class}_{policy_name}_seed_0"

    n_ep = 2 if args.smoke else args.n_deterministic_episodes

    if policy_name == "always_observe":
        policy = always_observe
    elif policy_name == "always_block":
        policy = always_block
    elif policy_name == "recommended_action":
        policy = recommended_action_policy
    elif policy_name == "rf_acting":
        rf_path = Path(args.rf_path)
        if not rf_path.exists():
            msg = f"missing RF detector at {rf_path}"
            logger.error(msg)
            return {
                "kind": "baseline", "ood_class": ood_class,
                "policy": policy_name, "seed": 0,
                "run_id": run_id, "ok": False, "error": msg,
                "rf_path": str(rf_path), "rf_sha256": None,
            }
        # Probe the env to discover the obs dim, then compute num_features
        # the same way scripts/benchmark/run_test_eval.py does (environment-design
        # frozen contract).
        spec = _ood_eval_env_spec()
        probe_env = _build_ood_env(args, ood_class, seed=0)
        try:
            obs0 = probe_env.reset()
            obs_dim = int(np.asarray(obs0).reshape(-1).size)
        finally:
            try:
                probe_env.close()
            except Exception:  # noqa: BLE001
                pass
        per_row = obs_dim // spec.window_size
        num_features = per_row // 2 if spec.include_deltas else per_row
        policy = RFActingPolicy(
            rf_path,
            num_features=num_features,
            window_size=spec.window_size,
            include_deltas=spec.include_deltas,
        )
    else:
        raise ValueError(f"unknown deterministic baseline {policy_name!r}")

    env = _build_ood_env(args, ood_class, seed=0)
    try:
        t0 = time.time()
        stats = run_policy(
            policy, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name=policy_name,
            latency_path=None,
            seed=0,
        )
        wallclock = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    out: Dict[str, Any] = {
        "kind": "baseline",
        "ood_class": ood_class,
        "policy": policy_name,
        "seed": 0,
        "run_id": run_id,
        "ok": True,
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "wallclock_seconds": wallclock,
        **stats,
    }
    if policy_name == "rf_acting":
        out["rf_path"] = str(args.rf_path)
        out["rf_sha256"] = _sha256(Path(args.rf_path))
    return out


# ---------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose >= 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.ood_classes = args.ood_classes[:1]
        args.policies = args.policies[:2]
        args.seeds = args.seeds[:1]
        logger.info(
            "SMOKE mode: 1 OOD class × 2 policies × 1 seed × 2 ep "
            "(approx 4 episodes total)."
        )

    t_start = time.time()
    results: List[Dict[str, Any]] = []

    for ood_class in args.ood_classes:
        for policy_name in args.policies:
            if policy_name in _RL_ALGOS:
                for seed in args.seeds:
                    results.append(_roll_rl(args, ood_class, policy_name, seed))
                    logger.info(
                        "F15 cell done: ood=%s algo=%s seed=%d ok=%s wc=%.1fs",
                        ood_class, policy_name, seed,
                        results[-1]["ok"],
                        results[-1].get("wallclock_seconds", 0.0),
                    )
            elif policy_name == "random":
                for seed in args.seeds:
                    results.append(_roll_random(args, ood_class, seed))
                    logger.info(
                        "F15 cell done: ood=%s policy=random seed=%d ok=%s wc=%.1fs",
                        ood_class, seed,
                        results[-1]["ok"],
                        results[-1].get("wallclock_seconds", 0.0),
                    )
            elif policy_name in _DETERMINISTIC_BASELINES:
                results.append(_roll_deterministic(args, ood_class, policy_name))
                logger.info(
                    "F15 cell done: ood=%s policy=%s ok=%s wc=%.1fs",
                    ood_class, policy_name,
                    results[-1]["ok"],
                    results[-1].get("wallclock_seconds", 0.0),
                )
            else:
                logger.warning("unknown policy %r; skipping", policy_name)

    # ---- F15 manifest (D7.7: hash-pin the upstream blue-team + benchmark manifests) ----
    splits_manifest = Path(args.splits_manifest)
    scaler_path = Path(args.dataset_path) / "scaler.joblib"
    rf_path = Path(args.rf_path)
    blue_team_sweep_manifest = Path(args.blue_team_runs) / "sweep_manifest.json"
    benchmark_eval_manifest = Path(args.benchmark_eval_manifest)

    eval_manifest = {
        "schema_version": "1.0",
        "phase": 7,
        "kind": "f15_ood_eval_manifest",
        "audit_finding": "AF1 — promote OOD-class robustness to Tier-1 "
                          "deliverable (2026-04-30 mentor audit).",
        "git_sha": _git_sha(),
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "input_hashes": {
            "splits_manifest": _sha256(splits_manifest),
            "scaler": _sha256(scaler_path),
            "rf_model": _sha256(rf_path),
            "blue_team_sweep_manifest": _sha256(blue_team_sweep_manifest),
            "benchmark_eval_manifest": _sha256(benchmark_eval_manifest),
        },
        "ood_classes": list(args.ood_classes),
        "policies": list(args.policies),
        "eval_env": {
            "exclude_ood": False,  # F15 specifically: keep OOD rows alive
            "window_size": _ood_eval_env_spec().window_size,
            "include_deltas": _ood_eval_env_spec().include_deltas,
            "max_steps": _ood_eval_env_spec().max_steps,
            "min_episode_length": _ood_eval_env_spec().min_episode_length,
            "p_defender_deescalation": _ood_eval_env_spec().p_defender_deescalation,
            "impact_is_terminal": _ood_eval_env_spec().impact_is_terminal,
            "_note": "environment-design frozen reward config (D7.4) — F15 isolates "
                      "the generalisation axis from the reward-shaping axis.",
        },
        "runs": results,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_failed": sum(1 for r in results if not r.get("ok")),
    }
    manifest_path = out_root / "eval_manifest.json"
    manifest_path.write_text(json.dumps(eval_manifest, indent=2))
    logger.info(
        "F15 OOD eval done: %d ok / %d failed in %.1fs; manifest -> %s",
        eval_manifest["n_ok"], eval_manifest["n_failed"],
        eval_manifest["wallclock_seconds"], manifest_path,
    )

    return 0 if eval_manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
