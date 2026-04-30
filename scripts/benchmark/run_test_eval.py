"""Phase-6 test-split evaluation sweeper (PLAN §3.1.4, C3).

Rolls every Phase-5 trained checkpoint and every non-RL baseline on the
**held-out ``test_balanced`` split** (D6.2) and writes:

- ``runs/phase6/<policy>/seed_<k>/eval_test.jsonl``  — schema-v1.0
  EpisodeRecord JSONL (one line per deterministic eval episode).
- ``runs/phase6/<policy>/seed_<k>/latency.jsonl``    — sidecar per-step
  inference duration in nanoseconds (used by F7).
- ``runs/phase6/eval_manifest.json`` — top-level manifest with
  SHA-256 hashes of every Phase-5 checkpoint, the RF model, the
  scaler, the splits manifest, plus the git SHA at production time.
  This is the input artefact every Phase-6 figure manifest will hash
  by reference (G6.7 / D6.9).

Usage::

    python -m scripts.benchmark.run_test_eval \\
        [--algos dqn ppo a2c] [--seeds 0 1 2 3 4] \\
        [--n-episodes 30] \\
        [--phase5-runs-root runs/phase5] \\
        [--out-root runs/phase6] \\
        [--rf-path artifacts/detector/random_forest.joblib] \\
        [--smoke]

Per D6.3 the default sweep is:

- 3 RL algos × 5 seeds × 30 deterministic episodes  = 450 episodes total
  (15 checkpoints rolled, one per (algo, seed)).
- 1 random-policy × 5 seeds × 30 episodes           = 150 episodes total.
- 4 deterministic baselines × 1 seed × 150 episodes = 600 episodes total
  (always_observe / always_block / recommended_action / rf_acting).

Total: ~1200 episodes, expected wallclock < 10 minutes on Apple silicon
CPU with the production env (max_steps=100).

The sweeper deliberately does NOT use subprocesses (cf. Phase 5's
``run_phase5.py``): we do not need clean PyTorch state per run because
no training happens, and a single-process sweep produces hashable
``runs/phase6/eval_manifest.json`` in one atomic write.

If ``--smoke`` is passed, the sweep shrinks to 1 algo × 1 seed × 2
episodes (and 2 episodes for each baseline) so CI / smoke runs verify
the wiring without burning CPU.
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
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.benchmark.run_test_eval")

_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------- helpers


def _sha256(path: Path) -> Optional[str]:
    """Return SHA-256 hex of ``path`` content; ``None`` if missing.

    Files are streamed in 1 MiB chunks so 100 MB+ checkpoints don't
    blow up memory. Returning ``None`` (rather than raising) lets
    optional inputs (e.g., the RF model when a baseline is skipped)
    be missing without aborting the sweep — the manifest records
    the absence faithfully.
    """
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    """Best-effort current git commit SHA.

    Falls back to ``"unknown"`` so the sweeper never crashes on a
    detached worktree or a stripped tarball.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001 — env-dependent
        return "unknown"


def _load_sb3_model(algo: str, model_path: Path, env: Any) -> Any:
    """Dispatch ``DQN/PPO/A2C.load(model_path, env=env)``.

    Phase 5 saves with the matching algo's ``.save()``; loading must
    round-trip with the same class. Importing inside the function
    keeps stable_baselines3 out of the module-import cost when
    callers only want the baseline-policy entrypoints.
    """
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


def _eval_env_spec() -> EnvConfigSerializable:
    """Phase-6 eval env spec: held-out test_balanced split (D6.2).

    Reward-shaping fields stay at the Phase-3 frozen defaults; only the
    split changes vs. Phase 5's ``val_balanced`` eval.
    """
    return EnvConfigSerializable(split="test_balanced", exclude_ood=True)


def _build_eval_env(args: argparse.Namespace, seed: Optional[int] = None) -> Any:
    """Build a fresh eval env on test_balanced for one rollout."""
    return make_eval_env(
        spec=_eval_env_spec(),
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )


# ---------------------------------------------------------------- argparse


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-6 RL benchmark — roll trained Phase-5 checkpoints "
                    "and non-RL baselines on test_balanced.",
    )
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo", "a2c"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--n-episodes", type=int, default=30,
        help="Episodes per (algo, seed) and per random-policy seed (D6.3).",
    )
    p.add_argument(
        "--n-deterministic-episodes", type=int, default=150,
        help="Episodes per deterministic baseline (D6.3); single seed.",
    )
    p.add_argument(
        "--phase5-runs-root", default="runs/phase5",
        help="Where the trained Phase-5 model.zip files live.",
    )
    p.add_argument("--out-root", default="runs/phase6")
    p.add_argument(
        "--generator-path", default="artifacts/generator/phase2",
    )
    p.add_argument(
        "--dataset-path", default="data/processed/ciciot2023",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument(
        "--rf-path", default="artifacts/detector/random_forest.joblib",
    )
    p.add_argument(
        "--baselines", nargs="+",
        default=[
            "random", "always_observe", "always_block",
            "recommended_action", "rf_acting",
        ],
        help="Subset of {random, always_observe, always_block, "
             "recommended_action, rf_acting} to roll. "
             "Pass an empty list to skip all baselines.",
    )
    p.add_argument(
        "--skip-trained", action="store_true",
        help="Skip the Phase-5 trained checkpoints. Useful for "
             "iterating on baselines only.",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 1 algo × 1 seed × 2 ep, 2 ep / baseline.")
    p.add_argument("--verbose", type=int, default=1)
    return p


# ---------------------------------------------------------------- per-run


def _roll_trained(
    args: argparse.Namespace,
    algo: str,
    seed: int,
) -> Dict[str, Any]:
    """Roll one Phase-5 (algo, seed) checkpoint on test_balanced.

    The function is the inner loop's worker; it owns env construction
    and tear-down so a per-run failure can never leak resources.
    """
    model_path = Path(args.phase5_runs_root) / algo / f"seed_{seed}" / "model.zip"
    out_dir = Path(args.out_root) / algo / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    latency_jsonl = out_dir / "latency.jsonl"
    run_id = f"{algo}_seed_{seed}_test"

    if not model_path.exists():
        msg = f"missing Phase-5 checkpoint at {model_path}"
        logger.error(msg)
        return {
            "kind": "trained", "algo": algo, "seed": seed,
            "run_id": run_id, "ok": False, "error": msg,
            "model_path": str(model_path),
            "model_sha256": None,
        }

    n_ep = 2 if args.smoke else args.n_episodes
    env = _build_eval_env(args, seed=seed)
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
            latency_path=latency_jsonl,
            seed=seed,
        )
        wallclock = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001 — best-effort
            pass

    return {
        "kind": "trained",
        "algo": algo,
        "seed": seed,
        "run_id": run_id,
        "ok": True,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "latency_jsonl": str(latency_jsonl),
        "latency_jsonl_sha256": _sha256(latency_jsonl),
        "wallclock_seconds": wallclock,
        **stats,
    }


def _roll_random(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    """Roll the random policy with one seed × n_episodes (D6.3)."""
    out_dir = Path(args.out_root) / "random" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    latency_jsonl = out_dir / "latency.jsonl"
    run_id = f"random_seed_{seed}_test"

    n_ep = 2 if args.smoke else args.n_episodes
    rng = np.random.default_rng(seed)

    def _seeded_random(obs: np.ndarray, info: Dict[str, Any]) -> int:
        # Bind ``rng`` so successive calls share the seeded generator.
        return random_policy(obs, info, rng=rng)

    env = _build_eval_env(args, seed=seed)
    try:
        t0 = time.time()
        stats = run_policy(
            _seeded_random, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name="random",
            latency_path=latency_jsonl,
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
        "policy": "random",
        "seed": seed,
        "run_id": run_id,
        "ok": True,
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "latency_jsonl": str(latency_jsonl),
        "latency_jsonl_sha256": _sha256(latency_jsonl),
        "wallclock_seconds": wallclock,
        **stats,
    }


def _roll_deterministic(
    args: argparse.Namespace,
    policy_name: str,
) -> Dict[str, Any]:
    """Roll a deterministic baseline once (single seed=0, n=150 episodes).

    Per D6.3, deterministic baselines (always-X, recommended-action,
    rf-acting) get one seed × 150 episodes for the same total n=150 as
    the seeded random baseline (5 × 30).
    """
    out_dir = Path(args.out_root) / policy_name / "seed_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    latency_jsonl = out_dir / "latency.jsonl"
    run_id = f"{policy_name}_seed_0_test"

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
                "kind": "baseline", "policy": policy_name, "seed": 0,
                "run_id": run_id, "ok": False, "error": msg,
                "rf_path": str(rf_path),
                "rf_sha256": None,
            }
        # Default env spec: window=5, F=29, deltas=True (Phase-3 frozen).
        spec = _eval_env_spec()
        # F is whatever the env reports at construction; use a probe
        # rollout instead of hard-coding 29 to stay robust to a
        # smaller-feature-matrix split.
        probe_env = _build_eval_env(args, seed=0)
        try:
            obs0 = probe_env.reset()
            obs_dim = int(np.asarray(obs0).reshape(-1).size)
        finally:
            try:
                probe_env.close()
            except Exception:  # noqa: BLE001
                pass
        per_row = obs_dim // spec.window_size
        if spec.include_deltas:
            num_features = per_row // 2
        else:
            num_features = per_row
        policy = RFActingPolicy(
            rf_path,
            num_features=num_features,
            window_size=spec.window_size,
            include_deltas=spec.include_deltas,
        )
    else:
        raise ValueError(f"unknown deterministic baseline {policy_name!r}")

    env = _build_eval_env(args, seed=0)
    try:
        t0 = time.time()
        stats = run_policy(
            policy, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name=policy_name,
            latency_path=latency_jsonl,
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
        "policy": policy_name,
        "seed": 0,
        "run_id": run_id,
        "ok": True,
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "latency_jsonl": str(latency_jsonl),
        "latency_jsonl_sha256": _sha256(latency_jsonl),
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
        # Shrink to a sane smoke configuration; first algo / first seed
        # only, plus the baselines as-given so the wiring is exercised.
        args.algos = args.algos[:1]
        args.seeds = args.seeds[:1]
        logger.info("SMOKE mode: 1 algo × 1 seed × 2 ep + 2 ep per baseline")

    t_start = time.time()
    results: List[Dict[str, Any]] = []

    # ---- trained checkpoints ----
    if not args.skip_trained:
        for algo in args.algos:
            for seed in args.seeds:
                results.append(_roll_trained(args, algo, seed))
                logger.info(
                    "trained run done: algo=%s seed=%d ok=%s wallclock=%.1fs",
                    algo, seed, results[-1]["ok"],
                    results[-1].get("wallclock_seconds", 0.0),
                )
    else:
        logger.info("--skip-trained set; skipping Phase-5 checkpoints")

    # ---- baselines ----
    for name in args.baselines:
        if name == "random":
            for seed in args.seeds:
                results.append(_roll_random(args, seed))
                logger.info(
                    "random seed=%d done: ok=%s wallclock=%.1fs",
                    seed, results[-1]["ok"],
                    results[-1].get("wallclock_seconds", 0.0),
                )
        elif name in {"always_observe", "always_block",
                      "recommended_action", "rf_acting"}:
            results.append(_roll_deterministic(args, name))
            logger.info(
                "%s done: ok=%s wallclock=%.1fs",
                name, results[-1]["ok"],
                results[-1].get("wallclock_seconds", 0.0),
            )
        else:
            logger.warning("unknown baseline %r; skipping", name)

    # ---- top-level manifest ----
    splits_manifest = Path(args.splits_manifest)
    scaler_path = Path(args.dataset_path) / "scaler.joblib"
    rf_path = Path(args.rf_path)

    eval_manifest = {
        "schema_version": "1.0",
        "phase": 6,
        "kind": "eval_manifest",
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
        },
        "eval_env": {
            "split": _eval_env_spec().split,
            "exclude_ood": _eval_env_spec().exclude_ood,
            "window_size": _eval_env_spec().window_size,
            "include_deltas": _eval_env_spec().include_deltas,
            "max_steps": _eval_env_spec().max_steps,
            "min_episode_length": _eval_env_spec().min_episode_length,
            "p_defender_deescalation": _eval_env_spec().p_defender_deescalation,
        },
        "runs": results,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_failed": sum(1 for r in results if not r.get("ok")),
    }
    manifest_path = out_root / "eval_manifest.json"
    manifest_path.write_text(json.dumps(eval_manifest, indent=2))
    logger.info(
        "phase-6 eval sweep done: %d ok / %d failed in %.1fs; manifest -> %s",
        eval_manifest["n_ok"], eval_manifest["n_failed"],
        eval_manifest["wallclock_seconds"], manifest_path,
    )

    return 0 if eval_manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
