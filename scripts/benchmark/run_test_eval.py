"""benchmark test-split evaluation sweeper (PLAN §3.1.4).

Rolls every Blue-Team trained checkpoint and every non-RL baseline on the
**held-out ``test_balanced`` split** (D6.2) and writes:

- ``runs/benchmark/<policy>/seed_<k>/eval_test.jsonl``  — schema-v1.0
  EpisodeRecord JSONL (one line per deterministic eval episode).
- ``runs/benchmark/<policy>/seed_<k>/latency.jsonl``    — sidecar per-step
  inference duration in nanoseconds (used by F7).
- ``runs/benchmark/eval_manifest.json`` — top-level manifest with
  SHA-256 hashes of every Blue-Team checkpoint, the RF model, the
  scaler, the splits manifest, plus the git SHA at production time.
  This is the input artefact every benchmark figure manifest will hash
  by reference (G6.7 / D6.9).

Usage::

    python -m scripts.benchmark.run_test_eval \\
        [--algos dqn ppo a2c] [--seeds 0 1 2 3 4] \\
        [--n-episodes 30] \\
        [--blue-team-runs-root runs/blue_team] \\
        [--out-root runs/benchmark] \\
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

The sweeper deliberately does NOT use subprocesses (cf. Blue-Team Training's
trainer): we do not need clean PyTorch state per run because
no training happens, and a single-process sweep produces hashable
``runs/benchmark/eval_manifest.json`` in one atomic write.

If ``--smoke`` is passed, the sweep shrinks to 1 algo × 1 seed × 2
episodes (and 2 episodes for each baseline) so CI / smoke runs verify
the wiring without burning CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _sha256(path: Path) -> str | None:
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
    try:  # noqa: SIM105
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001 — env-dependent
        return "unknown"


def _load_sb3_model(algo: str, model_path: Path, env: Any) -> Any:
    """Dispatch ``DQN/PPO/A2C.load(model_path, env=env)``.

    Blue-Team Training saves with the matching algo's ``.save()``; loading must
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


def _eval_env_spec(
    reward_mode: str = "outcome",
    *,
    aliasing_rate: float = 0.0,
    session_coherent: bool = False,
    no_post_transition_leak: bool = False,
    proximity_coupled: bool = True,
    proximity_min_escalation: float = 0.4,
) -> EnvConfigSerializable:
    """benchmark eval env spec: held-out test_balanced split (D6.2).

    Reward-shaping fields stay at the environment-design frozen defaults; only the
    split changes vs. Blue-Team's ``val_balanced`` eval.

    ``impact_is_terminal=False`` is set explicitly to match the training
    contract: agents are trained with ``impact_is_terminal=False`` (the primary
    reward contract), so the eval env must terminate IMPACT the same way.
    Without this the eval env would default to ``True`` and silently evaluate
    under a different terminal contract than training.

    ``reward_mode`` MUST match the training contract too (the primary deployment
    contract is ``"outcome"``). If agents were trained ``outcome`` but evaluated
    ``coupled``, the reported reward would measure a different objective than the
    one optimised. :func:`_assert_train_eval_contract` cross-checks this against
    each training manifest.

    The partial-observability redesign fields (``aliasing_rate``,
    ``session_coherent``, ``no_post_transition_leak``, ``proximity_coupled``,
    ``proximity_min_escalation``) MUST match the training contract: agents trained
    under observation aliasing / proximity-coupled dynamics must be benchmarked on
    the same regime, or the held-out reward measures a different (easier or harder)
    MDP than the one optimised.
    """
    return EnvConfigSerializable(
        split="test_balanced",
        exclude_ood=True,
        impact_is_terminal=False,
        reward_mode=reward_mode,
        aliasing_rate=aliasing_rate,
        session_coherent=session_coherent,
        no_post_transition_leak=no_post_transition_leak,
        proximity_coupled=proximity_coupled,
        proximity_min_escalation=proximity_min_escalation,
    )


# Env contract fields the eval env must share with the training run, or the
# eval number measures a different MDP than the one the agent optimised.
_BENCHMARK_PARITY_FIELDS = (
    "exclude_ood",
    "impact_is_terminal",
    "reward_mode",
    "tug_of_war",
    "p_onset",
    "p_onset_access",
    "p_down",
    "p_up",
    "p_down_isolate",
    "action_cost_scale",
    "aliasing_rate",
    "session_coherent",
    "no_post_transition_leak",
    "proximity_coupled",
    "proximity_min_escalation",
)


def _eval_env_spec_from_args(args: argparse.Namespace) -> EnvConfigSerializable:
    """Build the eval env spec from CLI args, threading the redesign contract."""
    return _eval_env_spec(
        reward_mode=getattr(args, "reward_mode", "outcome"),
        aliasing_rate=getattr(args, "aliasing_rate", 0.0),
        session_coherent=getattr(args, "session_coherent", False),
        no_post_transition_leak=getattr(args, "no_post_transition_leak", False),
        proximity_coupled=getattr(args, "proximity_coupled", False),
        proximity_min_escalation=getattr(args, "proximity_min_escalation", 0.4),
    )


def _assert_train_eval_contract(
    eval_spec: EnvConfigSerializable,
    run_root: Path,
    *,
    algo: str,
    seed: int,
) -> None:
    """Cross-script parity: fail if the benchmark eval contract diverges from
    the contract recorded in the training ``run_manifest.json``.

    This is the cross-process complement to
    :meth:`BlueTeamRunConfig.assert_train_eval_parity` (which only guards
    train-vs-eval *within* one training run). Here we guard
    training-vs-benchmark *across* scripts so the F5 number cannot silently be
    measured under a different attacker budget / reward mode than training.
    """
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        # Pre-manifest checkpoint; nothing to check against. Warn, don't fail.
        logger.warning(
            "no run_manifest.json under %s; cannot verify train/eval contract " "for %s seed %d",
            run_root,
            algo,
            seed,
        )
        return
    train_env = json.loads(manifest_path.read_text()).get("env", {})
    # Drop legacy manifest keys that the current EnvConfigSerializable no longer
    # accepts (e.g. the removed attacker-budget mechanism: attacker_budget,
    # budget_step_cost, budget_reset_cost, budget_cost_model). These fields are
    # not in _BENCHMARK_PARITY_FIELDS, so filtering them cannot weaken the parity
    # guard; it only lets us reconstruct the training spec for the fields that
    # still exist. Reconstructing older checkpoints under the current env is
    # intentional here (measuring pre-redesign checkpoints on the current MDP).
    known_fields = {f.name for f in dataclasses.fields(EnvConfigSerializable)}
    unknown_keys = sorted(set(train_env) - known_fields)
    if unknown_keys:
        logger.warning(
            "training manifest %s contains %d legacy env key(s) not in the "
            "current schema; dropping for parity reconstruction: %s",
            manifest_path,
            len(unknown_keys),
            ", ".join(unknown_keys),
        )
    train_env_known = {k: v for k, v in train_env.items() if k in known_fields}
    # Normalise the training reward_mode alias for an apples-to-apples compare.
    train_spec = EnvConfigSerializable(**train_env_known)
    mismatches = []
    for name in _BENCHMARK_PARITY_FIELDS:
        if name not in train_env:
            continue  # legacy manifest without this field; skip
        eval_val = getattr(eval_spec, name)
        train_val = getattr(train_spec, name)
        if eval_val != train_val:
            mismatches.append(f"  {name}: train={train_val!r} eval={eval_val!r}")
    if mismatches:
        raise ValueError(
            f"benchmark eval contract for {algo} seed {seed} diverges from its "
            f"training manifest ({manifest_path}) — the F5 number would measure "
            "a different MDP than training:\n" + "\n".join(mismatches)
        )


def _build_eval_env(args: argparse.Namespace, seed: int | None = None) -> Any:
    """Build a fresh eval env on test_balanced for one rollout."""
    return make_eval_env(
        spec=_eval_env_spec_from_args(args),
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )


# ---------------------------------------------------------------- argparse


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="benchmark RL benchmark — roll trained Blue-Team checkpoints "
        "and non-RL baselines on test_balanced.",
    )
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo", "a2c"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--n-episodes",
        type=int,
        default=30,
        help="Episodes per (algo, seed) and per random-policy seed (D6.3).",
    )
    p.add_argument(
        "--n-deterministic-episodes",
        type=int,
        default=150,
        help="Episodes per deterministic baseline (D6.3); single seed.",
    )
    p.add_argument(
        "--blue-team-runs-root",
        default="runs/blue_team",
        help="Where the trained Blue-Team model.zip files live.",
    )
    p.add_argument("--out-root", default="runs/benchmark")
    p.add_argument(
        "--reward-mode",
        default="outcome",
        choices=["outcome", "outcome_only", "coupled", "proportional"],
        help=(
            "Reward contract for the eval env; MUST match training. Default "
            "'outcome' is the primary deployment contract. Aliases "
            "outcome_only/proportional are normalised."
        ),
    )
    # Partial-observability redesign contract (must match training).
    p.add_argument(
        "--aliasing-rate",
        type=float,
        default=0.0,
        help="Observation aliasing rate alpha; MUST match training. Default 0.0.",
    )
    p.add_argument(
        "--session-coherent",
        action="store_true",
        help="Session-coherent (contiguous, without-replacement) sampling; MUST match training.",
    )
    p.add_argument(
        "--no-post-transition-leak",
        action="store_true",
        help="Sample refreshed obs from the pre-transition stage; MUST match training.",
    )
    p.add_argument(
        "--proximity-coupled",
        action="store_true",
        help="RESTRAIN-style proximity-coupled escalation/prevention; MUST match training.",
    )
    p.add_argument(
        "--proximity-min-escalation",
        type=float,
        default=0.4,
        help="Floor on proximity-scaled escalation; MUST match training. Default 0.4.",
    )
    p.add_argument(
        "--dataset-path",
        default="data/processed/ciciot2023",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument(
        "--rf-path",
        default="artifacts/detector/random_forest.joblib",
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        default=[
            "random",
            "always_observe",
            "always_block",
            "recommended_action",
            "rf_acting",
        ],
        help="Subset of {random, always_observe, always_block, "
        "recommended_action, rf_acting} to roll. "
        "Pass an empty list to skip all baselines.",
    )
    p.add_argument(
        "--skip-trained",
        action="store_true",
        help="Skip the Blue-Team trained checkpoints. Useful for iterating on baselines only.",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Smoke mode: 1 algo × 1 seed × 2 ep, 2 ep / baseline."
    )
    p.add_argument("--verbose", type=int, default=1)
    return p


# ---------------------------------------------------------------- per-run


def _roll_trained(
    args: argparse.Namespace,
    algo: str,
    seed: int,
) -> dict[str, Any]:
    """Roll one Blue-Team (algo, seed) checkpoint on test_balanced.

    The function is the inner loop's worker; it owns env construction
    and tear-down so a per-run failure can never leak resources.
    """
    # Prefer the best-eval checkpoint (written by the training EvalCallback);
    # fall back to the last-model checkpoint for pre-early-stop runs.
    run_root = Path(args.blue_team_runs_root) / algo / f"seed_{seed}"
    best_path = run_root / "best_model.zip"
    model_path = best_path if best_path.exists() else run_root / "model.zip"
    out_dir = Path(args.out_root) / algo / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    latency_jsonl = out_dir / "latency.jsonl"
    run_id = f"{algo}_seed_{seed}_test"

    if not model_path.exists():
        msg = f"missing Blue-Team checkpoint at {model_path}"
        logger.error(msg)
        return {
            "kind": "trained",
            "algo": algo,
            "seed": seed,
            "run_id": run_id,
            "ok": False,
            "error": msg,
            "model_path": str(model_path),
            "model_sha256": None,
        }

    # Cross-script parity: the eval contract must match what this checkpoint was
    # trained under, or the F5 reward measures a different MDP.
    _assert_train_eval_contract(
        _eval_env_spec_from_args(args),
        run_root,
        algo=algo,
        seed=seed,
    )

    n_ep = 2 if args.smoke else args.n_episodes
    env = _build_eval_env(args, seed=seed)
    try:  # noqa: SIM105
        model = _load_sb3_model(algo, model_path, env)
        policy = SB3PolicyAdapter(model, deterministic=True)
        t0 = time.time()
        stats = run_policy(
            policy,
            env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name=algo,
            latency_path=latency_jsonl,
            seed=seed,
        )
        wallclock = time.time() - t0
    finally:
        try:  # noqa: SIM105
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


def _roll_random(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Roll the random policy with one seed × n_episodes (D6.3)."""
    out_dir = Path(args.out_root) / "random" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    latency_jsonl = out_dir / "latency.jsonl"
    run_id = f"random_seed_{seed}_test"

    n_ep = 2 if args.smoke else args.n_episodes
    rng = np.random.default_rng(seed)

    def _seeded_random(obs: np.ndarray, info: dict[str, Any]) -> int:
        # Bind ``rng`` so successive calls share the seeded generator.
        return random_policy(obs, info, rng=rng)

    env = _build_eval_env(args, seed=seed)
    try:  # noqa: SIM105
        t0 = time.time()
        stats = run_policy(
            _seeded_random,
            env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name="random",
            latency_path=latency_jsonl,
            seed=seed,
        )
        wallclock = time.time() - t0
    finally:
        try:  # noqa: SIM105
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
) -> dict[str, Any]:
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
                "kind": "baseline",
                "policy": policy_name,
                "seed": 0,
                "run_id": run_id,
                "ok": False,
                "error": msg,
                "rf_path": str(rf_path),
                "rf_sha256": None,
            }
        # Default env spec: window=5, F=29, deltas=True (environment-design frozen).
        spec = _eval_env_spec_from_args(args)
        # F is whatever the env reports at construction; use a probe
        # rollout instead of hard-coding 29 to stay robust to a
        # smaller-feature-matrix split.
        probe_env = _build_eval_env(args, seed=0)
        try:  # noqa: SIM105
            obs0 = probe_env.reset()
            obs_dim = int(np.asarray(obs0).reshape(-1).size)
        finally:
            try:  # noqa: SIM105
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

    env = _build_eval_env(args, seed=0)
    try:  # noqa: SIM105
        t0 = time.time()
        stats = run_policy(
            policy,
            env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name=policy_name,
            latency_path=latency_jsonl,
            seed=0,
        )
        wallclock = time.time() - t0
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass

    out: dict[str, Any] = {
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


def main(argv: list[str] | None = None) -> int:
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
    results: list[dict[str, Any]] = []

    # ---- trained checkpoints ----
    if not args.skip_trained:
        for algo in args.algos:
            for seed in args.seeds:
                results.append(_roll_trained(args, algo, seed))
                logger.info(
                    "trained run done: algo=%s seed=%d ok=%s wallclock=%.1fs",
                    algo,
                    seed,
                    results[-1]["ok"],
                    results[-1].get("wallclock_seconds", 0.0),
                )
    else:
        logger.info("--skip-trained set; skipping Blue-Team checkpoints")

    # ---- baselines ----
    for name in args.baselines:
        if name == "random":
            for seed in args.seeds:
                results.append(_roll_random(args, seed))
                logger.info(
                    "random seed=%d done: ok=%s wallclock=%.1fs",
                    seed,
                    results[-1]["ok"],
                    results[-1].get("wallclock_seconds", 0.0),
                )
        elif name in {"always_observe", "always_block", "recommended_action", "rf_acting"}:
            results.append(_roll_deterministic(args, name))
            logger.info(
                "%s done: ok=%s wallclock=%.1fs",
                name,
                results[-1]["ok"],
                results[-1].get("wallclock_seconds", 0.0),
            )
        else:
            logger.warning("unknown baseline %r; skipping", name)

    # ---- top-level manifest ----
    splits_manifest = Path(args.splits_manifest)
    scaler_path = Path(args.dataset_path) / "scaler.joblib"
    rf_path = Path(args.rf_path)

    eval_manifest = {
        "schema_version": "1.1",
        "stage": "benchmark",
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
        # serialise the *actual* eval spec used, via asdict() so every
        # EnvConfigSerializable field — evasion_prob, impact_is_terminal,
        # proximity_coupled, … — is faithfully recorded.
        "eval_env": dataclasses.asdict(_eval_env_spec_from_args(args)),
        "runs": results,
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_failed": sum(1 for r in results if not r.get("ok")),
    }
    manifest_path = out_root / "eval_manifest.json"
    manifest_path.write_text(json.dumps(eval_manifest, indent=2))
    logger.info(
        "benchmark eval sweep done: %d ok / %d failed in %.1fs; manifest -> %s",
        eval_manifest["n_ok"],
        eval_manifest["n_failed"],
        eval_manifest["wallclock_seconds"],
        manifest_path,
    )

    return 0 if eval_manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
