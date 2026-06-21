"""Coupled-vs-decoupled reward ablation driver — the reward-design control.

Motivation
----------
The environment's *coupled* reward shapes every step by ``d = action -
recommended(stage)``: matching the hidden stage's recommended action earns a
proportionality bonus. Because the attacker dynamics are *also* keyed on the
same ``d``, the coupled task collapses to 5-way stage classification — a frozen
supervised classifier (RF-Acting) that simply predicts the stage and plays its
recommended action is near-optimal *by construction*, and a model-free RL agent
has nothing to discover beyond imitating that lookup table. Under coupling,
RF-Acting dominating RL is therefore an artefact of the reward design, not
evidence that RL is the wrong tool.

The *outcome* reward removes the per-step proportionality shaping and grades the
agent only on realised outcomes (prevention, compromise, benign-overreaction,
action cost). Under the outcome contract there is no per-step label to imitate,
so a frozen stage classifier no longer enjoys a structural advantage.

This driver runs the **same** train→benchmark pipeline under both reward modes
and reports, per mode, the head-to-head reward gap between RF-Acting (the
strongest deployable supervised baseline) and the best RL agent. The headline
read is the *change in that gap* between the coupled and outcome contracts:

    coupled :  RF-Acting >> best RL      (the mis-posed task)
    outcome :  gap shrinks / reverses    (RL is necessary)

Layout::

    runs/ablation/reward_coupling/
        sweep_manifest.json
        <mode>/                       # mode in {coupled, outcome}
            <algo>/seed_<k>/
                episodes.jsonl / eval.jsonl / run_manifest.json
                model.zip / best_model.zip
                eval_test.jsonl       # RL agent on test_balanced under <mode>
            rf_acting/
                eval_test.jsonl       # RF-Acting on test_balanced under <mode>

Both modes are run at the SAME operating point (proximity_coupled,
impact_is_terminal) so the only difference is the reward contract.

Usage::

    python -m scripts.ablation.run_reward_coupling \\
        [--modes coupled outcome] [--algos dqn ppo a2c] \\
        [--seeds 0 1 2 ...] \\
        [--total-timesteps 1000000] [--parallel 10] [--smoke]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark.baseline_policies import RFActingPolicy, SB3PolicyAdapter
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.ablation.run_reward_coupling")

_ROOT = Path(__file__).resolve().parents[2]

# The two reward contracts under test. ``coupled`` is the legacy per-step
# proportionality reward keyed on recommended(stage); ``outcome`` strips that
# shaping and grades only realised outcomes.
_DEFAULT_MODES: tuple[str, ...] = ("coupled", "outcome")
_DEFAULT_ALGOS: tuple[str, ...] = ("dqn", "ppo", "a2c")
_RL_ALGOS: frozenset[str] = frozenset(_DEFAULT_ALGOS)


def _sha256(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:  # noqa: SIM105
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _resolve_checkpoint(run_root: Path) -> Path:
    """Prefer the best eval checkpoint; fall back to the last model."""
    best = run_root / "best_model.zip"
    return best if best.exists() else run_root / "model.zip"


# ----------------------------------------------------------------- RL per cell


def _train_one_rl(args: argparse.Namespace, mode: str, algo: str, seed: int) -> dict[str, Any]:
    """Spawn a single ``train_agent`` run under reward_mode=<mode>."""
    out_dir = Path(args.out_root) / mode / algo / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    overrides: dict[str, Any] = {"reward_mode": mode}
    # Partial-observability redesign regime (must match the eval spec for parity).
    overrides["aliasing_rate"] = float(getattr(args, "aliasing_rate", 0.0))
    overrides["session_coherent"] = bool(getattr(args, "session_coherent", False))
    overrides["no_post_transition_leak"] = bool(getattr(args, "no_post_transition_leak", False))
    overrides["proximity_coupled"] = bool(getattr(args, "proximity_coupled", True))
    overrides["proximity_min_escalation"] = float(getattr(args, "proximity_min_escalation", 0.4))

    cmd: list[str] = [
        sys.executable,
        "-m",
        "scripts.blue_team.train_agent",
        "--algo",
        algo,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(args.total_timesteps),
        "--eval-freq",
        str(args.eval_freq),
        "--n-eval-episodes",
        str(args.n_eval_episodes),
        "--out-dir",
        str(out_dir),
        "--dataset-path",
        args.dataset_path,
        "--splits-manifest",
        args.splits_manifest,
        "--reward-overrides",
        json.dumps(overrides),
        "--impact-is-terminal",
        "false",
        "--verbose",
        "0",
    ]
    if args.smoke:
        cmd.append("--smoke")

    logger.info("coupling mode=%s algo=%s seed=%d → %s", mode, algo, seed, out_dir)
    t0 = time.time()
    with log_path.open("w") as log_fh:
        proc = subprocess.run(cmd, cwd=_ROOT, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
    wallclock = time.time() - t0
    ok = proc.returncode == 0

    test_eval_jsonl = out_dir / "eval_test.jsonl"
    test_eval_ok = False
    if ok:
        try:  # noqa: SIM105
            _eval_rl_on_test(args, mode, algo, seed, out_dir, test_eval_jsonl)
            test_eval_ok = test_eval_jsonl.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("mode=%s algo=%s seed=%d test-eval failed: %s", mode, algo, seed, exc)

    return {
        "mode": mode,
        "policy": algo,
        "seed": seed,
        "ok_train": ok,
        "ok_test_eval": test_eval_ok,
        "wallclock_seconds": wallclock,
        "out_dir": str(out_dir),
        "checkpoint": str(_resolve_checkpoint(out_dir)),
        "checkpoint_sha256": _sha256(_resolve_checkpoint(out_dir)),
        "test_eval_jsonl": str(test_eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(test_eval_jsonl),
        "returncode": proc.returncode,
    }


def _eval_spec_for_mode(
    mode: str,
    *,
    aliasing_rate: float = 0.0,
    session_coherent: bool = False,
    no_post_transition_leak: bool = False,
    proximity_coupled: bool = True,
    proximity_min_escalation: float = 0.4,
) -> EnvConfigSerializable:
    """Benchmark spec on test_balanced under the given reward contract.

    Mirrors the held-out benchmark operating point so coupling-ablation rewards
    are commensurable with F5: same split, impact non-terminal.
    The five partial-observability redesign fields MUST match the trained
    checkpoints' contract or the eval measures a different MDP.
    """
    return EnvConfigSerializable(
        split="test_balanced",
        exclude_ood=True,
        impact_is_terminal=False,
        reward_mode=mode,
        aliasing_rate=aliasing_rate,
        session_coherent=session_coherent,
        no_post_transition_leak=no_post_transition_leak,
        proximity_coupled=proximity_coupled,
        proximity_min_escalation=proximity_min_escalation,
    )


def _eval_rl_on_test(
    args: argparse.Namespace,
    mode: str,
    algo: str,
    seed: int,
    out_dir: Path,
    eval_jsonl_path: Path,
) -> None:
    """Roll the best checkpoint on test_balanced under reward_mode=<mode>."""
    spec = _eval_spec_for_mode(
        mode,
        aliasing_rate=getattr(args, "aliasing_rate", 0.0),
        session_coherent=getattr(args, "session_coherent", False),
        no_post_transition_leak=getattr(args, "no_post_transition_leak", False),
        proximity_coupled=getattr(args, "proximity_coupled", True),
        proximity_min_escalation=getattr(args, "proximity_min_escalation", 0.4),
    )
    env = make_eval_env(
        spec=spec,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )
    try:
        ckpt = _resolve_checkpoint(out_dir)
        a = algo.lower()
        if a == "ppo":
            from stable_baselines3 import PPO

            model = PPO.load(ckpt, env=env, device="cpu")
        elif a == "dqn":
            from stable_baselines3 import DQN

            model = DQN.load(ckpt, env=env, device="cpu")
        elif a == "a2c":
            from stable_baselines3 import A2C

            model = A2C.load(ckpt, env=env, device="cpu")
        else:
            raise ValueError(f"unknown algo {algo!r}")
        policy = SB3PolicyAdapter(model, deterministic=True)
        run_policy(
            policy,
            env,
            n_episodes=2 if args.smoke else args.n_deterministic_episodes,
            jsonl_path=eval_jsonl_path,
            run_id=f"coupling_{mode}_{algo}_seed_{seed}_test",
            policy_name=algo,
            latency_path=None,
            seed=seed,
        )
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------- RF-Acting cell


def _eval_rf_acting(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    """Roll RF-Acting on test_balanced under reward_mode=<mode>.

    RF-Acting is not trained on reward, so a single deterministic roll under
    each reward contract is the fair comparison: the SAME frozen classifier is
    scored by the coupled vs outcome reward function.
    """
    out_dir = Path(args.out_root) / mode / "rf_acting"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl_path = out_dir / "eval_test.jsonl"

    spec = _eval_spec_for_mode(
        mode,
        aliasing_rate=getattr(args, "aliasing_rate", 0.0),
        session_coherent=getattr(args, "session_coherent", False),
        no_post_transition_leak=getattr(args, "no_post_transition_leak", False),
        proximity_coupled=getattr(args, "proximity_coupled", True),
        proximity_min_escalation=getattr(args, "proximity_min_escalation", 0.4),
    )
    env = make_eval_env(
        spec=spec,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=0,
    )
    ok = False
    try:
        import joblib

        rf = joblib.load(args.rf_path)
        # Probe per-row feature width from the observation space.
        obs_dim = int(env.observation_space.shape[0])  # type: ignore[union-attr]
        per_row = obs_dim // spec.window_size
        num_features = per_row // 2 if spec.include_deltas else per_row
        policy = RFActingPolicy(
            rf,
            num_features=num_features,
            window_size=spec.window_size,
            include_deltas=spec.include_deltas,
        )
        run_policy(
            policy,
            env,
            n_episodes=2 if args.smoke else args.n_deterministic_episodes,
            jsonl_path=eval_jsonl_path,
            run_id=f"coupling_{mode}_rf_acting_test",
            policy_name="rf_acting",
            latency_path=None,
            seed=0,
        )
        ok = eval_jsonl_path.exists()
    except Exception as exc:  # noqa: BLE001
        logger.error("mode=%s rf_acting eval failed: %s", mode, exc)
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "mode": mode,
        "policy": "rf_acting",
        "seed": 0,
        "ok_train": True,  # frozen classifier; no training
        "ok_test_eval": ok,
        "out_dir": str(out_dir),
        "test_eval_jsonl": str(eval_jsonl_path),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl_path),
        "rf_path": str(args.rf_path),
        "rf_sha256": _sha256(Path(args.rf_path)),
    }


# ----------------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Coupled-vs-decoupled reward ablation: train+benchmark RL "
        "and RF-Acting under reward_mode in {coupled, outcome} and report the "
        "RF-minus-RL reward gap per mode (the reward-design control).",
    )
    p.add_argument("--modes", nargs="+", default=list(_DEFAULT_MODES))
    p.add_argument("--algos", nargs="+", default=list(_DEFAULT_ALGOS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Partial-observability redesign regime (applied to BOTH train + eval specs).
    p.add_argument("--aliasing-rate", type=float, default=0.0)
    p.add_argument("--session-coherent", action="store_true")
    p.add_argument("--no-post-transition-leak", action="store_true")
    p.add_argument("--proximity-coupled", action="store_true")
    p.add_argument("--proximity-min-escalation", type=float, default=0.4)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--n-deterministic-episodes", type=int, default=300)
    p.add_argument("--out-root", default="runs/ablation/reward_coupling")
    p.add_argument("--rf-path", default="artifacts/detector/random_forest.joblib")
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.modes = list(args.modes)[:2]
        args.algos = list(args.algos)[:1]
        args.seeds = list(args.seeds)[:1]
        logger.info("SMOKE: %s × %s × seed %s", args.modes, args.algos, args.seeds)

    # Build the RL training grid (mode × algo × seed).
    rl_grid = [
        (mode, algo, seed) for mode in args.modes for algo in args.algos for seed in args.seeds
    ]
    logger.info(
        "coupling ablation: %d modes × %d algos × %d seeds = %d RL runs + "
        "%d RF-Acting evals (%d worker(s))",
        len(args.modes),
        len(args.algos),
        len(args.seeds),
        len(rl_grid),
        len(args.modes),
        args.parallel,
    )

    t_start = time.time()
    results: list[dict[str, Any]] = []

    if args.parallel <= 1:
        for mode, algo, seed in rl_grid:
            results.append(_train_one_rl(args, mode, algo, seed))
            if not results[-1]["ok_train"] and not args.continue_on_failure:
                logger.error("run failed; aborting (use --continue-on-failure)")
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {
                ex.submit(_train_one_rl, args, mode, algo, seed): (mode, algo, seed)
                for mode, algo, seed in rl_grid
            }
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    # RF-Acting under each mode (frozen classifier; one deterministic roll).
    for mode in args.modes:
        results.append(_eval_rf_acting(args, mode))

    sweep_manifest = {
        "schema_version": "1.0",
        "kind": "reward_coupling_ablation_manifest",
        "git_sha": _git_sha(),
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "modes": list(args.modes),
        "algos": list(args.algos),
        "rl_algos": sorted(_RL_ALGOS),
        "runs": results,
        "n_ok_train": sum(1 for r in results if r["ok_train"]),
        "n_ok_test_eval": sum(1 for r in results if r["ok_test_eval"]),
        "n_failed": sum(1 for r in results if not r["ok_train"]),
    }
    manifest_path = out_root / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(sweep_manifest, indent=2))
    logger.info(
        "coupling ablation done: %d/%d ok in %.1fs; manifest -> %s",
        sweep_manifest["n_ok_test_eval"],
        len(results),
        sweep_manifest["wallclock_seconds"],
        manifest_path,
    )

    if sweep_manifest["n_failed"] and not args.continue_on_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
