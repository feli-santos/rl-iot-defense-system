"""Phase-7 F10 — Attack-aggressiveness sweep driver (PLAN §3.1.5).

Sweeps ``p_defender_deescalation ∈ {0.0, 0.2, 0.4, 0.6 (default), 0.8,
1.0}`` × PPO only (D7.2) × 5 seeds × 250K timesteps (D7.8). Total:
6 × 5 = **30 runs ≈ 1.5 h CPU walk-away**.

Aligned with IoTWarden Fig. 6 (Bhattacharjee et al., 2023): how does
the defender's value function shift as the attacker's aggressiveness
varies? p=0.0 means the defender NEVER wins a de-escalation roll
(harshest attacker); p=1.0 means the defender ALWAYS wins (easiest
attacker).

Each cell trains PPO with ``--p-defender-deescalation P`` and
evaluates on test_balanced AT THE SAME P. The recommended-action
oracle baseline is rolled separately under each p as a reference
curve (its mean reward shifts with p because the realiser's attack
success rate shifts even though the rule's behaviour does not).

Output layout::

    runs/phase7/aggressiveness/
        sweep_manifest.json           — top-level F10 manifest
        ppo_p<p>/seed_<k>/
            episodes.jsonl
            eval.jsonl
            run_manifest.json
            model.zip
            eval_test.jsonl           — test_balanced at this p
            train.log
        rule_p<p>/seed_0/
            eval_test.jsonl           — oracle rule at this p
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
from typing import Any, Dict, List, Optional

from src.benchmark.baseline_policies import (
    SB3PolicyAdapter,
    recommended_action_policy,
)
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.ablation.run_aggressiveness_sweep")

_ROOT = Path(__file__).resolve().parents[2]


_DEFAULT_P_VALUES: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


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


def _p_slug(p: float) -> str:
    """Filesystem-safe slug for a probability value, e.g. 0.6 -> '0p6'."""
    return f"{p:.1f}".replace(".", "p")


# --------------------------------------------------------------------- per-cell


def _train_ppo(
    args: argparse.Namespace, p: float, seed: int,
) -> Dict[str, Any]:
    """Train PPO at p_defender_deescalation=p for one seed."""
    out_dir = Path(args.out_root) / f"ppo_p{_p_slug(p)}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        sys.executable, "-m", "scripts.blue_team.train_agent",
        "--algo", "ppo",
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--eval-freq", str(args.eval_freq),
        "--n-eval-episodes", str(args.n_eval_episodes),
        "--out-dir", str(out_dir),
        "--generator-path", args.generator_path,
        "--dataset-path", args.dataset_path,
        "--splits-manifest", args.splits_manifest,
        "--p-defender-deescalation", str(p),
        "--verbose", "0",
    ]
    if args.smoke:
        cmd.append("--smoke")

    logger.info("F10 ppo p=%.1f seed=%d → %s", p, seed, out_dir)
    t0 = time.time()
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            cmd, cwd=_ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
            check=False,
        )
    wallclock = time.time() - t0
    ok = proc.returncode == 0

    # Test-eval at the same p.
    test_eval_jsonl = out_dir / "eval_test.jsonl"
    test_eval_ok = False
    if ok:
        try:
            _eval_ppo_on_test(args, p, seed, out_dir, test_eval_jsonl)
            test_eval_ok = test_eval_jsonl.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("F10 ppo p=%.1f seed=%d test-eval failed: %s",
                         p, seed, exc)

    logger.info(
        "F10 ppo p=%.1f seed=%d done train=%s test_eval=%s wc=%.1fs",
        p, seed, ok, test_eval_ok, wallclock,
    )

    return {
        "kind": "ppo",
        "p_defender_deescalation": p,
        "seed": seed,
        "ok_train": ok,
        "ok_test_eval": test_eval_ok,
        "wallclock_seconds": wallclock,
        "out_dir": str(out_dir),
        "model_path": str(out_dir / "model.zip"),
        "model_sha256": _sha256(out_dir / "model.zip"),
        "test_eval_jsonl": str(test_eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(test_eval_jsonl),
        "log_path": str(log_path),
        "returncode": proc.returncode,
    }


def _eval_ppo_on_test(
    args: argparse.Namespace, p: float, seed: int,
    out_dir: Path, eval_jsonl_path: Path,
) -> None:
    """Roll the trained PPO at this p on test_balanced (same p)."""
    spec = EnvConfigSerializable(
        split="test_balanced", exclude_ood=True,
        p_defender_deescalation=p,
    )
    env = make_eval_env(
        spec=spec,
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )
    try:
        from stable_baselines3 import PPO
        model = PPO.load(out_dir / "model.zip", env=env, device="cpu")
        policy = SB3PolicyAdapter(model, deterministic=True)
        run_policy(
            policy, env,
            n_episodes=2 if args.smoke else args.n_eval_episodes,
            jsonl_path=eval_jsonl_path,
            run_id=f"f10_ppo_p{_p_slug(p)}_seed_{seed}_test",
            policy_name="ppo",
            latency_path=None,
            seed=seed,
        )
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _roll_rule_baseline(
    args: argparse.Namespace, p: float,
) -> Dict[str, Any]:
    """Roll the recommended-action oracle baseline at this p (single
    seed=0, n=150 ep — same as Phase-6 D6.3 protocol).

    The rule's behaviour doesn't depend on p — but the realiser's
    attacker success rate does, so the rule's *mean reward* shifts
    with p. F10 wants both curves on the same axes.
    """
    out_dir = Path(args.out_root) / f"rule_p{_p_slug(p)}" / "seed_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    run_id = f"f10_rule_p{_p_slug(p)}_seed_0"

    spec = EnvConfigSerializable(
        split="test_balanced", exclude_ood=True,
        p_defender_deescalation=p,
    )
    env = make_eval_env(
        spec=spec,
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=0,
    )
    n_ep = 2 if args.smoke else args.n_deterministic_episodes
    try:
        t0 = time.time()
        run_policy(
            recommended_action_policy, env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name="recommended_action",
            latency_path=None,
            seed=0,
        )
        wallclock = time.time() - t0
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "kind": "rule",
        "p_defender_deescalation": p,
        "seed": 0,
        "ok": True,
        "wallclock_seconds": wallclock,
        "test_eval_jsonl": str(eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl),
    }


# ---------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 F10 — attack-aggressiveness sweep "
                    "(PLAN §3.1.5; PPO + oracle rule × 6 p values × 5 seeds, ~1.5 h CPU).",
    )
    p.add_argument(
        "--p-values", nargs="+", type=float, default=_DEFAULT_P_VALUES,
        help="p_defender_deescalation values to sweep.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="PPO seeds (recommended-action rule uses single seed=0).",
    )
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--n-deterministic-episodes", type=int, default=150,
                   help="Episodes for the recommended-action rule per p.")
    p.add_argument("--out-root", default="runs/phase7/aggressiveness")
    p.add_argument("--generator-path", default="artifacts/generator/phase2")
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument(
        "--parallel", type=int, default=1,
        help="Concurrent train subprocesses (default 1 = serial).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.p_values = args.p_values[:2]
        args.seeds = args.seeds[:1]

    t_start = time.time()
    ppo_results: List[Dict[str, Any]] = []
    rule_results: List[Dict[str, Any]] = []

    # PPO sweep.
    grid = [(p, s) for p in args.p_values for s in args.seeds]
    if args.parallel <= 1:
        for p, seed in grid:
            ppo_results.append(_train_ppo(args, p, seed))
            if not ppo_results[-1]["ok_train"] and not args.continue_on_failure:
                logger.error("aborting sweep (use --continue-on-failure)")
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_train_ppo, args, p, s): (p, s) for p, s in grid}
            for fut in concurrent.futures.as_completed(futs):
                ppo_results.append(fut.result())

    # Rule baseline at each p.
    for p in args.p_values:
        rule_results.append(_roll_rule_baseline(args, p))
        logger.info(
            "F10 rule p=%.1f done wc=%.1fs",
            p, rule_results[-1]["wallclock_seconds"],
        )

    sweep_manifest = {
        "schema_version": "1.0",
        "phase": 7,
        "kind": "f10_aggressiveness_sweep_manifest",
        "git_sha": _git_sha(),
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "p_values": list(args.p_values),
        "ppo_runs": ppo_results,
        "rule_runs": rule_results,
        "n_ppo_ok": sum(1 for r in ppo_results if r["ok_train"]),
        "n_ppo_failed": sum(1 for r in ppo_results if not r["ok_train"]),
    }
    sweep_manifest_path = out_root / "sweep_manifest.json"
    sweep_manifest_path.write_text(json.dumps(sweep_manifest, indent=2))
    logger.info(
        "F10 sweep done: %d/%d ppo trained, %d rule p-values rolled in %.1fs; -> %s",
        sweep_manifest["n_ppo_ok"], len(ppo_results),
        len(rule_results), sweep_manifest["wallclock_seconds"],
        sweep_manifest_path,
    )
    if sweep_manifest["n_ppo_failed"] and not args.continue_on_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
