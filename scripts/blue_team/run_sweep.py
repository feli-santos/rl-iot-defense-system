"""Blue-team sweep driver: run 3 algos x N seeds via subprocess.

PLAN §3.1.7. Why subprocess and not VecEnv:
- Each (algo, seed) gets a clean Python process, which means a clean
  PyTorch state, no cross-run RNG contamination, and survives a single
  failed run without taking the whole sweep down.
- Per-run JSONLs are independent files we hash-pin in the manifest;
  VecEnv would interleave the logs and we'd have to demultiplex.

Default sweep: ``DQN, PPO, A2C`` x seeds ``{0, 1, 2, 3, 4}`` x
``--total-timesteps`` (default 250 K). With a CPU running ~3 ms/step
this is ~3-7 h depending on the timestep budget.

Usage::

    python -m scripts.blue_team.run_sweep \\
        [--algos dqn ppo a2c] [--seeds 0 1 2 3 4] \\
        [--total-timesteps 250000] [--out-root runs/blue_team] \\
        [--parallel 1] [--smoke]

The driver writes ``runs/blue_team/sweep_manifest.json`` with one entry
per run referencing its ``run_manifest.json``; the figure scripts
consume the sweep manifest as the canonical "what was run".
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("scripts.blue_team.run_sweep")

_ROOT = Path(__file__).resolve().parents[2]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blue-team algo x seed sweep driver.")
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo", "a2c"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-timesteps", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--out-root", default="runs/blue_team")
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent subprocesses (default 1 = serial).",
    )
    p.add_argument(
        "--dataset-path",
        default="data/processed/ciciot2023",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="If a run crashes, log it and keep going. Default: stop.",
    )
    # Allow passing impact_is_terminal and reward overrides to the per-run
    # train_agent subprocess so the sweep driver can launch the primary
    # (impact_is_terminal=False) contract without code duplication.
    p.add_argument(
        "--impact-is-terminal",
        type=str,
        default=None,
        help=(
            "Forwarded to train_agent --impact-is-terminal. "
            "Use 'false' to enable the explicit IMPACT-row decision step. "
            "Default None preserves the environment-design frozen contract (True)."
        ),
    )
    p.add_argument(
        "--reward-overrides",
        type=str,
        default=None,
        help="JSON object forwarded to train_agent --reward-overrides.",
    )
    return p


def _run_one(args: argparse.Namespace, algo: str, seed: int) -> dict:
    """Spawn a single ``python -m scripts.blue_team.train_agent`` subprocess."""
    out_dir = Path(args.out_root) / algo / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists() and not getattr(args, "_force", False):
        logger.info("skipping algo=%s seed=%d (manifest already exists)", algo, seed)
        return {
            "algo": algo,
            "seed": seed,
            "ok": True,
            "wallclock_seconds": 0.0,
            "out_dir": str(out_dir),
            "run_manifest": str(manifest_path),
            "log_path": str(out_dir / "train.log"),
            "returncode": 0,
            "skipped": True,
        }
    log_path = out_dir / "train.log"

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
        "--verbose",
        "0",
    ]
    if args.smoke:
        cmd.append("--smoke")
    # Forward optional reward/env overrides to the subprocess
    if getattr(args, "impact_is_terminal", None) is not None:
        cmd.extend(["--impact-is-terminal", args.impact_is_terminal])
    if getattr(args, "reward_overrides", None) is not None:
        cmd.extend(["--reward-overrides", args.reward_overrides])

    logger.info("starting algo=%s seed=%d -> %s", algo, seed, out_dir)
    t0 = time.time()
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=_ROOT,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wallclock = time.time() - t0
    ok = proc.returncode == 0
    logger.info("done    algo=%s seed=%d ok=%s wallclock=%.1fs", algo, seed, ok, wallclock)

    return {
        "algo": algo,
        "seed": seed,
        "ok": ok,
        "wallclock_seconds": wallclock,
        "out_dir": str(out_dir),
        "run_manifest": str(out_dir / "run_manifest.json"),
        "log_path": str(log_path),
        "returncode": proc.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    grid = [(a, s) for a in args.algos for s in args.seeds]
    logger.info(
        "Blue-Team sweep: %d runs (%s) x (%s) on %d worker(s)",
        len(grid),
        args.algos,
        args.seeds,
        args.parallel,
    )
    t_start = time.time()
    results: list[dict] = []

    if args.parallel <= 1:
        for algo, seed in grid:
            results.append(_run_one(args, algo, seed))
            if not results[-1]["ok"] and not args.continue_on_failure:
                logger.error("run failed; aborting sweep (use --continue-on-failure to override)")
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_run_one, args, a, s): (a, s) for a, s in grid}
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    sweep_manifest = {
        "schema_version": "1.0",
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "runs": results,
        "n_ok": sum(1 for r in results if r["ok"]),
        "n_failed": sum(1 for r in results if not r["ok"]),
    }
    sweep_manifest_path = out_root / "sweep_manifest.json"
    sweep_manifest_path.write_text(json.dumps(sweep_manifest, indent=2))

    logger.info(
        "Blue-Team sweep done: %d ok / %d failed in %.1fs; manifest -> %s",
        sweep_manifest["n_ok"],
        sweep_manifest["n_failed"],
        sweep_manifest["wallclock_seconds"],
        sweep_manifest_path,
    )
    if sweep_manifest["n_failed"] and not args.continue_on_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
