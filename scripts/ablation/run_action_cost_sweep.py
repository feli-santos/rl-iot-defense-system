"""sensitivity-sweep FA_action_cost — action_cost_scale sensitivity sweep (C8).

Trains PPO × 3 seeds × action_cost_scale ∈ {0.5, 1.0, 2.0}.
Scale 1.0 is the primary-contract baseline; its runs are REUSED from
``runs/blue_team_primary/ppo/seed_{0,1,2}`` if they exist.

Usage::

    # Full sweep
    python -m scripts.ablation.run_action_cost_sweep \\
        --seeds 0 1 2 \\
        --scales 0.5 1.0 2.0 \\
        --out-root runs/ablation_action_cost \\
        --parallel 3

    # Smoke test
    python -m scripts.ablation.run_action_cost_sweep --smoke

Outputs::

    runs/ablation_action_cost/
        x0p5/ppo/seed_0/   ...  (new training runs)
        x1p0/ppo/seed_0/   ...  (symlink or copy from phase5_primary, or re-train)
        x2p0/ppo/seed_0/   ...  (new training runs)
        sweep_manifest.json

The sweep manifest has the shape::

    {
        "n_ok": ..., "n_failed": ..., "n_reused": ...,
        "runs": [{"scale": 1.0, "seed": 0, "ok": true, "out_dir": "..."}]
    }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

_PYTHON = sys.executable
_TRAIN_MODULE = "scripts.blue_team.train_agent"


# Maps scale value → directory-safe string (e.g. 0.5 → "x0p5")
def _scale_to_tag(scale: float) -> str:
    return "x" + f"{scale:.1f}".replace(".", "p")


def _run_cell(
    *,
    scale: float,
    seed: int,
    out_root: str,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    dataset_path: str,
    splits_manifest: str,
    smoke: bool,
    blue_team_primary_root: str | None,
) -> dict[str, Any]:
    """Train one (scale, seed) cell. Returns a result dict."""
    tag = _scale_to_tag(scale)
    out_dir = str(Path(out_root) / tag / "ppo" / f"seed_{seed}")

    # Reuse phase5_primary baseline for scale=1.0 if available
    if abs(scale - 1.0) < 1e-9 and blue_team_primary_root:
        primary_run = Path(blue_team_primary_root) / "ppo" / f"seed_{seed}"
        manifest_path = primary_run / "run_manifest.json"
        if manifest_path.exists():
            logger.info(
                "scale=1.0 seed=%d: reusing primary run at %s",
                seed,
                primary_run,
            )
            # Create a symlink (or record path) in out_dir
            out_path = Path(out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                try:
                    out_path.symlink_to(primary_run.resolve())
                except OSError:
                    # If symlink fails (e.g. cross-device), just record the path
                    out_path.mkdir(parents=True, exist_ok=True)
                    (out_path / "reused_from.txt").write_text(str(primary_run.resolve()))
            return {
                "scale": scale,
                "seed": seed,
                "ok": True,
                "out_dir": str(out_path),
                "reused": True,
                "wallclock": 0.0,
            }

    # Otherwise train fresh
    cmd = [
        _PYTHON,
        "-m",
        _TRAIN_MODULE,
        "--algo",
        "ppo",
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--eval-freq",
        str(eval_freq),
        "--n-eval-episodes",
        str(n_eval_episodes),
        "--out-dir",
        out_dir,
        "--reward-overrides",
        json.dumps(
            {
                "action_cost_scale": scale,
                "impact_is_terminal": False,  # match primary contract
            }
        ),
    ]
    if dataset_path:
        cmd += ["--dataset-path", dataset_path]
    if splits_manifest:
        cmd += ["--splits-manifest", splits_manifest]
    if smoke:
        cmd.append("--smoke")

    log_path = Path(out_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = False
    try:
        logger.info(
            "action_cost_sweep: scale=%.1f seed=%d -> %s",
            scale,
            seed,
            out_dir,
        )
        with open(log_path, "w") as flog:
            result = subprocess.run(
                cmd,
                stdout=flog,
                stderr=subprocess.STDOUT,
                cwd=str(_ROOT),
                timeout=7200,
            )
        ok = result.returncode == 0
        if not ok:
            logger.error(
                "scale=%.1f seed=%d FAILED (rc=%d); log: %s",
                scale,
                seed,
                result.returncode,
                log_path,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("scale=%.1f seed=%d EXCEPTION: %s", scale, seed, exc)
    wallclock = time.time() - t0
    if ok:
        logger.info(
            "done scale=%.1f seed=%d ok=%s wallclock=%.1fs",
            scale,
            seed,
            ok,
            wallclock,
        )
    return {
        "scale": scale,
        "seed": seed,
        "ok": ok,
        "out_dir": out_dir,
        "reused": False,
        "wallclock": wallclock,
    }


def run_sweep(
    *,
    scales: list[float],
    seeds: list[int],
    out_root: str,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    dataset_path: str,
    splits_manifest: str,
    parallel: int,
    smoke: bool,
    blue_team_primary_root: str | None,
) -> dict[str, Any]:
    cells = [(sc, sd) for sc in scales for sd in seeds]
    logger.info(
        "action_cost_sweep: %d cells (scales=%s seeds=%s) on %d worker(s)",
        len(cells),
        scales,
        seeds,
        parallel,
    )
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_cell,
                scale=sc,
                seed=sd,
                out_root=out_root,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                dataset_path=dataset_path,
                splits_manifest=splits_manifest,
                smoke=smoke,
                blue_team_primary_root=blue_team_primary_root,
            ): (sc, sd)
            for sc, sd in cells
        }
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    n_ok = sum(1 for r in results if r["ok"])
    n_failed = sum(1 for r in results if not r["ok"])
    n_reused = sum(1 for r in results if r.get("reused"))

    manifest = {
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_reused": n_reused,
        "scales": scales,
        "seeds": seeds,
        "runs": sorted(results, key=lambda r: (r["scale"], r["seed"])),
    }
    manifest_path = Path(out_root) / "sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "action_cost_sweep done: %d ok / %d failed / %d reused; manifest -> %s",
        n_ok,
        n_failed,
        n_reused,
        manifest_path,
    )
    return manifest


def main(argv: list | None = None) -> int:  # type: ignore[type-arg]
    p = argparse.ArgumentParser(
        description="sensitivity-sweep FA_action_cost — action_cost_scale sensitivity sweep (C8).",
    )
    p.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-root", default="runs/ablation_action_cost")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument(
        "--phase5-primary-root",
        default="runs/blue_team_primary",
        help="Root of the primary blue-team training runs (for scale=1.0 reuse).",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.smoke:
        args.scales = [0.5, 2.0]
        args.seeds = [0]

    manifest = run_sweep(
        scales=args.scales,
        seeds=args.seeds,
        out_root=args.out_root,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        parallel=args.parallel,
        smoke=args.smoke,
        blue_team_primary_root=args.phase5_primary_root,
    )
    print(
        f"OK: {manifest['n_ok']} / Failed: {manifest['n_failed']} / Reused: {manifest['n_reused']}"
    )
    return 0 if manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
