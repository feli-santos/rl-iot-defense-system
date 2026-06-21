"""sensitivity-sweep FA_window — window_size ablation (C22).

Trains PPO × 3 seeds × window_size ∈ {1, 3, 5, 10}.
Window size 5 is the primary-contract baseline; its runs are REUSED from
``runs/blue_team_primary/ppo/seed_{0,1,2}`` if they exist.

NOTE on obs_shape:
    window_size × features × 2 (with deltas) = window_size × 29 × 2
    w=1  → obs_shape=(58,)
    w=3  → obs_shape=(174,)
    w=5  → obs_shape=(290,)  ← primary baseline
    w=10 → obs_shape=(580,)

Usage::

    # Full sweep
    python -m scripts.ablation.run_window_ablation \\
        --seeds 0 1 2 \\
        --window-sizes 1 3 5 10 \\
        --out-root runs/ablation_window \\
        --parallel 3

    # Smoke test
    python -m scripts.ablation.run_window_ablation --smoke

Outputs::

    runs/ablation_window/
        w1/ppo/seed_0/   ...  (new training runs)
        w3/ppo/seed_0/   ...  (new training runs)
        w5/ppo/seed_0/   ...  (symlink to blue_team_primary or re-train)
        w10/ppo/seed_0/  ...  (new training runs)
        sweep_manifest.json
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

_PRIMARY_WINDOW_SIZE = 5  # matches Blue-Team default


def _window_to_tag(w: int) -> str:
    return f"w{w}"


def _run_cell(
    *,
    window_size: int,
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
    """Train one (window_size, seed) cell. Returns a result dict."""
    tag = _window_to_tag(window_size)
    out_dir = str(Path(out_root) / tag / "ppo" / f"seed_{seed}")

    # Reuse blue_team_primary baseline for window_size=5 if available
    if window_size == _PRIMARY_WINDOW_SIZE and blue_team_primary_root:
        primary_run = Path(blue_team_primary_root) / "ppo" / f"seed_{seed}"
        manifest_path = primary_run / "run_manifest.json"
        if manifest_path.exists():
            logger.info(
                "window_size=%d seed=%d: reusing primary run at %s",
                window_size,
                seed,
                primary_run,
            )
            out_path = Path(out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                try:
                    out_path.symlink_to(primary_run.resolve())
                except OSError:
                    out_path.mkdir(parents=True, exist_ok=True)
                    (out_path / "reused_from.txt").write_text(str(primary_run.resolve()))
            return {
                "window_size": window_size,
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
                "window_size": window_size,
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
            "window_ablation: w=%d seed=%d -> %s",
            window_size,
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
                "w=%d seed=%d FAILED (rc=%d); log: %s",
                window_size,
                seed,
                result.returncode,
                log_path,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("w=%d seed=%d EXCEPTION: %s", window_size, seed, exc)
    wallclock = time.time() - t0
    if ok:
        logger.info(
            "done w=%d seed=%d ok=%s wallclock=%.1fs",
            window_size,
            seed,
            ok,
            wallclock,
        )
    return {
        "window_size": window_size,
        "seed": seed,
        "ok": ok,
        "out_dir": out_dir,
        "reused": False,
        "wallclock": wallclock,
    }


def run_sweep(
    *,
    window_sizes: list[int],
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
    cells = [(w, sd) for w in window_sizes for sd in seeds]
    logger.info(
        "window_ablation: %d cells (windows=%s seeds=%s) on %d worker(s)",
        len(cells),
        window_sizes,
        seeds,
        parallel,
    )
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(
                _run_cell,
                window_size=w,
                seed=sd,
                out_root=out_root,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                dataset_path=dataset_path,
                splits_manifest=splits_manifest,
                smoke=smoke,
                blue_team_primary_root=blue_team_primary_root,
            ): (w, sd)
            for w, sd in cells
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
        "window_sizes": window_sizes,
        "seeds": seeds,
        "runs": sorted(results, key=lambda r: (r["window_size"], r["seed"])),
    }
    manifest_path = Path(out_root) / "sweep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "window_ablation done: %d ok / %d failed / %d reused; manifest -> %s",
        n_ok,
        n_failed,
        n_reused,
        manifest_path,
    )
    return manifest


def main(argv: list | None = None) -> int:  # type: ignore[type-arg]
    p = argparse.ArgumentParser(
        description="sensitivity-sweep FA_window — window_size ablation (C22).",
    )
    p.add_argument("--window-sizes", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-root", default="runs/ablation_window")
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
        "--blue-team-primary-root",
        default="runs/blue_team_primary",
        help="Root of the primary Blue-Team runs (for window_size=5 reuse).",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.smoke:
        args.window_sizes = [1, 10]
        args.seeds = [0]

    manifest = run_sweep(
        window_sizes=args.window_sizes,
        seeds=args.seeds,
        out_root=args.out_root,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        parallel=args.parallel,
        smoke=args.smoke,
        blue_team_primary_root=args.blue_team_primary_root,
    )
    print(
        f"OK: {manifest['n_ok']} / Failed: {manifest['n_failed']} / Reused: {manifest['n_reused']}"
    )
    return 0 if manifest["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
