"""Phase-7 F9 — Reward-component ablation sweep driver (PLAN §3.1.4 / D7.1).

Sparse one-at-a-time grid (D7.1): 5 reward components × {0.5×, 1×, 2×}
multipliers + 1 binary axis (impact_is_terminal ∈ {True, False}). The
"1×" centre cell is shared across all 6 axes (it's the Phase-5/6
baseline). Per-cell budget: PPO only (D7.2) × 5 seeds × 250K timesteps
(D7.8). Total cells: 12 (1 centre + 10 component off-centres + 1
binary off-centre). Total runs: 12 × 5 = 60 ≈ 6 h CPU.

Components swept (defaults from Phase-3 RESULTS §3):

    defense_success_bonus      = 250  → {125, 250, 500}
    penalty_missed_impact      = 150  → {75, 150, 300}
    reward_proportional        =   5  → {2.5, 5, 10}
    penalty_disproportionate   =   5  → {2.5, 5, 10}
    reward_benign_passive      =  10  → {5, 10, 20}

Plus impact_is_terminal ∈ {True (default, F9 baseline cell), False}.

Each cell produces a unique cell_id like ``def_success_bonus_x0.5``,
trains 5 seeds in subprocess (mirroring run_phase5.py), and after
training evaluates each checkpoint on test_balanced (reusing the
Phase-6 eval_runner harness).

Output layout::

    runs/phase7/reward_sweep/
        sweep_manifest.json                   — top-level F9 manifest
        <cell_id>/
            cell_config.json                  — the override JSON for this cell
            seed_<k>/
                episodes.jsonl                — train log
                eval.jsonl                    — eval log
                run_manifest.json             — per-run manifest
                model.zip                     — trained PPO ckpt
                eval_test.jsonl               — F6-style test_balanced eval
                train.log                     — stdout/stderr

Usage::

    python -m scripts.ablation.run_reward_sweep \\
        [--components defense_success_bonus penalty_missed_impact ...] \\
        [--multipliers 0.5 1.0 2.0] \\
        [--seeds 0 1 2 3 4] [--total-timesteps 250000] \\
        [--algo ppo] [--out-root runs/phase7/reward_sweep] \\
        [--n-eval-episodes 30] [--smoke]

The default sweep produces 12 cells × 5 seeds = 60 runs at PPO 250K
timesteps. The default --algo is "ppo" (D7.2).
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

import numpy as np

from src.benchmark.baseline_policies import SB3PolicyAdapter
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.ablation.run_reward_sweep")

_ROOT = Path(__file__).resolve().parents[2]


# Phase-3 defaults (from RESULTS §3 of docs/results/03_env/RESULTS.md)
_PHASE3_DEFAULTS: Dict[str, float] = {
    "defense_success_bonus":   250.0,
    "penalty_missed_impact":   150.0,
    "reward_proportional":       5.0,
    "penalty_disproportionate":  5.0,
    "reward_benign_passive":    10.0,
}

_DEFAULT_COMPONENTS: List[str] = list(_PHASE3_DEFAULTS.keys())
_DEFAULT_MULTIPLIERS: List[float] = [0.5, 1.0, 2.0]


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


def _slug(component: str, multiplier: float) -> str:
    """Cell-id from (component, multiplier). Used as directory name.

    Filesystem-safe: the multiplier dot is replaced with 'p' so the
    directory name doesn't contain '.', e.g.
    ``defense_success_bonus_x0p5``. The ``component`` part is left
    untouched (already an underscore-snake-case identifier).
    """
    if multiplier == 1.0:
        return f"{component}_x1p0_baseline"
    mult_str = f"{multiplier:.1f}".replace(".", "p")
    return f"{component}_x{mult_str}"


def _enumerate_cells(
    components: List[str],
    multipliers: List[float],
    *,
    impact_terminal_axis: bool = True,
) -> List[Dict[str, Any]]:
    """Build the sparse one-at-a-time cell list (D7.1).

    Returns a list of cells, each with:
      - cell_id: directory name
      - axis:    'reward' | 'impact_terminal'
      - component: name of the swept reward component (or None)
      - multiplier: the multiplier (or None)
      - reward_overrides: the dict to pass to train_agent.py via JSON
      - impact_is_terminal: True or False
    The "1×" centre cell is added once with axis='baseline'.
    """
    cells: List[Dict[str, Any]] = []

    # 1 centre cell (Phase-5/6 baseline).
    cells.append({
        "cell_id": "baseline_phase5_defaults",
        "axis": "baseline",
        "component": None,
        "multiplier": 1.0,
        "reward_overrides": {},
        "impact_is_terminal": True,
    })

    # 10 component off-centres (5 components × 2 off-centre multipliers).
    for component in components:
        if component not in _PHASE3_DEFAULTS:
            raise ValueError(
                f"unknown component {component!r}; valid: "
                f"{list(_PHASE3_DEFAULTS)}"
            )
        for mult in multipliers:
            if mult == 1.0:
                continue  # centre is shared
            value = _PHASE3_DEFAULTS[component] * mult
            cells.append({
                "cell_id": _slug(component, mult),
                "axis": "reward",
                "component": component,
                "multiplier": mult,
                "reward_overrides": {component: value},
                "impact_is_terminal": True,
            })

    # 1 binary off-centre (impact_is_terminal=False at otherwise-default).
    if impact_terminal_axis:
        cells.append({
            "cell_id": "impact_is_terminal_false",
            "axis": "impact_terminal",
            "component": "impact_is_terminal",
            "multiplier": None,
            "reward_overrides": {},
            "impact_is_terminal": False,
        })

    return cells


# --------------------------------------------------------------------- subprocess


def _run_one(
    args: argparse.Namespace,
    cell: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Spawn a single ``python -m scripts.blue_team.train_agent`` for a cell+seed."""
    out_dir = Path(args.out_root) / cell["cell_id"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd: List[str] = [
        sys.executable, "-m", "scripts.blue_team.train_agent",
        "--algo", args.algo,
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--eval-freq", str(args.eval_freq),
        "--n-eval-episodes", str(args.n_eval_episodes),
        "--out-dir", str(out_dir),
        "--generator-path", args.generator_path,
        "--dataset-path", args.dataset_path,
        "--splits-manifest", args.splits_manifest,
        "--verbose", "0",
    ]
    if cell["reward_overrides"]:
        cmd += ["--reward-overrides", json.dumps(cell["reward_overrides"])]
    if cell["impact_is_terminal"] is False:
        cmd += ["--impact-is-terminal", "false"]
    if args.smoke:
        cmd.append("--smoke")

    logger.info(
        "F9 cell=%s seed=%d → %s (overrides=%s impact_term=%s)",
        cell["cell_id"], seed, out_dir,
        cell["reward_overrides"], cell["impact_is_terminal"],
    )
    t0 = time.time()
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            cmd, cwd=_ROOT, stdout=log_fh, stderr=subprocess.STDOUT,
            check=False,
        )
    wallclock = time.time() - t0
    ok = proc.returncode == 0

    # Test-split eval right after training (Phase-6 eval_runner harness).
    test_eval_ok = False
    test_eval_jsonl = out_dir / "eval_test.jsonl"
    if ok:
        try:
            _eval_on_test_split(args, cell, seed, out_dir, test_eval_jsonl)
            test_eval_ok = test_eval_jsonl.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("F9 cell=%s seed=%d test-eval failed: %s",
                         cell["cell_id"], seed, exc)
            test_eval_ok = False

    logger.info(
        "F9 cell=%s seed=%d done train=%s test_eval=%s wc=%.1fs",
        cell["cell_id"], seed, ok, test_eval_ok, wallclock,
    )

    return {
        "cell_id": cell["cell_id"],
        "axis": cell["axis"],
        "component": cell["component"],
        "multiplier": cell["multiplier"],
        "reward_overrides": cell["reward_overrides"],
        "impact_is_terminal": cell["impact_is_terminal"],
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


def _eval_on_test_split(
    args: argparse.Namespace,
    cell: Dict[str, Any],
    seed: int,
    out_dir: Path,
    eval_jsonl_path: Path,
) -> None:
    """Roll the just-trained checkpoint on test_balanced under the
    *cell's* env config (so test rewards match training rewards).

    Loads PPO/DQN/A2C from out_dir/model.zip, builds an eval env on
    test_balanced with the cell's reward overrides applied, runs
    n_eval_episodes deterministic episodes, writes the JSONL.
    """
    spec = EnvConfigSerializable(split="test_balanced", exclude_ood=True)
    # Apply cell overrides to the eval spec (same protocol train_agent uses).
    for k, v in cell["reward_overrides"].items():
        setattr(spec, k, v)
    spec.impact_is_terminal = cell["impact_is_terminal"]

    env = make_eval_env(
        spec=spec,
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )
    try:
        a = args.algo.lower()
        if a == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(out_dir / "model.zip", env=env, device="cpu")
        elif a == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(out_dir / "model.zip", env=env, device="cpu")
        elif a == "a2c":
            from stable_baselines3 import A2C
            model = A2C.load(out_dir / "model.zip", env=env, device="cpu")
        else:
            raise ValueError(f"unknown algo {args.algo!r}")

        policy = SB3PolicyAdapter(model, deterministic=True)
        run_policy(
            policy, env,
            n_episodes=2 if args.smoke else args.n_eval_episodes,
            jsonl_path=eval_jsonl_path,
            run_id=f"f9_{cell['cell_id']}_{args.algo}_seed_{seed}_test",
            policy_name=args.algo,
            latency_path=None,  # not measured for F9 cells
            seed=seed,
        )
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-7 F9 — reward-component sweep (sparse one-at-a-time, "
                    "PPO 250K × 5 seeds × 12 cells ≈ 6 h CPU; D7.1 / D7.2 / D7.3).",
    )
    p.add_argument(
        "--components", nargs="+", default=_DEFAULT_COMPONENTS,
        help=f"Reward components to sweep (default: {_DEFAULT_COMPONENTS}).",
    )
    p.add_argument(
        "--multipliers", nargs="+", type=float, default=_DEFAULT_MULTIPLIERS,
        help="Multipliers to apply to each component's Phase-3 default.",
    )
    p.add_argument(
        "--no-impact-terminal-axis", action="store_true",
        help="Skip the impact_is_terminal=False cell (D7.3).",
    )
    p.add_argument("--algo", default="ppo", choices=("ppo", "dqn", "a2c"),
                   help="Algorithm to train per cell. Default ppo (D7.2).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-timesteps", type=int, default=250_000,
                   help="PPO timesteps per cell. Default 250K (D7.8 / D5.3.1).")
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--out-root", default="runs/phase7/reward_sweep")
    p.add_argument("--generator-path", default="artifacts/generator/phase2")
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument(
        "--parallel", type=int, default=1,
        help="Number of concurrent train subprocesses (default 1 = serial).",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: 1 cell × 1 seed × 5K timesteps.")
    p.add_argument(
        "--continue-on-failure", action="store_true",
        help="Keep going if a cell crashes (default: stop).",
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

    cells = _enumerate_cells(
        args.components, args.multipliers,
        impact_terminal_axis=not args.no_impact_terminal_axis,
    )

    if args.smoke:
        cells = cells[:1]
        args.seeds = args.seeds[:1]
        logger.info("SMOKE mode: 1 cell × 1 seed (~30 s).")

    # Persist per-cell config JSONs for downstream plotter.
    for cell in cells:
        cell_dir = out_root / cell["cell_id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "cell_config.json").write_text(json.dumps(cell, indent=2))

    grid = [(cell, seed) for cell in cells for seed in args.seeds]
    logger.info(
        "F9 sweep: %d cells × %d seeds = %d runs (algo=%s, %d worker(s))",
        len(cells), len(args.seeds), len(grid), args.algo, args.parallel,
    )

    t_start = time.time()
    results: List[Dict[str, Any]] = []

    if args.parallel <= 1:
        for cell, seed in grid:
            results.append(_run_one(args, cell, seed))
            if not results[-1]["ok_train"] and not args.continue_on_failure:
                logger.error("run failed; aborting (use --continue-on-failure)")
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {
                ex.submit(_run_one, args, cell, seed): (cell["cell_id"], seed)
                for cell, seed in grid
            }
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    sweep_manifest = {
        "schema_version": "1.0",
        "phase": 7,
        "kind": "f9_reward_sweep_manifest",
        "git_sha": _git_sha(),
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "phase3_defaults": _PHASE3_DEFAULTS,
        "cells": cells,
        "runs": results,
        "n_ok_train": sum(1 for r in results if r["ok_train"]),
        "n_ok_test_eval": sum(1 for r in results if r["ok_test_eval"]),
        "n_failed": sum(1 for r in results if not r["ok_train"]),
    }
    sweep_manifest_path = out_root / "sweep_manifest.json"
    sweep_manifest_path.write_text(json.dumps(sweep_manifest, indent=2))
    logger.info(
        "F9 sweep done: %d/%d trained, %d/%d test-evaled in %.1fs; manifest -> %s",
        sweep_manifest["n_ok_train"], len(results),
        sweep_manifest["n_ok_test_eval"], len(results),
        sweep_manifest["wallclock_seconds"], sweep_manifest_path,
    )

    if sweep_manifest["n_failed"] and not args.continue_on_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
