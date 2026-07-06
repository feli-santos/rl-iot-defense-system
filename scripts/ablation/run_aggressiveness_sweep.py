"""ablation F10 — Environment-difficulty sweep driver (PLAN §3.1.5).

Sweeps the tug-of-war de-escalation success probability
``p_down ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`` × PPO only (D7.2) × 10 seeds,
all under the LOCKED primary reward contract (``reward_mode='outcome'``,
``impact_is_terminal=False``, ``aliasing_rate=0.4``, session-coherent).
In the canonical mode (``--load-ppo-from``) the fixed deterministic-5M
α=0.4 PPO defender is LOADED and re-evaluated (not retrained) under each
shifted ``p_down``, so the curve isolates how a single deployed defender
generalizes as the attacker's de-escalation behavior drifts.

``p_down`` is the live tug-of-war knob that governs how *forgiving the
environment is to a correct defender*: when the defender plays the
proportional (recommended) action, the attacker is pushed one stage
back down the kill chain with probability ``p_down`` (and holds
otherwise). This replaces the legacy ``p_defender_deescalation``
parameter, which is INERT under the headline ``tug_of_war=True``
dynamics (it only affects the deprecated ``tug_of_war=False`` path).

Conceptually aligned with IoTWarden Fig. 6 (Bhattacharjee et al.,
2023): how does the fixed defender's realized reward shift as the
environment difficulty varies? ``p_down=0.0`` means a correct
defender action NEVER pushes the attacker back (harshest environment —
the attacker can only be held, never reversed); ``p_down=1.0`` means a
correct action ALWAYS reverses one stage (easiest environment). A fixed
policy trained at the headline ``p_down=0.90`` therefore earns more as
``p_down`` rises and can go negative at the harshest setting.

Each cell evaluates the fixed PPO defender on test_balanced under the
on-contract override set merged with ``{"p_down": P}``. The
recommended-action
oracle baseline is rolled separately under each p as a reference
curve (its mean reward shifts with p because the de-escalation success
rate shifts even though the rule's behaviour does not).

Output layout::

    runs/ablation/aggressiveness/
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
from typing import Any

from src.benchmark.baseline_policies import SB3PolicyAdapter, recommended_action_policy
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.ablation.run_aggressiveness_sweep")

_ROOT = Path(__file__).resolve().parents[2]


_DEFAULT_P_VALUES: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# Locked primary reward contract (matches the headline redesign sweep:
# runs/redesign_5M_det/*/sweep_manifest.json). F10 MUST train + evaluate under this
# contract so its numbers are directly comparable to Chapter 4; the swept
# knob (p_down) is merged on top per cell. Previously this sweep inherited the
# EnvConfigSerializable dataclass defaults (reward_mode='proportional',
# impact_is_terminal=True), which is OFF-CONTRACT.
# NB: the finite ``attacker_budget`` and ``session_coherent`` knobs were
# retired in commit 1e86f68 (no fixed intrusion budget; session-coherence is
# now intrinsic), so they are deliberately ABSENT here — passing them would
# raise ValueError under the current EnvConfigSerializable schema.
_CONTRACT_OVERRIDES: dict[str, Any] = {
    "reward_mode": "outcome",
    "aliasing_rate": 0.4,
    "no_post_transition_leak": True,
    "proximity_coupled": True,
    "impact_is_terminal": False,
}


def _cell_overrides(p: float) -> dict[str, Any]:
    """Canonical on-contract override set for an F10 cell at p_down=p."""
    merged = dict(_CONTRACT_OVERRIDES)
    merged["p_down"] = p
    return merged


# Load-mode (``--load-ppo-from``) evaluates a FIXED, already-trained defender
# (the deterministic-5M headline PPO at alpha_04) across the swept knob instead
# of retraining a fresh PPO per cell. To match that checkpoint's training MDP
# exactly we must add ``session_coherent=True`` (the det-5M alpha sweep sets it;
# the legacy per-cell train contract above omits it). This reframes F10 from
# "PPO retrained per difficulty" to "a fixed det-5M defender evaluated under
# shifted p_down difficulty" (fixed-policy generalization across attacker
# de-escalation difficulty).
_LOAD_MODE_OVERRIDES: dict[str, Any] = {
    "reward_mode": "outcome",
    "aliasing_rate": 0.4,
    "session_coherent": True,
    "no_post_transition_leak": True,
    "proximity_coupled": True,
    "proximity_min_escalation": 0.4,
    "impact_is_terminal": False,
}


def _load_cell_overrides(p: float) -> dict[str, Any]:
    """On-contract override set for a load-mode F10 cell at p_down=p."""
    merged = dict(_LOAD_MODE_OVERRIDES)
    merged["p_down"] = p
    return merged


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
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _p_slug(p: float) -> str:
    """Filesystem-safe slug for a probability value, e.g. 0.6 -> '0p6'."""
    return f"{p:.1f}".replace(".", "p")


# --------------------------------------------------------------------- per-cell


def _load_and_eval_ppo(
    args: argparse.Namespace,
    p: float,
    seed: int,
) -> dict[str, Any]:
    """Load the fixed det-5M PPO checkpoint and evaluate it at p_down=p.

    No retraining: loads ``<load_ppo_from>/ppo/seed_<seed>/best_model.zip``
    (falling back to ``model.zip``) and rolls it on ``test_balanced`` under
    the load-mode POMDP contract at this p_down. Writes ``eval_test.jsonl``
    into the normal ``ppo_p<slug>/seed_<k>/`` layout so the F10 plotter is
    unchanged.
    """
    out_dir = Path(args.out_root) / f"ppo_p{_p_slug(p)}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.load_ppo_from) / "ppo" / f"seed_{seed}"
    ckpt = ckpt_dir / "best_model.zip"
    if not ckpt.exists():
        ckpt = ckpt_dir / "model.zip"
    test_eval_jsonl = out_dir / "eval_test.jsonl"

    t0 = time.time()
    ok = False
    error: str | None = None
    if not ckpt.exists():
        error = f"checkpoint not found: {ckpt}"
        logger.error("F10 load p=%.1f seed=%d: %s", p, seed, error)
    else:
        try:
            spec = EnvConfigSerializable(
                split="test_balanced",
                exclude_ood=True,
                **_load_cell_overrides(p),
            )
            env = make_eval_env(
                spec=spec,
                dataset_path=args.dataset_path,
                splits_manifest=args.splits_manifest,
                seed=seed,
            )
            try:
                from stable_baselines3 import PPO

                model = PPO.load(ckpt, env=env, device="cpu")
                policy = SB3PolicyAdapter(model, deterministic=True)
                run_policy(
                    policy,
                    env,
                    n_episodes=2 if args.smoke else args.n_eval_episodes,
                    jsonl_path=test_eval_jsonl,
                    run_id=f"f10_ppo_p{_p_slug(p)}_seed_{seed}_test",
                    policy_name="ppo",
                    latency_path=None,
                    seed=seed,
                )
                ok = test_eval_jsonl.exists()
            finally:
                try:  # noqa: SIM105
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.error("F10 load p=%.1f seed=%d eval failed: %s", p, seed, exc)
    wallclock = time.time() - t0
    logger.info(
        "F10 load p=%.1f seed=%d done eval=%s wc=%.1fs",
        p,
        seed,
        ok,
        wallclock,
    )
    return {
        "kind": "ppo",
        "mode": "load",
        "p_down": p,
        "seed": seed,
        "ok_train": ok,  # keep key name for plotter/manifest compatibility
        "ok_test_eval": ok,
        "wallclock_seconds": wallclock,
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "test_eval_jsonl": str(test_eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(test_eval_jsonl),
        "error": error,
    }


def _train_ppo(
    args: argparse.Namespace,
    p: float,
    seed: int,
) -> dict[str, Any]:
    """Train PPO at p_down=p (tug-of-war de-escalation success) for one seed."""
    out_dir = Path(args.out_root) / f"ppo_p{_p_slug(p)}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        sys.executable,
        "-m",
        "scripts.blue_team.train_agent",
        "--algo",
        "ppo",
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
        json.dumps(_cell_overrides(p)),
        "--verbose",
        "0",
    ]
    if args.smoke:
        cmd.append("--smoke")

    logger.info("F10 ppo p=%.1f seed=%d → %s", p, seed, out_dir)
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

    # Test-eval at the same p.
    test_eval_jsonl = out_dir / "eval_test.jsonl"
    test_eval_ok = False
    if ok:
        try:
            _eval_ppo_on_test(args, p, seed, out_dir, test_eval_jsonl)
            test_eval_ok = test_eval_jsonl.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("F10 ppo p=%.1f seed=%d test-eval failed: %s", p, seed, exc)

    logger.info(
        "F10 ppo p=%.1f seed=%d done train=%s test_eval=%s wc=%.1fs",
        p,
        seed,
        ok,
        test_eval_ok,
        wallclock,
    )

    return {
        "kind": "ppo",
        "p_down": p,
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
    args: argparse.Namespace,
    p: float,
    seed: int,
    out_dir: Path,
    eval_jsonl_path: Path,
) -> None:
    """Roll the trained PPO at this p on test_balanced (same p).

    Reads the just-finished run's ``run_manifest.json`` to mirror the
    training-time env shape (window_size, max_steps, etc.) so the
    trained model's observation space matches the eval env's. This
    is the same robustness fix as F9's _eval_on_test_split.
    """
    run_manifest_path = out_dir / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text())
        spec = EnvConfigSerializable(**manifest["eval_env"])
        spec.split = "test_balanced"
        spec.p_down = p  # F10's eval matches train p_down
    else:
        spec = EnvConfigSerializable(
            split="test_balanced",
            exclude_ood=True,
            **_cell_overrides(p),  # on-contract: outcome reward + p_down
        )
    env = make_eval_env(
        spec=spec,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=seed,
    )
    try:
        from stable_baselines3 import PPO

        model = PPO.load(out_dir / "model.zip", env=env, device="cpu")
        policy = SB3PolicyAdapter(model, deterministic=True)
        run_policy(
            policy,
            env,
            n_episodes=2 if args.smoke else args.n_eval_episodes,
            jsonl_path=eval_jsonl_path,
            run_id=f"f10_ppo_p{_p_slug(p)}_seed_{seed}_test",
            policy_name="ppo",
            latency_path=None,
            seed=seed,
        )
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _roll_rule_baseline(
    args: argparse.Namespace,
    p: float,
) -> dict[str, Any]:
    """Roll the recommended-action oracle baseline at this p_down (single
    seed=0, n=150 ep — same as benchmark D6.3 protocol).

    The rule's behaviour doesn't depend on p_down — but the
    de-escalation success rate does, so the rule's *mean reward* shifts
    with p_down. F10 wants both curves on the same axes.

    Smoke note: the rule baseline doesn't load a trained model, so
    the env shape is decoupled from training; we use environment-design frozen
    defaults (window_size=5, max_steps=100) for non-smoke and shrink
    them to match the smoke train spec when --smoke is passed.
    """
    out_dir = Path(args.out_root) / f"rule_p{_p_slug(p)}" / "seed_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_jsonl = out_dir / "eval_test.jsonl"
    run_id = f"f10_rule_p{_p_slug(p)}_seed_0"

    if args.smoke:
        # Match the smoke env shape used by train_agent.py --smoke
        # (window_size=4, max_steps=20). Same realiser pool either
        # way (rule doesn't load a model).
        spec = EnvConfigSerializable(
            split="test_balanced",
            exclude_ood=True,
            window_size=4,
            max_steps=20,
            min_episode_length=5,
            **_cell_overrides(p),  # on-contract: outcome reward + p_down
        )
    else:
        spec = EnvConfigSerializable(
            split="test_balanced",
            exclude_ood=True,
            **_cell_overrides(p),  # on-contract: outcome reward + p_down
        )
    env = make_eval_env(
        spec=spec,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest,
        seed=0,
    )
    n_ep = 2 if args.smoke else args.n_deterministic_episodes
    try:
        t0 = time.time()
        run_policy(
            recommended_action_policy,
            env,
            n_episodes=n_ep,
            jsonl_path=eval_jsonl,
            run_id=run_id,
            policy_name="recommended_action",
            latency_path=None,
            seed=0,
        )
        wallclock = time.time() - t0
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass

    return {
        "kind": "rule",
        "p_down": p,
        "seed": 0,
        "ok": True,
        "wallclock_seconds": wallclock,
        "test_eval_jsonl": str(eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl),
    }


# ---------------------------------------------------------------- main


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F10 — environment-difficulty (p_down) sweep "
        "(PLAN §3.1.5; PPO + oracle rule × 6 p_down values × seeds, ~1.5 h CPU).",
    )
    p.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=_DEFAULT_P_VALUES,
        help="p_down (tug-of-war de-escalation success) values to sweep.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="PPO seeds (recommended-action rule uses single seed=0).",
    )
    p.add_argument("--total-timesteps", type=int, default=1_500_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument(
        "--n-deterministic-episodes",
        type=int,
        default=150,
        help="Episodes for the recommended-action rule per p.",
    )
    p.add_argument("--out-root", default="runs/ablation/aggressiveness")
    p.add_argument(
        "--load-ppo-from",
        default=None,
        help=(
            "If set, DO NOT retrain: load the fixed PPO checkpoint at "
            "<dir>/ppo/seed_<k>/best_model.zip and evaluate it across the "
            "p_down sweep (fixed-policy generalization). E.g. "
            "runs/redesign_5M_det/alpha_04. The load-mode env carries "
            "session_coherent=True to match the det-5M training contract."
        ),
    )
    p.add_argument("--dataset-path", default="data/processed/ciciot2023")
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    p.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Concurrent train subprocesses (default 1 = serial).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
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
    ppo_results: list[dict[str, Any]] = []
    rule_results: list[dict[str, Any]] = []

    # PPO sweep. In load mode we evaluate a fixed checkpoint across the sweep
    # instead of retraining a fresh PPO per cell.
    _ppo_fn = _load_and_eval_ppo if args.load_ppo_from else _train_ppo
    if args.load_ppo_from:
        logger.info(
            "F10 LOAD MODE: evaluating fixed PPO from %s (no retraining)",
            args.load_ppo_from,
        )
    grid = [(p, s) for p in args.p_values for s in args.seeds]
    if args.parallel <= 1:
        for p, seed in grid:
            ppo_results.append(_ppo_fn(args, p, seed))
            if not ppo_results[-1]["ok_train"] and not args.continue_on_failure:
                logger.error("aborting sweep (use --continue-on-failure)")
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_ppo_fn, args, p, s): (p, s) for p, s in grid}
            for fut in concurrent.futures.as_completed(futs):
                ppo_results.append(fut.result())

    # Rule baseline at each p.
    for p in args.p_values:
        rule_results.append(_roll_rule_baseline(args, p))
        logger.info(
            "F10 rule p=%.1f done wc=%.1fs",
            p,
            rule_results[-1]["wallclock_seconds"],
        )

    sweep_manifest = {
        "schema_version": "1.0",
        "stage": "ablation",
        "kind": "f10_aggressiveness_sweep_manifest",
        "git_sha": _git_sha(),
        "started_at": datetime.fromtimestamp(t_start, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallclock_seconds": time.time() - t_start,
        "args": vars(args),
        "mode": "load" if args.load_ppo_from else "train",
        "load_ppo_from": args.load_ppo_from,
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
        sweep_manifest["n_ppo_ok"],
        len(ppo_results),
        len(rule_results),
        sweep_manifest["wallclock_seconds"],
        sweep_manifest_path,
    )
    if sweep_manifest["n_ppo_failed"] and not args.continue_on_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
