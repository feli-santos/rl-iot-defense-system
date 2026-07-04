"""ablation F17 — Evasive-persistence robustness sweep driver.

Sweeps ``evasion_prob ∈ {0.0, 0.25, 0.5, 0.75}`` × PPO only × 10 seeds,
all under the LOCKED primary reward contract (``reward_mode='outcome'``,
``impact_is_terminal=False``, ``aliasing_rate=0.4``, session-coherent).
In the canonical mode (``--load-ppo-from``) the fixed deterministic-5M
α=0.4 PPO defender is LOADED and re-evaluated (not retrained) under each
evasion level, so the curve isolates the effect of the attacker mechanic
on a single deployed policy. Total: 4 × 10 evaluations.

``evasion_prob`` models an *evasive-persistence* attacker (adversarial_env.py
"post-detection hardening"): after the attacker senses defensive force
(BLOCK/ISOLATE) at a pre-commit stage (RECON/ACCESS), it hardens against the
NEXT eviction — on a subsequent proportional defender step the de-escalation
pushdown is resisted with probability ``evasion_prob``. The correct defensive
response still holds the line (the attacker does not advance), so the mechanic
never rewards mis-forcing; it only makes the attacker harder to remove. At
``evasion_prob=0`` this reduces to the standard Markov attacker (byte-identical
RNG stream).

Output layout::

    runs/ablation/evasion/
        sweep_manifest.json
        ppo_e<e>/seed_<k>/
            episodes.jsonl
            eval.jsonl
            run_manifest.json
            model.zip
            eval_test.jsonl
            train.log
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

from src.benchmark.baseline_policies import SB3PolicyAdapter
from src.benchmark.eval_runner import run_policy
from src.blue_team.env_factory import make_eval_env
from src.blue_team.run_config import EnvConfigSerializable

logger = logging.getLogger("scripts.ablation.run_evasion_sweep")

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_EVASION_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75]


# Locked primary reward contract (matches the headline redesign sweep). F17
# MUST train + evaluate under this contract so its numbers are directly
# comparable to Chapter 4; the swept knob (evasion_prob) is merged on top per
# cell. Previously this sweep inherited the EnvConfigSerializable dataclass
# defaults (reward_mode='coupled', impact_is_terminal=True, aliasing_rate=0.0),
# which is OFF-CONTRACT. NOTE: attacker_budget no longer exists (finite
# intrusion budget was retired); the budget=40 mentioned in old docstrings is
# obsolete.
_CONTRACT_OVERRIDES: dict[str, Any] = {
    "reward_mode": "outcome",
    "aliasing_rate": 0.4,
    "no_post_transition_leak": True,
    "proximity_coupled": True,
    "impact_is_terminal": False,
}


def _cell_overrides(e: float) -> dict[str, Any]:
    """Canonical on-contract override set for an F17 cell at evasion_prob=e."""
    merged = dict(_CONTRACT_OVERRIDES)
    merged["evasion_prob"] = e
    return merged


# Load-mode (``--load-ppo-from``) evaluates a FIXED, already-trained defender
# (the deterministic-5M headline PPO at alpha_04) across the evasion sweep
# instead of retraining a fresh PPO per cell. To match that checkpoint's
# training MDP exactly we add ``session_coherent=True`` (the det-5M alpha sweep
# sets it; the legacy per-cell train contract above omits it). This is the
# cleaner robustness test: does the SAME trained defender degrade gracefully
# as the attacker becomes evasive?
_LOAD_MODE_OVERRIDES: dict[str, Any] = {
    "reward_mode": "outcome",
    "aliasing_rate": 0.4,
    "session_coherent": True,
    "no_post_transition_leak": True,
    "proximity_coupled": True,
    "proximity_min_escalation": 0.4,
    "impact_is_terminal": False,
}


def _load_cell_overrides(e: float) -> dict[str, Any]:
    """On-contract override set for a load-mode F17 cell at evasion_prob=e."""
    merged = dict(_LOAD_MODE_OVERRIDES)
    merged["evasion_prob"] = e
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


def _e_slug(e: float) -> str:
    """Filesystem-safe slug for an evasion value, e.g. 0.25 -> '0p25'."""
    return f"{e:.2f}".replace(".", "p")


def _overrides_json(e: float) -> str:
    return json.dumps(_cell_overrides(e))


# --------------------------------------------------------------------- per-cell


def _load_and_eval_ppo(args: argparse.Namespace, e: float, seed: int) -> dict[str, Any]:
    """Load the fixed det-5M PPO checkpoint and evaluate it at evasion_prob=e.

    No retraining: loads ``<load_ppo_from>/ppo/seed_<seed>/best_model.zip``
    (falling back to ``model.zip``) and rolls it on ``test_balanced`` under
    the load-mode POMDP contract at this evasion level. Writes
    ``eval_test.jsonl`` into the normal ``ppo_e<slug>/seed_<k>/`` layout so
    the F17 plotter is unchanged.
    """
    out_dir = Path(args.out_root) / f"ppo_e{_e_slug(e)}" / f"seed_{seed}"
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
        logger.error("F17 load e=%.2f seed=%d: %s", e, seed, error)
    else:
        try:
            spec = EnvConfigSerializable(
                split="test_balanced",
                exclude_ood=True,
                **_load_cell_overrides(e),
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
                    run_id=f"f17_ppo_e{_e_slug(e)}_seed_{seed}_test",
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
            logger.error("F17 load e=%.2f seed=%d eval failed: %s", e, seed, exc)
    wallclock = time.time() - t0
    logger.info(
        "F17 load e=%.2f seed=%d done eval=%s wc=%.1fs",
        e,
        seed,
        ok,
        wallclock,
    )
    return {
        "kind": "ppo",
        "mode": "load",
        "evasion_prob": e,
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


def _train_ppo(args: argparse.Namespace, e: float, seed: int) -> dict[str, Any]:
    """Train PPO at evasion_prob=e under the locked outcome contract."""
    out_dir = Path(args.out_root) / f"ppo_e{_e_slug(e)}" / f"seed_{seed}"
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
        _overrides_json(e),
        "--verbose",
        "0",
    ]
    if args.smoke:
        cmd.append("--smoke")

    logger.info("F17 ppo e=%.2f seed=%d → %s", e, seed, out_dir)
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

    test_eval_jsonl = out_dir / "eval_test.jsonl"
    test_eval_ok = False
    if ok:
        try:
            _eval_ppo_on_test(args, e, seed, out_dir, test_eval_jsonl)
            test_eval_ok = test_eval_jsonl.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("F17 ppo e=%.2f seed=%d test-eval failed: %s", e, seed, exc)

    logger.info(
        "F17 ppo e=%.2f seed=%d done train=%s test_eval=%s wc=%.1fs",
        e,
        seed,
        ok,
        test_eval_ok,
        wallclock,
    )

    return {
        "kind": "ppo",
        "evasion_prob": e,
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
    e: float,
    seed: int,
    out_dir: Path,
    eval_jsonl_path: Path,
) -> None:
    """Roll the trained PPO at this evasion_prob on test_balanced.

    Reads the just-finished run's ``run_manifest.json`` to mirror the
    training-time env shape (window_size, max_steps, evasion) so
    the trained model's observation space matches the eval env's.
    """
    run_manifest_path = out_dir / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = json.loads(run_manifest_path.read_text())
        spec = EnvConfigSerializable(**manifest["eval_env"])
        spec.split = "test_balanced"
        # Belt-and-braces: ensure the eval cell carries the on-contract
        # knobs even if the manifest round-trip dropped them.
        for _k, _v in _cell_overrides(e).items():
            setattr(spec, _k, _v)
    else:
        spec = EnvConfigSerializable(
            split="test_balanced",
            exclude_ood=True,
            **_cell_overrides(e),
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
            run_id=f"f17_ppo_e{_e_slug(e)}_seed_{seed}_test",
            policy_name="ppo",
            latency_path=None,
            seed=seed,
        )
    finally:
        try:  # noqa: SIM105
            env.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------- driver


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ablation F17 — evasion-reactive sweep.",
    )
    p.add_argument(
        "--evasion-values",
        nargs="+",
        type=float,
        default=_DEFAULT_EVASION_VALUES,
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--total-timesteps", type=int, default=1_500_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--n-eval-episodes", type=int, default=300)
    p.add_argument("--out-root", default="runs/ablation/evasion")
    p.add_argument(
        "--load-ppo-from",
        default=None,
        help=(
            "If set, DO NOT retrain: load the fixed PPO checkpoint at "
            "<dir>/ppo/seed_<k>/best_model.zip and evaluate it across the "
            "evasion sweep (fixed-policy robustness). E.g. "
            "runs/redesign_5M_det/alpha_04. The load-mode env carries "
            "session_coherent=True to match the det-5M training contract."
        ),
    )
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument(
        "--dataset-path",
        default="data/processed/ciciot2023",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue-on-failure", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [(e, seed) for e in args.evasion_values for seed in args.seeds]
    n_workers = max(1, int(args.parallel))
    _ppo_fn = _load_and_eval_ppo if args.load_ppo_from else _train_ppo
    if args.load_ppo_from:
        logger.info(
            "F17 LOAD MODE: evaluating fixed PPO from %s (no retraining)",
            args.load_ppo_from,
        )
    logger.info(
        "F17 evasion sweep: %d evasion × %d seeds = %d runs (%d worker(s))",
        len(args.evasion_values),
        len(args.seeds),
        len(cells),
        n_workers,
    )

    t0 = time.time()
    results: list[dict[str, Any]] = []
    if n_workers == 1:
        for e, seed in cells:
            results.append(_ppo_fn(args, e, seed))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_ppo_fn, args, e, seed): (e, seed) for e, seed in cells}
            for fut in concurrent.futures.as_completed(futs):
                e, seed = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.error("F17 ppo e=%.2f seed=%d crashed: %s", e, seed, exc)
                    if not args.continue_on_failure:
                        raise
                    results.append(
                        {
                            "kind": "ppo",
                            "evasion_prob": e,
                            "seed": seed,
                            "ok_train": False,
                            "ok_test_eval": False,
                            "error": str(exc),
                        }
                    )

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r.get("ok_train") and r.get("ok_test_eval"))
    n_fail = len(results) - n_ok

    manifest = {
        "schema_version": "1.0",
        "figure": "F17",
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "load" if args.load_ppo_from else "train",
        "load_ppo_from": args.load_ppo_from,
        "evasion_values": list(args.evasion_values),
        "seeds": list(args.seeds),
        "total_timesteps": args.total_timesteps,
        "n_episodes": args.n_eval_episodes,
        "elapsed_seconds": elapsed,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "runs": results,
    }
    (out_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "F17 evasion sweep done: %d ok / %d fail in %.1fs → %s",
        n_ok,
        n_fail,
        elapsed,
        out_root / "sweep_manifest.json",
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
