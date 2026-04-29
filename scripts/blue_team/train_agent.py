"""Phase-5 single (algo, seed) training entrypoint.

PLAN §3.1.6.

Usage:

    python -m scripts.blue_team.train_agent \\
        --algo {dqn,ppo,a2c} \\
        --seed N \\
        [--total-timesteps 500000] \\
        [--out-dir runs/<algo>/seed_<N>] \\
        [--smoke]

The ``--smoke`` flag drops the run to 50 K timesteps with eval every
10 K — useful for the smoke test in ``tests/test_blue_team_train_agent.py``
and for the Phase-5 step 5.4 audit (run a smoke before committing the
full sweep).

Outputs (per run):

    out_dir/
        episodes.jsonl     — one line per training episode
        eval.jsonl         — one line per eval episode (every eval_freq)
        run_manifest.json  — frozen config + post-run telemetry
        model.zip          — saved SB3 model (loadable by Phase 7)
        train.log          — stdout/stderr (when run via the sweep driver)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Ensure the project root is on sys.path when invoked as a script (not
# `python -m`). This lets the Phase-2 scripts/red_team/train_lstm.py
# pattern continue to work.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from src.algorithms.adversarial_algorithm import (  # noqa: E402
    AdversarialAlgorithm,
    AdversarialAlgorithmConfig,
)
from src.blue_team import (  # noqa: E402
    BlueTeamRunConfig,
    EnvConfigSerializable,
    EpisodeJSONLCallback,
    EvalToJSONLCallback,
    make_eval_env,
    make_train_env,
)
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402

logger = logging.getLogger("scripts.blue_team.train_agent")


# Default per-algo hyperparameters locked in PLAN §8 D5.4. Phase 8
# may sweep these, but Phase 5 ships exactly these values.
DEFAULT_HPARAMS: Dict[str, Dict[str, Any]] = {
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "dqn": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}


def _git_sha() -> str:
    """Best-effort short SHA of the producing commit (untracked changes are noted)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=_ROOT
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=_ROOT
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover — best effort
        return "unknown"


def _seed_everything(seed: int) -> None:
    """Set every seed channel SB3 might consume."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CPU runs cheap and deterministic-ish.
    torch.set_num_threads(1)


def build_run_config(args: argparse.Namespace) -> BlueTeamRunConfig:
    """Translate CLI args into a :class:`BlueTeamRunConfig`."""
    if args.algo not in DEFAULT_HPARAMS:
        raise ValueError(f"unknown algo {args.algo!r}")

    if args.smoke:
        total_timesteps = 5_000
        eval_freq = 1_000
        n_eval_episodes = 5
        env_spec = EnvConfigSerializable(
            split="train", exclude_ood=True,
            min_episode_length=5, max_steps=20,
            window_size=4, include_deltas=True,
        )
        eval_spec = EnvConfigSerializable(
            split="val_balanced", exclude_ood=True,
            min_episode_length=5, max_steps=20,
            window_size=4, include_deltas=True,
        )
    else:
        total_timesteps = args.total_timesteps
        eval_freq = args.eval_freq
        n_eval_episodes = args.n_eval_episodes
        env_spec = EnvConfigSerializable(
            split="train", exclude_ood=True,
            min_episode_length=20, max_steps=100,
            window_size=5, include_deltas=True,
        )
        eval_spec = EnvConfigSerializable(
            split="val_balanced", exclude_ood=True,
            min_episode_length=20, max_steps=100,
            window_size=5, include_deltas=True,
        )

    out_dir = args.out_dir or f"runs/{args.algo}/seed_{args.seed}"

    return BlueTeamRunConfig(
        algo=args.algo,
        seed=args.seed,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        out_dir=out_dir,
        generator_path=args.generator_path,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest or "",
        env=env_spec,
        eval_env=eval_spec,
        algo_hparams=dict(DEFAULT_HPARAMS[args.algo]),
        notes=("smoke" if args.smoke else ""),
    )


def train(cfg: BlueTeamRunConfig, *, verbose: int = 0) -> Dict[str, Any]:
    """Execute one training run end-to-end.

    Returns a dict of post-run telemetry written into ``run_manifest.json``.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "phase-5 train: algo=%s seed=%d total_timesteps=%d out_dir=%s",
        cfg.algo, cfg.seed, cfg.total_timesteps, out_dir,
    )

    _seed_everything(cfg.seed)

    splits_manifest: Optional[str] = (
        cfg.splits_manifest if cfg.splits_manifest else None
    )

    train_env = make_train_env(
        spec=cfg.env,
        generator_path=cfg.generator_path,
        dataset_path=cfg.dataset_path,
        splits_manifest=splits_manifest,
        seed=cfg.seed,
    )
    eval_env = make_eval_env(
        spec=cfg.eval_env,
        generator_path=cfg.generator_path,
        dataset_path=cfg.dataset_path,
        splits_manifest=splits_manifest,
        seed=cfg.seed + 10_000,  # disjoint RNG pool
    )

    alg_config = AdversarialAlgorithmConfig(
        algorithm_type=cfg.algo,
        policy="MlpPolicy",
        total_timesteps=cfg.total_timesteps,
        verbose=verbose,
        tensorboard_log=None,
        **{k: v for k, v in cfg.algo_hparams.items()
           if k in AdversarialAlgorithmConfig.__dataclass_fields__},
    )
    alg = AdversarialAlgorithm(alg_config)
    model = alg.create_model(train_env)
    model.set_random_seed(cfg.seed)

    train_jsonl = out_dir / "episodes.jsonl"
    eval_jsonl = out_dir / "eval.jsonl"
    cb_train = EpisodeJSONLCallback(
        out_path=train_jsonl,
        run_id=cfg.run_id,
        algo=cfg.algo,
        seed=cfg.seed,
        flush_every=10,
    )
    cb_eval = EvalToJSONLCallback(
        eval_env=eval_env,
        out_path=eval_jsonl,
        run_id=cfg.run_id,
        algo=cfg.algo,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
    )
    cb = CallbackList([cb_train, cb_eval])

    t0 = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=cb,
        progress_bar=False,
    )
    wallclock = time.time() - t0

    # Persist the model + manifest.
    model_path = out_dir / "model.zip"
    model.save(str(model_path))

    n_train_episodes = _count_jsonl_lines(train_jsonl)
    n_eval_episodes = _count_jsonl_lines(eval_jsonl)

    manifest_path = out_dir / "run_manifest.json"
    extra = {
        "wallclock_seconds": wallclock,
        "n_episodes_train": n_train_episodes,
        "n_episodes_eval": n_eval_episodes,
        "git_sha": _git_sha(),
    }
    cfg.write_manifest(manifest_path, **extra)

    train_env.close()
    eval_env.close()

    logger.info(
        "phase-5 train done: wallclock=%.1fs episodes_train=%d episodes_eval=%d",
        wallclock, n_train_episodes, n_eval_episodes,
    )
    return {
        **extra,
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
    }


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


# --------------------------------------------------------------- CLI


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-5 single (algo, seed) RL Blue-Team training run."
    )
    p.add_argument("--algo", required=True, choices=("ppo", "dqn", "a2c"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--total-timesteps", type=int, default=500_000,
        help="Total training timesteps (default 500K, matches PLAN D5.3).",
    )
    p.add_argument(
        "--eval-freq", type=int, default=25_000,
        help="Run an eval block every N timesteps (default 25K, PLAN D5.5).",
    )
    p.add_argument(
        "--n-eval-episodes", type=int, default=30,
        help="Episodes per eval block (default 30, PLAN D5.5).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output dir (default runs/<algo>/seed_<seed>).",
    )
    p.add_argument(
        "--generator-path", default="artifacts/generator/phase2",
        help="Path to Phase-2 generator artefact directory.",
    )
    p.add_argument(
        "--dataset-path", default="data/processed/ciciot2023",
        help="Path to processed CICIoT2023 dataset directory.",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
        help=(
            "Path to Phase-1 splits manifest. Use empty string to disable "
            "split-aware sampling (synthetic tests only)."
        ),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Reduce to 5K timesteps + tiny eval grid; for quick smoke runs.",
    )
    p.add_argument(
        "--verbose", type=int, default=0, choices=(0, 1, 2),
        help="SB3 verbosity (0/1/2).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = build_run_config(args)
    out = train(cfg, verbose=args.verbose)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
