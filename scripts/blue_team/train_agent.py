"""Blue-Team single (algo, seed) training entrypoint.

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
and for the Blue-Team step 5.4 audit (run a smoke before committing the
full sweep).

Outputs (per run):

    out_dir/
        episodes.jsonl     — one line per training episode
        eval.jsonl         — one line per eval episode (every eval_freq)
        run_manifest.json  — frozen config + post-run telemetry
        model.zip          — saved SB3 model (loadable by ablation)
        train.log          — stdout/stderr (when run via the sweep driver)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is on sys.path when invoked as a script (not
# `python -m`), so direct-script invocation resolves project imports.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

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

logger = logging.getLogger("scripts.blue_team.train_agent")


# Default per-algo hyperparameters locked in PLAN §8 D5.4. Ablation &
# Robustness may sweep these, but Blue-Team Training ships exactly these values.
DEFAULT_HPARAMS: dict[str, dict[str, Any]] = {
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
    # DQN tuned for the sparse outcome-reward, long-horizon (100-step)
    # regime: a larger replay buffer, slower target updates, lower
    # learning rate, and longer exploration schedule reduce the eval-
    # reward instability seen with the SB3 defaults.
    "dqn": {
        "learning_rate": 5e-4,
        "buffer_size": 200_000,
        "learning_starts": 5_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "target_update_interval": 5_000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    },
    # A2C tuned for the long credit-assignment horizon: the SB3 default
    # n_steps=5 is far too myopic for a 100-step episode with sparse
    # terminal (outcome) reward, so the rollout length is raised to span
    # roughly two episodes and GAE + entropy are enabled for stability.
    "a2c": {
        "learning_rate": 7e-4,
        "n_steps": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}


def _git_sha() -> str:
    """Best-effort short SHA of the producing commit (untracked changes are noted)."""
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=_ROOT)
            .decode()
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT).decode().strip()
        )
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


def _apply_env_overrides(
    spec: EnvConfigSerializable,
    *,
    reward_overrides: dict[str, Any] | None = None,
    p_defender_deescalation: float | None = None,
    impact_is_terminal: bool | None = None,
) -> EnvConfigSerializable:
    """Return a copy of ``spec`` with per-field overrides applied.

    ablation §3.1.2 / D7.3. Validates that every key in
    ``reward_overrides`` is a valid :class:`EnvConfigSerializable`
    field name; raises ``ValueError`` with the bad key otherwise.

    Precedence (highest first): explicit kwargs (``p_defender_deescalation``,
    ``impact_is_terminal``) > ``reward_overrides`` JSON > ``spec`` defaults.
    """
    valid_fields = {f.name for f in dataclasses.fields(EnvConfigSerializable)}
    merged: dict[str, Any] = dataclasses.asdict(spec)

    if reward_overrides:
        bad = sorted(set(reward_overrides) - valid_fields)
        if bad:
            raise ValueError(
                f"--reward-overrides contains unknown field(s): {bad!r}. "
                f"Valid fields: {sorted(valid_fields)!r}"
            )
        merged.update(reward_overrides)

    if p_defender_deescalation is not None:
        merged["p_defender_deescalation"] = float(p_defender_deescalation)

    if impact_is_terminal is not None:
        merged["impact_is_terminal"] = bool(impact_is_terminal)

    return EnvConfigSerializable(**merged)


def build_run_config(args: argparse.Namespace) -> BlueTeamRunConfig:
    """Translate CLI args into a :class:`BlueTeamRunConfig`."""
    if args.algo not in DEFAULT_HPARAMS:
        raise ValueError(f"unknown algo {args.algo!r}")

    if args.smoke:
        total_timesteps = 5_000
        eval_freq = 1_000
        n_eval_episodes = 5
        env_spec = EnvConfigSerializable(
            split="train",
            exclude_ood=True,
            min_episode_length=5,
            max_steps=20,
            window_size=4,
            include_deltas=True,
        )
        eval_spec = EnvConfigSerializable(
            split="val_balanced",
            exclude_ood=True,
            min_episode_length=5,
            max_steps=20,
            window_size=4,
            include_deltas=True,
        )
    else:
        total_timesteps = args.total_timesteps
        eval_freq = args.eval_freq
        n_eval_episodes = args.n_eval_episodes
        env_spec = EnvConfigSerializable(
            split="train",
            exclude_ood=True,
            min_episode_length=20,
            max_steps=100,
            window_size=5,
            include_deltas=True,
        )
        eval_spec = EnvConfigSerializable(
            split="val_balanced",
            exclude_ood=True,
            min_episode_length=20,
            max_steps=100,
            window_size=5,
            include_deltas=True,
        )

    # ablation §3.1.2 / D7.3: apply per-field overrides from
    # --reward-overrides / --p-defender-deescalation / --impact-is-terminal.
    # Defaults preserve byte-for-byte Blue-Team behaviour.
    reward_overrides_obj: dict[str, Any] | None = None
    if getattr(args, "reward_overrides", None):
        reward_overrides_obj = json.loads(args.reward_overrides)
        if not isinstance(reward_overrides_obj, dict):
            raise ValueError(
                f"--reward-overrides must be a JSON object; got "
                f"{type(reward_overrides_obj).__name__}"
            )
    p_dee = getattr(args, "p_defender_deescalation", None)
    impact_term = getattr(args, "impact_is_terminal", None)
    env_spec = _apply_env_overrides(
        env_spec,
        reward_overrides=reward_overrides_obj,
        p_defender_deescalation=p_dee,
        impact_is_terminal=impact_term,
    )
    eval_spec = _apply_env_overrides(
        eval_spec,
        reward_overrides=reward_overrides_obj,
        p_defender_deescalation=p_dee,
        impact_is_terminal=impact_term,
    )

    out_dir = args.out_dir or f"runs/{args.algo}/seed_{args.seed}"

    # Smoke runs disable early-stop so the tiny grid runs deterministically
    # to its cap; full runs honour the CLI flags.
    early_stop = (not args.smoke) and getattr(args, "early_stop", True)

    return BlueTeamRunConfig(
        algo=args.algo,
        seed=args.seed,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        early_stop=early_stop,
        early_stop_patience=getattr(args, "early_stop_patience", 10),
        early_stop_min_evals=getattr(args, "early_stop_min_evals", 10),
        out_dir=out_dir,
        dataset_path=args.dataset_path,
        splits_manifest=args.splits_manifest or "",
        env=env_spec,
        eval_env=eval_spec,
        algo_hparams=dict(DEFAULT_HPARAMS[args.algo]),
        notes=("smoke" if args.smoke else ""),
    )


def train(cfg: BlueTeamRunConfig, *, verbose: int = 0) -> dict[str, Any]:
    """Execute one training run end-to-end.

    Returns a dict of post-run telemetry written into ``run_manifest.json``.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "blue-team train: algo=%s seed=%d total_timesteps=%d out_dir=%s",
        cfg.algo,
        cfg.seed,
        cfg.total_timesteps,
        out_dir,
    )

    _seed_everything(cfg.seed)

    splits_manifest: str | None = cfg.splits_manifest if cfg.splits_manifest else None

    train_env = make_train_env(
        spec=cfg.env,
        dataset_path=cfg.dataset_path,
        splits_manifest=splits_manifest,
        seed=cfg.seed,
    )
    eval_env = make_eval_env(
        spec=cfg.eval_env,
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
        **{
            k: v
            for k, v in cfg.algo_hparams.items()
            if k in AdversarialAlgorithmConfig.__dataclass_fields__
        },
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
    callbacks: list[Any] = [cb_train, cb_eval]

    # Best-checkpoint + early-stop on the eval-reward plateau. SB3's
    # EvalCallback needs its OWN VecEnv (it resets/steps it during the eval
    # block), so we build a third env here with a disjoint seed pool. It
    # writes ``best_model.zip`` whenever the mean eval reward improves; the
    # benchmark + OOD harnesses load that checkpoint, so a slow-converging
    # algorithm is never penalised for the fixed ``total_timesteps`` cap.
    sb3_eval_env = None
    if cfg.early_stop:
        sb3_eval_env = make_eval_env(
            spec=cfg.eval_env,
            dataset_path=cfg.dataset_path,
            splits_manifest=splits_manifest,
            seed=cfg.seed + 20_000,  # disjoint from train (seed) + eval (+10k)
        )
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=cfg.early_stop_patience,
            min_evals=cfg.early_stop_min_evals,
            verbose=verbose,
        )
        sb3_eval_cb = EvalCallback(
            sb3_eval_env,
            best_model_save_path=str(out_dir),
            log_path=None,
            eval_freq=cfg.eval_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=stop_cb,
            verbose=verbose,
        )
        callbacks.append(sb3_eval_cb)

    cb = CallbackList(callbacks)

    t0 = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=cb,
        progress_bar=False,
    )
    wallclock = time.time() - t0

    # Persist the last model + manifest. ``best_model.zip`` (written by the
    # EvalCallback) is the canonical checkpoint for downstream eval; we keep
    # ``model.zip`` (the last model) for back-compat and diagnostics. If
    # early-stop never wrote a best checkpoint (e.g. it stopped before the
    # first eval), fall back to the last model so downstream never breaks.
    model_path = out_dir / "model.zip"
    model.save(str(model_path))
    best_model_path = out_dir / "best_model.zip"
    if not best_model_path.exists():
        model.save(str(best_model_path))

    n_train_episodes = _count_jsonl_lines(train_jsonl)
    n_eval_episodes = _count_jsonl_lines(eval_jsonl)

    manifest_path = out_dir / "run_manifest.json"
    extra = {
        "wallclock_seconds": wallclock,
        "n_episodes_train": n_train_episodes,
        "n_episodes_eval": n_eval_episodes,
        "git_sha": _git_sha(),
        "early_stopped": bool(cfg.early_stop and model.num_timesteps < cfg.total_timesteps),
        "actual_timesteps": int(model.num_timesteps),
        "best_model_path": str(best_model_path),
    }
    cfg.write_manifest(manifest_path, **extra)

    train_env.close()
    eval_env.close()
    if sb3_eval_env is not None:
        sb3_eval_env.close()

    logger.info(
        "blue-team train done: wallclock=%.1fs episodes_train=%d episodes_eval=%d",
        wallclock,
        n_train_episodes,
        n_eval_episodes,
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
    p = argparse.ArgumentParser(description="Blue-Team single (algo, seed) RL training run.")
    p.add_argument("--algo", required=True, choices=("ppo", "dqn", "a2c"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default 500K, matches PLAN D5.3).",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Run an eval block every N timesteps (default 25K, PLAN D5.5).",
    )
    p.add_argument(
        "--n-eval-episodes",
        type=int,
        default=30,
        help="Episodes per eval block (default 30, PLAN D5.5).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output dir (default runs/<algo>/seed_<seed>).",
    )
    p.add_argument(
        "--dataset-path",
        default="data/processed/ciciot2023",
        help="Path to processed CICIoT2023 dataset directory.",
    )
    p.add_argument(
        "--splits-manifest",
        default="data/processed/ciciot2023/splits/manifest.json",
        help=(
            "Path to dataset-prep splits manifest. Use empty string to disable "
            "split-aware sampling (synthetic tests only)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Reduce to 5K timesteps + tiny eval grid; for quick smoke runs.",
    )
    p.add_argument(
        "--no-early-stop",
        dest="early_stop",
        action="store_false",
        help="Disable eval-plateau early-stopping (train the full cap).",
    )
    p.set_defaults(early_stop=True)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after N consecutive evals with no eval-reward improvement.",
    )
    p.add_argument(
        "--early-stop-min-evals",
        type=int,
        default=10,
        help="Never early-stop before this many evals have run.",
    )
    p.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="SB3 verbosity (0/1/2).",
    )
    # ----- ablation §3.1.2 / D7.3 overrides (default off; preserve Blue-Team) -----
    p.add_argument(
        "--reward-overrides",
        type=str,
        default=None,
        help=(
            "JSON object overriding individual EnvConfigSerializable fields "
            "(reward coefficients, lifecycle, impact_is_terminal). Example: "
            "'{\"defense_success_bonus\": 500}'. Applied to BOTH training "
            "and eval env specs. Default None preserves Blue-Team behaviour."
        ),
    )
    p.add_argument(
        "--p-defender-deescalation",
        type=float,
        default=None,
        help=(
            "Override AdversarialEnvConfig.p_defender_deescalation. "
            "Convenience knob for the F10 attack-aggressiveness sweep "
            "(takes precedence over the same field in --reward-overrides "
            "if both are supplied). Default None preserves Blue-Team 0.6."
        ),
    )
    p.add_argument(
        "--impact-is-terminal",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help=(
            "Override AdversarialEnvConfig.impact_is_terminal "
            "(true/false). Default None preserves the dataclass default "
            "(False, the primary training + benchmark contract): the agent "
            "gets an explicit IMPACT-row decision step before termination. "
            "Set --impact-is-terminal true only for the reward-mis-"
            "specification case study (D7.3)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
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
