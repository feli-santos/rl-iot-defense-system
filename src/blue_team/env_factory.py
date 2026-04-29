"""Split-aware env factories for Phase-5 RL training.

Two public functions:

- :func:`make_train_env` — builds an :class:`AdversarialIoTEnv` whose
  :class:`RealizationEngine` is restricted to the Phase-1 *train* split
  (with OOD-attack rows excluded). Wrapped in SB3's
  :class:`Monitor` and a :class:`DummyVecEnv` so the SB3 algorithms see
  the canonical vectorised interface.
- :func:`make_eval_env` — same plumbing but pointed at a different
  split (default ``val_balanced``) and *without* a ``Monitor`` log
  file. Used both by SB3's eval rollouts and by Phase-5's
  :class:`PhaseFiveEvalCallback`.

Both factories accept a ``BlueTeamRunConfig.env`` /
``BlueTeamRunConfig.eval_env`` :class:`EnvConfigSerializable` and the
Phase-1 paths.

The factories deliberately do NOT call ``env.reset(seed=...)``; SB3's
``learn(...)`` and ``evaluate_policy(...)`` paths take care of that and
will mis-align the RNG if we try to seed twice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.blue_team.run_config import EnvConfigSerializable
from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv
from src.utils.realization_engine import RealizationEngine

logger = logging.getLogger(__name__)


def _build_env_config(spec: EnvConfigSerializable) -> AdversarialEnvConfig:
    """Translate a serialisable env spec into an :class:`AdversarialEnvConfig`.

    Reward-shaping fields keep their Phase-3 frozen defaults; only the
    Phase-5-tweakable fields are forwarded.
    """
    return AdversarialEnvConfig(
        max_steps=spec.max_steps,
        min_episode_length=spec.min_episode_length,
        p_defender_deescalation=spec.p_defender_deescalation,
        window_size=spec.window_size,
        include_deltas=spec.include_deltas,
    )


def _build_env(
    *,
    spec: EnvConfigSerializable,
    generator_path: Union[str, Path],
    dataset_path: Union[str, Path],
    splits_manifest: Optional[Union[str, Path]],
    seed: Optional[int],
) -> AdversarialIoTEnv:
    """Assemble an unwrapped :class:`AdversarialIoTEnv` with split-aware
    feature sampling.

    ``splits_manifest`` may be ``None`` for synthetic-data tests that do
    not carry a Phase-1 manifest; in that case we fall back to a vanilla
    :class:`RealizationEngine` over the entire dataset directory.
    """
    env_cfg = _build_env_config(spec)
    env = AdversarialIoTEnv(
        generator_path=Path(generator_path),
        dataset_path=Path(dataset_path),
        config=env_cfg,
        device="cpu",
    )

    if splits_manifest is not None:
        engine = RealizationEngine.from_split_manifest(
            data_path=dataset_path,
            splits_manifest=splits_manifest,
            split_name=spec.split,
            exclude_ood=spec.exclude_ood,
            seed=seed,
        )
        env._realization_engine = engine  # type: ignore[attr-defined]
        # Re-pin num_features in case the split-restricted engine was
        # built from a smaller feature matrix; in practice this is the
        # same ``features.npy`` so the shape doesn't change, but we
        # guard against drift.
        env._num_features = engine.num_features  # type: ignore[attr-defined]
        logger.debug(
            "RealizationEngine restricted to split=%s (exclude_ood=%s)",
            spec.split, spec.exclude_ood,
        )
    return env


def make_train_env(
    *,
    spec: EnvConfigSerializable,
    generator_path: Union[str, Path],
    dataset_path: Union[str, Path],
    splits_manifest: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    monitor_path: Optional[Union[str, Path]] = None,
) -> DummyVecEnv:
    """Build the *training* env wrapped in Monitor + DummyVecEnv.

    Args:
        spec: ``EnvConfigSerializable`` from ``BlueTeamRunConfig.env``.
        generator_path: Path to ``artifacts/generator/phase2``.
        dataset_path: Path to ``data/processed/ciciot2023``.
        splits_manifest: Path to the Phase-1 ``splits/manifest.json``.
            ``None`` short-circuits the split restriction (synthetic tests).
        seed: Optional seed for the underlying :class:`RealizationEngine`.
            SB3 will additionally call ``env.reset(seed=...)`` in
            ``learn(...)``; both seeds are consumed.
        monitor_path: Optional CSV path for SB3's :class:`Monitor`. When
            ``None``, ``Monitor`` runs in non-recording mode; the
            Phase-5 :class:`EpisodeJSONLCallback` is the canonical log.

    Returns:
        :class:`DummyVecEnv` wrapping a single :class:`Monitor` env.
    """
    def _factory() -> gym.Env:
        env = _build_env(
            spec=spec,
            generator_path=generator_path,
            dataset_path=dataset_path,
            splits_manifest=splits_manifest,
            seed=seed,
        )
        # Monitor is required by SB3's evaluate_policy() and by
        # learn()'s ep_info_buffer; we use it primarily so the env
        # returns ``info["episode"]`` on done. The CSV side-effect is
        # opt-in via monitor_path.
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)

    return DummyVecEnv([_factory])


def make_eval_env(
    *,
    spec: EnvConfigSerializable,
    generator_path: Union[str, Path],
    dataset_path: Union[str, Path],
    splits_manifest: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
) -> DummyVecEnv:
    """Build the *evaluation* env. Same plumbing as :func:`make_train_env`,
    no Monitor CSV side-effect."""
    return make_train_env(
        spec=spec,
        generator_path=generator_path,
        dataset_path=dataset_path,
        splits_manifest=splits_manifest,
        seed=seed,
        monitor_path=None,
    )


__all__ = ["make_eval_env", "make_train_env"]
