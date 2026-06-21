"""Split-aware env factories for Blue-Team RL training.

Two public functions:

- :func:`make_train_env` — builds an :class:`AdversarialIoTEnv` whose
  :class:`RealizationEngine` is restricted to the dataset-prep *train* split
  (with OOD-attack rows excluded). Wrapped in SB3's
  :class:`Monitor` and a :class:`DummyVecEnv` so the SB3 algorithms see
  the canonical vectorised interface.
- :func:`make_eval_env` — same plumbing but pointed at a different
  split (caller-supplied via ``spec.split`` — typically
  ``"val_balanced"`` for Blue-Team eval, ``"test_balanced"`` for
  benchmark / ablation evaluation) and *without* a ``Monitor`` log
  file. Used both by SB3's eval rollouts and by Blue-Team's
  :class:`EvalToJSONLCallback`. **Step-5 F4 / Step-8 doc-fix:**
  earlier docstrings claimed "default `val_balanced`"; the function
  imposes no default, the split is always caller-supplied via the
  ``spec: EnvConfigSerializable`` argument.

Both factories accept a ``BlueTeamRunConfig.env`` /
``BlueTeamRunConfig.eval_env`` :class:`EnvConfigSerializable` and the
dataset-prep paths.

The factories deliberately do NOT call ``env.reset(seed=...)``; SB3's
``learn(...)`` and ``evaluate_policy(...)`` paths take care of that and
will mis-align the RNG if we try to seed twice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.blue_team.run_config import EnvConfigSerializable
from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv
from src.utils.realization_engine import RealizationEngine

logger = logging.getLogger(__name__)


def _build_env_config(spec: EnvConfigSerializable) -> AdversarialEnvConfig:
    """Translate a serialisable env spec into an :class:`AdversarialEnvConfig`.

    Blue-Team forwarded only the lifecycle + sampling fields; reward
    coefficients kept their environment-design frozen defaults. ablation (PLAN
    §3.1.2 / D7.3) forwards the **full** field set so the F9
    reward-component sweep can override individual coefficients
    per-cell. ``EnvConfigSerializable`` defaults match
    :class:`AdversarialEnvConfig` defaults, so when nothing is
    overridden the resulting env config is byte-for-byte identical
    to the Blue-Team baseline.
    """
    return AdversarialEnvConfig(
        # Lifecycle + sampling (Blue-Team fields)
        max_steps=spec.max_steps,
        min_episode_length=spec.min_episode_length,
        p_defender_deescalation=spec.p_defender_deescalation,
        window_size=spec.window_size,
        include_deltas=spec.include_deltas,
        # Tug-of-war dynamics (headline contract)
        tug_of_war=spec.tug_of_war,
        p_onset=spec.p_onset,
        p_onset_access=spec.p_onset_access,
        p_down=spec.p_down,
        p_up=spec.p_up,
        p_down_isolate=spec.p_down_isolate,
        # ablation D7.3
        impact_is_terminal=spec.impact_is_terminal,
        # Stage-prediction ablation (review 2.4.1)
        stage_detector_path=spec.stage_detector_path,
        include_stage_pred=spec.include_stage_pred,
        # Non-monotonic attacker stress-test (review 2.4.3)
        retreat_prob=spec.retreat_prob,
        # Evasion-before-commit reactive attacker
        evasion_prob=spec.evasion_prob,
        # Proximity-coupled prevention model
        prevention_bonus=spec.prevention_bonus,
        # Reward shaping (ablation F9 axes)
        action_cost_scale=spec.action_cost_scale,
        reward_proportional=spec.reward_proportional,
        penalty_disproportionate=spec.penalty_disproportionate,
        proportional_bonus_cap=spec.proportional_bonus_cap,
        reward_deescalation=spec.reward_deescalation,
        deescalation_bonus_cap=spec.deescalation_bonus_cap,
        reward_mode=spec.reward_mode,
        impact_penalty=spec.impact_penalty,
        penalty_missed_impact=spec.penalty_missed_impact,
        defense_success_bonus=spec.defense_success_bonus,
        reward_benign_passive=spec.reward_benign_passive,
        penalty_overreact_benign=spec.penalty_overreact_benign,
        penalty_block_benign=spec.penalty_block_benign,
        penalty_block_recon=spec.penalty_block_recon,
        # Lagrangian FPR penalty (review 2.2 / Direction 6)
        fpr_penalty_beta=spec.fpr_penalty_beta,
        # Partial-observability redesign (sequential POMDP)
        aliasing_rate=spec.aliasing_rate,
        session_coherent=spec.session_coherent,
        no_post_transition_leak=spec.no_post_transition_leak,
        proximity_coupled=spec.proximity_coupled,
        proximity_min_escalation=spec.proximity_min_escalation,
    )


def _build_env(
    *,
    spec: EnvConfigSerializable,
    dataset_path: str | Path,
    splits_manifest: str | Path | None,
    seed: int | None,
) -> AdversarialIoTEnv:
    """Assemble an unwrapped :class:`AdversarialIoTEnv` with split-aware
    feature sampling.

    ``splits_manifest`` may be ``None`` for synthetic-data tests that do
    not carry a dataset-prep manifest; in that case we fall back to a vanilla
    :class:`RealizationEngine` over the entire dataset directory.
    """
    env_cfg = _build_env_config(spec)
    env = AdversarialIoTEnv(
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
            spec.split,
            spec.exclude_ood,
        )
    return env


def make_train_env(
    *,
    spec: EnvConfigSerializable,
    dataset_path: str | Path,
    splits_manifest: str | Path | None = None,
    seed: int | None = None,
    monitor_path: str | Path | None = None,
) -> DummyVecEnv:
    """Build the *training* env wrapped in Monitor + DummyVecEnv.

    Args:
        spec: ``EnvConfigSerializable`` from ``BlueTeamRunConfig.env``.
        dataset_path: Path to ``data/processed/ciciot2023``.
        splits_manifest: Path to the dataset-prep ``splits/manifest.json``.
            ``None`` short-circuits the split restriction (synthetic tests).
        seed: Optional seed for the underlying :class:`RealizationEngine`.
            SB3 will additionally call ``env.reset(seed=...)`` in
            ``learn(...)``; both seeds are consumed.
        monitor_path: Optional CSV path for SB3's :class:`Monitor`. When
            ``None``, ``Monitor`` runs in non-recording mode; the
            Blue-Team :class:`EpisodeJSONLCallback` is the canonical log.

    Returns:
        :class:`DummyVecEnv` wrapping a single :class:`Monitor` env.
    """

    def _factory() -> gym.Env:
        env = _build_env(
            spec=spec,
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
    dataset_path: str | Path,
    splits_manifest: str | Path | None = None,
    seed: int | None = None,
) -> DummyVecEnv:
    """Build the *evaluation* env. Same plumbing as :func:`make_train_env`,
    no Monitor CSV side-effect."""
    return make_train_env(
        spec=spec,
        dataset_path=dataset_path,
        splits_manifest=splits_manifest,
        seed=seed,
        monitor_path=None,
    )


__all__ = ["make_eval_env", "make_train_env"]
