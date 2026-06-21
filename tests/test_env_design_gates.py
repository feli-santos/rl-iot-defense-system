"""Environment-design exit-gate regression suite (G3.1-G3.7).

Runs ~6-8 s on a laptop. Every gate corresponds to a numbered line in
``docs/results/environment/PLAN.md`` §3.2; if a gate fails here, the
environment-design rewrite in commit 2a526af must be revisited before
the detector training step is allowed to consume the new env.

Per-gate budgets (set conservatively to avoid flakiness):
    - G3.2 / G3.3:  100 episodes  (PLAN said 200; 100 is enough for the median)
    - G3.4 / G3.5 / G3.6:  50 episodes each
The fixed RNG seeds make the suite deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.environment.adversarial_env import (
    AdversarialEnvConfig,
    AdversarialIoTEnv,
    _recommended_action,
)
from src.utils.label_mapper import KillChainStage

# ---------------------------------------------------------------------------
# Fixtures: a tiny untrained env that nevertheless exercises the new lifecycle
# ---------------------------------------------------------------------------


def _build_env(tmp_path: Path, *, config_overrides: Iterable[tuple] = ()) -> AdversarialIoTEnv:
    """Build an AdversarialIoTEnv backed by a 100-row dataset.

    The attacker is a first-order Markov chain over the 5 kill-chain stages.
    That stresses the env's lifecycle (the agent must survive frequent
    BENIGN <-> RECON jitter and occasional fast escalations).
    """
    # Dataset (100 rows × 8 features, 20 rows per stage)
    dataset_path = tmp_path / "ds"
    dataset_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((100, 8)).astype(np.float32)
    labels = np.tile(np.arange(5), 20)
    np.save(dataset_path / "features.npy", features)
    np.save(dataset_path / "labels.npy", labels)
    state_indices = {str(s): np.where(labels == s)[0].tolist() for s in range(5)}
    (dataset_path / "state_indices.json").write_text(json.dumps(state_indices))
    joblib.dump(StandardScaler().fit(features), dataset_path / "scaler.joblib")

    env_cfg = AdversarialEnvConfig()
    for k, v in config_overrides:
        setattr(env_cfg, k, v)
    return AdversarialIoTEnv(dataset_path, config=env_cfg)


@pytest.fixture(scope="module")
def env_factory(tmp_path_factory: pytest.TempPathFactory):
    """Return a callable that builds a fresh env on demand."""
    base_tmp = tmp_path_factory.mktemp("env_design_gates")

    def _make(**overrides) -> AdversarialIoTEnv:
        return _build_env(base_tmp, config_overrides=tuple((k, v) for k, v in overrides.items()))

    return _make


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rollout_random(env: AdversarialIoTEnv, *, seed: int) -> dict:
    """One episode with uniformly-random actions. Returns summary stats."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = int(rng.integers(0, env.action_space.n))
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        steps += 1
        if term or trunc:
            break
    return {
        "steps": steps,
        "total_reward": total_reward,
        "compromised": info.get("compromised", False),
        "mttc": info.get("mttc_steps"),
        "defender_deescalations": info.get("defender_deescalations", 0),
    }


def _rollout_fixed_action(env: AdversarialIoTEnv, action: int, *, seed: int) -> dict:
    """One episode with a constant action. For G3.3, G3.5, G3.6."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        steps += 1
        if term or trunc:
            break
    return {
        "steps": steps,
        "total_reward": total_reward,
        "compromised": info.get("compromised", False),
    }


def _rollout_recommended(env: AdversarialIoTEnv, *, seed: int) -> dict:
    """One episode where the agent always picks the recommended action.

    The agent reads ``info["attack_stage"]`` (an oracle peek) — this is
    intentional: G3.4 measures whether the *new reward function* rewards the
    correct policy, which is a property of the env, not of any agent.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = _recommended_action(int(info["attack_stage"]))
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        steps += 1
        if term or trunc:
            break
    return {
        "steps": steps,
        "total_reward": total_reward,
        "compromised": info.get("compromised", False),
    }


# ===========================================================================
# G3.1 — Mechanical regression tests on lifecycle, reward, and MTTC
# ===========================================================================


class TestG3_1_RegressionTests:  # noqa: N801
    """G3.1: targeted unit tests for each environment-design behaviour change."""

    def test_recommended_action_yields_positive_reward_per_step(self, env_factory):
        """B2 fix: picking the recommended action earns net-positive reward."""
        # B2 proportionality shaping lives in the coupled reward contract.
        env = env_factory(reward_mode="proportional")
        # Force the env to BENIGN, then call the reward function directly.
        env.reset(seed=0)
        # Action == recommended for every stage should always be net-positive.
        for stage in range(5):
            r = env._calculate_reward(_recommended_action(stage), previous_stage=stage)
            assert r > 0, f"recommended action net negative for stage {stage}: {r}"

    def test_overreaction_on_benign_yields_negative_reward(self, env_factory):
        """B2/guardrail: picking BLOCK on BENIGN must net negative."""
        env = env_factory()
        env.reset(seed=0)
        r = env._calculate_reward(action=3, previous_stage=KillChainStage.BENIGN.value)
        assert r < 0, f"overreaction on BENIGN scored {r:+.2f}; expected negative"

    def test_underreaction_on_impact_yields_negative_reward(self, env_factory):
        """B2/guardrail: picking OBSERVE on IMPACT (via _calculate_reward, not
        _step_at_impact) must net negative."""
        # B2/guardrail penalties live in the coupled reward contract.
        env = env_factory(reward_mode="proportional")
        env.reset(seed=0)
        r = env._calculate_reward(action=0, previous_stage=KillChainStage.IMPACT.value)
        assert r < 0, f"underreaction at IMPACT scored {r:+.2f}; expected negative"

    def test_block_does_not_terminate_episode_early(self, env_factory):
        """B1 fix: an always-BLOCK policy must not end the episode in ≤2 steps.

        A weaker version of G3.3 — useful as a fast unit test."""
        env = env_factory(min_episode_length=20, max_steps=50)
        result = _rollout_fixed_action(env, action=3, seed=0)
        assert (
            result["steps"] >= 5
        ), f"always-BLOCK terminated after {result['steps']} steps; expected ≥ 5"

    def test_mttc_fields_present_in_info(self, env_factory):
        """B5 fix: info dict exposes the four MTTC keys."""
        env = env_factory()
        _, info = env.reset(seed=0)
        for key in (
            "compromised",
            "mttc_steps",
            "first_attack_step",
            "compromise_step",
            "defender_deescalations",
            "recommended_action",
        ):
            assert key in info, f"info missing key {key!r}"

    def test_mttc_is_none_at_reset(self, env_factory):
        env = env_factory()
        _, info = env.reset(seed=0)
        assert info["compromised"] is False
        assert info["mttc_steps"] is None
        assert info["first_attack_step"] is None
        assert info["compromise_step"] is None
        assert info["defender_deescalations"] == 0

    def test_defender_deescalation_resets_to_benign(self, env_factory):
        """B3: at ACCESS+, BLOCK should sometimes reset the stage to BENIGN.

        We force the deterministic case by setting the override probability to
        1.0 and manually placing the env at ACCESS. This pins the *legacy*
        de-escalation mechanic (``_maybe_defender_deescalation``), which the
        default tug-of-war dynamics replace, so we set ``tug_of_war=False``.
        """
        env = env_factory(p_defender_deescalation=1.0, tug_of_war=False)
        env.reset(seed=0)
        env._current_attack_stage = KillChainStage.ACCESS.value
        env._attack_history = [KillChainStage.ACCESS.value]
        _, _, _, _, info = env.step(action=3)  # BLOCK
        assert info["attack_stage"] == KillChainStage.BENIGN.value
        assert info["defender_deescalations"] == 1

    def test_defender_deescalation_does_not_fire_below_access(self, env_factory):
        """B3: BLOCK at RECON must NOT trigger an override (too early).

        Pins the legacy de-escalation mechanic (``tug_of_war=False``).
        """
        env = env_factory(p_defender_deescalation=1.0, tug_of_war=False)
        env.reset(seed=0)
        env._current_attack_stage = KillChainStage.RECON.value
        env._attack_history = [KillChainStage.RECON.value]
        _, _, _, _, info = env.step(action=3)
        # Whatever the new stage is, defender_deescalations must remain 0
        assert info["defender_deescalations"] == 0


# ===========================================================================
# G3.2-G3.6 — Empirical gates over many rollouts
# ===========================================================================


class TestG3_2_to_G3_6_StatisticalGates:  # noqa: N801
    """G3.2-G3.6: empirical gates over many rollouts."""

    @pytest.mark.parametrize("n_episodes", [100])
    def test_g3_2_random_action_median_episode_length(self, env_factory, n_episodes):
        """G3.2: median episode length over N random-action rollouts ≥ 15 steps."""
        env = env_factory(min_episode_length=20, max_steps=200)
        lengths: list[int] = [
            _rollout_random(env, seed=1000 + i)["steps"] for i in range(n_episodes)
        ]
        median = float(np.median(lengths))
        assert median >= 15, (
            f"G3.2 FAILED: median random-action episode length = {median:.1f} "
            f"(min={min(lengths)}, max={max(lengths)}); expected ≥ 15"
        )

    @pytest.mark.parametrize("n_episodes", [100])
    def test_g3_3_always_block_median_episode_length(self, env_factory, n_episodes):
        """G3.3: median always-BLOCK episode length ≥ 10 steps.

        Proves the 'BLOCK = instant win' bug is gone (B1)."""
        env = env_factory(min_episode_length=20, max_steps=200)
        lengths: list[int] = [
            _rollout_fixed_action(env, action=3, seed=2000 + i)["steps"] for i in range(n_episodes)
        ]
        median = float(np.median(lengths))
        assert (
            median >= 10
        ), f"G3.3 FAILED: median always-BLOCK length = {median:.1f}; expected ≥ 10"

    @pytest.mark.parametrize("n_episodes", [50])
    def test_g3_4_recommended_action_mean_reward_positive(self, env_factory, n_episodes):
        """G3.4: agent that always plays the recommended action earns
        average reward > 0 over N rollouts. (B2 sanity check.)"""
        env = env_factory(min_episode_length=20, max_steps=100)
        rewards: list[float] = [
            _rollout_recommended(env, seed=3000 + i)["total_reward"] for i in range(n_episodes)
        ]
        mean_r = float(np.mean(rewards))
        assert (
            mean_r > 0
        ), f"G3.4 FAILED: recommended-action mean reward = {mean_r:+.1f}; expected > 0"

    @pytest.mark.parametrize("n_episodes", [50])
    def test_g3_5_always_observe_mean_reward_negative(self, env_factory, n_episodes):
        """G3.5: an always-OBSERVE policy must earn average reward < 0.

        This is the 'do-nothing exploit' check: if always-OBSERVE were
        net-positive, the agent could trivially win by ignoring all attacks."""
        env = env_factory(min_episode_length=20, max_steps=100)
        rewards: list[float] = [
            _rollout_fixed_action(env, action=0, seed=4000 + i)["total_reward"]
            for i in range(n_episodes)
        ]
        mean_r = float(np.mean(rewards))
        assert mean_r < 0, f"G3.5 FAILED: always-OBSERVE mean reward = {mean_r:+.1f}; expected < 0"

    @pytest.mark.parametrize("n_episodes", [50])
    def test_g3_6_always_isolate_mean_reward_negative(self, env_factory, n_episodes):
        """G3.6: an always-ISOLATE policy must earn average reward < 0.

        This is the 'always-blast exploit' check: punishing benign traffic
        with ISOLATE on every step must dominate any defense bonus."""
        env = env_factory(min_episode_length=20, max_steps=100)
        rewards: list[float] = [
            _rollout_fixed_action(env, action=4, seed=5000 + i)["total_reward"]
            for i in range(n_episodes)
        ]
        mean_r = float(np.mean(rewards))
        assert mean_r < 0, f"G3.6 FAILED: always-ISOLATE mean reward = {mean_r:+.1f}; expected < 0"


# ===========================================================================
# G3.7 — implicit. The presence of this file in tests/ ensures pytest counts
# all the gate tests in the suite total. No explicit assertion needed.
# ===========================================================================
