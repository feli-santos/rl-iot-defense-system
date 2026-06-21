"""Ablation: ``impact_is_terminal`` env-config flag (D7.3).

Pins the ``AdversarialEnvConfig.impact_is_terminal`` codepath. The default
value is ``True``, which preserves the environment-design frozen contract
byte-for-byte; ``False`` enables a separate explicit IMPACT-row decision
step before termination, used as one binary axis of the F9 reward-component
sweep (PLAN §3.1.4 / D7.3).

Synthetic-only — uses the same tiny untrained LSTM + 100-row dataset
fixture as ``tests/test_env_design_gates.py``. No real-data dependency.
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
)
from src.utils.label_mapper import KillChainStage

# ---------------------------------------------------------------------------
# Fixture (mirrors tests/test_env_design_gates.py:_build_env)
# ---------------------------------------------------------------------------


def _build_env(
    tmp_path: Path,
    *,
    config_overrides: Iterable[tuple] = (),
) -> AdversarialIoTEnv:
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

    # These tests pin the ``impact_is_terminal`` contract by monkey-patching
    # ``_advance_attack`` (the legacy autonomous-Markov advance) via the
    # ``_force_into_impact`` helper to drive the env deterministically to
    # IMPACT. The default tug-of-war dynamics route progression through
    # ``_advance_tug_of_war`` instead, bypassing that patch, so we pin the
    # legacy path here. ``impact_is_terminal`` semantics are orthogonal to the
    # progression rule, so this does not weaken the contract under test.
    #
    # We also disable the per-episode proportional-bonus cap
    # (``proportional_bonus_cap=None``). The ``_force_into_impact`` helper rolls
    # ~20 BENIGN warmup steps, each earning the +5 proportionality bonus; with
    # the default cap (100 = 20*5) that budget is fully consumed before the
    # IMPACT-arrival step, which would zero out the +5 the arrival-step reward
    # assertions expect. The cap is an orthogonal reward-shaping concern, so we
    # switch it off to isolate the ``impact_is_terminal`` reward shape.
    # These tests pin the *coupled* (proportional) reward shape (e.g. the +5
    # proportionality band on the IMPACT-arrival step), so the base config uses
    # ``reward_mode="proportional"``; the default deployment contract is now the
    # sparse ``"outcome"`` mode, which would strip that shaping.
    env_cfg = AdversarialEnvConfig(
        tug_of_war=False,
        proportional_bonus_cap=None,
        reward_mode="proportional",
    )
    for k, v in config_overrides:
        setattr(env_cfg, k, v)
    return AdversarialIoTEnv(dataset_path, config=env_cfg)


@pytest.fixture(scope="function")
def env_factory(tmp_path: Path):
    """Function-scoped fixture: each test gets a fresh env.

    These tests deliberately mutate env internals (force-set
    ``_current_attack_stage``, monkey-patch ``_advance_attack``) to
    drive the env to IMPACT deterministically. A module-scoped fixture
    would let those mutations leak across tests, so we pay the
    per-test cost of rebuilding the tiny LSTM + 100-row dataset
    (≈ 0.4 s per test) in exchange for state isolation.
    """

    def _make(**overrides) -> AdversarialIoTEnv:
        return _build_env(tmp_path, config_overrides=tuple((k, v) for k, v in overrides.items()))

    return _make


# ---------------------------------------------------------------------------
# Helpers — drive the env to IMPACT deterministically by manipulating the
# internal stage. The LSTM's stochastic transitions don't reliably reach
# IMPACT in a small number of steps; for these tests we want a deterministic
# IMPACT-arrival step so the assertions can be made without flakiness.
# ---------------------------------------------------------------------------


def _force_into_impact(
    env: AdversarialIoTEnv,
    action_into_impact: int,
    *,
    seed: int = 0,
) -> tuple:
    """Drive ``env`` to a step where the env *just transitioned* to IMPACT.

    Returns the ``(obs, reward, terminated, truncated, info)`` tuple from
    that step. The agent's action on the IMPACT-arrival step is
    ``action_into_impact`` — relevant only for the
    ``impact_is_terminal=True`` Phase-3 frozen branch (in the False
    branch, the IMPACT-arrival action does NOT trigger the terminal
    reward shape, only the next step does).

    Implementation: monkey-patch ``_advance_attack`` from the very first
    step so the LSTM is never consulted; the patched advance keeps the
    env at BENIGN throughout the warmup (so the lifecycle clamp never
    sees an IMPACT transition to clamp), then on the *target* step
    flips to IMPACT. We also set the env's stage to MANEUVER right
    before the target step so the IMPACT inline-terminal branch's
    ``decision_stage`` is MANEUVER (not BENIGN), matching the
    "natural" MANEUVER → IMPACT transition that Phase-3's reward
    formula was designed for.
    """
    obs, info = env.reset(seed=seed)

    # Replace _advance_attack with a deterministic stub that keeps the
    # attack at BENIGN. This isolates the test from LSTM stochasticity
    # AND prevents accidental IMPACT transitions during warmup.
    original_advance = env._advance_attack

    def _stay_benign() -> None:
        env._current_attack_stage = KillChainStage.BENIGN.value
        env._attack_history.append(env._current_attack_stage)

    env._advance_attack = _stay_benign  # type: ignore[method-assign]
    try:
        # Roll past min_episode_length with OBSERVE — env stays BENIGN.
        while env._step_count < env._config.min_episode_length:
            obs, r, term, trunc, info = env.step(0)  # OBSERVE
            if term or trunc:
                raise RuntimeError(
                    "Env terminated unexpectedly during warmup; the "
                    "_stay_benign stub should keep the env BENIGN forever."
                )

        # Target step: previous_attack_stage = MANEUVER, advance lands
        # on IMPACT.
        env._current_attack_stage = KillChainStage.MANEUVER.value
        env._attack_history.append(env._current_attack_stage)

        def _force_impact_advance() -> None:
            env._current_attack_stage = KillChainStage.IMPACT.value
            env._attack_history.append(env._current_attack_stage)

        env._advance_attack = _force_impact_advance  # type: ignore[method-assign]
        result = env.step(action_into_impact)
    finally:
        env._advance_attack = original_advance  # type: ignore[method-assign]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImpactIsTerminalDefault:
    """Default ``impact_is_terminal=True`` preserves the environment-design
    frozen contract byte-for-byte."""

    def test_default_value_is_true(self, env_factory):
        """A bare ``AdversarialEnvConfig()`` has ``impact_is_terminal=True``."""
        cfg = AdversarialEnvConfig()
        assert cfg.impact_is_terminal is True, (
            "Default impact_is_terminal must be True to preserve the "
            "environment-design frozen contract. Changing the default would "
            "invalidate blue-team and benchmark trained checkpoints."
        )

    def test_default_terminates_at_impact_arrival(self, env_factory):
        """With the default config, the step that lands on IMPACT
        terminates the episode (environment-design frozen lifecycle)."""
        env = env_factory()  # impact_is_terminal=True by default
        obs, reward, terminated, truncated, info = _force_into_impact(
            env,
            action_into_impact=4,  # ISOLATE
        )
        assert terminated is True, (
            "Environment-design frozen contract: episode must terminate the same "
            "step IMPACT arrives when impact_is_terminal=True (default)."
        )
        assert info["attack_stage"] == KillChainStage.IMPACT.value

    def test_default_isolate_at_impact_arrival_nets_partial_mitigation(self, env_factory):
        """With the default, ISOLATE on the IMPACT-arrival step earns
        the inline terminal reward shape: -impact_penalty
        +defense_success_bonus + the proportionality reward for picking
        ISOLATE on MANEUVER (the decision-time stage)."""
        env = env_factory()
        obs, reward, terminated, truncated, info = _force_into_impact(env, action_into_impact=4)
        # MANEUVER -> ISOLATE: action_cost=-0.8, prop_band(|4-3|=1)=+5,
        # then -impact_penalty(200) +defense_success_bonus(250) inline.
        # Expected: -0.8 + 5 - 200 + 250 = +54.2
        assert reward == pytest.approx(54.2, abs=0.5), (
            f"Expected ~+54.2 for ISOLATE on IMPACT-arrival under "
            f"impact_is_terminal=True; got {reward:+.4f}."
        )


class TestImpactIsTerminalFalse:
    """``impact_is_terminal=False`` defers the terminal reward to an
    explicit IMPACT-row decision on the *next* step."""

    def test_false_does_not_terminate_at_impact_arrival(self, env_factory):
        """When False, the IMPACT-arrival step returns terminated=False
        — the agent gets one more step to make an explicit IMPACT-row
        decision."""
        env = env_factory(impact_is_terminal=False)
        obs, reward, terminated, truncated, info = _force_into_impact(env, action_into_impact=4)
        assert terminated is False, (
            "impact_is_terminal=False must NOT terminate on the "
            "IMPACT-arrival step; the agent gets a separate explicit "
            "IMPACT-row decision step."
        )
        assert (
            info["attack_stage"] == KillChainStage.IMPACT.value
        ), "The env still transitioned the attack to IMPACT; only termination is deferred."

    def test_false_arrival_step_skips_inline_terminal_reward(self, env_factory):
        """When False, the IMPACT-arrival step's reward is just the
        non-IMPACT decision reward (MANEUVER -> ISOLATE proportional
        bonus minus action cost), NOT the terminal -impact_penalty
        +defense_success_bonus shape."""
        env = env_factory(impact_is_terminal=False)
        obs, reward, terminated, truncated, info = _force_into_impact(env, action_into_impact=4)
        # MANEUVER -> ISOLATE: action_cost=-0.8, prop_band=+5, no
        # terminal reward applied. Expected: ~+4.2.
        assert reward == pytest.approx(4.2, abs=0.5), (
            f"Expected ~+4.2 (non-terminal MANEUVER->ISOLATE reward) on "
            f"IMPACT-arrival under impact_is_terminal=False; got "
            f"{reward:+.4f}. The terminal IMPACT-penalty shape must "
            f"only be applied on the explicit IMPACT-row step."
        )

    def test_false_explicit_impact_row_decision_isolate(self, env_factory):
        """When False, calling step() AGAIN after IMPACT arrival routes
        through ``_step_at_impact`` and applies the canonical IMPACT-row
        terminal reward for ISOLATE: -impact_penalty -ISOLATE_cost
        +defense_success_bonus = -200 -0.8 +250 = +49.2."""
        env = env_factory(impact_is_terminal=False)
        # Step 1: arrival (any action), no termination.
        _force_into_impact(env, action_into_impact=0)
        # Step 2: explicit IMPACT-row decision = ISOLATE.
        obs, reward, terminated, truncated, info = env.step(4)
        assert terminated is True, (
            "After the IMPACT-arrival step (impact_is_terminal=False), "
            "the next step() call must route through _step_at_impact "
            "and terminate the episode."
        )
        assert reward == pytest.approx(49.2, abs=0.5), (
            f"Expected ~+49.2 for ISOLATE in the explicit IMPACT-row "
            f"decision step; got {reward:+.4f}. This is the canonical "
            f"_step_at_impact reward formula."
        )

    def test_false_explicit_impact_row_decision_observe_full_penalty(self, env_factory):
        """OBSERVE in the explicit IMPACT-row decision incurs the
        canonical -impact_penalty -penalty_missed_impact = -350 reward
        shape (no defense_success_bonus, no proportional reward)."""
        env = env_factory(impact_is_terminal=False)
        _force_into_impact(env, action_into_impact=0)
        obs, reward, terminated, truncated, info = env.step(0)  # OBSERVE
        assert terminated is True
        # OBSERVE: -impact_penalty(200) -ISOLATE_cost(0.0)
        #          -penalty_missed_impact(150) = -350.
        assert reward == pytest.approx(
            -350.0, abs=0.5
        ), f"Expected ~-350.0 for OBSERVE in explicit IMPACT-row decision; got {reward:+.4f}."

    def test_false_outcome_label_preserved_on_arrival_step(self, env_factory):
        """When False and IMPACT arrives via _advance_attack (not via
        defender_deescalation override), the arrival step's
        info["outcome"] label must be 'ongoing', NOT 'compromised' or
        'impact_*'. The terminal labelling happens on the explicit
        IMPACT-row step."""
        env = env_factory(impact_is_terminal=False)
        obs, reward, terminated, truncated, info = _force_into_impact(env, action_into_impact=4)
        assert info["outcome"] == "ongoing", (
            f"On IMPACT-arrival under impact_is_terminal=False, outcome "
            f"must remain 'ongoing' (the agent has not yet picked an "
            f"IMPACT response); got '{info['outcome']}'. The terminal "
            f"label is set on the explicit IMPACT-row step."
        )
