"""Tests for the Phase-6 baseline policies (PLAN §3.3, C2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.baseline_policies import (
    _RECOMMENDED_BY_STAGE,
    RFActingPolicy,
    SB3PolicyAdapter,
    always_block,
    always_observe,
    random_policy,
    recommended_action_policy,
)

# --------------------------------------------------------------- closed-form


class TestRandomPolicy:
    def test_returns_int_in_action_space(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = random_policy(np.zeros(10), {}, rng=rng)
            assert isinstance(a, int)
            assert 0 <= a <= 4

    def test_seeded_rng_is_reproducible(self) -> None:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        seq_a = [random_policy(np.zeros(1), {}, rng=rng_a) for _ in range(20)]
        seq_b = [random_policy(np.zeros(1), {}, rng=rng_b) for _ in range(20)]
        assert seq_a == seq_b

    def test_unseeded_rng_still_returns_valid_action(self) -> None:
        # rng=None path. Just sanity-check the return type/range; the
        # values are non-deterministic by design.
        a = random_policy(np.zeros(1), {})
        assert isinstance(a, int)
        assert 0 <= a <= 4


class TestConstantPolicies:
    def test_always_observe_returns_zero(self) -> None:
        assert always_observe(np.zeros(290), {}) == 0
        assert always_observe(np.ones(1), {"attack_stage": 4}) == 0

    def test_always_block_returns_three(self) -> None:
        assert always_block(np.zeros(290), {}) == 3
        assert always_block(np.zeros(1), {"recommended_action": 0}) == 3


class TestRecommendedActionPolicy:
    @pytest.mark.parametrize("stage,expected", list(_RECOMMENDED_BY_STAGE.items()))
    def test_passes_through_info_field(self, stage: int, expected: int) -> None:
        # The Phase-3 env writes recommended_action; the policy is a
        # one-line look-up. Pinning the *exact* mapping here doubles as
        # a regression guard against drift in adversarial_env.py.
        info = {"recommended_action": expected, "attack_stage": stage}
        assert recommended_action_policy(np.zeros(1), info) == expected

    def test_missing_key_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            recommended_action_policy(np.zeros(1), {})


# --------------------------------------------------------------- RF baseline


class _StubRF:
    """Sklearn-shaped RF whose ``predict`` is a deterministic function
    of the first feature value. Useful because we can pre-compute the
    expected action without committing the test to a real classifier."""

    def __init__(self, mapping: dict[float, int]) -> None:
        self._mapping = dict(mapping)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D401 — sklearn API
        out = []
        for row in np.asarray(X):
            key = float(row[0])
            out.append(self._mapping.get(key, 0))
        return np.asarray(out, dtype=np.int64)


class TestRFActingPolicy:
    @pytest.mark.parametrize(
        "predicted_stage, expected_action",
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
    )
    def test_maps_predicted_stage_to_recommended_action(
        self,
        predicted_stage: int,
        expected_action: int,
    ) -> None:
        # Stub RF: returns `predicted_stage` whenever the first feature
        # is exactly the marker value 0.5. Latest-step features start
        # with 0.5; the rest of the obs is irrelevant.
        rf = _StubRF({0.5: predicted_stage})
        F, W = 29, 5
        # Build an obs of shape (W, 2F) with the LATEST row's first
        # raw feature set to 0.5 and everything else zero.
        window = np.zeros((W, 2 * F), dtype=np.float32)
        window[W - 1, 0] = 0.5
        obs = window.flatten()

        pol = RFActingPolicy(rf, num_features=F, window_size=W, include_deltas=True)
        assert pol(obs, {"recommended_action": -1}) == expected_action

    def test_no_deltas_layout(self) -> None:
        rf = _StubRF({1.0: 3})
        F, W = 8, 4
        # Without deltas the row stride is F (not 2F).
        window = np.zeros((W, F), dtype=np.float32)
        window[W - 1, 0] = 1.0
        obs = window.flatten()
        pol = RFActingPolicy(
            rf,
            num_features=F,
            window_size=W,
            include_deltas=False,
        )
        # Predicted stage 3 (MANEUVER) → recommended action 3 (BLOCK).
        assert pol(obs, {}) == 3

    def test_wrong_obs_size_raises(self) -> None:
        rf = _StubRF({0.0: 0})
        pol = RFActingPolicy(rf, num_features=29, window_size=5)
        with pytest.raises(ValueError, match="expected obs of size"):
            pol(np.zeros(13), {})

    def test_out_of_range_stage_prediction_raises(self) -> None:
        rf = _StubRF({0.0: 7})  # invalid stage
        pol = RFActingPolicy(rf, num_features=29, window_size=5)
        # Need an obs of the right size; first feature is 0.0 by default.
        obs = np.zeros(5 * 29 * 2, dtype=np.float32)
        with pytest.raises(ValueError, match="out-of-range"):
            pol(obs, {})

    def test_non_predict_object_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a sklearn-like classifier"):
            RFActingPolicy(object(), num_features=29, window_size=5)

    def test_invalid_dimensions_raise(self) -> None:
        rf = _StubRF({})
        with pytest.raises(ValueError, match="num_features must be >= 1"):
            RFActingPolicy(rf, num_features=0)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            RFActingPolicy(rf, window_size=0)


# --------------------------------------------------------------- SB3 adapter


class _StubSB3Model:
    """Returns a fixed scripted action sequence (round-robin)."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = list(actions)
        self._i = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True):  # noqa: D401
        # noqa: ARG002 — `deterministic` is recorded for assertion
        self.last_deterministic = deterministic
        a = self._actions[self._i % len(self._actions)]
        self._i += 1
        # SB3 Discrete action-space returns shape (n_envs,) array.
        n_envs = obs.shape[0] if obs.ndim == 2 else 1
        return np.full((n_envs,), a, dtype=np.int64), None


class TestSB3PolicyAdapter:
    def test_round_trips_scripted_actions(self) -> None:
        model = _StubSB3Model([2, 4, 0])
        ad = SB3PolicyAdapter(model)
        # 1-D obs → adapter must add the batch dim before calling predict.
        obs1d = np.zeros(290, dtype=np.float32)
        assert ad(obs1d, {}) == 2
        # 2-D obs (already batched) is also OK.
        obs2d = np.zeros((1, 290), dtype=np.float32)
        assert ad(obs2d, {}) == 4
        assert ad(obs1d, {}) == 0
        assert model.last_deterministic is True

    def test_deterministic_flag_propagates(self) -> None:
        model = _StubSB3Model([1])
        ad = SB3PolicyAdapter(model, deterministic=False)
        ad(np.zeros(10), {})
        assert model.last_deterministic is False

    def test_non_predict_object_raises(self) -> None:
        with pytest.raises(TypeError, match="must expose .predict"):
            SB3PolicyAdapter(object())

    def test_feedforward_adapter_is_not_recurrent(self) -> None:
        # A plain (non-RecurrentPPO) model must use the byte-identical
        # feedforward path: no LSTM state, predict called without
        # state/episode_start kwargs.
        model = _StubSB3Model([3])
        ad = SB3PolicyAdapter(model)
        assert ad._is_recurrent is False
        assert ad(np.zeros(10), {"decision_step": 0}) == 3


class RecurrentPPO:  # noqa: N801 — name must match the SB3-contrib class
    """Stub whose class name triggers the adapter's recurrent path.

    ``SB3PolicyAdapter`` detects recurrence via
    ``type(model).__name__ == "RecurrentPPO"``. This stub records the
    ``state`` and ``episode_start`` it was called with and returns a
    monotonically-incrementing fake LSTM state so the adapter's
    carry/reset behaviour is observable.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        self.calls.append(
            {
                "state_in": state,
                "episode_start": None
                if episode_start is None
                else bool(np.asarray(episode_start).reshape(-1)[0]),
                "deterministic": deterministic,
            }
        )
        self._counter += 1
        new_state = (f"lstm_state_{self._counter}",)
        n_envs = obs.shape[0] if obs.ndim == 2 else 1
        return np.full((n_envs,), 1, dtype=np.int64), new_state


class TestRecurrentSB3PolicyAdapter:
    def test_detected_as_recurrent(self) -> None:
        ad = SB3PolicyAdapter(RecurrentPPO())
        assert ad._is_recurrent is True
        assert ad._lstm_state is None

    def test_episode_start_resets_then_carries_state(self) -> None:
        model = RecurrentPPO()
        ad = SB3PolicyAdapter(model)
        obs = np.zeros(8, dtype=np.float32)

        # First decision of episode: decision_step == 0 → episode_start True,
        # state reset to None before predict.
        ad(obs, {"decision_step": 0})
        assert model.calls[0]["episode_start"] is True
        assert model.calls[0]["state_in"] is None
        # Adapter must store the returned LSTM state.
        assert ad._lstm_state == ("lstm_state_1",)

        # Mid-episode step: episode_start False, carries the prior state in.
        ad(obs, {"decision_step": 1})
        assert model.calls[1]["episode_start"] is False
        assert model.calls[1]["state_in"] == ("lstm_state_1",)
        assert ad._lstm_state == ("lstm_state_2",)

        # New episode boundary: decision_step back to 0 → reset to None.
        ad(obs, {"decision_step": 0})
        assert model.calls[2]["episode_start"] is True
        assert model.calls[2]["state_in"] is None

    def test_missing_decision_step_defaults_to_episode_start(self) -> None:
        # An empty info dict is treated as the start of an episode
        # (decision_step defaults to 0), which is the safe reset behaviour.
        model = RecurrentPPO()
        ad = SB3PolicyAdapter(model)
        ad(np.zeros(8), {})
        assert model.calls[0]["episode_start"] is True
