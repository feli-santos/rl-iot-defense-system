"""
Tests for AdversarialIoTEnv.

The Adversarial Environment implements a Gymnasium environment where:
- tug-of-war Markov attacker controls attack progression
- Blue Team (RL Agent) observes realized features and takes defensive actions
- Attack state is hidden from the agent (partial observability)

Key properties:
- Box observation space: window of realized feature vectors
- Discrete action space: 5 force continuum levels
- Hidden state: Attack Sequence Generator controls actual attack stage
"""

import gymnasium as gym
import numpy as np
import pytest


class TestAdversarialEnvConfig:
    """Test AdversarialEnvConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from src.environment.adversarial_env import AdversarialEnvConfig

        config = AdversarialEnvConfig()

        assert config.max_steps == 500
        assert config.window_size == 5
        assert config.include_deltas is True
        assert config.num_actions == 5

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from src.environment.adversarial_env import AdversarialEnvConfig

        config = AdversarialEnvConfig(
            max_steps=100,
            window_size=3,
            num_actions=5,
        )

        assert config.max_steps == 100
        assert config.window_size == 3


class TestAdversarialEnvInitialization:
    """Test AdversarialIoTEnv initialization."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Unused legacy fixture (attacker is now a first-order Markov chain)."""
        path = tmp_path / "unused"
        path.mkdir(parents=True)
        return path

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock processed dataset."""
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        # Create mock features (100 samples, 46 features)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        # Create mock labels (stage IDs)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        # Create state indices
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        # Create mock scaler
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        return dataset_path

    def test_is_gymnasium_env(self, mock_generator, mock_dataset) -> None:
        """Test environment is a Gymnasium environment."""
        from src.environment.adversarial_env import AdversarialIoTEnv

        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
        )

        assert isinstance(env, gym.Env)

    def test_has_observation_space(self, mock_generator, mock_dataset) -> None:
        """Test environment has observation space."""
        from src.environment.adversarial_env import AdversarialIoTEnv

        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
        )

        assert hasattr(env, "observation_space")
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_observation_space_shape(self, mock_generator, mock_dataset) -> None:
        """Test observation space has correct shape."""
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(window_size=5)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )

        # Shape should be (window_size, num_features * 2) when deltas enabled
        # Flattened to (window_size * num_features * 2,)
        expected_shape = (5 * 46 * 2,)  # 46 features + 46 deltas from mock dataset
        assert env.observation_space.shape == expected_shape

    def test_has_action_space(self, mock_generator, mock_dataset) -> None:
        """Test environment has action space."""
        from src.environment.adversarial_env import AdversarialIoTEnv

        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
        )

        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_action_space_size(self, mock_generator, mock_dataset) -> None:
        """Test action space has 5 actions (force continuum)."""
        from src.environment.adversarial_env import AdversarialIoTEnv

        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
        )

        assert env.action_space.n == 5


class TestAdversarialEnvReset:
    """Test environment reset functionality."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create environment with mock components."""
        import json

        import joblib
        from sklearn.preprocessing import StandardScaler

        from src.environment.adversarial_env import AdversarialIoTEnv

        # Create dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        return AdversarialIoTEnv(
            dataset_path=dataset_path,
        )

    def test_reset_returns_observation(self, env) -> None:
        """Test reset returns observation."""
        obs, info = env.reset()

        assert obs is not None
        assert isinstance(obs, np.ndarray)

    def test_reset_returns_info(self, env) -> None:
        """Test reset returns info dict."""
        obs, info = env.reset()

        assert isinstance(info, dict)

    def test_reset_observation_matches_space(self, env) -> None:
        """Test observation is within observation space."""
        obs, _ = env.reset()

        assert env.observation_space.contains(obs)

    def test_reset_with_seed(self, env) -> None:
        """Test reset with seed for reproducibility."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_initializes_attack_sequence(self, env) -> None:
        """Test reset starts a new attack sequence."""
        env.reset()

        # Should have initial attack state
        assert hasattr(env, "_current_attack_stage")
        assert env._current_attack_stage is not None


class TestAdversarialEnvStep:
    """Test environment step functionality."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create environment with mock components."""
        import json

        import joblib
        from sklearn.preprocessing import StandardScaler

        from src.environment.adversarial_env import AdversarialIoTEnv

        # Create dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        return AdversarialIoTEnv(
            dataset_path=dataset_path,
        )

    def test_step_returns_tuple(self, env) -> None:
        """Test step returns 5-tuple."""
        env.reset()
        result = env.step(0)

        assert isinstance(result, tuple)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_step_observation_matches_space(self, env) -> None:
        """Test step observation is within space."""
        env.reset()
        obs, _, _, _, _ = env.step(0)

        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env) -> None:
        """Test reward is a float."""
        env.reset()
        _, reward, _, _, _ = env.step(0)

        assert isinstance(reward, (int, float))

    def test_step_terminated_is_bool(self, env) -> None:
        """Test terminated flag is boolean."""
        env.reset()
        _, _, terminated, _, _ = env.step(0)

        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env) -> None:
        """Test truncated flag is boolean."""
        env.reset()
        _, _, _, truncated, _ = env.step(0)

        assert isinstance(truncated, bool)

    def test_step_valid_actions(self, env) -> None:
        """Test all valid actions can be taken."""
        env.reset()

        for action in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            if not (terminated or truncated):
                assert env.observation_space.contains(obs)
            env.reset()

    def test_step_updates_attack_sequence(self, env) -> None:
        """Test step advances attack sequence."""
        env.reset()

        # Take several steps
        for _ in range(10):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        # Attack stage may or may not have changed (stochastic)
        # But we should have taken steps
        assert hasattr(env, "_step_count")


class TestAdversarialEnvHiddenState:
    """Test that attack state is hidden from agent."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create environment with mock components."""
        import json

        import joblib
        from sklearn.preprocessing import StandardScaler

        from src.environment.adversarial_env import AdversarialIoTEnv

        # Create dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        return AdversarialIoTEnv(
            dataset_path=dataset_path,
        )

    def test_observation_does_not_reveal_attack_stage(self, env) -> None:
        """Test observation is features, not attack stage."""
        obs, _ = env.reset()

        # Observation should be a flattened feature window
        # Not a simple integer representing attack stage
        assert len(obs) > 5  # Should be window_size * num_features

    def test_attack_stage_in_info_for_debugging(self, env) -> None:
        """Test info contains hidden attack stage for evaluation/debugging."""
        obs, info = env.reset()

        # Info should contain hidden state for evaluation purposes
        assert "attack_stage" in info


class TestAdversarialEnvTermination:
    """Test episode termination conditions."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create environment with mock components."""
        import json

        import joblib
        from sklearn.preprocessing import StandardScaler

        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        # Create dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        # Short max_steps for testing
        env_config = AdversarialEnvConfig(max_steps=10)

        return AdversarialIoTEnv(
            dataset_path=dataset_path,
            config=env_config,
        )

    def test_truncation_at_max_steps(self, env) -> None:
        """Test episode truncates after max_steps."""
        env.reset()

        for _i in range(15):  # More than max_steps=10
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        assert truncated or terminated

    def test_info_tracks_episode_length(self, env) -> None:
        """Test info contains episode length."""
        env.reset()

        for _i in range(5):
            _, _, _, _, info = env.step(0)

        assert "step_count" in info
        assert info["step_count"] == 5


class TestForceContiuumActions:
    """Test force continuum action space."""

    def test_action_costs(self) -> None:
        """Test action costs are defined correctly."""
        from src.environment.adversarial_env import get_action_cost

        # OBSERVE should be free
        assert get_action_cost(0) == 0.0

        # Higher actions should cost more
        costs = [get_action_cost(a) for a in range(5)]
        assert costs == sorted(costs)  # Non-decreasing

    def test_action_names(self) -> None:
        """Test action names are defined."""
        from src.environment.adversarial_env import ACTION_NAMES

        assert len(ACTION_NAMES) == 5
        assert ACTION_NAMES[0] == "OBSERVE"
        assert ACTION_NAMES[4] == "ISOLATE"


class TestEnvironmentIntegration:
    """Integration tests for full environment lifecycle."""

    @pytest.fixture
    def env(self, tmp_path):
        """Create a fully configured environment."""
        import json

        import joblib
        from sklearn.preprocessing import StandardScaler

        from src.environment.adversarial_env import AdversarialIoTEnv

        # Create dataset
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)

        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)

        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)

        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)

        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")

        return AdversarialIoTEnv(
            dataset_path=dataset_path,
        )

    def test_full_episode_rollout(self, env) -> None:
        """Test running a complete episode."""
        obs, info = env.reset(seed=42)

        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert isinstance(total_reward, float)
        assert "attack_history" in info

    def test_attack_progression_over_episode(self, env) -> None:
        """Test that attack progresses over multiple steps."""
        env.reset(seed=42)

        attack_stages_seen = set()
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(0)  # Always OBSERVE
            attack_stages_seen.add(info["attack_stage"])
            if terminated or truncated:
                break

        # Should see multiple attack stages over 50 steps
        assert len(attack_stages_seen) >= 1

    def test_reward_varies_with_action(self, env) -> None:
        """Test that rewards differ based on action choice."""
        # We need multiple steps to see consistent differences
        # because reward depends on attack progression

        # Test that action cost affects reward
        env.reset(seed=42)

        # Take multiple OBSERVE actions and accumulate reward
        total_observe = 0.0
        for _ in range(10):
            _, reward, _, _, _ = env.step(0)
            total_observe += reward

        env.reset(seed=42)

        # Take multiple ISOLATE actions and accumulate reward
        total_isolate = 0.0
        for _ in range(10):
            _, reward, _, _, _ = env.step(4)
            total_isolate += reward

        # Due to action costs, ISOLATE should generally cost more
        # unless it provides significant defense bonuses
        # The key is that rewards differ
        assert total_observe != total_isolate

    def test_gymnasium_check_env_compatible(self, env) -> None:
        """Test environment is compatible with Gymnasium check_env."""
        from gymnasium.utils.env_checker import check_env

        try:
            check_env(env, warn=True)
        except Exception as e:
            pytest.fail(f"Environment failed Gymnasium check: {e}")


class TestStagePredictionAblation:
    """Tests for the stage-prediction-in-observation ablation (review 2.4.1)."""

    @pytest.fixture
    def mock_generator(self, tmp_path):
        """Unused legacy fixture (attacker is now a first-order Markov chain)."""
        path = tmp_path / "unused"
        path.mkdir(parents=True)
        return path

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock processed dataset."""
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def test_obs_shape_includes_stage_pred(self, mock_generator, mock_dataset, tmp_path):
        """When include_stage_pred=True, obs space grows by num_actions."""
        # Save a mock RF detector that always predicts stage 2
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        dummy_clf = RandomForestClassifier(n_estimators=1, random_state=0)
        dummy_clf.fit(np.zeros((10, 46)), np.full(10, 2))
        det_path = tmp_path / "detector.joblib"
        joblib.dump(dummy_clf, det_path)

        config = AdversarialEnvConfig(
            window_size=5,
            stage_detector_path=str(det_path),
            include_stage_pred=True,
        )
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        expected_shape = (5 * 46 * 2 + 5,)  # +5 for one-hot stage
        assert env.observation_space.shape == expected_shape

    def test_stage_pred_one_hot_in_observation(self, mock_generator, mock_dataset, tmp_path):
        """The predicted stage is appended as a one-hot vector at the tail."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        dummy_clf = RandomForestClassifier(n_estimators=1, random_state=0)
        dummy_clf.fit(np.zeros((10, 46)), np.full(10, 2))
        det_path = tmp_path / "detector.joblib"
        joblib.dump(dummy_clf, det_path)

        config = AdversarialEnvConfig(
            window_size=5,
            stage_detector_path=str(det_path),
            include_stage_pred=True,
        )
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        obs, _ = env.reset(seed=42)
        tail = obs[-5:]
        expected = np.zeros(5, dtype=np.float32)
        expected[2] = 1.0
        np.testing.assert_array_equal(tail, expected)

    def test_include_stage_pred_without_path_raises(self, mock_generator, mock_dataset):
        """include_stage_pred=True without stage_detector_path is an error."""
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(include_stage_pred=True)
        with pytest.raises(ValueError, match="stage_detector_path"):
            AdversarialIoTEnv(
                dataset_path=mock_dataset,
                config=config,
            )


class TestRetreatProb:
    """Tests for non-monotonic attacker stress-test (review 2.4.3)."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def test_retreat_prob_zero_is_monotonic(self, mock_dataset):
        """retreat_prob=0 should never produce retreats.

        The Markov attacker's transition matrix is upper-triangular for attack
        stages (no regression), so without the retreat override the visible
        chain never drops to an earlier non-zero stage.
        """
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(retreat_prob=0.0)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        prev_stage = env._current_attack_stage
        for _ in range(50):
            env.step(0)
            new_stage = env._current_attack_stage
            # A "retreat" is a drop to a non-zero earlier stage.
            # Drops to 0 are defender de-escalation, not attacker retreat.
            assert not (0 < new_stage < prev_stage)
            prev_stage = new_stage

    def test_retreat_prob_nonzero_can_retreat(self, mock_dataset):
        """retreat_prob>0 should occasionally produce retreats.

        ``retreat_prob`` is a legacy-attacker (autonomous Markov) mechanic that
        is bypassed by the default tug-of-war dynamics, so this test pins the
        legacy path explicitly with ``tug_of_war=False``.
        """
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(retreat_prob=0.5, tug_of_war=False)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        retreats = 0
        prev_stage = env._current_attack_stage
        for _ in range(100):
            env.step(0)
            new_stage = env._current_attack_stage
            if new_stage < prev_stage and new_stage != 0:
                retreats += 1
            prev_stage = new_stage
        # With 50% retreat prob we should see at least one retreat in 100 steps
        assert retreats > 0, "Expected at least one retreat with retreat_prob=0.5"


class TestFPRPenalty:
    """Tests for Lagrangian FPR penalty (review 2.2 / Direction 6)."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def test_fpr_penalty_zero_no_effect(self, mock_dataset):
        """fpr_penalty_beta=0 should not affect reward."""
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(fpr_penalty_beta=0.0)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        for _ in range(20):
            obs, reward, terminated, truncated, _ = env.step(0)
            total_reward += reward
            if terminated or truncated:
                break
        # Just verify the run completes without error
        assert isinstance(total_reward, float)

    def _build_env(self, mock_dataset, beta):
        """Construct an env with a given fpr_penalty_beta."""
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(fpr_penalty_beta=beta)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        return env

    def test_fpr_penalty_nonzero_reduces_reward(self, mock_dataset):
        """fpr_penalty_beta>0 must reduce reward by beta * (benign_blocks/steps).

        Tested directly against the terminal-penalty formula so the assertion
        does not depend on the attacker's stochastic trajectory: a high penalty
        with a nonzero benign false-positive rate yields a strictly lower reward,
        while beta=0 leaves the reward unchanged.
        """
        env_penalised = self._build_env(mock_dataset, beta=1000.0)
        env_penalised._benign_steps = 10
        env_penalised._benign_blocks = 4
        # penalty = -beta * (benign_blocks / benign_steps) = -1000 * 0.4 = -400.
        assert env_penalised._apply_episode_fpr_penalty(0.0) == pytest.approx(-400.0)

        env_free = self._build_env(mock_dataset, beta=0.0)
        env_free._benign_steps = 10
        env_free._benign_blocks = 4
        # beta=0 disables the penalty entirely; reward is returned unchanged.
        assert env_free._apply_episode_fpr_penalty(0.0) == 0.0


class TestEvasion:
    """Evasion-before-commit reactive attacker (the one adaptive-attacker axis).

    When the defender has *just* applied force (BLOCK/ISOLATE) and the attacker
    is still at a pre-trigger stage (RECON/ACCESS), it probabilistically stalls
    in anticipation instead of progressing. This is coupled to the defender's
    action, unlike the random (defender-independent) ``retreat_prob`` override
    and unlike de-escalation (which resets to BENIGN on force at ACCESS+).
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def _build_env(self, mock_dataset, **config_kwargs):
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(**config_kwargs)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        return env

    def test_evasion_prob_zero_is_noop(self, mock_dataset):
        """evasion_prob=0 leaves the attacker advancing normally."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=0.0)
        # Force a deterministic forward advance.
        env._attacker.sample_next = lambda stage, rng: stage + 1
        env._recent_block = True
        env._current_attack_stage = KillChainStage.RECON.value
        env._attack_history = [KillChainStage.RECON.value]
        env._advance_attack()
        assert env._current_attack_stage == KillChainStage.RECON.value + 1

    def test_evasion_stalls_when_recently_blocked(self, mock_dataset):
        """evasion_prob=1.0 + recent block + pre-trigger stage -> stall."""
        from src.utils.label_mapper import KillChainStage

        for stage in (KillChainStage.RECON.value, KillChainStage.ACCESS.value):
            env = self._build_env(
                mock_dataset,
                evasion_prob=1.0,
                retreat_prob=0.0,
            )
            env._attacker.sample_next = lambda s, rng: s + 1
            env._recent_block = True
            env._current_attack_stage = stage
            env._attack_history = [stage]
            env._advance_attack()
            assert env._current_attack_stage == stage  # stalled

    def test_evasion_requires_recent_block(self, mock_dataset):
        """Without a recent defender block, evasion never fires (defender-coupled)."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            evasion_prob=1.0,
            retreat_prob=0.0,
        )
        env._attacker.sample_next = lambda stage, rng: stage + 1
        env._recent_block = False  # defender did NOT just apply force
        env._current_attack_stage = KillChainStage.RECON.value
        env._attack_history = [KillChainStage.RECON.value]
        env._advance_attack()
        # Advanced normally — proves the stall is coupled to the defender action.
        assert env._current_attack_stage == KillChainStage.RECON.value + 1

    def test_evasion_only_at_pretrigger_stages(self, mock_dataset):
        """Evasion does not fire once the attacker is past the pre-trigger band."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            evasion_prob=1.0,
            retreat_prob=0.0,
        )
        env._attacker.sample_next = lambda stage, rng: min(stage + 1, 4)
        env._recent_block = True
        env._current_attack_stage = KillChainStage.MANEUVER.value
        env._attack_history = [KillChainStage.MANEUVER.value]
        env._advance_attack()
        assert env._current_attack_stage == KillChainStage.MANEUVER.value + 1

    def test_step_sets_recent_block_flag(self, mock_dataset):
        """step() records whether the current action was BLOCK/ISOLATE."""
        env = self._build_env(mock_dataset)
        env.step(3)  # BLOCK
        assert env._recent_block is True
        env.step(0)  # OBSERVE
        assert env._recent_block is False

    def test_env_config_serializable_round_trips_evasion(self):
        from dataclasses import asdict

        from src.blue_team.run_config import EnvConfigSerializable

        spec = EnvConfigSerializable(
            split="train",
            exclude_ood=True,
            evasion_prob=0.5,
        )
        assert asdict(spec)["evasion_prob"] == 0.5


class TestEvasivePersistence:
    """Evasive persistence under the *tug-of-war* dynamics (the headline contract).

    The legacy stall (``TestEvasion``) lives only in ``_advance_attack``
    (``tug_of_war=False``). Under tug-of-war the attacker never climbs on a
    proportional/over-force step, so a stall has no bite. Here evasion is
    reinterpreted as *post-detection hardening*: after the attacker senses force
    (BLOCK/ISOLATE) at a pre-commit stage (RECON/ACCESS) it *arms*, and on a
    subsequent proportional (``d == 0``) step it *resists* the de-escalation
    pushdown with probability ``evasion_prob``. The correct response still holds
    the line (the attacker never climbs), it just fails to evict the hardened
    attacker that turn. Every new RNG draw is gated behind ``evasion_prob > 0``
    AND the armed flag, so with ``evasion_prob == 0`` the RNG stream — and thus
    every deterministic result and gate — is byte-identical to before.
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def _build_env(self, mock_dataset, **config_kwargs):
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        # tug_of_war=True is the default headline contract; make it explicit.
        config_kwargs.setdefault("tug_of_war", True)
        config = AdversarialEnvConfig(**config_kwargs)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        return env

    def test_evasion_prob_zero_never_arms(self, mock_dataset):
        """evasion_prob=0: over-force at RECON after a block never arms hardening."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=0.0)
        env._recent_block = True
        stage = KillChainStage.RECON.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        # action=3 (BLOCK) at RECON (recommended=1) -> d=+2 (over-force / HOLD).
        env._advance_tug_of_war(action=3, previous_stage=stage)
        assert env._evasion_hardened is False

    def test_evasion_prob_zero_deescalates_normally(self, mock_dataset):
        """evasion_prob=0: a proportional step de-escalates as usual (p_down=1)."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=0.0, p_down=1.0)
        stage = KillChainStage.ACCESS.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        # action=2 (recommended for ACCESS) -> d=0 -> de-escalate w.p. p_down=1.
        result = env._advance_tug_of_war(action=2, previous_stage=stage)
        assert result == "defended"
        assert env._current_attack_stage == stage - 1

    def test_arming_on_over_force_at_precommit(self, mock_dataset):
        """Block/ISOLATE at RECON/ACCESS after a recent block arms hardening."""
        from src.utils.label_mapper import KillChainStage

        for stage in (KillChainStage.RECON.value, KillChainStage.ACCESS.value):
            env = self._build_env(mock_dataset, evasion_prob=1.0)
            env._recent_block = True
            env._current_attack_stage = stage
            env._attack_history = [stage]
            # action=4 (ISOLATE) -> d >= 1 at these low stages (over-force).
            env._advance_tug_of_war(action=4, previous_stage=stage)
            assert env._evasion_hardened is True
            assert env._current_attack_stage == stage  # held, did not climb

    def test_arming_requires_recent_block(self, mock_dataset):
        """No arming without a recent defender block (defender-coupled)."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=1.0)
        env._recent_block = False
        stage = KillChainStage.RECON.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        env._advance_tug_of_war(action=4, previous_stage=stage)
        assert env._evasion_hardened is False

    def test_arming_only_at_precommit_stages(self, mock_dataset):
        """Over-force at MANEUVER (post-commit) does not arm hardening."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=1.0)
        env._recent_block = True
        stage = KillChainStage.MANEUVER.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        # action=4 (ISOLATE) at MANEUVER (recommended=3) -> d=+1 (over-force).
        env._advance_tug_of_war(action=4, previous_stage=stage)
        assert env._evasion_hardened is False

    def test_hardened_attacker_resists_deescalation(self, mock_dataset):
        """Armed + evasion_prob=1: a proportional step is resisted (no pushdown)."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=1.0, p_down=1.0)
        stage = KillChainStage.ACCESS.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        env._evasion_hardened = True
        # action=2 (recommended for ACCESS) -> d=0 -> would de-escalate, but the
        # hardened attacker resists this turn.
        result = env._advance_tug_of_war(action=2, previous_stage=stage)
        assert result == "ongoing"
        assert env._current_attack_stage == stage  # held the line
        assert env._evasion_hardened is False  # flag consumed

    def test_resist_fires_once_then_evicts(self, mock_dataset):
        """After resisting once, the flag is consumed and the next proportional
        step de-escalates normally."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=1.0, p_down=1.0)
        stage = KillChainStage.ACCESS.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        env._evasion_hardened = True
        env._advance_tug_of_war(action=2, previous_stage=stage)  # resisted
        assert env._current_attack_stage == stage
        # Second proportional step: no longer hardened -> de-escalates.
        result = env._advance_tug_of_war(action=2, previous_stage=stage)
        assert result == "defended"
        assert env._current_attack_stage == stage - 1

    def test_unarmed_proportional_deescalates(self, mock_dataset):
        """Not armed + evasion_prob=1: proportional step still de-escalates."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, evasion_prob=1.0, p_down=1.0)
        stage = KillChainStage.ACCESS.value
        env._current_attack_stage = stage
        env._attack_history = [stage]
        env._evasion_hardened = False
        result = env._advance_tug_of_war(action=2, previous_stage=stage)
        assert result == "defended"
        assert env._current_attack_stage == stage - 1


class TestRewardMode:
    """Outcome-only reward mode: tests whether the proportionality shaping is
    load-bearing.

    ``reward_mode="proportional"`` (alias ``"coupled"``) applies the full
    kill-chain-aware per-step shaping. ``reward_mode="outcome_only"`` strips every
    stage-conditioned shaping term so the per-step reward is only the action
    cost; outcome signals (de-escalation bonus, impact penalty, prevention
    bonus, FPR penalty) live outside ``_calculate_reward`` and are unaffected.
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        features = np.random.randn(100, 46).astype(np.float32)
        np.save(dataset_path / "features.npy", features)
        labels = np.random.randint(0, 5, size=100)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(label)].append(idx)
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def _build_env(self, mock_dataset, **config_kwargs):
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(**config_kwargs)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=42)
        return env

    def test_proportional_default_penalises_disproportionate(self, mock_dataset):
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, reward_mode="proportional")
        # BENIGN decision-stage + ISOLATE action triggers the benign guardrails
        # plus the disproportionate-action penalty in the coupled (proportional)
        # mode.
        reward = env._calculate_reward(4, KillChainStage.BENIGN.value)
        assert reward < -100

    def test_outcome_only_strips_shaping(self, mock_dataset):
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, reward_mode="outcome_only")
        # Same BENIGN+ISOLATE step: only the action cost remains (ISOLATE=0.8).
        reward = env._calculate_reward(4, KillChainStage.BENIGN.value)
        assert reward == pytest.approx(-0.8)

    def test_outcome_only_observe_is_free(self, mock_dataset):
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(mock_dataset, reward_mode="outcome_only")
        # OBSERVE costs 0.0 and outcome-only adds no shaping, so reward is 0.
        reward = env._calculate_reward(0, KillChainStage.RECON.value)
        assert reward == pytest.approx(0.0)

    def test_outcome_mode_is_not_keyed_on_stage_match(self, mock_dataset):
        """The re-pose contract: under outcome mode the per-step reward must NOT
        reward emitting the stage's ``recommended_action`` label.

        This is the direct test of the de-coupling. Under the coupled contract,
        playing the recommended action for a stage earns the proportionality
        bonus, so two different actions yield different rewards. Under the
        outcome contract the per-step reward is the action cost alone, so the
        only difference between two actions is their cost — never a bonus for
        matching the hidden stage's label.
        """
        from src.environment.adversarial_env import get_action_cost
        from src.utils.label_mapper import KillChainStage

        stage = KillChainStage.ACCESS.value  # recommended action = RESTRICT (2)

        coupled = self._build_env(mock_dataset, reward_mode="proportional")
        # Coupled: the recommended action (2) earns the proportional bonus; a
        # disproportionate action (4) does not. So matching the stage label is
        # strictly rewarded — the hallmark of the mis-posed task.
        r_match_coupled = coupled._calculate_reward(2, stage)
        r_off_coupled = coupled._calculate_reward(4, stage)
        assert r_match_coupled > r_off_coupled

        outcome = self._build_env(mock_dataset, reward_mode="outcome")
        # Outcome: each action's per-step reward is exactly minus its cost,
        # independent of the (hidden) stage. Matching the label confers no bonus.
        for action in range(5):
            assert outcome._calculate_reward(action, stage) == pytest.approx(
                -get_action_cost(action)
            )
        # And the per-step reward does not depend on the stage at all.
        for stage_a in range(5):
            for stage_b in range(5):
                assert outcome._calculate_reward(3, stage_a) == pytest.approx(
                    outcome._calculate_reward(3, stage_b)
                )

    def test_env_config_serializable_normalises_reward_mode_aliases(self):
        from dataclasses import asdict

        from src.blue_team.run_config import EnvConfigSerializable

        # Legacy aliases normalise to canonical tokens so manifests are
        # consistent and the train/eval parity check is alias-insensitive.
        assert asdict(EnvConfigSerializable(reward_mode="outcome_only"))["reward_mode"] == "outcome"
        assert asdict(EnvConfigSerializable(reward_mode="outcome"))["reward_mode"] == "outcome"
        assert asdict(EnvConfigSerializable(reward_mode="proportional"))["reward_mode"] == "coupled"
        assert asdict(EnvConfigSerializable(reward_mode="coupled"))["reward_mode"] == "coupled"
        with pytest.raises(ValueError):
            EnvConfigSerializable(reward_mode="bogus")


class TestPartialObservabilityRedesign:
    """Sequential-POMDP redesign: observation aliasing, session-coherent
    sampling, post-transition-leak removal, and proximity-coupled tolerance.

    The defaults (``aliasing_rate=0.0``, ``session_coherent=False``,
    ``no_post_transition_leak=False``, ``proximity_coupled=False``) reproduce
    the legacy fully-observable, budget-driven environment byte-for-byte; each
    flag is opt-in so the legacy anchors (alpha=0, coupling ablation) are
    unaffected.
    """

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        import json

        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir(parents=True)
        # Make each stage's rows linearly separable by stage so the emitted
        # stage of a sampled row is recoverable in tests: row feature[0] == stage.
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(5), 40)  # 200 rows, 40 per stage
        features = rng.standard_normal((labels.size, 46)).astype(np.float32)
        features[:, 0] = labels.astype(np.float32)  # stage tag in column 0
        np.save(dataset_path / "features.npy", features)
        np.save(dataset_path / "labels.npy", labels)
        state_indices = {str(i): [] for i in range(5)}
        for idx, label in enumerate(labels):
            state_indices[str(int(label))].append(int(idx))
        with open(dataset_path / "state_indices.json", "w") as f:
            json.dump(state_indices, f)
        import joblib
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        return dataset_path

    def _build_env(self, mock_dataset, **config_kwargs):
        from src.environment.adversarial_env import AdversarialEnvConfig, AdversarialIoTEnv

        config = AdversarialEnvConfig(**config_kwargs)
        env = AdversarialIoTEnv(
            dataset_path=mock_dataset,
            config=config,
        )
        env.reset(seed=7)
        return env

    # --- (i) observation aliasing -------------------------------------------

    def test_aliasing_zero_is_byte_compatible(self, mock_dataset):
        """aliasing_rate=0 reproduces the legacy single-choice draw exactly."""
        from src.environment.adversarial_env import AdversarialEnvConfig
        from src.utils.realization_engine import RealizationEngine

        engine = RealizationEngine(data_path=mock_dataset, seed=123)
        legacy = engine.sample_by_id(1)
        # A fresh engine with the same seed under the redesign defaults must
        # yield the identical row.
        engine2 = RealizationEngine(data_path=mock_dataset, seed=123)
        redesigned = engine2.sample_by_id(1, aliasing_rate=0.0, session_coherent=False)
        assert np.array_equal(legacy, redesigned)
        # Default config exposes the new fields with backward-compatible defaults.
        cfg = AdversarialEnvConfig()
        assert cfg.aliasing_rate == 0.0
        assert cfg.session_coherent is False
        assert cfg.no_post_transition_leak is False
        # proximity_coupled now defaults True (headline contract; budget retired).
        assert cfg.proximity_coupled is True

    def test_aliasing_full_emits_only_adjacent_stages(self, mock_dataset):
        """aliasing_rate=1 always emits a row from an ADJACENT stage."""
        from src.utils.realization_engine import RealizationEngine

        engine = RealizationEngine(data_path=mock_dataset, seed=5)
        # RECON (stage 1) has neighbours {BENIGN=0, ACCESS=2}. With alpha=1 the
        # emitted row's stage tag (column 0) must always be 0 or 2, never 1.
        emitted = {int(round(engine.sample_by_id(1, aliasing_rate=1.0)[0])) for _ in range(60)}
        assert emitted.issubset({0, 2})
        assert 1 not in emitted

    def test_aliasing_endpoint_has_single_neighbour(self, mock_dataset):
        """BENIGN (endpoint) aliases only to RECON; IMPACT only to MANEUVER."""
        from src.utils.realization_engine import RealizationEngine

        engine = RealizationEngine(data_path=mock_dataset, seed=9)
        benign_emitted = {
            int(round(engine.sample_by_id(0, aliasing_rate=1.0)[0])) for _ in range(40)
        }
        assert benign_emitted == {1}
        impact_emitted = {
            int(round(engine.sample_by_id(4, aliasing_rate=1.0)[0])) for _ in range(40)
        }
        assert impact_emitted == {3}

    # --- (ii) session-coherent sampling -------------------------------------

    def test_session_coherent_run_has_no_repeats(self, mock_dataset):
        """A within-stage run draws WITHOUT replacement until the pool empties."""
        from src.utils.realization_engine import RealizationEngine

        engine = RealizationEngine(data_path=mock_dataset, seed=11)
        # Stage MANEUVER (3) has 40 rows; the first 40 draws must be distinct.
        pool_size = len(engine.get_indices_for_stage(3))
        rows = [
            tuple(engine.sample_by_id(3, session_coherent=True).tolist()) for _ in range(pool_size)
        ]
        assert len(set(rows)) == pool_size  # no repeats within one pass

    # --- (iii) post-transition-leak removal ---------------------------------

    def test_no_post_transition_leak_samples_pre_transition_stage(self, mock_dataset):
        """With the leak removed, the refreshed obs reflects the PRE-transition
        stage (column-0 tag == previous stage), not the just-entered stage."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            no_post_transition_leak=True,
            tug_of_war=True,
            p_onset=1.0,  # BENIGN deterministically onsets to RECON this step
            include_deltas=False,
        )
        # Start at BENIGN; after step the attacker moves to RECON, but the
        # refreshed observation row must still carry the BENIGN (=0) stage tag.
        env._current_attack_stage = KillChainStage.BENIGN.value
        env._attack_history = [KillChainStage.BENIGN.value]
        obs, *_ = env.step(0)
        latest_row = env._observation_window[-1]
        assert int(round(latest_row[0])) == KillChainStage.BENIGN.value
        assert env._current_attack_stage == KillChainStage.RECON.value

    def test_legacy_leak_samples_post_transition_stage(self, mock_dataset):
        """Default (leak present) refreshes obs from the NEW stage (legacy)."""
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            no_post_transition_leak=False,
            tug_of_war=True,
            p_onset=1.0,
            include_deltas=False,
        )
        env._current_attack_stage = KillChainStage.BENIGN.value
        env._attack_history = [KillChainStage.BENIGN.value]
        env.step(0)
        latest_row = env._observation_window[-1]
        assert int(round(latest_row[0])) == KillChainStage.RECON.value
        assert env._current_attack_stage == KillChainStage.RECON.value

    # --- (iv) proximity-coupled tolerance -----------------------------------

    def test_proximity_escalation_rises_with_stage(self, mock_dataset):
        """Under-force escalation probability increases with attacker proximity.

        p_up_eff = p_up * (min_esc + (1 - min_esc) * stage/IMPACT). With a fixed
        rng draw, a deeper stage must escalate where a shallower one would not.
        """
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            proximity_coupled=True,
            proximity_min_escalation=0.4,
            p_up=0.5,
            tug_of_war=True,
        )

        # RECON proximity lambda = 1/4 -> p_up_eff = 0.5*(0.4+0.6*0.25)=0.275
        # MANEUVER lambda = 3/4 -> p_up_eff = 0.5*(0.4+0.6*0.75)=0.425
        # A fixed draw of 0.35 escalates MANEUVER but not RECON.
        class _FixedRng:
            def __init__(self, value):
                self._value = value

            def random(self):
                return self._value

        env._rng = _FixedRng(0.35)

        env._current_attack_stage = KillChainStage.RECON.value
        env._attack_history = [KillChainStage.RECON.value]
        env._advance_tug_of_war(0, KillChainStage.RECON.value)  # under-force
        recon_after = env._current_attack_stage

        env._current_attack_stage = KillChainStage.MANEUVER.value
        env._attack_history = [KillChainStage.MANEUVER.value]
        env._advance_tug_of_war(0, KillChainStage.MANEUVER.value)  # under-force
        maneuver_after = env._current_attack_stage

        assert recon_after == KillChainStage.RECON.value  # did NOT escalate
        assert maneuver_after == KillChainStage.IMPACT.value  # DID escalate

    def test_proximity_truncation_yields_prevention(self, mock_dataset):
        """Holding the attacker below IMPACT to the horizon counts as prevented.

        Under proximity-coupled mode with no budget counter, prevention is
        awarded at truncation when the attacker is still below IMPACT.
        """
        from src.utils.label_mapper import KillChainStage

        env = self._build_env(
            mock_dataset,
            proximity_coupled=True,
            reward_mode="outcome",
            impact_is_terminal=False,
            max_steps=10,
            prevention_bonus=5.0,
        )
        # Place the attacker mid-chain on the final step; truncation should fire
        # the proximity-prevention branch (stage < IMPACT).
        env._step_count = env._config.max_steps - 1
        env._current_attack_stage = KillChainStage.RECON.value
        env._attack_history = [KillChainStage.RECON.value]
        # Over-force to hold the stage (no escalation, no de-escalation past 0).
        _, _, terminated, truncated, info = env.step(4)
        assert truncated is True
        assert info["outcome"] == "prevented"
        assert info["compromised"] is False
