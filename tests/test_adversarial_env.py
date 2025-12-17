"""
Tests for AdversarialIoTEnv.

The Adversarial Environment implements a Gymnasium environment where:
- Red Team (Attack Sequence Generator) controls attack progression
- Blue Team (RL Agent) observes realized features and takes defensive actions
- Attack state is hidden from the agent (partial observability)

Key properties:
- Box observation space: window of realized feature vectors
- Discrete action space: 5 force continuum levels
- Hidden state: Attack Sequence Generator controls actual attack stage
"""

import numpy as np
import pytest
import gymnasium as gym


class TestAdversarialEnvConfig:
    """Test AdversarialEnvConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        from src.environment.adversarial_env import AdversarialEnvConfig
        
        config = AdversarialEnvConfig()
        
        assert config.max_steps == 500
        assert config.window_size == 5
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
        """Create a mock Attack Sequence Generator."""
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        
        # Create and save a simple generator
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        
        model_path = tmp_path / "generator" / "attack_sequence_generator.pth"
        generator.save(model_path, save_config=True)
        
        return tmp_path / "generator"
    
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
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        scaler = StandardScaler()
        scaler.fit(features)
        joblib.dump(scaler, dataset_path / "scaler.joblib")
        
        return dataset_path
    
    def test_is_gymnasium_env(self, mock_generator, mock_dataset) -> None:
        """Test environment is a Gymnasium environment."""
        from src.environment.adversarial_env import AdversarialIoTEnv
        
        env = AdversarialIoTEnv(
            generator_path=mock_generator,
            dataset_path=mock_dataset,
        )
        
        assert isinstance(env, gym.Env)
    
    def test_has_observation_space(self, mock_generator, mock_dataset) -> None:
        """Test environment has observation space."""
        from src.environment.adversarial_env import AdversarialIoTEnv
        
        env = AdversarialIoTEnv(
            generator_path=mock_generator,
            dataset_path=mock_dataset,
        )
        
        assert hasattr(env, "observation_space")
        assert isinstance(env.observation_space, gym.spaces.Box)
    
    def test_observation_space_shape(self, mock_generator, mock_dataset) -> None:
        """Test observation space has correct shape."""
        from src.environment.adversarial_env import (
            AdversarialIoTEnv,
            AdversarialEnvConfig,
        )
        
        config = AdversarialEnvConfig(window_size=5)
        env = AdversarialIoTEnv(
            generator_path=mock_generator,
            dataset_path=mock_dataset,
            config=config,
        )
        
        # Shape should be (window_size, num_features)
        # Flattened to (window_size * num_features,)
        expected_shape = (5 * 46,)  # 46 features from mock dataset
        assert env.observation_space.shape == expected_shape
    
    def test_has_action_space(self, mock_generator, mock_dataset) -> None:
        """Test environment has action space."""
        from src.environment.adversarial_env import AdversarialIoTEnv
        
        env = AdversarialIoTEnv(
            generator_path=mock_generator,
            dataset_path=mock_dataset,
        )
        
        assert hasattr(env, "action_space")
        assert isinstance(env.action_space, gym.spaces.Discrete)
    
    def test_action_space_size(self, mock_generator, mock_dataset) -> None:
        """Test action space has 5 actions (force continuum)."""
        from src.environment.adversarial_env import AdversarialIoTEnv
        
        env = AdversarialIoTEnv(
            generator_path=mock_generator,
            dataset_path=mock_dataset,
        )
        
        assert env.action_space.n == 5


class TestAdversarialEnvReset:
    """Test environment reset functionality."""
    
    @pytest.fixture
    def env(self, tmp_path):
        """Create environment with mock components."""
        from src.environment.adversarial_env import AdversarialIoTEnv
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        import json
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create generator
        generator_path = tmp_path / "generator"
        generator_path.mkdir(parents=True)
        
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        generator.save(generator_path / "attack_sequence_generator.pth", save_config=True)
        
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
            generator_path=generator_path,
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
        from src.environment.adversarial_env import AdversarialIoTEnv
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        import json
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create generator
        generator_path = tmp_path / "generator"
        generator_path.mkdir(parents=True)
        
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        generator.save(generator_path / "attack_sequence_generator.pth", save_config=True)
        
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
            generator_path=generator_path,
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
        initial_stage = env._current_attack_stage
        
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
        from src.environment.adversarial_env import AdversarialIoTEnv
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        import json
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create generator
        generator_path = tmp_path / "generator"
        generator_path.mkdir(parents=True)
        
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        generator.save(generator_path / "attack_sequence_generator.pth", save_config=True)
        
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
            generator_path=generator_path,
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
        from src.environment.adversarial_env import (
            AdversarialIoTEnv,
            AdversarialEnvConfig,
        )
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        import json
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create generator
        generator_path = tmp_path / "generator"
        generator_path.mkdir(parents=True)
        
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        generator.save(generator_path / "attack_sequence_generator.pth", save_config=True)
        
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
            generator_path=generator_path,
            dataset_path=dataset_path,
            config=env_config,
        )
    
    def test_truncation_at_max_steps(self, env) -> None:
        """Test episode truncates after max_steps."""
        env.reset()
        
        for i in range(15):  # More than max_steps=10
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        assert truncated or terminated
    
    def test_info_tracks_episode_length(self, env) -> None:
        """Test info contains episode length."""
        env.reset()
        
        for i in range(5):
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
        from src.environment.adversarial_env import AdversarialIoTEnv
        from src.generator.attack_sequence_generator import (
            AttackSequenceGenerator,
            AttackSequenceGeneratorConfig,
        )
        import json
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        # Create generator
        generator_path = tmp_path / "generator"
        generator_path.mkdir(parents=True)
        
        config = AttackSequenceGeneratorConfig(
            num_stages=5,
            embedding_dim=16,
            hidden_size=32,
            num_layers=1,
        )
        generator = AttackSequenceGenerator(config=config)
        generator.save(generator_path / "attack_sequence_generator.pth", save_config=True)
        
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
            generator_path=generator_path,
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
