"""
Tests for RL Algorithms with AdversarialIoTEnv.

Tests DQN, PPO, and A2C algorithms with the new adversarial environment.
"""

import numpy as np
import pytest
from pathlib import Path


class TestAlgorithmFactory:
    """Test AlgorithmFactory with new environment."""
    
    @pytest.fixture
    def mock_env(self, tmp_path):
        """Create a mock environment for testing."""
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
    
    def test_dqn_with_box_obs(self, mock_env) -> None:
        """Test DQN works with Box observation space."""
        from stable_baselines3 import DQN
        
        model = DQN(
            "MlpPolicy",
            mock_env,
            learning_rate=1e-4,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            verbose=0,
        )
        
        # Should be able to take a few training steps
        model.learn(total_timesteps=200)
        
        # Should be able to predict
        obs, _ = mock_env.reset()
        action, _ = model.predict(obs)
        assert 0 <= action < 5
    
    def test_ppo_with_box_obs(self, mock_env) -> None:
        """Test PPO works with Box observation space."""
        from stable_baselines3 import PPO
        
        model = PPO(
            "MlpPolicy",
            mock_env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            verbose=0,
        )
        
        # Should be able to take a few training steps
        model.learn(total_timesteps=200)
        
        # Should be able to predict
        obs, _ = mock_env.reset()
        action, _ = model.predict(obs)
        assert 0 <= action < 5
    
    def test_a2c_with_box_obs(self, mock_env) -> None:
        """Test A2C works with Box observation space."""
        from stable_baselines3 import A2C
        
        model = A2C(
            "MlpPolicy",
            mock_env,
            learning_rate=7e-4,
            n_steps=5,
            verbose=0,
        )
        
        # Should be able to take a few training steps
        model.learn(total_timesteps=200)
        
        # Should be able to predict
        obs, _ = mock_env.reset()
        action, _ = model.predict(obs)
        assert 0 <= action < 5


class TestAdversarialAlgorithmConfig:
    """Test AdversarialAlgorithmConfig dataclass."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        from src.algorithms.adversarial_algorithm import AdversarialAlgorithmConfig
        
        config = AdversarialAlgorithmConfig()
        
        assert config.algorithm_type == "ppo"
        assert config.policy == "MlpPolicy"
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from src.algorithms.adversarial_algorithm import AdversarialAlgorithmConfig
        
        config = AdversarialAlgorithmConfig(
            algorithm_type="dqn",
            total_timesteps=100000,
        )
        
        assert config.algorithm_type == "dqn"
        assert config.total_timesteps == 100000


class TestAdversarialAlgorithm:
    """Test AdversarialAlgorithm class."""
    
    @pytest.fixture
    def mock_env(self, tmp_path):
        """Create a mock environment for testing."""
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
    
    def test_create_dqn_model(self, mock_env) -> None:
        """Test creating a DQN model."""
        from src.algorithms.adversarial_algorithm import (
            AdversarialAlgorithm,
            AdversarialAlgorithmConfig,
        )
        
        config = AdversarialAlgorithmConfig(algorithm_type="dqn")
        alg = AdversarialAlgorithm(config)
        
        model = alg.create_model(mock_env)
        assert model is not None
    
    def test_create_ppo_model(self, mock_env) -> None:
        """Test creating a PPO model."""
        from src.algorithms.adversarial_algorithm import (
            AdversarialAlgorithm,
            AdversarialAlgorithmConfig,
        )
        
        config = AdversarialAlgorithmConfig(algorithm_type="ppo")
        alg = AdversarialAlgorithm(config)
        
        model = alg.create_model(mock_env)
        assert model is not None
    
    def test_create_a2c_model(self, mock_env) -> None:
        """Test creating an A2C model."""
        from src.algorithms.adversarial_algorithm import (
            AdversarialAlgorithm,
            AdversarialAlgorithmConfig,
        )
        
        config = AdversarialAlgorithmConfig(algorithm_type="a2c")
        alg = AdversarialAlgorithm(config)
        
        model = alg.create_model(mock_env)
        assert model is not None
    
    def test_train_model(self, mock_env, tmp_path) -> None:
        """Test training a model."""
        from src.algorithms.adversarial_algorithm import (
            AdversarialAlgorithm,
            AdversarialAlgorithmConfig,
        )
        
        config = AdversarialAlgorithmConfig(
            algorithm_type="ppo",
            total_timesteps=100,
        )
        alg = AdversarialAlgorithm(config)
        
        model = alg.create_model(mock_env)
        trained_model = alg.train(model, total_timesteps=100)
        
        assert trained_model is not None
    
    def test_save_load_model(self, mock_env, tmp_path) -> None:
        """Test saving and loading a model."""
        from src.algorithms.adversarial_algorithm import (
            AdversarialAlgorithm,
            AdversarialAlgorithmConfig,
        )
        
        config = AdversarialAlgorithmConfig(algorithm_type="dqn")
        alg = AdversarialAlgorithm(config)
        
        model = alg.create_model(mock_env)
        
        # Save model
        model_path = tmp_path / "test_model"
        alg.save_model(model, model_path)
        
        # Load model
        loaded_model = alg.load_model(model_path, mock_env)
        assert loaded_model is not None
        
        # Should produce same predictions
        obs, _ = mock_env.reset(seed=42)
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)
        assert action1 == action2
