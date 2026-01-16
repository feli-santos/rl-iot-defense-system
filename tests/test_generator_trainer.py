"""
Tests for GeneratorTrainer.

This module tests the training functionality for the Attack Sequence
Generator (LSTM next-token predictor).
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from src.generator.attack_sequence_generator import AttackSequenceGenerator
from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from src.training.generator_trainer import (
    GeneratorTrainer,
    GeneratorTrainingConfig,
)


class TestGeneratorTrainingConfig:
    """Tests for GeneratorTrainingConfig."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = GeneratorTrainingConfig()
        
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.sequence_length >= 3

    def test_custom_config(self) -> None:
        """Should accept custom configuration."""
        config = GeneratorTrainingConfig(
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            sequence_length=10,
        )
        
        assert config.epochs == 50
        assert config.batch_size == 64


class TestGeneratorTrainerInitialization:
    """Tests for GeneratorTrainer initialization."""

    @pytest.fixture
    def trainer(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create trainer with temp output directory."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            output_dir=temp_generator_dir,
        )
        return GeneratorTrainer(config=config)

    def test_init_creates_trainer(self, trainer: GeneratorTrainer) -> None:
        """Should create trainer instance."""
        assert trainer is not None

    def test_init_creates_model(self, trainer: GeneratorTrainer) -> None:
        """Should create the generator model."""
        assert trainer.model is not None
        assert isinstance(trainer.model, AttackSequenceGenerator)


class TestGeneratorTrainerDataPreparation:
    """Tests for training data preparation."""

    @pytest.fixture
    def trainer(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create trainer."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            sequence_length=5,
            output_dir=temp_generator_dir,
        )
        return GeneratorTrainer(config=config)

    def test_prepare_data_from_episodes(
        self, trainer: GeneratorTrainer
    ) -> None:
        """Should prepare training data from episodes."""
        episode_gen = EpisodeGenerator(seed=42)
        episodes = episode_gen.generate_batch(n=50)
        
        train_loader, val_loader = trainer.prepare_data(episodes)
        
        assert train_loader is not None
        assert val_loader is not None

    def test_data_loader_yields_correct_shape(
        self, trainer: GeneratorTrainer
    ) -> None:
        """DataLoader should yield correctly shaped batches."""
        episode_gen = EpisodeGenerator(seed=42)
        episodes = episode_gen.generate_batch(n=50)
        
        train_loader, _ = trainer.prepare_data(episodes)
        
        x_batch, y_batch = next(iter(train_loader))
        
        # x: (batch_size, sequence_length)
        assert x_batch.ndim == 2
        assert x_batch.shape[1] == trainer.config.sequence_length
        
        # y: (batch_size,)
        assert y_batch.ndim == 1

    def test_training_input_is_stage_episodes(self, trainer: GeneratorTrainer) -> None:
        """Training should work on stage ID episodes, not feature vectors."""
        # Generate stage episodes (list of lists of integers)
        episode_gen = EpisodeGenerator(seed=42)
        episodes = episode_gen.generate_batch(n=20)
        
        # Verify episodes are stage IDs (0-4)
        assert all(isinstance(ep, list) for ep in episodes)
        assert all(all(0 <= s < 5 for s in ep) for ep in episodes)
        
        # Should successfully prepare data
        train_loader, val_loader = trainer.prepare_data(episodes)
        assert train_loader is not None
        assert val_loader is not None

    def test_target_is_next_stage_after_window(self, trainer: GeneratorTrainer) -> None:
        """Target should be the next stage after the input window (next-token prediction)."""
        # Create a known episode
        episodes = [[0, 1, 1, 2, 3, 4, 4]]
        
        # Convert to training data with sequence_length=3
        ep_gen = EpisodeGenerator()
        X, y = ep_gen.to_numpy(episodes, sequence_length=3)
        
        # Expected pairs for [0,1,1,2,3,4,4]:
        # X[0] = [0,1,1], y[0] = 2
        # X[1] = [1,1,2], y[1] = 3
        # X[2] = [1,2,3], y[2] = 4
        # X[3] = [2,3,4], y[3] = 4
        
        assert X.shape == (4, 3), f"Expected (4, 3), got {X.shape}"
        assert y.shape == (4,), f"Expected (4,), got {y.shape}"
        
        # Verify first pair
        assert list(X[0]) == [0, 1, 1]
        assert y[0] == 2
        
        # Verify second pair
        assert list(X[1]) == [1, 1, 2]
        assert y[1] == 3


class TestGeneratorTrainerImbalanceMitigation:
    """Tests for imbalance mitigation features."""

    @pytest.fixture
    def trainer_with_class_weights(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create trainer with class weights enabled."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            sequence_length=5,
            output_dir=temp_generator_dir,
            use_class_weights=True,
            seed=42,
        )
        return GeneratorTrainer(config=config)

    @pytest.fixture
    def trainer_with_sampler(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create trainer with weighted sampler enabled."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            sequence_length=5,
            output_dir=temp_generator_dir,
            use_weighted_sampler=True,
            seed=42,
        )
        return GeneratorTrainer(config=config)

    def test_class_weights_applied(
        self, trainer_with_class_weights: GeneratorTrainer
    ) -> None:
        """Should apply class weights to loss function."""
        episode_gen = EpisodeGenerator(seed=42)
        episodes = episode_gen.generate_batch(n=50)
        
        # Prepare data should configure weighted loss
        train_loader, _ = trainer_with_class_weights.prepare_data(episodes)
        
        # Loss function should have weights
        assert hasattr(trainer_with_class_weights._criterion, 'weight')
        assert trainer_with_class_weights._criterion.weight is not None
        assert trainer_with_class_weights._criterion.weight.shape == (5,)

    def test_weighted_sampler_applied(
        self, trainer_with_sampler: GeneratorTrainer
    ) -> None:
        """Should use weighted sampler for balanced batches."""
        episode_gen = EpisodeGenerator(seed=42)
        episodes = episode_gen.generate_batch(n=50)
        
        train_loader, _ = trainer_with_sampler.prepare_data(episodes)
        
        # Sampler should be WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        assert train_loader.sampler is not None
        assert isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_seed_makes_split_deterministic(
        self, temp_generator_dir: Path
    ) -> None:
        """Seed should make train/val split reproducible."""
        config1 = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            output_dir=temp_generator_dir,
            seed=42,
        )
        config2 = GeneratorTrainingConfig(
            epochs=2,
            batch_size=8,
            output_dir=temp_generator_dir,
            seed=42,
        )
        
        trainer1 = GeneratorTrainer(config=config1)
        trainer2 = GeneratorTrainer(config=config2)
        
        episode_gen = EpisodeGenerator(seed=100)
        episodes = episode_gen.generate_batch(n=50)
        
        train_loader1, val_loader1 = trainer1.prepare_data(episodes)
        train_loader2, val_loader2 = trainer2.prepare_data(episodes)
        
        # Compare validation sets (they don't shuffle, so should be identical)
        x1_val, y1_val = next(iter(val_loader1))
        x2_val, y2_val = next(iter(val_loader2))
        
        # Should be identical
        assert torch.equal(x1_val, x2_val)
        assert torch.equal(y1_val, y2_val)


class TestGeneratorTrainerEvaluation:
    """Tests for model evaluation."""

    @pytest.fixture
    def trainer(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create and train a simple trainer for evaluation tests."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=16,
            sequence_length=5,
            output_dir=temp_generator_dir,
            seed=42,
        )
        trainer = GeneratorTrainer(config=config)
        
        # Train briefly
        ep_config = EpisodeGeneratorConfig(
            num_episodes=50,
            min_length=10,
            max_length=20,
        )
        episode_gen = EpisodeGenerator(config=ep_config, seed=42)
        episodes = episode_gen.generate_all()
        trainer.train(episodes)
        
        return trainer

    def test_evaluate_returns_comprehensive_metrics(
        self, trainer: GeneratorTrainer
    ) -> None:
        """Evaluate should return comprehensive metrics."""
        episode_gen = EpisodeGenerator(seed=100)
        test_episodes = episode_gen.generate_batch(n=20)
        
        metrics = trainer.evaluate(test_episodes)
        
        # Check presence of key metrics
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "perplexity" in metrics
        assert "macro_f1" in metrics
        assert "transition_accuracy" in metrics
        
        # Check per-class metrics for all stages
        for stage in range(5):
            assert f"precision_stage_{stage}" in metrics
            assert f"recall_stage_{stage}" in metrics
            assert f"f1_stage_{stage}" in metrics

    def test_evaluate_metrics_in_valid_range(
        self, trainer: GeneratorTrainer
    ) -> None:
        """Evaluation metrics should be in valid ranges."""
        episode_gen = EpisodeGenerator(seed=100)
        test_episodes = episode_gen.generate_batch(n=20)
        
        metrics = trainer.evaluate(test_episodes)
        
        # Accuracy, precision, recall, F1 should be in [0, 1]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["transition_accuracy"] <= 1.0
        
        # Perplexity should be positive
        assert metrics["perplexity"] > 0
        
        # Loss should be non-negative
        assert metrics["loss"] >= 0

    def test_evaluate_stores_confusion_matrix(
        self, trainer: GeneratorTrainer
    ) -> None:
        """Evaluate should store confusion matrix."""
        episode_gen = EpisodeGenerator(seed=100)
        test_episodes = episode_gen.generate_batch(n=20)
        
        trainer.evaluate(test_episodes)
        
        cm = trainer.last_confusion_matrix
        assert cm is not None
        assert cm.shape == (5, 5)
        assert cm.dtype == np.int32


class TestGeneratorTrainerTraining:
    """Tests for model training."""

    @pytest.fixture
    def trainer(self, temp_generator_dir: Path) -> GeneratorTrainer:
        """Create trainer with minimal config for fast testing."""
        config = GeneratorTrainingConfig(
            epochs=3,
            batch_size=16,
            sequence_length=5,
            output_dir=temp_generator_dir,
        )
        return GeneratorTrainer(config=config)

    @pytest.fixture
    def training_episodes(self) -> list:
        """Generate training episodes."""
        ep_config = EpisodeGeneratorConfig(
            num_episodes=100,
            min_length=10,
            max_length=20,
        )
        episode_gen = EpisodeGenerator(config=ep_config, seed=42)
        return episode_gen.generate_all()

    def test_train_runs_without_error(
        self,
        trainer: GeneratorTrainer,
        training_episodes: list,
    ) -> None:
        """Training should complete without errors."""
        results = trainer.train(training_episodes)
        
        assert results is not None

    def test_train_returns_history(
        self,
        trainer: GeneratorTrainer,
        training_episodes: list,
    ) -> None:
        """Training should return loss history."""
        results = trainer.train(training_episodes)
        
        assert "train_losses" in results
        assert "val_losses" in results
        assert len(results["train_losses"]) == trainer.config.epochs

    def test_train_loss_decreases(
        self,
        trainer: GeneratorTrainer,
        training_episodes: list,
    ) -> None:
        """Training loss should generally decrease."""
        results = trainer.train(training_episodes)
        
        train_losses = results["train_losses"]
        
        # Loss should decrease from first to last epoch
        assert train_losses[-1] < train_losses[0]

    def test_train_saves_model(
        self,
        trainer: GeneratorTrainer,
        training_episodes: list,
        temp_generator_dir: Path,
    ) -> None:
        """Training should save model to output directory."""
        trainer.train(training_episodes)
        
        model_path = temp_generator_dir / "attack_sequence_generator.pth"
        assert model_path.exists()

    def test_train_saves_config(
        self,
        trainer: GeneratorTrainer,
        training_episodes: list,
        temp_generator_dir: Path,
    ) -> None:
        """Training should save config to output directory."""
        trainer.train(training_episodes)
        
        config_path = temp_generator_dir / "config.json"
        assert config_path.exists()


class TestGeneratorTrainerEvaluation:
    """Tests for model evaluation."""

    @pytest.fixture
    def trained_trainer(
        self, temp_generator_dir: Path
    ) -> GeneratorTrainer:
        """Create and train a trainer."""
        config = GeneratorTrainingConfig(
            epochs=3,
            batch_size=16,
            sequence_length=5,
            output_dir=temp_generator_dir,
        )
        trainer = GeneratorTrainer(config=config)
        
        ep_config = EpisodeGeneratorConfig(num_episodes=50, min_length=10)
        episodes = EpisodeGenerator(config=ep_config, seed=42).generate_all()
        trainer.train(episodes)
        
        return trainer

    def test_evaluate_returns_metrics(
        self, trained_trainer: GeneratorTrainer
    ) -> None:
        """Evaluation should return performance metrics."""
        episodes = EpisodeGenerator(seed=123).generate_batch(n=20)
        
        metrics = trained_trainer.evaluate(episodes)
        
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_accuracy_is_valid(
        self, trained_trainer: GeneratorTrainer
    ) -> None:
        """Accuracy should be between 0 and 1."""
        episodes = EpisodeGenerator(seed=123).generate_batch(n=20)
        
        metrics = trained_trainer.evaluate(episodes)
        
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestGeneratorTrainerPersistence:
    """Tests for saving and loading trained models."""

    @pytest.fixture
    def trained_trainer(
        self, temp_generator_dir: Path
    ) -> GeneratorTrainer:
        """Create and train a trainer."""
        config = GeneratorTrainingConfig(
            epochs=2,
            batch_size=16,
            output_dir=temp_generator_dir,
        )
        trainer = GeneratorTrainer(config=config)
        
        episodes = EpisodeGenerator(seed=42).generate_batch(n=50)
        trainer.train(episodes)
        
        return trainer

    def test_load_trained_model(
        self,
        trained_trainer: GeneratorTrainer,
        temp_generator_dir: Path,
    ) -> None:
        """Should load trained model from disk."""
        loaded = GeneratorTrainer.load(temp_generator_dir)
        
        assert loaded is not None
        assert loaded.model is not None

    def test_loaded_model_produces_same_output(
        self,
        trained_trainer: GeneratorTrainer,
        temp_generator_dir: Path,
    ) -> None:
        """Loaded model should produce same predictions."""
        loaded = GeneratorTrainer.load(temp_generator_dir)
        
        test_input = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        
        trained_trainer.model.eval()
        loaded.model.eval()
        
        with torch.no_grad():
            orig_out = trained_trainer.model(test_input)
            loaded_out = loaded.model(test_input)
        
        assert torch.allclose(orig_out, loaded_out)
