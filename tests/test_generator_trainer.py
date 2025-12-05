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
