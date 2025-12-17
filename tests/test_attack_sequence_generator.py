"""
Tests for AttackSequenceGenerator.

This module tests the LSTM-based attack sequence generator that
predicts the next Kill Chain stage given a history of stages.
Per PRD Section 4, this is a next-token predictor with:
- Embedding layer -> Stacked LSTM -> Dense output head
- Temperature-scaled softmax for stochastic generation
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from src.generator.attack_sequence_generator import (
    AttackSequenceGenerator,
    AttackSequenceGeneratorConfig,
)


class TestAttackSequenceGeneratorConfig:
    """Tests for AttackSequenceGeneratorConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = AttackSequenceGeneratorConfig()
        
        assert config.num_stages == 5
        assert config.embedding_dim > 0
        assert config.hidden_size > 0
        assert config.num_layers >= 1
        assert 0.0 <= config.dropout < 1.0
        assert config.temperature > 0

    def test_custom_config(self) -> None:
        """Should accept custom configuration."""
        config = AttackSequenceGeneratorConfig(
            embedding_dim=64,
            hidden_size=128,
            num_layers=3,
            temperature=0.5,
        )
        
        assert config.embedding_dim == 64
        assert config.hidden_size == 128
        assert config.num_layers == 3
        assert config.temperature == 0.5


class TestAttackSequenceGeneratorArchitecture:
    """Tests for model architecture."""

    @pytest.fixture
    def model(self) -> AttackSequenceGenerator:
        """Create model with default config."""
        return AttackSequenceGenerator()

    @pytest.fixture
    def custom_model(self) -> AttackSequenceGenerator:
        """Create model with custom config."""
        config = AttackSequenceGeneratorConfig(
            embedding_dim=32,
            hidden_size=64,
            num_layers=2,
        )
        return AttackSequenceGenerator(config=config)

    def test_is_nn_module(self, model: AttackSequenceGenerator) -> None:
        """Should be a PyTorch Module."""
        assert isinstance(model, torch.nn.Module)

    def test_has_embedding_layer(self, model: AttackSequenceGenerator) -> None:
        """Should have an embedding layer."""
        assert hasattr(model, 'embedding')
        assert isinstance(model.embedding, torch.nn.Embedding)
        assert model.embedding.num_embeddings == 5  # 5 stages

    def test_has_lstm_layer(self, model: AttackSequenceGenerator) -> None:
        """Should have LSTM layers."""
        assert hasattr(model, 'lstm')
        assert isinstance(model.lstm, torch.nn.LSTM)

    def test_has_output_layer(self, model: AttackSequenceGenerator) -> None:
        """Should have output (dense) layer."""
        assert hasattr(model, 'fc')
        assert isinstance(model.fc, torch.nn.Linear)
        assert model.fc.out_features == 5  # 5 stages

    def test_custom_dimensions(self, custom_model: AttackSequenceGenerator) -> None:
        """Custom config should set correct dimensions."""
        assert custom_model.embedding.embedding_dim == 32
        assert custom_model.lstm.hidden_size == 64
        assert custom_model.lstm.num_layers == 2


class TestAttackSequenceGeneratorForward:
    """Tests for forward pass."""

    @pytest.fixture
    def model(self) -> AttackSequenceGenerator:
        """Create model."""
        return AttackSequenceGenerator()

    def test_forward_single_sequence(self, model: AttackSequenceGenerator) -> None:
        """Forward pass with single sequence."""
        # Batch size 1, sequence length 5
        x = torch.tensor([[0, 1, 1, 2, 3]], dtype=torch.long)
        
        logits = model(x)
        
        assert logits.shape == (1, 5)  # (batch, num_stages)

    def test_forward_batch(self, model: AttackSequenceGenerator) -> None:
        """Forward pass with batch of sequences."""
        # Batch size 8, sequence length 10
        x = torch.randint(0, 5, (8, 10), dtype=torch.long)
        
        logits = model(x)
        
        assert logits.shape == (8, 5)  # (batch, num_stages)

    def test_forward_returns_logits(self, model: AttackSequenceGenerator) -> None:
        """Should return raw logits (not probabilities)."""
        x = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        
        logits = model(x)
        
        # Logits can be any real number (not 0-1)
        assert logits.min() < 0 or logits.max() > 1

    def test_forward_different_sequence_lengths(
        self, model: AttackSequenceGenerator
    ) -> None:
        """Should handle different sequence lengths."""
        x5 = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        x10 = torch.randint(0, 5, (1, 10), dtype=torch.long)
        x20 = torch.randint(0, 5, (1, 20), dtype=torch.long)
        
        logits5 = model(x5)
        logits10 = model(x10)
        logits20 = model(x20)
        
        # All should have same output shape
        assert logits5.shape == logits10.shape == logits20.shape == (1, 5)


class TestAttackSequenceGeneratorInference:
    """Tests for inference (sampling next stage)."""

    @pytest.fixture
    def model(self) -> AttackSequenceGenerator:
        """Create model."""
        model = AttackSequenceGenerator()
        model.eval()
        return model

    def test_predict_next_returns_distribution(
        self, model: AttackSequenceGenerator
    ) -> None:
        """predict_next should return probability distribution."""
        history = [0, 1, 1, 2]
        
        probs = model.predict_next(history)
        
        assert probs.shape == (5,)
        assert np.isclose(probs.sum(), 1.0)
        assert all(p >= 0 for p in probs)

    def test_sample_next_returns_valid_stage(
        self, model: AttackSequenceGenerator
    ) -> None:
        """sample_next should return valid stage ID."""
        history = [0, 1, 1, 2]
        
        next_stage = model.sample_next(history)
        
        assert 0 <= next_stage <= 4

    def test_sample_next_is_stochastic(
        self, model: AttackSequenceGenerator
    ) -> None:
        """sample_next should produce different results (stochastic)."""
        history = [0, 0, 0, 0]  # All benign
        
        samples = [model.sample_next(history) for _ in range(50)]
        
        # Should see some variation
        unique_samples = set(samples)
        assert len(unique_samples) > 1

    def test_temperature_affects_sampling(
        self, model: AttackSequenceGenerator
    ) -> None:
        """Higher temperature should increase randomness."""
        history = [0, 1, 2, 3]
        
        # Low temperature (more deterministic)
        samples_low = [
            model.sample_next(history, temperature=0.1)
            for _ in range(50)
        ]
        
        # High temperature (more random)
        samples_high = [
            model.sample_next(history, temperature=2.0)
            for _ in range(50)
        ]
        
        # High temperature should have more variety
        unique_low = len(set(samples_low))
        unique_high = len(set(samples_high))
        
        # High temp should have >= unique values
        assert unique_high >= unique_low

    def test_generate_sequence_returns_list(
        self, model: AttackSequenceGenerator
    ) -> None:
        """generate_sequence should return list of stages."""
        sequence = model.generate_sequence(
            start_history=[0],
            length=10,
        )
        
        assert isinstance(sequence, list)
        assert len(sequence) == 10
        assert all(0 <= s <= 4 for s in sequence)

    def test_generate_sequence_starts_with_history(
        self, model: AttackSequenceGenerator
    ) -> None:
        """Generated sequence should start with given history."""
        start = [0, 1, 2]
        
        sequence = model.generate_sequence(
            start_history=start,
            length=10,
        )
        
        assert sequence[:3] == start


class TestAttackSequenceGeneratorSaveLoad:
    """Tests for model persistence."""

    @pytest.fixture
    def model(self) -> AttackSequenceGenerator:
        """Create trained model."""
        model = AttackSequenceGenerator()
        # Simulate training by modifying weights
        with torch.no_grad():
            model.fc.weight.fill_(0.5)
        return model

    def test_save_model(
        self, model: AttackSequenceGenerator, temp_generator_dir: Path
    ) -> None:
        """Should save model to disk."""
        save_path = temp_generator_dir / "model.pth"
        
        model.save(save_path)
        
        assert save_path.exists()

    def test_load_model(
        self, model: AttackSequenceGenerator, temp_generator_dir: Path
    ) -> None:
        """Should load model from disk."""
        save_path = temp_generator_dir / "model.pth"
        model.save(save_path)
        
        loaded = AttackSequenceGenerator.load(save_path)
        
        assert isinstance(loaded, AttackSequenceGenerator)

    def test_loaded_model_matches_original(
        self, model: AttackSequenceGenerator, temp_generator_dir: Path
    ) -> None:
        """Loaded model should produce same outputs."""
        save_path = temp_generator_dir / "model.pth"
        model.save(save_path)
        
        loaded = AttackSequenceGenerator.load(save_path)
        
        # Compare weights
        orig_weights = model.fc.weight.data
        loaded_weights = loaded.fc.weight.data
        
        assert torch.allclose(orig_weights, loaded_weights)

    def test_save_with_config(
        self, model: AttackSequenceGenerator, temp_generator_dir: Path
    ) -> None:
        """Should save config alongside model."""
        save_path = temp_generator_dir / "model.pth"
        
        model.save(save_path, save_config=True)
        
        config_path = temp_generator_dir / "config.json"
        assert config_path.exists()


class TestAttackSequenceGeneratorGradients:
    """Tests for training compatibility."""

    @pytest.fixture
    def model(self) -> AttackSequenceGenerator:
        """Create model in training mode."""
        model = AttackSequenceGenerator()
        model.train()
        return model

    def test_gradients_flow(self, model: AttackSequenceGenerator) -> None:
        """Gradients should flow through the model."""
        x = torch.randint(0, 5, (4, 10), dtype=torch.long)
        y = torch.randint(0, 5, (4,), dtype=torch.long)
        
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        # Check gradients exist
        assert model.embedding.weight.grad is not None
        assert model.fc.weight.grad is not None

    def test_parameters_are_trainable(
        self, model: AttackSequenceGenerator
    ) -> None:
        """All parameters should be trainable."""
        for param in model.parameters():
            assert param.requires_grad
