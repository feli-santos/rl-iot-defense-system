"""
Tests for RealizationEngine.

This module tests the realization engine that samples feature vectors
from the CICIoT2023 dataset based on abstract Kill Chain states.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.utils.realization_engine import RealizationEngine
from src.utils.label_mapper import KillChainStage


class TestRealizationEngineInitialization:
    """Tests for RealizationEngine initialization."""

    def test_init_with_mock_dataset(
        self, mock_dataset: dict[str, Any]
    ) -> None:
        """Should initialize correctly with mock dataset."""
        engine = RealizationEngine(data_path=mock_dataset["path"])
        
        assert engine is not None
        assert engine.num_features > 0
        assert engine.num_samples > 0

    def test_init_loads_state_indices(
        self, mock_dataset: dict[str, Any]
    ) -> None:
        """Should load state indices mapping."""
        engine = RealizationEngine(data_path=mock_dataset["path"])
        
        # Should have indices for all 5 stages
        for stage_id in range(5):
            indices = engine.get_indices_for_stage(stage_id)
            assert len(indices) > 0

    def test_init_loads_scaler(
        self, mock_dataset: dict[str, Any]
    ) -> None:
        """Should load the feature scaler."""
        engine = RealizationEngine(data_path=mock_dataset["path"])
        
        assert engine.scaler is not None

    def test_init_invalid_path_raises(self) -> None:
        """Should raise error for invalid data path."""
        with pytest.raises(FileNotFoundError):
            RealizationEngine(data_path=Path("/nonexistent/path"))


class TestRealizationEngineSampling:
    """Tests for feature vector sampling."""

    @pytest.fixture
    def engine(self, mock_dataset: dict[str, Any]) -> RealizationEngine:
        """Create engine with mock dataset."""
        return RealizationEngine(data_path=mock_dataset["path"])

    def test_sample_for_benign_state(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should sample feature vector for BENIGN state."""
        features = engine.sample(KillChainStage.BENIGN)
        
        assert features is not None
        assert features.shape == (num_features,)
        assert features.dtype == np.float32

    def test_sample_for_attack_state(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should sample feature vector for IMPACT state."""
        features = engine.sample(KillChainStage.IMPACT)
        
        assert features is not None
        assert features.shape == (num_features,)

    def test_sample_by_stage_id(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should sample by integer stage ID."""
        features = engine.sample_by_id(stage_id=2)  # ACCESS
        
        assert features is not None
        assert features.shape == (num_features,)

    def test_sample_is_stochastic(
        self, engine: RealizationEngine
    ) -> None:
        """Multiple samples should be different (stochastic)."""
        samples = [engine.sample(KillChainStage.IMPACT) for _ in range(10)]
        
        # At least some samples should differ
        unique_samples = len(set(tuple(s) for s in samples))
        assert unique_samples > 1

    def test_sample_batch(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should sample a batch of feature vectors."""
        batch_size = 16
        features = engine.sample_batch(
            stage=KillChainStage.RECON,
            batch_size=batch_size,
        )
        
        assert features.shape == (batch_size, num_features)

    def test_sample_invalid_stage_raises(
        self, engine: RealizationEngine
    ) -> None:
        """Should raise error for invalid stage ID."""
        with pytest.raises(ValueError):
            engine.sample_by_id(stage_id=10)


class TestRealizationEngineIndices:
    """Tests for state-to-indices functionality."""

    @pytest.fixture
    def engine(self, mock_dataset: dict[str, Any]) -> RealizationEngine:
        """Create engine with mock dataset."""
        return RealizationEngine(data_path=mock_dataset["path"])

    def test_get_indices_for_each_stage(
        self, engine: RealizationEngine
    ) -> None:
        """Should have indices for all stages."""
        for stage in KillChainStage:
            indices = engine.get_indices_for_stage(stage.value)
            assert len(indices) > 0

    def test_indices_are_valid_row_numbers(
        self, engine: RealizationEngine
    ) -> None:
        """Indices should be valid dataset row numbers."""
        indices = engine.get_indices_for_stage(0)
        
        for idx in indices:
            assert 0 <= idx < engine.num_samples

    def test_get_stage_sample_counts(
        self, engine: RealizationEngine
    ) -> None:
        """Should report sample counts per stage."""
        counts = engine.get_stage_sample_counts()
        
        assert len(counts) == 5
        assert all(count > 0 for count in counts.values())


class TestRealizationEngineNormalization:
    """Tests for feature normalization."""

    @pytest.fixture
    def engine(self, mock_dataset: dict[str, Any]) -> RealizationEngine:
        """Create engine with mock dataset."""
        return RealizationEngine(data_path=mock_dataset["path"])

    def test_samples_are_normalized(
        self, engine: RealizationEngine
    ) -> None:
        """Sampled features should be normalized (roughly zero mean, unit var)."""
        # Get many samples
        samples = np.array([
            engine.sample(KillChainStage.IMPACT) for _ in range(100)
        ])
        
        # Check approximate normalization
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        
        # Most features should be close to zero mean and unit variance
        # (allowing for sampling variance)
        assert np.abs(mean).mean() < 1.0
        assert 0.5 < std.mean() < 2.0

    def test_raw_sample_option(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should optionally return raw (unnormalized) features."""
        raw = engine.sample(KillChainStage.BENIGN, normalize=False)
        
        assert raw is not None
        assert raw.shape == (num_features,)


class TestRealizationEngineStats:
    """Tests for statistics and utility methods."""

    @pytest.fixture
    def engine(self, mock_dataset: dict[str, Any]) -> RealizationEngine:
        """Create engine with mock dataset."""
        return RealizationEngine(data_path=mock_dataset["path"])

    def test_num_features_property(
        self, engine: RealizationEngine, num_features: int
    ) -> None:
        """Should report correct number of features."""
        assert engine.num_features == num_features

    def test_num_samples_property(
        self, engine: RealizationEngine
    ) -> None:
        """Should report total number of samples."""
        assert engine.num_samples == 500  # 5 stages * 100 samples each

    def test_stage_distribution(
        self, engine: RealizationEngine
    ) -> None:
        """Should report distribution of samples across stages."""
        distribution = engine.get_stage_distribution()
        
        assert len(distribution) == 5
        assert sum(distribution.values()) == engine.num_samples

    def test_get_random_stage_weighted(
        self, engine: RealizationEngine
    ) -> None:
        """Should sample random stage weighted by sample counts."""
        stages = [engine.get_random_stage_weighted() for _ in range(100)]
        
        # All stages should be valid
        assert all(0 <= s <= 4 for s in stages)
        
        # Should see variety
        assert len(set(stages)) > 1
