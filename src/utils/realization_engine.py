"""
Realization Engine for Adversarial IoT Environment.

This module provides the RealizationEngine class that samples feature
vectors from the CICIoT2023 dataset based on abstract Kill Chain states.
Per PRD Section 5.1, this translates abstract states into concrete
numerical observations for the Blue Team.

The engine:
1. Loads pre-processed dataset with state indices
2. Samples feature vectors matching the current Kill Chain stage
3. Returns normalized feature vectors as environment observations
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.label_mapper import KillChainStage

logger = logging.getLogger(__name__)


class RealizationEngine:
    """Samples real CICIoT2023 features based on Kill Chain states.
    
    This class acts as the "Oracle" that translates abstract attack
    stages into concrete feature vector observations. It maintains
    an index of dataset rows by Kill Chain stage and samples from
    the appropriate subset when requested.
    
    Attributes:
        data_path: Path to processed dataset directory.
        features: Full normalized feature matrix (num_samples, num_features).
        state_indices: Mapping from stage ID to list of row indices.
        scaler: StandardScaler used for normalization.
        num_features: Number of features per sample.
        num_samples: Total number of samples in dataset.
    
    Example:
        >>> engine = RealizationEngine(Path("data/processed/ciciot2023"))
        >>> observation = engine.sample(KillChainStage.IMPACT)
        >>> observation.shape
        (46,)
    """
    
    def __init__(
        self,
        data_path: Path,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the realization engine.
        
        Args:
            data_path: Path to processed dataset directory containing
                features.npy, state_indices.json, and scaler.joblib.
            seed: Random seed for reproducibility (optional).
        
        Raises:
            FileNotFoundError: If required files are missing.
        """
        self._data_path = Path(data_path)
        self._rng = np.random.default_rng(seed)
        
        self._validate_data_path()
        self._load_artifacts()
        
        logger.info(
            f"RealizationEngine initialized with {self.num_samples} samples, "
            f"{self.num_features} features"
        )
    
    def _validate_data_path(self) -> None:
        """Validate that required files exist."""
        required_files = [
            "features.npy",
            "state_indices.json",
            "scaler.joblib",
        ]
        
        for filename in required_files:
            filepath = self._data_path / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required file not found: {filepath}"
                )
    
    def _load_artifacts(self) -> None:
        """Load dataset artifacts from disk."""
        # Load features
        self._features: np.ndarray = np.load(
            self._data_path / "features.npy"
        ).astype(np.float32)
        
        # Load raw features if available (for normalize=False option)
        raw_features_path = self._data_path / "features_raw.npy"
        if raw_features_path.exists():
            self._raw_features: Optional[np.ndarray] = np.load(
                raw_features_path
            ).astype(np.float32)
        else:
            self._raw_features = None
        
        # Load state indices
        with open(self._data_path / "state_indices.json", "r") as f:
            raw_indices = json.load(f)
        
        # Convert string keys to int
        self._state_indices: dict[int, list[int]] = {
            int(k): v for k, v in raw_indices.items()
        }
        
        # Load scaler
        self._scaler: StandardScaler = joblib.load(
            self._data_path / "scaler.joblib"
        )
        
        # Cache counts for weighted sampling
        self._stage_counts = {
            stage_id: len(indices)
            for stage_id, indices in self._state_indices.items()
        }
        self._total_samples = sum(self._stage_counts.values())
        
        logger.debug(f"Loaded state indices: {self._stage_counts}")
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def num_features(self) -> int:
        """Get number of features per sample."""
        return self._features.shape[1]
    
    @property
    def num_samples(self) -> int:
        """Get total number of samples in dataset."""
        return self._features.shape[0]
    
    @property
    def scaler(self) -> StandardScaler:
        """Get the feature scaler."""
        return self._scaler
    
    # =========================================================================
    # Sampling Methods
    # =========================================================================
    
    def sample(
        self,
        stage: KillChainStage,
        normalize: bool = True,
    ) -> np.ndarray:
        """Sample a feature vector for the given Kill Chain stage.
        
        Args:
            stage: Kill Chain stage to sample from.
            normalize: Whether to return normalized features (default True).
        
        Returns:
            Feature vector of shape (num_features,).
        """
        return self.sample_by_id(stage.value, normalize=normalize)
    
    def sample_by_id(
        self,
        stage_id: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Sample a feature vector by integer stage ID.
        
        Args:
            stage_id: Integer stage ID (0-4).
            normalize: Whether to return normalized features.
        
        Returns:
            Feature vector of shape (num_features,).
        
        Raises:
            ValueError: If stage_id is out of range.
        """
        if stage_id not in self._state_indices:
            raise ValueError(
                f"Invalid stage_id: {stage_id}. Must be 0-4."
            )
        
        indices = self._state_indices[stage_id]
        selected_idx = self._rng.choice(indices)
        
        if normalize or self._raw_features is None:
            return self._features[selected_idx].copy()
        else:
            return self._raw_features[selected_idx].copy()
    
    def sample_batch(
        self,
        stage: Union[KillChainStage, int],
        batch_size: int,
        normalize: bool = True,
    ) -> np.ndarray:
        """Sample a batch of feature vectors.
        
        Args:
            stage: Kill Chain stage or integer ID.
            batch_size: Number of samples to return.
            normalize: Whether to return normalized features.
        
        Returns:
            Feature matrix of shape (batch_size, num_features).
        """
        stage_id = stage.value if isinstance(stage, KillChainStage) else stage
        
        if stage_id not in self._state_indices:
            raise ValueError(f"Invalid stage: {stage_id}")
        
        indices = self._state_indices[stage_id]
        selected_indices = self._rng.choice(indices, size=batch_size)
        
        features = self._raw_features if not normalize and self._raw_features is not None else self._features
        return features[selected_indices].copy()
    
    # =========================================================================
    # Index Access
    # =========================================================================
    
    def get_indices_for_stage(self, stage_id: int) -> list[int]:
        """Get dataset indices for a given stage.
        
        Args:
            stage_id: Integer stage ID (0-4).
        
        Returns:
            List of row indices in the dataset.
        """
        return self._state_indices.get(stage_id, []).copy()
    
    def get_stage_sample_counts(self) -> dict[int, int]:
        """Get number of samples per stage.
        
        Returns:
            Dictionary mapping stage IDs to sample counts.
        """
        return self._stage_counts.copy()
    
    def get_stage_distribution(self) -> dict[int, int]:
        """Get distribution of samples across stages.
        
        Returns:
            Dictionary mapping stage IDs to sample counts.
        """
        return self._stage_counts.copy()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_random_stage_weighted(self) -> int:
        """Sample a random stage weighted by sample counts.
        
        Stages with more samples are more likely to be selected.
        Useful for training data augmentation.
        
        Returns:
            Integer stage ID (0-4).
        """
        stages = list(self._stage_counts.keys())
        weights = [self._stage_counts[s] / self._total_samples for s in stages]
        
        return int(self._rng.choice(stages, p=weights))
    
    def get_features_for_indices(
        self,
        indices: list[int],
        normalize: bool = True,
    ) -> np.ndarray:
        """Get features for specific dataset indices.
        
        Args:
            indices: List of row indices.
            normalize: Whether to return normalized features.
        
        Returns:
            Feature matrix of shape (len(indices), num_features).
        """
        features = self._raw_features if not normalize and self._raw_features is not None else self._features
        return features[indices].copy()
