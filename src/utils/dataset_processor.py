"""
CICIoT2023 Dataset Processor

Processes raw CICIoT2023 dataset for Attack Sequence Generator training
and Adversarial Environment integration (PRD compliant).

Key outputs:
- features.npy: Normalized feature matrix for RealizationEngine
- labels.npy: CICIoT2023 label strings for each sample
- state_indices.json: Mapping from Kill Chain stages to dataset row indices
- scaler.joblib: StandardScaler for feature normalization
- metadata.json: Dataset metadata and configuration
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.label_mapper import AbstractStateLabelMapper, KillChainStage

logger = logging.getLogger(__name__)


@dataclass
class DataProcessingConfig:
    """Configuration for dataset processing with configurable splits and EDA recommendations."""
    dataset_path: Path
    output_path: Path
    sample_size: int
    sequence_length: int
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4
    random_state: int = 42
    # Feature engineering options
    feature_selection: bool = False
    variance_threshold: float = 0.01  # Remove features with variance below this
    correlation_threshold: float = 0.95  # Remove redundant features above this correlation
    feature_keep_keywords: Optional[List[str]] = None
    sampling_strategy: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate split ratios sum to 1.0."""
        total_split = self.train_split + self.val_split + self.test_split
        if not np.isclose(total_split, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")


class CICIoTProcessor:
    """
    Processes CICIoT2023 dataset with configurable train/validation/test splits.
    
    Features:
    - Configurable data splits from config file
    - Comprehensive preprocessing with scaling and encoding
    - Sequence generation for LSTM training
    - Artifact saving for reproducible training
    """
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.output_path = Path(config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Preprocessing artifacts
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_columns: List[str] = []
        self.class_names: List[str] = []
        
        logger.info(f"Initialized CICIoT processor with splits: "
                   f"train={config.train_split}, val={config.val_split}, test={config.test_split}")
    
    def process_dataset(self) -> Dict[str, Any]:
        """
        Process the complete CICIoT2023 dataset with configurable splits.
        
        Returns:
            Processing results dictionary
        """
        logger.info("Starting CICIoT2023 dataset processing...")
        
        try:
            # Load raw data
            raw_data = self._load_raw_data()
            logger.info(f"Loaded {len(raw_data):,} raw samples")
            
            # Sample data if needed
            if self.config.sample_size < len(raw_data):
                raw_data = raw_data.sample(n=self.config.sample_size, random_state=self.config.random_state)
                logger.info(f"Sampled to {len(raw_data):,} samples")
            
            # Clean and preprocess data
            processed_data = self._preprocess_data(raw_data)
            logger.info(f"Preprocessed data shape: {processed_data.shape}")
            
            # Split data with configurable ratios
            train_data, val_data, test_data = self._split_data(processed_data)
            
            # Generate sequences for LSTM
            train_sequences = self._generate_sequences(train_data)
            val_sequences = self._generate_sequences(val_data)
            test_sequences = self._generate_sequences(test_data)
            
            # Save processed data
            self._save_processed_data(train_sequences, val_sequences, test_sequences)
            
            # Save preprocessing artifacts
            self._save_artifacts()
            
            results = {
                'total_samples': len(raw_data),
                'train_samples': len(train_sequences[0]),
                'val_samples': len(val_sequences[0]),
                'test_samples': len(test_sequences[0]),
                'feature_count': len(self.feature_columns),
                'class_count': len(self.class_names),
                'sequence_length': self.config.sequence_length,
                'splits': {
                    'train': self.config.train_split,
                    'val': self.config.val_split,
                    'test': self.config.test_split
                }
            }
            
            logger.info("Dataset processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            raise
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw CICIoT2023 dataset from CSV files."""
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find CSV files in the dataset directory
        csv_files = list(dataset_path.glob("**/*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Load and combine all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                dataframes.append(df)
                logger.debug(f"Loaded {csv_file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        
        # Combine all dataframes
        combined_data = pd.concat(dataframes, ignore_index=True)
        return combined_data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data with class balancing and feature selection."""
        logger.info("Preprocessing data...")
        
        # Drop rows with missing values
        data = data.dropna()
        
        # Identify feature and target columns
        target_column = data.columns[-1]
        feature_columns = data.columns[:-1].tolist()
        
        # Separate features and targets
        X = data[feature_columns].copy()
        y = data[target_column].copy()
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Clean inf/NaN values before scaling (critical for numerical stability)
        X = self._clean_numerical_data(X)
        
        # Feature selection - remove zero/low variance features and correlations
        if hasattr(self.config, 'feature_selection') and self.config.feature_selection:
            X = self._apply_feature_selection(X)
            logger.info(f"After feature selection: {X.shape[1]} features")
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Encode target labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store class names and feature columns
        self.class_names = self.label_encoder.classes_.tolist()
        self.feature_columns = X.columns.tolist()
        
        # Apply class balancing if configured
        if hasattr(self.config, 'sampling_strategy') and self.config.sampling_strategy == 'balanced':
            X, y_encoded = self._apply_class_balancing(X, y_encoded)
            logger.info(f"After class balancing: {len(X)} samples")
        
        # Combine features and targets
        processed_data = X.copy()
        processed_data['target'] = y_encoded
        
        logger.info(f"Final preprocessing: {len(self.feature_columns)} features, {len(self.class_names)} classes")
        return processed_data

    def _clean_numerical_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean inf and NaN values from numerical data.
        
        Strategy:
        - Replace inf with large finite values (preserves scale)
        - Fill NaN with column median (robust to outliers)
        """
        # Replace inf with large finite values
        X = X.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min])
        
        # Fill remaining NaN with median
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
        
        inf_count = np.isinf(X.values).sum()
        nan_count = X.isna().sum().sum()
        if inf_count > 0 or nan_count > 0:
            logger.warning(f"Cleaned {inf_count} inf and {nan_count} NaN values")
        
        return X
    
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove zero/low variance and highly correlated features.
        
        Three-stage process:
        1. Remove zero-variance features
        2. Remove low-variance features (below threshold)
        3. Remove redundant highly-correlated features
        """
        from sklearn.feature_selection import VarianceThreshold
        
        original_count = X.shape[1]
        
        # Stage 1: Remove zero variance features
        selector = VarianceThreshold(threshold=0)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        logger.info(f"Stage 1: Removed {original_count - len(selected_features)} zero-variance features")
        
        # Stage 2: Remove low variance features (below configured threshold)
        variance_threshold = getattr(self.config, 'variance_threshold', 0.01)
        keep_features = self._get_keep_features(X.columns)
        if len(selected_features) > 10:  # Only if we have enough features
            variances = X.var()
            high_var_features = variances[variances >= variance_threshold].index.tolist()
            kept_set = set(high_var_features).union(keep_features)
            kept_ordered = [col for col in X.columns if col in kept_set]
            removed_count = len(selected_features) - len(kept_ordered)
            X = X[kept_ordered]
            logger.info(f"Stage 2: Removed {removed_count} low-variance features (threshold={variance_threshold})")
        
        # Stage 3: Remove highly correlated features
        correlation_threshold = getattr(self.config, 'correlation_threshold', 0.95)
        X = self._remove_correlated_features(
            X,
            threshold=correlation_threshold,
            keep_features=keep_features,
        )
        
        logger.info(f"Feature selection complete: {original_count} -> {X.shape[1]} features")
        return X
    
    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95,
        keep_features: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Remove highly correlated features to reduce redundancy.
        
        For each pair of features with correlation > threshold,
        remove the one with lower variance (less informative).
        """
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        keep_features = keep_features or set()
        for column in upper_tri.columns:
            # Find features correlated above threshold
            correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
            if correlated_features:
                # Keep the feature with higher variance, drop the others
                variances = X[[column] + correlated_features].var()
                features_to_drop = variances.nsmallest(len(correlated_features)).index.tolist()
                drop_candidates = features_to_drop[:-1]  # Keep the highest variance one
                to_drop.update([feat for feat in drop_candidates if feat not in keep_features])
        
        if to_drop:
            logger.info(f"Stage 3: Removed {len(to_drop)} highly correlated features (threshold={threshold})")
            X = X.drop(columns=list(to_drop))
        
        return X

    def _get_keep_features(self, columns: List[str]) -> set[str]:
        """Determine features to keep regardless of selection thresholds.

        Args:
            columns: Available feature columns.

        Returns:
            Set of feature names to preserve.
        """
        keywords = getattr(self.config, 'feature_keep_keywords', None) or []
        if not keywords:
            return set()

        lowered_keywords = [keyword.lower() for keyword in keywords]
        keep_features = {
            col for col in columns
            if any(keyword in col.lower() for keyword in lowered_keywords)
        }

        if keep_features:
            logger.info(
                "Preserving %d features due to keyword matches: %s",
                len(keep_features),
                sorted(keep_features)[:10],
            )

        return keep_features

    def _apply_class_balancing(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply class balancing for severe imbalance."""
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # Use SMOTE + RandomUnderSampler for severe imbalance
        over = SMOTE(sampling_strategy=0.1, random_state=self.config.random_state)  # Oversample to 10%
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=self.config.random_state) # Undersample to 50%
        
        pipeline = Pipeline(steps=[('over', over), ('under', under)])
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
        
        return pd.DataFrame(X_balanced, columns=X.columns), y_balanced
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets with configurable ratios."""
        logger.info(f"Splitting data: train={self.config.train_split}, "
                   f"val={self.config.val_split}, test={self.config.test_split}")
        
        # First split: separate test set
        test_size = self.config.test_split
        temp_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=self.config.random_state,
            stratify=data['target']
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.config.val_split / (self.config.train_split + self.config.val_split)
        train_data, val_data = train_test_split(
            temp_data,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=temp_data['target']
        )
        
        logger.info(f"Split sizes - Train: {len(train_data):,}, "
                   f"Val: {len(val_data):,}, Test: {len(test_data):,}")
        
        return train_data, val_data, test_data
    
    def _generate_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sequences for LSTM training."""
        features = data[self.feature_columns].values
        targets = data['target'].values
        
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.config.sequence_length + 1):
            sequence = features[i:i + self.config.sequence_length]
            target = targets[i + self.config.sequence_length - 1]
            
            sequences.append(sequence)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def _save_processed_data(self, train_sequences: Tuple[np.ndarray, np.ndarray],
                           val_sequences: Tuple[np.ndarray, np.ndarray],
                           test_sequences: Tuple[np.ndarray, np.ndarray]) -> None:
        """Save processed sequences to disk."""
        logger.info("Saving processed data...")
        
        # Save training data
        np.save(self.output_path / "train_sequences.npy", train_sequences[0])
        np.save(self.output_path / "train_targets.npy", train_sequences[1])
        
        # Save validation data
        np.save(self.output_path / "val_sequences.npy", val_sequences[0])
        np.save(self.output_path / "val_targets.npy", val_sequences[1])
        
        # Save test data
        np.save(self.output_path / "test_sequences.npy", test_sequences[0])
        np.save(self.output_path / "test_targets.npy", test_sequences[1])
        
        logger.info(f"Processed data saved to {self.output_path}")
    
    def _save_artifacts(self) -> None:
        """Save preprocessing artifacts for reproducible training."""
        logger.info("Saving preprocessing artifacts...")
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, self.output_path / "scaler.joblib")
        
        # Save label encoder
        if self.label_encoder:
            joblib.dump(self.label_encoder, self.output_path / "label_encoder.joblib")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'class_names': self.class_names,
            'sequence_length': self.config.sequence_length,
            'splits': {
                'train': self.config.train_split,
                'val': self.config.val_split,
                'test': self.config.test_split
            }
        }
        
        with open(self.output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Preprocessing artifacts saved")
    
    def process_for_adversarial_env(self) -> Dict[str, Any]:
        """
        Process dataset for the Adversarial IoT Environment (PRD compliant).
        
        Creates:
        - features.npy: Normalized feature matrix (num_samples, num_features)
        - labels.npy: Original CICIoT2023 labels (strings)
        - state_indices.json: Kill Chain stage to row indices mapping
        - scaler.joblib: Feature scaler
        - metadata.json: Dataset metadata
        
        Returns:
            Processing results dictionary
        """
        logger.info("Processing dataset for Adversarial Environment...")
        
        try:
            # Load raw data
            raw_data = self._load_raw_data()
            logger.info(f"Loaded {len(raw_data):,} raw samples")
            
            # Sample data if needed
            if self.config.sample_size < len(raw_data):
                raw_data = raw_data.sample(
                    n=self.config.sample_size,
                    random_state=self.config.random_state
                )
                logger.info(f"Sampled to {len(raw_data):,} samples")
            
            # Extract features and labels
            features, labels = self._extract_features_and_labels(raw_data)
            
            # Store raw features before processing
            raw_features_copy = features.copy()
            
            # Convert to DataFrame for feature engineering
            features_df = pd.DataFrame(features, columns=self.feature_columns)
            
            # Clean inf/NaN values
            features_df = self._clean_numerical_data(features_df)
            
            # Apply feature selection if enabled
            if hasattr(self.config, 'feature_selection') and self.config.feature_selection:
                features_df = self._apply_feature_selection(features_df)
                self.feature_columns = features_df.columns.tolist()
                logger.info(f"After feature selection: {len(self.feature_columns)} features")
            
            # Convert back to numpy
            features = features_df.values
            
            # Normalize features
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(features)
            
            # Build state indices using AbstractStateLabelMapper
            state_indices = self._build_state_indices(labels)
            
            # Save artifacts
            self._save_adversarial_artifacts(
                normalized_features,
                labels,
                state_indices,
                features,  # Save processed (but not scaled) features
            )
            
            results = {
                'total_samples': len(normalized_features),
                'num_features': normalized_features.shape[1],
                'num_stages': 5,
                'stage_counts': {
                    int(k): len(v) for k, v in state_indices.items()
                },
                'class_names': list(set(labels)),
            }
            
            logger.info("Adversarial environment dataset processing completed")
            return results
            
        except Exception as e:
            logger.error(f"Adversarial env processing failed: {e}")
            raise
    
    def _extract_features_and_labels(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label array from raw data."""
        # Identify target column (usually last column named 'label' or similar)
        target_column = data.columns[-1]
        feature_columns = data.columns[:-1].tolist()
        
        # Drop rows with missing values
        data = data.dropna()
        
        # Extract features
        X = data[feature_columns].copy()
        
        # Handle categorical features (convert to numeric)
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].astype('category').cat.codes
        
        # Convert to numpy
        features = X.values.astype(np.float32)
        labels = data[target_column].values.astype(str)
        
        self.feature_columns = feature_columns
        
        logger.info(f"Extracted {features.shape[1]} features, {len(set(labels))} unique labels")
        return features, labels
    
    def _build_state_indices(self, labels: np.ndarray) -> Dict[int, List[int]]:
        """Build mapping from Kill Chain stages to dataset row indices.
        
        Uses AbstractStateLabelMapper to convert CICIoT2023 labels to
        abstract Kill Chain stages (0-4), then groups row indices by stage.
        """
        mapper = AbstractStateLabelMapper()
        state_indices: Dict[int, List[int]] = {i: [] for i in range(5)}
        
        unknown_labels = set()
        
        for idx, label in enumerate(labels):
            stage_id = mapper.get_stage_id_safe(label, default=-1)
            
            if stage_id == -1:
                unknown_labels.add(label)
                # Default unknown labels to BENIGN for safety
                stage_id = 0
            
            state_indices[stage_id].append(idx)
        
        if unknown_labels:
            logger.warning(
                f"Found {len(unknown_labels)} unknown labels mapped to BENIGN: "
                f"{list(unknown_labels)[:5]}..."
            )
        
        # Log distribution
        for stage_id, indices in state_indices.items():
            stage_name = mapper.get_stage_name(stage_id)
            logger.info(f"Stage {stage_id} ({stage_name}): {len(indices):,} samples")
        
        return state_indices
    
    def _save_adversarial_artifacts(
        self,
        normalized_features: np.ndarray,
        labels: np.ndarray,
        state_indices: Dict[int, List[int]],
        raw_features: np.ndarray,
    ) -> None:
        """Save artifacts for RealizationEngine and Adversarial Environment."""
        logger.info(f"Saving adversarial environment artifacts to {self.output_path}...")
        
        # Save normalized features
        np.save(self.output_path / "features.npy", normalized_features.astype(np.float32))
        
        # Save raw features (for normalize=False option)
        np.save(self.output_path / "features_raw.npy", raw_features.astype(np.float32))
        
        # Save labels (as strings)
        np.save(self.output_path / "labels.npy", labels)
        
        # Save state indices
        with open(self.output_path / "state_indices.json", "w") as f:
            json.dump({str(k): v for k, v in state_indices.items()}, f)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, self.output_path / "scaler.joblib")
        
        # Compute stage distribution statistics
        stage_counts = {int(k): len(v) for k, v in state_indices.items()}
        total_samples = sum(stage_counts.values())
        stage_percentages = {k: v / total_samples * 100 for k, v in stage_counts.items()}
        
        # Compute imbalance ratio
        if stage_counts:
            majority_count = max(stage_counts.values())
            minority_count = min(stage_counts.values())
            imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        # Save metadata with enhanced statistics
        metadata = {
            'num_samples': len(normalized_features),
            'num_features': normalized_features.shape[1],
            'num_stages': 5,
            'feature_columns': self.feature_columns,
            'stage_counts': stage_counts,
            'stage_percentages': stage_percentages,
            'imbalance_ratio': imbalance_ratio,
            'feature_selection_enabled': getattr(self.config, 'feature_selection', False),
            'variance_threshold': getattr(self.config, 'variance_threshold', 0.01),
            'correlation_threshold': getattr(self.config, 'correlation_threshold', 0.95),
            'feature_keep_keywords': getattr(self.config, 'feature_keep_keywords', None),
            'sampling_strategy': getattr(self.config, 'sampling_strategy', None),
        }
        with open(self.output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Adversarial environment artifacts saved successfully")
    
    def load_artifacts(self) -> Dict[str, Any]:
        """Load preprocessing artifacts."""
        artifacts = {}
        
        # Load scaler
        scaler_path = self.output_path / "scaler.joblib"
        if scaler_path.exists():
            artifacts['scaler'] = joblib.load(scaler_path)
        
        # Load label encoder
        encoder_path = self.output_path / "label_encoder.joblib"
        if encoder_path.exists():
            artifacts['label_encoder'] = joblib.load(encoder_path)
        
        # Load metadata
        metadata_path = self.output_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                artifacts['metadata'] = json.load(f)
        
        return artifacts