"""
CICIoT2023 Dataset Processor

Processes raw CICIoT2023 dataset for Attack Sequence Generator training
and Adversarial Environment integration.

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
    min_samples_per_stage: int = 2000  # Floor for each Kill Chain stage in env dataset
    sampling_mode: str = "default"  # default | smart_balanced
    benign_target_count: Optional[int] = None
    max_samples_per_attack_class: Optional[int] = None
    
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
        self.sampling_info: Dict[str, Any] = {}
        self.feature_selection_info: Dict[str, Any] = {
            'enabled': bool(getattr(config, 'feature_selection', False)),
            'original_feature_count': 0,
            'final_feature_count': 0,
            'dropped_zero_variance': [],
            'dropped_low_variance': [],
            'dropped_high_correlation': [],
            'dropped_total': [],
        }
        self.split_info: Dict[str, Any] = {}
        
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
            raw_data = self._sample_raw_data(raw_data)
            logger.info(f"Sampled to {len(raw_data):,} samples")

            # Split BEFORE feature selection / scaling to prevent leakage
            target_column = raw_data.columns[-1]
            train_raw, val_raw, test_raw = self._split_by_target(raw_data, target_column)

            # Build shared label encoder from sampled data labels for robust transforms
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(raw_data[target_column].astype(str))
            self.class_names = self.label_encoder.classes_.tolist()

            train_data = self._preprocess_split(train_raw, target_column, fit_transformers=True)
            val_data = self._preprocess_split(val_raw, target_column, fit_transformers=False)
            test_data = self._preprocess_split(test_raw, target_column, fit_transformers=False)

            logger.info(
                "Preprocessed split shapes - Train: %s, Val: %s, Test: %s",
                train_data.shape,
                val_data.shape,
                test_data.shape,
            )
            
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
                },
                'sampling': self.sampling_info,
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

    def _preprocess_split(
        self,
        split_data: pd.DataFrame,
        target_column: str,
        fit_transformers: bool,
    ) -> pd.DataFrame:
        """Preprocess a split with train-only-fitted transformers.

        Args:
            split_data: DataFrame containing features + target column.
            target_column: Target column name.
            fit_transformers: Whether to fit feature selection/scaler on this split.

        Returns:
            Processed DataFrame with scaled features and encoded target.
        """
        split_data = split_data.dropna().copy()
        feature_columns = [col for col in split_data.columns if col != target_column]

        X = split_data[feature_columns].copy()
        y = split_data[target_column].astype(str).copy()

        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].astype('category').cat.codes

        X = self._clean_numerical_data(X)

        if fit_transformers:
            if self.config.feature_selection:
                X = self._apply_feature_selection(X)
            self.feature_columns = X.columns.tolist()
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if not self.feature_columns:
                raise ValueError("Feature columns not initialized from training split")
            if self.scaler is None:
                raise ValueError("Scaler not initialized from training split")
            X = X[self.feature_columns]
            X_scaled = self.scaler.transform(X)

        if self.label_encoder is None:
            raise ValueError("Label encoder not initialized")
        y_encoded = self.label_encoder.transform(y)

        processed_data = pd.DataFrame(X_scaled, columns=self.feature_columns, index=split_data.index)
        processed_data['target'] = y_encoded
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
        original_features = X.columns.tolist()
        
        # Stage 1: Remove zero variance features
        selector = VarianceThreshold(threshold=0)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        zero_var_dropped = [col for col in original_features if col not in selected_features]
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        logger.info(f"Stage 1: Removed {original_count - len(selected_features)} zero-variance features")
        
        # Stage 2: Remove low variance features (below configured threshold)
        variance_threshold = getattr(self.config, 'variance_threshold', 0.01)
        keep_features = self._get_keep_features(X.columns)
        low_var_dropped: List[str] = []
        if len(selected_features) > 10:  # Only if we have enough features
            variances = X.var()
            high_var_features = variances[variances >= variance_threshold].index.tolist()
            kept_set = set(high_var_features).union(keep_features)
            kept_ordered = [col for col in X.columns if col in kept_set]
            low_var_dropped = [col for col in X.columns if col not in kept_ordered]
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

        corr_dropped = sorted(getattr(self, '_last_correlated_drops', []))
        final_features = X.columns.tolist()
        dropped_total = [col for col in original_features if col not in final_features]
        self.feature_selection_info = {
            'enabled': True,
            'original_feature_count': len(original_features),
            'final_feature_count': len(final_features),
            'dropped_zero_variance': sorted(zero_var_dropped),
            'dropped_low_variance': sorted(low_var_dropped),
            'dropped_high_correlation': corr_dropped,
            'dropped_total': sorted(dropped_total),
        }
        
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
        variances = X.var()
        keep_features = keep_features or set()
        to_drop: set[str] = set()

        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                feat_i = cols[i]
                feat_j = cols[j]

                if feat_i in to_drop or feat_j in to_drop:
                    continue

                corr_val = corr_matrix.iloc[i, j]
                if not np.isfinite(corr_val) or corr_val <= threshold:
                    continue

                var_i = float(variances[feat_i])
                var_j = float(variances[feat_j])
                i_protected = feat_i in keep_features
                j_protected = feat_j in keep_features

                if i_protected and j_protected:
                    # Do not allow both protected features to bypass threshold.
                    if var_i < var_j:
                        drop_feat = feat_i
                    elif var_j < var_i:
                        drop_feat = feat_j
                    else:
                        drop_feat = max(feat_i, feat_j)
                elif i_protected:
                    drop_feat = feat_j
                elif j_protected:
                    drop_feat = feat_i
                else:
                    if var_i < var_j:
                        drop_feat = feat_i
                    elif var_j < var_i:
                        drop_feat = feat_j
                    else:
                        drop_feat = max(feat_i, feat_j)

                to_drop.add(drop_feat)

        if to_drop:
            logger.info(
                "Stage 3: Removed %d highly correlated features (threshold=%s)",
                len(to_drop),
                threshold,
            )
            X = X.drop(columns=sorted(to_drop))

        self._last_correlated_drops = sorted(to_drop)

        if X.shape[1] > 1:
            remaining_corr = X.corr().abs()
            upper_vals = remaining_corr.where(
                np.triu(np.ones(remaining_corr.shape), k=1).astype(bool)
            ).stack()
            if not upper_vals.empty:
                logger.info("Stage 3 post-check max |corr|: %.4f", float(upper_vals.max()))

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

    def _split_by_target(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by raw target labels, with robust fallback for singleton classes."""
        logger.info(
            "Splitting raw data by target: train=%s, val=%s, test=%s",
            self.config.train_split,
            self.config.val_split,
            self.config.test_split,
        )

        try:
            temp_data, test_data = train_test_split(
                data,
                test_size=self.config.test_split,
                random_state=self.config.random_state,
                stratify=data[target_column],
            )
            val_size_adjusted = self.config.val_split / (self.config.train_split + self.config.val_split)
            train_data, val_data = train_test_split(
                temp_data,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=temp_data[target_column],
            )
        except ValueError as exc:
            logger.warning(
                "Stratified split failed (%s). Falling back to non-stratified split.",
                exc,
            )
            temp_data, test_data = train_test_split(
                data,
                test_size=self.config.test_split,
                random_state=self.config.random_state,
                stratify=None,
            )
            val_size_adjusted = self.config.val_split / (self.config.train_split + self.config.val_split)
            train_data, val_data = train_test_split(
                temp_data,
                test_size=val_size_adjusted,
                random_state=self.config.random_state,
                stratify=None,
            )

        logger.info(
            "Raw split sizes - Train: %s, Val: %s, Test: %s",
            len(train_data),
            len(val_data),
            len(test_data),
        )
        return train_data, val_data, test_data

    def _sample_raw_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Sample raw data based on configured strategy."""
        if self.config.sample_size >= len(raw_data):
            self.sampling_info = {
                'mode': 'none',
                'original_rows': len(raw_data),
                'sampled_rows': len(raw_data),
            }
            return raw_data

        mode = getattr(self.config, 'sampling_mode', 'default')
        strategy = (self.config.sampling_strategy or '').lower()

        if mode == 'smart_balanced' or strategy == 'balanced':
            sampled = self._smart_balanced_sample(raw_data)
            logger.info("Smart-balanced sampled to %s rows", len(sampled))
            return sampled

        sampled = raw_data.sample(n=self.config.sample_size, random_state=self.config.random_state)
        self.sampling_info = {
            'mode': 'random',
            'original_rows': len(raw_data),
            'sampled_rows': len(sampled),
        }
        return sampled

    def _smart_balanced_sample(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Create a balanced subset with custom benign quota and attack cap.

        Policy:
        - Reserve a Benign quota first.
        - Allocate remaining budget to attack classes with a derived cap.
        - Keep all minority classes below cap.
        """
        label_col = raw_data.columns[-1]
        target_n = min(self.config.sample_size, len(raw_data))
        rng_state = self.config.random_state

        labels = raw_data[label_col].astype(str)
        benign_mask = labels.str.lower() == 'benigntraffic'

        benign_df = raw_data[benign_mask]
        attack_df = raw_data[~benign_mask]

        default_benign_quota = int(target_n * 0.20)
        benign_quota = self.config.benign_target_count if self.config.benign_target_count is not None else default_benign_quota
        benign_take = min(max(0, benign_quota), len(benign_df), target_n)

        sampled_parts: List[pd.DataFrame] = []
        if benign_take > 0:
            sampled_parts.append(benign_df.sample(n=benign_take, random_state=rng_state))

        attack_budget = max(0, target_n - benign_take)
        sampled_attack_counts: Dict[str, int] = {}

        if attack_budget > 0 and not attack_df.empty:
            attack_counts = attack_df[label_col].value_counts()
            num_attack_classes = len(attack_counts)
            derived_cap = max(1, attack_budget // max(1, num_attack_classes))
            class_cap = self.config.max_samples_per_attack_class or derived_cap

            for class_name in attack_counts.index:
                class_rows = attack_df[attack_df[label_col] == class_name]
                take_n = min(len(class_rows), class_cap)
                if take_n > 0:
                    sampled_cls = class_rows.sample(n=take_n, random_state=rng_state)
                    sampled_parts.append(sampled_cls)
                    sampled_attack_counts[str(class_name)] = int(take_n)

            sampled_attack_total = sum(sampled_attack_counts.values())
            if sampled_attack_total > attack_budget:
                # In edge cases, enforce budget exactly.
                attack_concat = pd.concat(
                    [part for part in sampled_parts if not ((part[label_col].astype(str).str.lower() == 'benigntraffic').all())],
                    ignore_index=False,
                )
                attack_trimmed = attack_concat.sample(n=attack_budget, random_state=rng_state)
                sampled_parts = [part for part in sampled_parts if (part[label_col].astype(str).str.lower() == 'benigntraffic').all()]
                sampled_parts.append(attack_trimmed)
                sampled_attack_total = attack_budget

            final_attack_cap = class_cap
        else:
            sampled_attack_total = 0
            final_attack_cap = 0
            derived_cap = 0

        if not sampled_parts:
            sampled = raw_data.sample(n=target_n, random_state=rng_state)
        else:
            sampled = pd.concat(sampled_parts, ignore_index=False)
            if len(sampled) > target_n:
                sampled = sampled.sample(n=target_n, random_state=rng_state)

        sampled = sampled.sample(frac=1, random_state=rng_state).reset_index(drop=True)

        sampled_counts = sampled[label_col].astype(str).value_counts().to_dict()
        self.sampling_info = {
            'mode': 'smart_balanced',
            'original_rows': len(raw_data),
            'sampled_rows': len(sampled),
            'target_rows': target_n,
            'benign_target_count': benign_quota,
            'benign_taken': int(sampled_counts.get('BenignTraffic', 0)),
            'attack_budget': attack_budget,
            'derived_attack_cap': int(derived_cap),
            'effective_attack_cap': int(final_attack_cap),
            'sampled_label_counts': sampled_counts,
        }
        return sampled
    
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
    
    def _stratified_sample_for_env(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Sample with per-stage floors to prevent minority stage starvation.

        Takes at least ``min_samples_per_stage`` rows from each Kill Chain stage
        before filling the remaining budget proportionally from the rest of the
        dataset.  This guarantees the RL RealizationEngine has a meaningful pool
        of rows for every stage, including ACCESS which is otherwise ~0.08 %.
        """
        label_col = raw_data.columns[-1]
        mapper = AbstractStateLabelMapper()
        min_floor = getattr(self.config, 'min_samples_per_stage', 2000)
        target_n = self.config.sample_size
        rng_state = self.config.random_state

        stage_ids = raw_data[label_col].astype(str).map(
            lambda lbl: mapper.get_stage_id_safe(lbl, default=0)
        )

        sampled_parts: List[pd.DataFrame] = []
        used_indices: set = set()

        # Floor pass: guarantee min_floor rows per stage
        for sid in range(5):
            group = raw_data[stage_ids == sid]
            n_take = min(min_floor, len(group))
            taken = group.sample(n=n_take, random_state=rng_state)
            sampled_parts.append(taken)
            used_indices.update(taken.index)

        current_n = sum(len(p) for p in sampled_parts)
        remaining = target_n - current_n

        # Fill remaining budget from rows not yet selected
        if remaining > 0:
            residual = raw_data.drop(index=list(used_indices))
            n_extra = min(remaining, len(residual))
            if n_extra > 0:
                sampled_parts.append(
                    residual.sample(n=n_extra, random_state=rng_state)
                )

        result = pd.concat(sampled_parts)
        return result.sample(frac=1, random_state=rng_state).reset_index(drop=True)

    def process_for_adversarial_env(self) -> Dict[str, Any]:
        """
        Process dataset for the Adversarial IoT Environment.
        
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
            
            # Sample with configured strategy
            raw_data = self._sample_raw_data(raw_data)

            target_column = raw_data.columns[-1]
            train_raw, val_raw, test_raw = self._split_by_target(raw_data, target_column)
            self.split_info = {
                'train_samples': int(len(train_raw)),
                'val_samples': int(len(val_raw)),
                'test_samples': int(len(test_raw)),
                'train_ratio': float(self.config.train_split),
                'val_ratio': float(self.config.val_split),
                'test_ratio': float(self.config.test_split),
            }

            # Build shared feature matrix by split
            train_features, train_labels = self._extract_features_and_labels(train_raw)
            val_features, val_labels = self._extract_features_and_labels(val_raw)
            test_features, test_labels = self._extract_features_and_labels(test_raw)

            train_df = pd.DataFrame(train_features, columns=self.feature_columns)
            val_df = pd.DataFrame(val_features, columns=self.feature_columns)
            test_df = pd.DataFrame(test_features, columns=self.feature_columns)

            train_df = self._clean_numerical_data(train_df)
            val_df = self._clean_numerical_data(val_df)
            test_df = self._clean_numerical_data(test_df)

            if self.config.feature_selection:
                train_df = self._apply_feature_selection(train_df)
                self.feature_columns = train_df.columns.tolist()
                val_df = val_df[self.feature_columns]
                test_df = test_df[self.feature_columns]
                logger.info("After feature selection: %d features", len(self.feature_columns))
            else:
                self.feature_columns = train_df.columns.tolist()
                self.feature_selection_info = {
                    'enabled': False,
                    'original_feature_count': len(self.feature_columns),
                    'final_feature_count': len(self.feature_columns),
                    'dropped_zero_variance': [],
                    'dropped_low_variance': [],
                    'dropped_high_correlation': [],
                    'dropped_total': [],
                }

            # Fit scaler only on training set
            self.scaler = StandardScaler()
            train_scaled = self.scaler.fit_transform(train_df)
            val_scaled = self.scaler.transform(val_df)
            test_scaled = self.scaler.transform(test_df)

            normalized_features = np.vstack([train_scaled, val_scaled, test_scaled])
            labels = np.concatenate([train_labels, val_labels, test_labels])
            raw_features = np.vstack([train_df.values, val_df.values, test_df.values])
            
            # Build state indices using AbstractStateLabelMapper
            state_indices = self._build_state_indices(labels)
            
            # Save artifacts
            self._save_adversarial_artifacts(
                normalized_features,
                labels,
                state_indices,
                raw_features,  # Save processed (but not scaled) features
            )
            
            results = {
                'total_samples': len(normalized_features),
                'num_features': normalized_features.shape[1],
                'num_stages': 5,
                'split_info': self.split_info,
                'stage_counts': {
                    int(k): len(v) for k, v in state_indices.items()
                },
                'class_names': list(set(labels)),
                'sampling': self.sampling_info,
                'feature_selection_info': self.feature_selection_info,
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
            'split_info': self.split_info,
            'stage_counts': stage_counts,
            'stage_percentages': stage_percentages,
            'imbalance_ratio': imbalance_ratio,
            'feature_selection_enabled': getattr(self.config, 'feature_selection', False),
            'variance_threshold': getattr(self.config, 'variance_threshold', 0.01),
            'correlation_threshold': getattr(self.config, 'correlation_threshold', 0.95),
            'feature_keep_keywords': getattr(self.config, 'feature_keep_keywords', None),
            'feature_selection_info': self.feature_selection_info,
            'sampling_strategy': getattr(self.config, 'sampling_strategy', None),
            'sampling_mode': getattr(self.config, 'sampling_mode', 'default'),
            'sampling_info': self.sampling_info,
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