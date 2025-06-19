"""
CICIoT2023 Dataset Processor

Processes raw CICIoT2023 dataset for LSTM training and RL environment integration.
Enhanced with configurable train/validation/test splits and comprehensive preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json

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
    # Add EDA recommendations
    feature_selection: bool = False
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
        
        # Feature selection - remove zero/low variance features
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

    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove zero and low variance features."""
        from sklearn.feature_selection import VarianceThreshold
        
        # Remove zero variance features
        selector = VarianceThreshold(threshold=0)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Remove low variance features (bottom 5%)
        if len(selected_features) > 10:  # Keep at least 10 features
            variances = pd.DataFrame(X_selected, columns=selected_features).var()
            threshold = variances.quantile(0.05)
            high_var_features = variances[variances > threshold].index.tolist()
            X_final = X[high_var_features]
        else:
            X_final = pd.DataFrame(X_selected, columns=selected_features)
        
        logger.info(f"Feature selection: {X.shape[1]} -> {X_final.shape[1]} features")
        return X_final

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