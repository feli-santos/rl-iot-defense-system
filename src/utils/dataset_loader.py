"""
CICIoT2023 Data Loader

Provides efficient data loading utilities for the processed CICIoT2023 dataset
to support LSTM training and RL environment integration.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for CICIoT data loader."""
    data_path: Path
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    sequence_length: int = 5


class CICIoTDataset(Dataset):
    """PyTorch Dataset for CICIoT2023 data."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences [batch_size, seq_len, n_features]
            labels: Target labels [batch_size]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        
        logger.info(f"Dataset initialized: {len(self.sequences)} samples, "
                   f"sequence shape: {self.sequences.shape}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, label)
        """
        return self.sequences[idx], self.labels[idx]


class CICIoTDataLoader:
    """
    Main data loader for processed CICIoT2023 dataset.
    Compatible with new dataset processor format.
    """
    
    def __init__(self, config: LoaderConfig) -> None:
        """
        Initialize data loader.
        
        Args:
            config: Loader configuration
        """
        self.config = config
        self.data_path = Path(config.data_path)
        
        # Preprocessing artifacts
        self.scaler = None
        self.label_encoder = None
        self.metadata: Dict[str, Any] = {}
        
        # Load artifacts
        self._load_artifacts()
        
        logger.info(f"CICIoT data loader initialized for path: {self.data_path}")
    
    def _load_artifacts(self) -> None:
        """Load preprocessing artifacts from new processor format."""
        try:
            # Load scaler
            scaler_path = self.data_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
            
            # Load label encoder
            encoder_path = self.data_path / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Label encoder loaded successfully")
            else:
                logger.warning(f"Label encoder not found at {encoder_path}")
            
            # Load metadata
            metadata_path = self.data_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info("Metadata loaded successfully")
            else:
                logger.warning(f"Metadata not found at {metadata_path}")
                
        except Exception as e:
            logger.error(f"Failed to load preprocessing artifacts: {e}")
            raise FileNotFoundError(f"Preprocessing artifacts not found or corrupted: {e}")
    
    def load_lstm_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load LSTM training data.
        
        Returns:
            Tuple of (sequences_dict, targets_dict) for train/val/test splits
        """
        logger.info("Loading LSTM sequences and targets...")
        
        try:
            # Load sequences
            train_sequences = np.load(self.data_path / "train_sequences.npy")
            val_sequences = np.load(self.data_path / "val_sequences.npy")
            test_sequences = np.load(self.data_path / "test_sequences.npy")
            
            # Load targets
            train_targets = np.load(self.data_path / "train_targets.npy")
            val_targets = np.load(self.data_path / "val_targets.npy")
            test_targets = np.load(self.data_path / "test_targets.npy")
            
            sequences = {
                'train': train_sequences,
                'val': val_sequences,
                'test': test_sequences
            }
            
            targets = {
                'train': train_targets,
                'val': val_targets,
                'test': test_targets
            }
            
            logger.info(f"LSTM data loaded - Train: {len(train_sequences)}, "
                       f"Val: {len(val_sequences)}, Test: {len(test_sequences)}")
            
            return sequences, targets
            
        except FileNotFoundError as e:
            logger.error(f"LSTM data files not found: {e}")
            raise FileNotFoundError(f"LSTM sequences not found in {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to load LSTM data: {e}")
            raise
    
    def get_lstm_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for LSTM training.
        
        Returns:
            Dictionary of DataLoaders for train/val/test splits
        """
        logger.info("Creating LSTM DataLoaders...")
        
        # Load data
        sequences, targets = self.load_lstm_data()
        
        # Create datasets
        datasets = {}
        for split in ['train', 'val', 'test']:
            datasets[split] = CICIoTDataset(sequences[split], targets[split])
        
        # Create data loaders
        dataloaders = {}
        
        # Training loader with shuffling
        dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        # Validation and test loaders without shuffling
        for split in ['val', 'test']:
            dataloaders[split] = DataLoader(
                datasets[split],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False
            )
        
        logger.info("LSTM DataLoaders created successfully")
        return dataloaders
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get feature information from metadata.
        
        Returns:
            Feature information dictionary
        """
        if not self.metadata:
            raise ValueError("Metadata not loaded. Cannot provide feature info.")
        
        return {
            'n_features': len(self.metadata.get('feature_columns', [])),
            'feature_columns': self.metadata.get('feature_columns', []),
            'n_classes': len(self.metadata.get('class_names', [])),
            'class_names': self.metadata.get('class_names', []),
            'sequence_length': self.metadata.get('sequence_length', self.config.sequence_length)
        }
    
    def get_class_names(self) -> List[str]:
        """Get class names from label encoder or metadata."""
        if self.label_encoder is not None:
            return self.label_encoder.classes_.tolist()
        elif 'class_names' in self.metadata:
            return self.metadata['class_names']
        else:
            raise ValueError("Class names not available")
    
    def load_rl_states(self) -> np.ndarray:
        """
        Load states for RL environment (uses test sequences as environment states).
        
        Returns:
            Array of states for RL environment
        """
        logger.info("Loading RL environment states...")
        
        try:
            # Use test sequences as RL states
            test_sequences = np.load(self.data_path / "test_sequences.npy")
            
            # Flatten sequences for RL state representation
            # Shape: [n_samples, seq_len * n_features]
            rl_states = test_sequences.reshape(test_sequences.shape[0], -1)
            
            logger.info(f"RL states loaded: {rl_states.shape}")
            return rl_states
            
        except FileNotFoundError as e:
            logger.error(f"RL state data not found: {e}")
            raise FileNotFoundError(f"Test sequences not found for RL states: {e}")
        except Exception as e:
            logger.error(f"Failed to load RL states: {e}")
            raise