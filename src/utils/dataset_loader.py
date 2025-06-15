"""
CICIoT2023 Data Loader

Provides efficient data loading utilities for the processed CICIoT2023 dataset
to support LSTM training and RL environment integration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Generator
import pickle
import logging
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

@dataclass
class LoaderConfig:
    """Configuration for data loading"""
    data_path: Path
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 2
    pin_memory: bool = True


class CICIoTDataset(Dataset):
    """PyTorch Dataset for CICIoT2023 data"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class CICIoTDataLoader:
    """
    Main data loader for processed CICIoT2023 dataset
    """
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.artifacts: Dict = {}
        self.data_splits: Dict[str, pd.DataFrame] = {}
        self.lstm_data: Dict[str, np.ndarray] = {}
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load preprocessing artifacts"""
        artifacts_path = self.config.data_path / "preprocessing_artifacts.pkl"
        
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Preprocessing artifacts not found: {artifacts_path}")
        
        with open(artifacts_path, 'rb') as f:
            self.artifacts = pickle.load(f)
        
        logger.info("Loaded preprocessing artifacts")
    
    def load_data_splits(self) -> Dict[str, pd.DataFrame]:
        """Load train/validation/test data splits"""
        if self.data_splits:
            return self.data_splits
        
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            file_path = self.config.data_path / f"{split}_data.parquet"
            if file_path.exists():
                self.data_splits[split] = pd.read_parquet(file_path)
                logger.info(f"Loaded {split} data: {len(self.data_splits[split])} records")
        
        return self.data_splits
    
    def load_lstm_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load LSTM sequences and labels"""
        sequences_path = self.config.data_path / "lstm_sequences.npy"
        labels_path = self.config.data_path / "lstm_labels.npy"
        
        if not (sequences_path.exists() and labels_path.exists()):
            raise FileNotFoundError("LSTM data files not found")
        
        sequences = np.load(sequences_path)
        labels = np.load(labels_path)
        
        logger.info(f"Loaded LSTM data: {sequences.shape} sequences, {labels.shape} labels")
        return sequences, labels
    
    def load_rl_states(self) -> np.ndarray:
        """Load RL state representations"""
        states_path = self.config.data_path / "rl_states.npy"
        
        if not states_path.exists():
            raise FileNotFoundError("RL states file not found")
        
        states = np.load(states_path)
        logger.info(f"Loaded RL states: {states.shape}")
        return states
    
    def get_lstm_dataloaders(self) -> Dict[str, DataLoader]:
        """Get PyTorch DataLoaders for LSTM training"""
        sequences, labels = self.load_lstm_data()
        
        # Create datasets
        dataset = CICIoTDataset(sequences, labels)
        
        # Split dataset (assuming we want to use the same split ratios)
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'validation': DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        }
        
        return dataloaders
    
    def get_feature_info(self) -> Dict:
        """Get feature information for model configuration"""
        return {
            'n_features': self.artifacts['feature_stats']['n_features'],
            'feature_names': self.artifacts['feature_stats']['feature_names'],
            'feature_groups': self.artifacts['feature_groups'],
            'n_classes': len(self.artifacts['label_encoder'].classes_),
            'class_names': list(self.artifacts['label_encoder'].classes_),
            'attack_categories': self.artifacts['attack_categories'],
            'scaler': self.artifacts['scaler'],
            'label_encoder': self.artifacts['label_encoder']
        }
    
    def get_rl_state_info(self) -> Dict:
        """Get RL state space information"""
        rl_states = self.load_rl_states()
        
        return {
            'state_dim': rl_states.shape[1],
            'n_samples': rl_states.shape[0],
            'state_bounds': {
                'min': rl_states.min(axis=0),
                'max': rl_states.max(axis=0),
                'mean': rl_states.mean(axis=0),
                'std': rl_states.std(axis=0)
            }
        }