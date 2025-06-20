"""
LSTM Attack Predictor Training

Handles training of LSTM attack predictor using processed CICIoT2023 dataset.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import torch
import mlflow
import mlflow.pytorch

from predictor.attack import LSTMAttackPredictor, LSTMConfig, DataTrainer
from utils.dataset_loader import CICIoTDataLoader, LoaderConfig

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    LSTM trainer for attack prediction using CICIoT2023 dataset.
    Compatible with new dataset processor format.
    """
    
    def __init__(self, config: Dict[str, Any], data_path: Path):
        """
        Initialize LSTM trainer.
        
        Args:
            config: Main configuration dictionary
            data_path: Path to processed dataset
        """
        self.config = config
        self.data_path = data_path
        
        # Create data loader to get feature info
        self.data_loader = self._create_data_loader()
        
        # Get actual feature info from processed data
        feature_info = self.data_loader.get_feature_info()
        
        # Update config with actual feature count
        self.config['lstm']['model']['input_size'] = feature_info['n_features']
        
        # Create LSTM config
        self.lstm_config = self._create_lstm_config()
        
        logger.info(f"LSTM trainer initialized with {feature_info['n_features']} features, "
                   f"{feature_info['n_classes']} classes")
    
    def _create_data_loader(self) -> CICIoTDataLoader:
        """Create data loader configuration."""
        loader_config = LoaderConfig(
            data_path=self.data_path,
            batch_size=self.config['lstm']['training']['batch_size'],
            num_workers=self.config['lstm']['data']['num_workers'],
            pin_memory=self.config['lstm']['data']['pin_memory'],
            sequence_length=self.config['lstm']['data']['sequence_length']
        )
        
        return CICIoTDataLoader(loader_config)
    
    def _create_lstm_config(self) -> LSTMConfig:
        """Create LSTM configuration from main config."""
        lstm_cfg = self.config['lstm']
        
        # Get number of classes from data loader
        feature_info = self.data_loader.get_feature_info()
        
        return LSTMConfig(
            input_size=lstm_cfg['model']['input_size'],
            hidden_size=lstm_cfg['model']['hidden_size'],
            num_layers=lstm_cfg['model']['num_layers'],
            num_classes=feature_info['n_classes'],
            dropout=lstm_cfg['model']['dropout'],
            batch_size=lstm_cfg['training']['batch_size'],
            learning_rate=lstm_cfg['training']['learning_rate'],
            num_epochs=lstm_cfg['training']['epochs'],
            sequence_length=lstm_cfg['data']['sequence_length'],
            model_save_path=self.config['models']['lstm']['save_path']
        )
    
    def train(self) -> Dict[str, Any]:
        """
        Train LSTM attack predictor.
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting LSTM attack predictor training...")
        
        # Create trainer with data loader
        trainer = DataTrainer(self.lstm_config, self.data_loader)
        
        # Train model
        history = trainer.train()
        
        # Evaluate model
        eval_results = trainer.evaluate_detailed()
        
        # Prepare results
        results = {
            'model_path': self.lstm_config.model_save_path,
            'test_accuracy': eval_results['accuracy'],
            'macro_f1': eval_results['macro_avg_f1'],
            'weighted_f1': eval_results['weighted_avg_f1'],
            'training_history': history,
            'evaluation_results': eval_results
        }
        
        logger.info(f"LSTM training completed. Test accuracy: {eval_results['accuracy']:.4f}")
        
        return results