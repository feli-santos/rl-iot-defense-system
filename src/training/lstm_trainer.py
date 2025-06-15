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

from models.lstm_attack_predictor import RealDataLSTMPredictor, LSTMConfig, RealDataTrainer
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    LSTM trainer for attack prediction using CICIoT2023 dataset.
    """
    
    def __init__(self, config: Dict[str, Any], data_path: Path):
        self.config = config
        self.data_path = data_path
        self.lstm_config = self._create_lstm_config()
        
        logger.info("LSTM trainer initialized")
    
    def _create_lstm_config(self) -> LSTMConfig:
        """Create LSTM configuration from main config"""
        lstm_cfg = self.config['lstm']
        
        return LSTMConfig(
            input_size=lstm_cfg['model']['input_size'],
            hidden_size=lstm_cfg['model']['hidden_size'],
            num_layers=lstm_cfg['model']['num_layers'],
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
        
        # Create trainer
        trainer = RealDataTrainer(self.lstm_config, self.data_path)
        
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