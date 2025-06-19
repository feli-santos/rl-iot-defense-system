"""
Configuration Management

Enhanced configuration loader with validation and environment variable support.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConfigPaths:
    """Configuration paths structure"""
    data_raw: Path
    data_processed: Path
    models_lstm: Path
    models_rl: Path
    results_logs: Path
    results_plots: Path
    results_reports: Path


class ConfigLoader:
    """
    Configuration loader with validation and environment variable substitution.
    """
    
    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.paths: Optional[ConfigPaths] = None
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            # Validate configuration
            self._validate_config(config)
            
            # Create paths structure
            self.paths = self._create_paths(config)
            
            # Ensure directories exist
            self._ensure_directories()
            
            self.config = config
            logger.info(f"Configuration loaded from {config_path}")
            
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute environment variables in config"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return os.path.expandvars(config)
        else:
            return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_sections = ['dataset', 'lstm', 'environment', 'rl', 'models']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate dataset config
        dataset_config = config['dataset']
        required_dataset_keys = ['name', 'processed_path', 'sequence_length']
        for key in required_dataset_keys:
            if key not in dataset_config:
                raise ValueError(f"Missing required dataset config key: {key}")
        
        # Validate model paths
        models_config = config['models']
        if 'lstm' not in models_config or 'save_path' not in models_config['lstm']:
            raise ValueError("Missing LSTM model save path in configuration")
        
        logger.info("Configuration validation passed")
    
    def _create_paths(self, config: Dict[str, Any]) -> ConfigPaths:
        """Create paths structure from configuration"""
        return ConfigPaths(
            data_raw=Path(config['dataset'].get('raw_path', 'data/raw')),
            data_processed=Path(config['dataset']['processed_path']),
            models_lstm=Path(config['models']['lstm']['save_path']).parent,
            models_rl=Path(config['models']['rl']['save_dir']),
            results_logs=Path(config.get('logging', {}).get('log_dir', 'results/logs')),
            results_plots=Path(config.get('results', {}).get('plots_dir', 'results/plots')),
            results_reports=Path(config.get('results', {}).get('reports_dir', 'results/reports'))
        )
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        if self.paths is None:
            return
        
        directories = [
            self.paths.data_processed,
            self.paths.models_lstm,
            self.paths.models_rl,
            self.paths.results_logs,
            self.paths.results_plots,
            self.paths.results_reports
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("All required directories created/verified")
    
    def get_lstm_config(self) -> Dict[str, Any]:
        """Get LSTM-specific configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        return self.config['lstm']
    
    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL-specific configuration"""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        return self.config['rl']
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get algorithm-specific hyperparameters"""
        rl_config = self.get_rl_config()
        
        if 'algorithms' not in rl_config:
            raise ValueError("No algorithm configurations found")
        
        if algorithm not in rl_config['algorithms']:
            available = list(rl_config['algorithms'].keys())
            raise ValueError(f"Algorithm '{algorithm}' not configured. Available: {available}")
        
        return rl_config['algorithms'][algorithm]
    
    def get_paths(self) -> ConfigPaths:
        """Get paths structure"""
        if self.paths is None:
            raise ValueError("Configuration not loaded")
        return self.paths