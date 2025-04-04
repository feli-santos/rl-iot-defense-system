import yaml
from typing import Dict, Any
import os
from pathlib import Path

class Config:
    def __init__(self, config_path: str = None):
        # Get the directory where this config.py file is located
        current_dir = Path(__file__).parent
        
        # Default to config.yml in the same directory as config.py
        if config_path is None:
            config_path = str(current_dir / "../config.yml")
        
        # Try loading the config file
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config file not found at: {config_path}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Config.py location: {current_dir}"
            )
        
        # Create attributes from the YAML structure
        for section, values in self._config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    setattr(self, f"{section.upper()}_{key.upper()}", value)
            else:
                setattr(self, section.upper(), values)
    
    def __getattr__(self, name: str) -> Any:
        """Fallback to prevent AttributeError"""
        return None

# Load the configuration
config = Config()

# Example usage:
if __name__ == "__main__":
    print(f"Learning rate: {config.DQN_LEARNING_RATE}")
    print(f"Hidden layers: {config.NETWORK_HIDDEN_LAYERS}")
    print(f"Log directory: {config.TRAINING_LOG_DIR}")