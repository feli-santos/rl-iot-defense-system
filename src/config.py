import yaml
from typing import Dict, Any
import os

class Config:
    def __init__(self, config_path: str = "config.yml"):
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
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