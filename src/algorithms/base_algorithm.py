"""
Base Algorithm Interface

Defines the common interface that all RL algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm


class BaseAlgorithm(ABC):
    """Abstract base class for all RL algorithms"""
    
    def __init__(self, config: Any, algorithm_name: str):
        self.config = config
        self.algorithm_name = algorithm_name
        self.model: Optional[SB3BaseAlgorithm] = None
        self.env: Optional[VecEnv] = None
        
    @abstractmethod
    def create_model(self, env: VecEnv, training_manager: Any) -> SB3BaseAlgorithm:
        """Create and configure the RL model"""
        pass
        
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get algorithm-specific hyperparameters"""
        pass
        
    @abstractmethod
    def train(self, model: SB3BaseAlgorithm, training_manager: Any) -> SB3BaseAlgorithm:
        """Train the model"""
        pass
        
    @abstractmethod
    def get_total_timesteps(self) -> int:
        """Get total training timesteps for this algorithm"""
        pass
        
    def save_model(self, model: SB3BaseAlgorithm, path: str) -> None:
        """Save the trained model"""
        model.save(path)
        
    def load_model(self, path: str, env: VecEnv) -> SB3BaseAlgorithm:
        """Load a trained model"""
        # This will be implemented by subclasses with specific model types
        raise NotImplementedError("Subclasses must implement load_model")
        
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about the algorithm"""
        return {
            "name": self.algorithm_name,
            "hyperparameters": self.get_hyperparameters(),
            "total_timesteps": self.get_total_timesteps()
        }