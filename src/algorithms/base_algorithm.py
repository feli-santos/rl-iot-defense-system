"""
Base Algorithm Interface

Defines the common interface for all RL algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os


class BaseAlgorithm(ABC):
    """Base class for all RL algorithms"""
    
    def __init__(self, config: Any):
        self.config = config
        self.algorithm_name = self.__class__.__name__.replace('Algorithm', '').upper()
        
    @abstractmethod
    def create_model(self, env, training_manager):
        """Create the RL model"""
        pass
        
    @abstractmethod
    def train(self, model, training_manager):
        """Train the model"""
        pass
        
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for this algorithm"""
        pass
        
    @abstractmethod
    def save_model(self, model, path: str):
        """Save the trained model"""
        pass
        
    @abstractmethod
    def load_model(self, path: str):
        """Load a trained model"""
        pass