"""
Algorithm Factory

Factory pattern implementation for creating RL algorithms.
"""

from typing import Dict, Any, Type
from .base_algorithm import BaseAlgorithm
from .dqn_algorithm import DQNAlgorithm
from .ppo_algorithm import PPOAlgorithm
from .sac_algorithm import SACAlgorithm


class AlgorithmFactory:
    """Factory for creating RL algorithms"""
    
    _algorithms: Dict[str, Type[BaseAlgorithm]] = {
        "DQN": DQNAlgorithm,
        "PPO": PPOAlgorithm,
        "SAC": SACAlgorithm
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, config: Any) -> BaseAlgorithm:
        """
        Create an algorithm instance
        
        Args:
            algorithm_name: Name of the algorithm ("DQN", "PPO", "SAC")
            config: Configuration object
            
        Returns:
            BaseAlgorithm: Algorithm instance
            
        Raises:
            ValueError: If algorithm name is not supported
        """
        algorithm_name = algorithm_name.upper()
        
        if algorithm_name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not supported. Available: {available}")
            
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(config)
    
    @classmethod
    def get_available_algorithms(cls) -> list:
        """Get list of available algorithm names"""
        return list(cls._algorithms.keys())
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        """
        Register a new algorithm
        
        Args:
            name: Algorithm name
            algorithm_class: Algorithm class
        """
        cls._algorithms[name.upper()] = algorithm_class