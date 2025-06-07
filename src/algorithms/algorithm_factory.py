"""
Algorithm Factory

Creates algorithm instances based on configuration.
"""

from algorithms.dqn_algorithm import DQNAlgorithm
from algorithms.ppo_algorithm import PPOAlgorithm
from algorithms.a2c_algorithm import A2CAlgorithm


class AlgorithmFactory:
    """Factory for creating RL algorithms"""
    
    _algorithms = {
        'DQN': DQNAlgorithm,
        'PPO': PPOAlgorithm,
        'A2C': A2CAlgorithm,  
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, config):
        """Create an algorithm instance"""
        algorithm_name = algorithm_name.upper()
        
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                           f"Available algorithms: {list(cls._algorithms.keys())}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(config)
        
    @classmethod
    def get_available_algorithms(cls):
        """Get list of available algorithms"""
        return list(cls._algorithms.keys())