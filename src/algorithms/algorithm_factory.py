"""
Enhanced Algorithm Factory with Real Attack Prediction Support

Factory for creating RL algorithms that work with the enhanced IoT environment
using real attack prediction from CICIoT2023 dataset.
"""

from typing import Dict, Any, Type, Optional
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.dqn.policies import MultiInputPolicy as DQNMultiInputPolicy
from stable_baselines3.ppo.policies import MultiInputPolicy as PPOMultiInputPolicy
from stable_baselines3.a2c.policies import MultiInputPolicy as A2CMultiInputPolicy
import logging

from environment import IoTEnv, EnvironmentConfig

logger = logging.getLogger(__name__)


class AlgorithmFactory:
    """
    Factory for creating RL algorithms with enhanced IoT environment support.
    
    Automatically configures algorithms to work with Dict observation spaces
    and real attack prediction from trained LSTM models.
    """
    
    # Algorithm registry mapping names to classes and policies
    ALGORITHMS: Dict[str, Dict[str, Any]] = {
        'dqn': {
            'class': DQN,
            'policy': DQNMultiInputPolicy
        },
        'ppo': {
            'class': PPO,
            'policy': PPOMultiInputPolicy
        },
        'a2c': {
            'class': A2C,
            'policy': A2CMultiInputPolicy
        }
    }
    
    # Default hyperparameters for each algorithm
    DEFAULT_HYPERPARAMS = {
        'dqn': {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'max_grad_norm': 10,
        },
        'ppo': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        },
        'a2c': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'rms_prop_eps': 1e-5,
            'use_rms_prop': True,
            'use_sde': False,
        }
    }
    
    @classmethod
    def create_environment(cls, config: Optional[EnvironmentConfig] = None) -> IoTEnv:
        """
        Create enhanced IoT environment with real attack prediction.
        
        Args:
            config: Optional environment configuration
            
        Returns:
            Configured IoT environment
        """
        if config is None:
            config = EnvironmentConfig(
                max_steps=1000,
                attack_probability=0.3,
                sequence_length=10,
                model_path="models/saved/lstm_real_data.pth",
                data_path="data/processed/ciciot2023"
            )
        
        env = IoTEnv(config)
        logger.info(f"Created enhanced IoT environment with real attack prediction")
        return env
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, env: IoTEnv, 
                        hyperparams: Optional[Dict[str, Any]] = None,
                        verbose: int = 1) -> BaseAlgorithm:
        """
        Create RL algorithm instance configured for enhanced IoT environment.
        
        Args:
            algorithm_name: Name of the algorithm ('dqn', 'ppo', 'a2c')
            env: Enhanced IoT environment instance
            hyperparams: Optional hyperparameter overrides
            verbose: Verbosity level for training output
            
        Returns:
            Configured algorithm instance
            
        Raises:
            ValueError: If algorithm name is not supported
        """
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name not in cls.ALGORITHMS:
            available = list(cls.ALGORITHMS.keys())
            raise ValueError(f"Algorithm '{algorithm_name}' not supported. "
                           f"Available: {available}")
        
        # Get algorithm class and policy
        algorithm_info = cls.ALGORITHMS[algorithm_name]
        algorithm_class = algorithm_info['class']
        policy_class = algorithm_info['policy']
        
        # Merge default hyperparameters with user overrides
        final_hyperparams = cls.DEFAULT_HYPERPARAMS[algorithm_name].copy()
        if hyperparams:
            final_hyperparams.update(hyperparams)
        
        # Create algorithm with appropriate MultiInputPolicy for Dict observations
        algorithm = algorithm_class(
            policy=policy_class,
            env=env,
            verbose=verbose,
            **final_hyperparams
        )
        
        logger.info(f"Created {algorithm_name.upper()} algorithm with enhanced environment support")
        logger.info(f"Using policy: {policy_class.__name__}")
        logger.info(f"Using hyperparameters: {final_hyperparams}")
        
        return algorithm
    
    @classmethod
    def create_algorithm_with_env(cls, algorithm_name: str,
                                 env_config: Optional[EnvironmentConfig] = None,
                                 hyperparams: Optional[Dict[str, Any]] = None,
                                 verbose: int = 1) -> tuple[BaseAlgorithm, IoTEnv]:
        """
        Convenience method to create both environment and algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            env_config: Optional environment configuration
            hyperparams: Optional hyperparameter overrides
            verbose: Verbosity level
            
        Returns:
            Tuple of (algorithm, environment)
        """
        # Create environment
        env = cls.create_environment(env_config)
        
        # Create algorithm
        algorithm = cls.create_algorithm(algorithm_name, env, hyperparams, verbose)
        
        return algorithm, env
    
    @classmethod
    def get_default_hyperparams(cls, algorithm_name: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of default hyperparameters
        """
        algorithm_name = algorithm_name.lower()
        if algorithm_name not in cls.DEFAULT_HYPERPARAMS:
            raise ValueError(f"No default hyperparameters for '{algorithm_name}'")
        
        return cls.DEFAULT_HYPERPARAMS[algorithm_name].copy()
    
    @classmethod
    def list_algorithms(cls) -> list[str]:
        """
        List all available algorithms.
        
        Returns:
            List of available algorithm names
        """
        return list(cls.ALGORITHMS.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary with algorithm class, policy, and default hyperparameters
        """
        algorithm_name = algorithm_name.lower()
        if algorithm_name not in cls.ALGORITHMS:
            raise ValueError(f"Algorithm '{algorithm_name}' not supported")
        
        return {
            'class': cls.ALGORITHMS[algorithm_name]['class'],
            'policy': cls.ALGORITHMS[algorithm_name]['policy'],
            'default_hyperparams': cls.DEFAULT_HYPERPARAMS[algorithm_name].copy()
        }