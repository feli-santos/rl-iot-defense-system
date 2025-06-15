"""
RL Defense Agent Training

Handles training of RL defense agents using enhanced IoT environment
with real attack prediction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

from algorithms.algorithm_factory import AlgorithmFactory
from environment import EnvironmentConfig
from training.training_manager import TrainingManager

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    RL trainer for defense agents using enhanced IoT environment.
    """
    
    def __init__(self, config: Dict[str, Any], data_path: Path, 
                 lstm_model_path: Optional[Path] = None):
        self.config = config
        self.data_path = data_path
        self.lstm_model_path = lstm_model_path
        
        logger.info("RL trainer initialized")
    
    def train(self) -> Dict[str, Any]:
        """
        Train RL defense agent.
        
        Returns:
            Training results dictionary
        """
        rl_config = self.config['rl']
        algorithm_name = rl_config['algorithm']
        
        logger.info(f"Starting RL training with {algorithm_name.upper()}")
        
        # Create environment configuration
        env_config = EnvironmentConfig(
            max_steps=self.config['environment']['max_steps'],
            attack_probability=self.config['environment']['attack_probability'],
            state_history_length=self.config['environment']['state_history_length'],
            action_history_length=self.config['environment']['action_history_length'],
            reward_scale=self.config['environment']['reward_scale'],
            model_path=str(self.lstm_model_path) if self.lstm_model_path else None,
            data_path=str(self.data_path)
        )
        
        # Get algorithm-specific hyperparameters
        hyperparams = rl_config['algorithms'][algorithm_name]
        
        # Create algorithm and environment
        algorithm, env = AlgorithmFactory.create_algorithm_with_env(
            algorithm_name=algorithm_name,
            env_config=env_config,
            hyperparams=hyperparams,
            verbose=1
        )
        
        # Create training manager
        training_manager = TrainingManager(
            algorithm=algorithm,
            experiment_name=f"{algorithm_name}_enhanced_iot",
            save_path=Path(self.config['models']['rl']['save_dir'])
        )
        
        # Train agent
        training_results = training_manager.train(
            total_timesteps=rl_config['training']['total_timesteps'],
            eval_freq=rl_config['training']['eval_freq'],
            n_eval_episodes=rl_config['training']['n_eval_episodes']
        )
        
        # Clean up
        env.close()
        
        results = {
            'algorithm': algorithm_name,
            'model_path': training_results['model_path'],
            'final_reward': training_results['final_mean_reward'],
            'best_reward': training_results['best_mean_reward'],
            'training_time': training_results['training_time']
        }
        
        logger.info(f"RL training completed. Final reward: {results['final_reward']:.2f}")
        
        return results