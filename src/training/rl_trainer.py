"""
RL Defense Agent Training

Handles training of RL defense agents using enhanced IoT environment
with real attack prediction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import traceback

from algorithms.algorithm_factory import AlgorithmFactory
from environment.environment import EnvironmentConfig
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
        try:
            rl_config = self.config['rl']
            algorithm_name = rl_config['algorithm']
            
            print(f"🎯 Training Configuration:")
            print(f"   • Algorithm: {algorithm_name.upper()}")
            print(f"   • LSTM Model: {self.lstm_model_path}")
            print(f"   • Data Path: {self.data_path}")
            print(f"   • Timesteps: {rl_config['training']['total_timesteps']:,}")
            print()
            
            logger.info(f"Starting RL training with {algorithm_name.upper()}")
            
            # Validate LSTM model path
            if self.lstm_model_path and not self.lstm_model_path.exists():
                raise FileNotFoundError(f"LSTM model not found: {self.lstm_model_path}")
            
            # Validate data path
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data path not found: {self.data_path}")
            
            # Create environment configuration
            print("🔧 Creating environment configuration...")
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
            print("⚙️  Loading algorithm hyperparameters...")
            if algorithm_name not in rl_config['algorithms']:
                raise KeyError(f"Algorithm '{algorithm_name}' not found in config")
            
            hyperparams = rl_config['algorithms'][algorithm_name]
            logger.info(f"Hyperparameters: {hyperparams}")
            
            # Create algorithm and environment
            print("🏗️  Creating algorithm and environment...")
            algorithm, env = AlgorithmFactory.create_algorithm_with_env(
                algorithm_name=algorithm_name,
                env_config=env_config,
                hyperparams=hyperparams,
                verbose=1
            )
            
            print("✅ Algorithm and environment created successfully")
            
            # Test environment
            print("🧪 Testing environment...")
            obs = env.reset()
            logger.info(f"Initial observation shape: {obs}")
            
            # Create training manager
            print("📊 Setting up training manager...")
            training_manager = TrainingManager(
                algorithm=algorithm,
                experiment_name=f"{algorithm_name}_enhanced_iot",
                save_path=Path(self.config['models']['rl']['save_dir'])
            )
            
            # Train agent
            print("🚀 Starting training...")
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
            
        except Exception as e:
            logger.error(f"RL training failed with error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"💥 Detailed error information:")
            print(f"   • Error type: {type(e).__name__}")
            print(f"   • Error message: {str(e)}")
            print(f"   • Traceback:")
            traceback.print_exc()
            raise