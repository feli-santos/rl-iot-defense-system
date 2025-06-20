"""
RL Defense Agent Training

Handles training of RL defense agents using IoT environment
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
    RL trainer for defense agents using IoT environment.
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
            algorithm_name = rl_config['algorithm'].lower()
            
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
            
            hyperparams = rl_config['algorithms'][algorithm_name].copy()
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
            obs, _ = env.reset()
            logger.info(f"Initial observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not dict'}")
            
            # Create training manager
            print("📊 Setting up training manager...")
            training_manager = TrainingManager(
                algorithm=algorithm,
                experiment_name=algorithm_name,
                save_path=Path(self.config['models']['rl']['save_dir'])
            )
            
            # Prepare consolidated parameters for single logging
            training_params = {
                'algorithm': algorithm_name,
                'total_timesteps': rl_config['training']['total_timesteps'],
                'eval_freq': rl_config['training']['eval_freq'],
                'n_eval_episodes': rl_config['training']['n_eval_episodes'],
                'lstm_model_path': str(self.lstm_model_path) if self.lstm_model_path else 'None',
                'data_path': str(self.data_path),
                'environment_max_steps': self.config['environment']['max_steps'],
                'environment_attack_probability': self.config['environment']['attack_probability'],
                **{f"hp_{k}": v for k, v in hyperparams.items()}  # Prefix hyperparams to avoid conflicts
            }
            
            # Start MLflow run with all parameters
            print("🔬 Starting MLflow tracking...")
            run_name = f"{algorithm_name}_iot_defense_{int(training_params['total_timesteps']/1000)}k"
            training_manager.start_run(run_name=run_name)
            
            try:
                # Log all parameters at once to avoid conflicts
                training_manager.log_params(training_params)
                
                # Train agent using correct method name
                print("🚀 Starting training...")
                training_results = training_manager.train_algorithm(
                    algorithm=algorithm,
                    total_timesteps=rl_config['training']['total_timesteps'],
                    eval_freq=rl_config['training']['eval_freq'],
                    n_eval_episodes=rl_config['training']['n_eval_episodes'],
                    save_freq=rl_config['training'].get('save_freq', 12500)
                )
                
                # Log final results with unique metric names
                final_metrics = {
                    'final_mean_reward': training_results.get('final_evaluation', {}).get('mean_reward', 0.0),
                    'final_std_reward': training_results.get('final_evaluation', {}).get('std_reward', 0.0),
                    'best_mean_reward': training_results.get('best_mean_reward', 0.0),
                    'training_duration_seconds': training_results.get('training_time', 0.0),
                    'total_episodes': training_results.get('total_episodes', 0),
                    'success': 1 if training_results.get('success', False) else 0
                }
                
                training_manager.log_metrics(final_metrics)
                
                # Log final results
                if training_results.get('success', False):
                    print("✅ RL training completed successfully!")
                    print(f"   • Final mean reward: {final_metrics['final_mean_reward']:.3f} ± {final_metrics['final_std_reward']:.3f}")
                    print(f"   • Best model reward: {final_metrics['best_mean_reward']:.3f}")
                    print(f"   • Training time: {final_metrics['training_duration_seconds']:.1f}s")
                    print(f"   • Total episodes: {final_metrics['total_episodes']}")
                    print(f"   • Model saved to: {training_results.get('final_model_path', 'N/A')}")
                else:
                    print("❌ RL training failed!")
                    print(f"   • Error: {training_results.get('error', 'Unknown error')}")
                
                # Prepare return results
                results = {
                    'algorithm': algorithm_name,
                    'success': training_results.get('success', False),
                    'final_reward': final_metrics['final_mean_reward'],
                    'final_std': final_metrics['final_std_reward'],
                    'best_reward': final_metrics['best_mean_reward'],
                    'training_time': final_metrics['training_duration_seconds'],
                    'total_episodes': final_metrics['total_episodes'],
                    'model_path': training_results.get('final_model_path', ''),
                    'best_model_path': training_results.get('best_model_path', ''),
                    'mlflow_run_id': training_manager.current_run.info.run_id if training_manager.current_run else None
                }
                
                logger.info(f"RL training completed. Final reward: {results['final_reward']:.3f}")
                
                return results
                
            finally:
                # Always end MLflow run
                training_manager.end_run()
                
                # Clean up environment
                try:
                    env.close()
                    logger.info("Environment closed successfully")
                except Exception as cleanup_error:
                    logger.warning(f"Environment cleanup failed: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"RL training failed with error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"💥 Detailed error information:")
            print(f"   • Error type: {type(e).__name__}")
            print(f"   • Error message: {str(e)}")
            print(f"   • Traceback:")
            traceback.print_exc()
            
            # Return error results
            return {
                'algorithm': rl_config.get('algorithm', 'unknown').lower(),
                'success': False,
                'error': str(e),
                'final_reward': 0.0,
                'final_std': 0.0,
                'best_reward': 0.0,
                'training_time': 0.0,
                'total_episodes': 0,
                'model_path': '',
                'best_model_path': '',
                'mlflow_run_id': None
            }