"""
Training Manager

Orchestrates training experiments with MLflow tracking and artifact management.
Provides unified interface for LSTM and RL training workflows.
"""

import time
import uuid
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import mlflow
import mlflow.pytorch
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

logger = logging.getLogger(__name__)


class MLflowLoggingCallback(BaseCallback):
    """Custom callback for logging training metrics to MLflow"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def _on_step(self) -> bool:
        """Called at each training step"""
        # Log episode-level metrics when episode ends
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_info = info['episode']
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                
                # Log to MLflow every log_freq episodes
                if len(self.episode_rewards) % self.log_freq == 0:
                    try:
                        mean_reward = np.mean(self.episode_rewards[-self.log_freq:])
                        mean_length = np.mean(self.episode_lengths[-self.log_freq:])
                        
                        mlflow.log_metrics({
                            'episode_reward_mean': mean_reward,
                            'episode_length_mean': mean_length,
                            'total_episodes': len(self.episode_rewards)
                        }, step=self.num_timesteps)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics: {e}")
        
        return True


class TrainingManager:
    """
    Manages training experiments with comprehensive tracking and artifact storage.
    
    Provides unified interface for both LSTM and RL training with MLflow integration,
    automatic checkpointing, and detailed performance monitoring.
    """
    
    def __init__(self, algorithm: BaseAlgorithm, experiment_name: str, 
                 save_path: Path) -> None:
        """
        Initialize training manager.
        
        Args:
            algorithm: RL algorithm instance to train
            experiment_name: Name for MLflow experiment
            save_path: Path to save trained models
        """
        self.algorithm = algorithm
        self.experiment_name = experiment_name
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Create a unique run ID
        self.run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Setup artifact directories
        self.base_artifact_path = Path("./artifacts")
        self.run_artifact_path = self.base_artifact_path / self.run_id
        self.models_path = self.run_artifact_path / "models"
        self.logs_path = self.run_artifact_path / "logs"
        self.plots_path = self.run_artifact_path / "plots"
        
        # Add save_path attribute for compatibility
        self.save_path = self.run_artifact_path
        
        # Create directories
        self._create_directories()
        
        # MLflow tracking
        self.mlflow_run = None
        self.best_model_metric = -np.inf
        self.best_model_path: Optional[Path] = None
        
        logger.info(f"Initialized TrainingManager for {experiment_name}")
    
    def _create_directories(self) -> None:
        """Create all necessary directories"""
        for path in [self.run_artifact_path, self.models_path, self.logs_path, self.plots_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> None:
        """
        Start MLflow run
        
        Args:
            run_name: Optional custom run name
            nested: Whether this is a nested run
        """
        try:
            # End any existing run first
            if mlflow.active_run() is not None:
                mlflow.end_run()
            
            mlflow.set_experiment(self.experiment_name)
            self.mlflow_run = mlflow.start_run(
                run_name=run_name or self.run_id,
                nested=nested
            )
            
            # Log basic configuration
            if self.config:
                if hasattr(self.config, '__dict__'):
                    mlflow.log_params(self.config.__dict__)
                elif isinstance(self.config, dict):
                    mlflow.log_params(self.config)
            
            logger.info(f"Started MLflow run: {self.mlflow_run.info.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        try:
            if self.mlflow_run:
                mlflow.end_run()
                logger.info("Ended MLflow run")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        try:
            # Filter out non-numeric values and convert to proper types
            numeric_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    numeric_metrics[k] = float(v)
                elif isinstance(v, str) and k == 'algorithm':
                    # Skip algorithm as it should be logged as param, not metric
                    continue
            
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow"""
        try:
            # Convert complex objects to strings
            string_params = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in params.items()
            }
            mlflow.log_params(string_params)
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def train_algorithm(self, 
                       algorithm: BaseAlgorithm,
                       total_timesteps: int,
                       eval_freq: int = 10000,
                       n_eval_episodes: int = 10,
                       save_freq: int = 50000) -> Dict[str, Any]:
        """
        Train RL algorithm with evaluation and checkpointing
        
        Args:
            algorithm: RL algorithm instance
            total_timesteps: Total training timesteps
            eval_freq: Frequency of evaluation during training
            n_eval_episodes: Number of episodes for evaluation
            save_freq: Frequency of model saving
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {total_timesteps:,} timesteps")
        
        # Log training parameters
        self.log_params({
            'algorithm': algorithm.__class__.__name__,
            'total_timesteps': total_timesteps,
            'eval_freq': eval_freq,
            'n_eval_episodes': n_eval_episodes,
            'save_freq': save_freq
        })
        
        # Setup callbacks
        callbacks = []
        
        # Evaluation callback
        if eval_freq > 0:
            eval_callback = EvalCallback(
                eval_env=algorithm.get_env(),
                best_model_save_path=str(self.models_path / "best_model"),
                log_path=str(self.logs_path / "evaluations"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # MLflow logging callback
        mlflow_callback = MLflowLoggingCallback(log_freq=max(1000, eval_freq // 5))
        callbacks.append(mlflow_callback)
        
        try:
            # Train the algorithm
            start_time = datetime.now()
            
            algorithm.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save final model
            final_model_path = self.models_path / "final_model"
            algorithm.save(str(final_model_path))
            
            # Final evaluation
            final_eval = self.evaluate_algorithm(algorithm, n_episodes=n_eval_episodes)
            
            # Prepare results
            results = {
                'total_timesteps': total_timesteps,
                'training_time': training_time,
                'final_model_path': str(final_model_path),
                'final_evaluation': final_eval,
                'success': True
            }
            
            # Add best model info if available
            if hasattr(eval_callback, 'best_mean_reward'):
                results['best_mean_reward'] = eval_callback.best_mean_reward
                results['best_model_path'] = str(self.models_path / "best_model")
            else:
                results['best_mean_reward'] = final_eval.get('mean_reward', 0.0)
            
            # Log final metrics
            self.log_metrics({
                'training_time_seconds': training_time,
                'final_mean_reward': final_eval.get('mean_reward', 0.0),
                'final_std_reward': final_eval.get('std_reward', 0.0)
            })
            
            logger.info(f"Training completed successfully in {training_time:.2f}s")
            return results
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': training_time
            }
    
    def evaluate_algorithm(self, 
                          algorithm: BaseAlgorithm,
                          n_episodes: int = 20,
                          deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate trained algorithm
        
        Args:
            algorithm: Trained RL algorithm
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating algorithm for {n_episodes} episodes")
        
        try:
            # Get environment - handle both VecEnv and regular env
            env = algorithm.get_env()
            
            # For VecEnv, we need to handle differently
            if hasattr(env, 'envs'):
                # This is a VecEnv
                eval_env = env
            else:
                # Regular environment
                eval_env = env
            
            # Evaluate policy
            mean_reward, std_reward = evaluate_policy(
                algorithm,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                return_episode_rewards=False
            )
            
            # Detailed episode analysis (reduced to avoid action errors)
            episode_rewards = []
            episode_lengths = []
            
            # Only do detailed analysis for a few episodes to avoid errors
            for episode_idx in range(min(3, n_episodes)):
                try:
                    if hasattr(eval_env, 'envs'):
                        obs = eval_env.reset()
                        single_env = eval_env.envs[0]
                    else:
                        obs, _ = eval_env.reset()
                        single_env = eval_env
                    
                    episode_reward = 0
                    episode_length = 0
                    terminated = False
                    truncated = False
                    
                    max_steps = 50  # Limit steps to avoid long episodes
                    
                    while not (terminated or truncated) and episode_length < max_steps:
                        # Get action from algorithm
                        action, _ = algorithm.predict(obs, deterministic=deterministic)
                        
                        # Ensure action is valid (0, 1, 2, or 3)
                        if hasattr(action, 'item'):
                            action = action.item()
                        action = int(action)
                        
                        # Validate action
                        if action not in [0, 1, 2, 3]:
                            logger.warning(f"Invalid action {action}, using 0")
                            action = 0
                        
                        # Step environment
                        if hasattr(eval_env, 'envs'):
                            obs, reward, terminated, truncated, info = eval_env.step([action])
                            reward = reward[0]
                            terminated = terminated[0]
                            truncated = truncated[0]
                        else:
                            obs, reward, terminated, truncated, info = eval_env.step(action)
                        
                        episode_reward += reward
                        episode_length += 1
                    
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
                except Exception as e:
                    logger.warning(f"Episode {episode_idx} failed: {e}")
                    continue
            
            results = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'n_episodes': n_episodes,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'mean_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                'success_rate': float(len([r for r in episode_rewards if r > 0]) / len(episode_rewards)) if episode_rewards else 0.0
            }
            
            logger.info(f"Evaluation completed: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'n_episodes': n_episodes,
                'error': str(e)
            }
    
    def log_model(self, model: torch.nn.Module, name: str) -> None:
        """
        Log PyTorch model to MLflow
        
        Args:
            model: PyTorch model to log
            name: Name for the model artifact
        """
        try:
            mlflow.pytorch.log_model(model, name)
            logger.info(f"Logged model: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log model {name}: {e}")
    
    def log_figure(self, figure: plt.Figure, name: str) -> None:
        """
        Log matplotlib figure to MLflow and save locally
        
        Args:
            figure: Matplotlib figure to log
            name: Name for the figure
        """
        try:
            # Save locally
            figure_path = self.plots_path / f"{name}.png"
            figure.savefig(figure_path, dpi=300, bbox_inches='tight')
            
            # Log to MLflow
            mlflow.log_artifact(str(figure_path))
            
            logger.info(f"Logged figure: {name}")
            
        except Exception as e:
            logger.error(f"Failed to log figure {name}: {e}")
    
    def plot_training_curves(self, 
                            train_metrics: Dict[str, List[float]], 
                            val_metrics: Dict[str, List[float]], 
                            title: str = "Training Curves") -> plt.Figure:
        """
        Plot training curves
        
        Args:
            train_metrics: Dictionary of training metrics over time
            val_metrics: Dictionary of validation metrics over time
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot reward curves
        if 'reward' in train_metrics:
            axes[0, 0].plot(train_metrics['reward'], label='Training', alpha=0.7)
        if 'reward' in val_metrics:
            axes[0, 0].plot(val_metrics['reward'], label='Validation', alpha=0.7)
        axes[0, 0].set_title('Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss curves (if available)
        if 'loss' in train_metrics:
            axes[0, 1].plot(train_metrics['loss'], label='Training Loss', alpha=0.7)
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot episode length
        if 'episode_length' in train_metrics:
            axes[1, 0].plot(train_metrics['episode_length'], alpha=0.7)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot exploration rate (if available)
        if 'exploration_rate' in train_metrics:
            axes[1, 1].plot(train_metrics['exploration_rate'], alpha=0.7)
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_best_model(self, 
                       model: Union[BaseAlgorithm, torch.nn.Module], 
                       metric_value: float, 
                       metric_name: str = "mean_reward", 
                       mode: str = "max") -> bool:
        """
        Save model if it's the best so far
        
        Args:
            model: Model to potentially save
            metric_value: Current metric value
            metric_name: Name of the metric being tracked
            mode: 'max' for higher is better, 'min' for lower is better
            
        Returns:
            True if model was saved as best, False otherwise
        """
        is_best = False
        
        if mode == "max":
            is_best = metric_value > self.best_model_metric
        else:
            is_best = metric_value < self.best_model_metric
        
        if is_best:
            self.best_model_metric = metric_value
            self.best_model_path = self.models_path / f"best_{metric_name}_model"
            
            if isinstance(model, BaseAlgorithm):
                model.save(str(self.best_model_path))
            elif isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), f"{self.best_model_path}.pth")
            
            logger.info(f"Saved new best model with {metric_name}={metric_value:.4f}")
            
            # Log to MLflow
            self.log_metrics({f'best_{metric_name}': metric_value})
        
        return is_best
    
    def get_artifact_path(self, artifact_type: str) -> Path:
        """Get path for specific artifact type"""
        paths = {
            'models': self.models_path,
            'logs': self.logs_path,
            'plots': self.plots_path,
            'run': self.run_artifact_path
        }
        return paths.get(artifact_type, self.run_artifact_path)