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
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import mlflow
import mlflow.pytorch
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

logger = logging.getLogger(__name__)


class MLflowCallback(BaseCallback):
    """
    MLflow callback that captures comprehensive RL training metrics
    for DQN, PPO, and A2C algorithms with detailed performance insights.
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_timestep = 0
        
        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        # Reward analysis
        self.reward_buffer: List[float] = []
        self.cumulative_rewards: List[float] = []
        
        # Training metrics
        self.loss_values: List[float] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropy_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.explained_variances: List[float] = []
        
        # Performance metrics
        self.fps_values: List[float] = []
        self.exploration_rates: List[float] = []
        self.clipfracs: List[float] = []  # PPO specific
        self.kl_divergences: List[float] = []  # PPO specific
        
        # Time tracking
        self.step_times: List[float] = []
        self.last_time = time.time()
        
        # Action distribution tracking
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 defense actions
        
    def _on_training_start(self) -> None:
        """Called when training starts"""
        try:
            mlflow.log_metrics({
                'training/timesteps_total': 0,
                'training/episodes_total': 0,
                'training/started': 1,
                'performance/fps': 0.0
            }, step=0)
            
            if self.verbose > 0:
                print("📊 MLflow callback initialized - comprehensive RL metrics logging enabled")
                
        except Exception as e:
            logger.warning(f"Failed to log training start: {e}")
    
    def _on_step(self) -> bool:
        """Called at each training step with comprehensive metric capture"""
        try:
            # Track step timing for FPS calculation
            current_time = time.time()
            step_time = current_time - self.last_time
            self.step_times.append(step_time)
            self.last_time = current_time
            
            # Capture episode data
            self._capture_episode_data()
            
            # Capture training metrics
            self._capture_training_metrics()
            
            # Track actions for distribution analysis
            self._track_action_distribution()
            
            # Log metrics at specified frequency
            if self.num_timesteps - self.last_log_timestep >= self.log_freq:
                self._log_comprehensive_metrics()
                self.last_log_timestep = self.num_timesteps
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in MLflow callback step: {e}")
            return True
    
    def _capture_episode_data(self) -> None:
        """Capture detailed episode completion data"""
        try:
            # Method 1: Check infos for episode data (most reliable)
            if hasattr(self.locals, 'infos') and self.locals.get('infos'):
                for info in self.locals['infos']:
                    if isinstance(info, dict) and 'episode' in info:
                        episode_info = info['episode']
                        if 'r' in episode_info and 'l' in episode_info:
                            episode_reward = float(episode_info['r'])
                            episode_length = int(episode_info['l'])
                            
                            self.episode_rewards.append(episode_reward)
                            self.episode_lengths.append(episode_length)
                            self.episode_count += 1
                            
                            # Log individual episode completion
                            mlflow.log_metrics({
                                'episodes/reward': episode_reward,
                                'episodes/length': episode_length,
                                'episodes/count': self.episode_count
                            }, step=self.num_timesteps)
            
            # Method 2: Track current episode progress
            if hasattr(self.locals, 'rewards') and self.locals.get('rewards') is not None:
                rewards = self.locals['rewards']
                if hasattr(rewards, '__iter__'):
                    # VecEnv case
                    for reward in rewards:
                        self.reward_buffer.append(float(reward))
                        self.current_episode_reward += float(reward)
                else:
                    # Single env case
                    self.reward_buffer.append(float(rewards))
                    self.current_episode_reward += float(rewards)
                
                self.current_episode_length += 1
            
            # Check for episode termination
            if hasattr(self.locals, 'dones') and self.locals.get('dones') is not None:
                dones = self.locals['dones']
                if hasattr(dones, '__iter__'):
                    for done in dones:
                        if done:
                            self._finalize_episode()
                else:
                    if dones:
                        self._finalize_episode()
                        
        except Exception as e:
            logger.debug(f"Could not capture episode data: {e}")
    
    def _finalize_episode(self) -> None:
        """Finalize episode tracking when episode ends"""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            # Reset for next episode
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
    
    def _capture_training_metrics(self) -> None:
        """Capture algorithm-specific training metrics"""
        try:
            model = self.model
            
            # DQN-specific metrics
            if hasattr(model, 'logger') and model.logger is not None:
                logger_data = model.logger.name_to_value
                
                # Loss metrics
                for loss_key in ['train/loss', 'loss', 'train/q_loss']:
                    if loss_key in logger_data:
                        self.loss_values.append(float(logger_data[loss_key]))
                        break
                
                # Learning rate
                for lr_key in ['train/learning_rate', 'learning_rate']:
                    if lr_key in logger_data:
                        self.learning_rates.append(float(logger_data[lr_key]))
                        break
                
                # Exploration rate (DQN epsilon)
                for eps_key in ['train/exploration_rate', 'exploration_rate', 'eps']:
                    if eps_key in logger_data:
                        self.exploration_rates.append(float(logger_data[eps_key]))
                        break
            
            # PPO/A2C-specific metrics
            if hasattr(self.locals, 'logger') and self.locals['logger'] is not None:
                logger_data = self.locals['logger'].name_to_value
                
                # Policy and value losses
                loss_mappings = {
                    'train/policy_loss': self.policy_losses,
                    'train/value_loss': self.value_losses,
                    'train/entropy_loss': self.entropy_losses,
                    'train/explained_variance': self.explained_variances,
                    'train/clip_fraction': self.clipfracs,
                    'train/kl_divergence': self.kl_divergences
                }
                
                for key, storage in loss_mappings.items():
                    if key in logger_data:
                        storage.append(float(logger_data[key]))
                
                # Learning rate for PPO/A2C
                if 'train/learning_rate' in logger_data:
                    self.learning_rates.append(float(logger_data['train/learning_rate']))
            
            # Calculate FPS
            if len(self.step_times) >= 10:
                recent_times = self.step_times[-10:]
                fps = 1.0 / (np.mean(recent_times) + 1e-8)
                self.fps_values.append(fps)
                
        except Exception as e:
            logger.debug(f"Could not capture training metrics: {e}")
    
    def _track_action_distribution(self) -> None:
        """Track action distribution for policy analysis"""
        try:
            if hasattr(self.locals, 'actions') and self.locals.get('actions') is not None:
                actions = self.locals['actions']
                if hasattr(actions, '__iter__'):
                    for action in actions:
                        action_idx = int(action)
                        if action_idx in self.action_counts:
                            self.action_counts[action_idx] += 1
                else:
                    action_idx = int(actions)
                    if action_idx in self.action_counts:
                        self.action_counts[action_idx] += 1
        except Exception as e:
            logger.debug(f"Could not track action distribution: {e}")
    
    def _log_comprehensive_metrics(self) -> None:
        """Log comprehensive RL training metrics to MLflow"""
        try:
            metrics = {
                'training/timesteps': self.num_timesteps,
                'training/episodes_total': self.episode_count
            }
            
            # === REWARD METRICS ===
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                all_rewards = self.episode_rewards
                
                metrics.update({
                    'rewards/episode_mean_recent': float(np.mean(recent_rewards)),
                    'rewards/episode_std_recent': float(np.std(recent_rewards)),
                    'rewards/episode_min_recent': float(np.min(recent_rewards)),
                    'rewards/episode_max_recent': float(np.max(recent_rewards)),
                    'rewards/episode_mean_all': float(np.mean(all_rewards)),
                    'rewards/episode_best': float(np.max(all_rewards)),
                    'rewards/episode_worst': float(np.min(all_rewards)),
                })
                
                # Reward trend analysis
                if len(all_rewards) >= 20:
                    early_rewards = np.mean(all_rewards[:10])
                    recent_rewards_mean = np.mean(recent_rewards[-10:])
                    metrics['rewards/improvement'] = float(recent_rewards_mean - early_rewards)
                
                # Success rate (positive rewards)
                positive_rewards = [r for r in recent_rewards if r > 0]
                metrics['rewards/success_rate'] = float(len(positive_rewards) / len(recent_rewards))
            
            # === EPISODE LENGTH METRICS ===
            if self.episode_lengths:
                recent_lengths = self.episode_lengths[-100:]
                metrics.update({
                    'episodes/length_mean': float(np.mean(recent_lengths)),
                    'episodes/length_std': float(np.std(recent_lengths)),
                    'episodes/length_min': float(np.min(recent_lengths)),
                    'episodes/length_max': float(np.max(recent_lengths)),
                })
            
            # === TRAINING LOSS METRICS ===
            if self.loss_values:
                recent_losses = self.loss_values[-10:]
                metrics.update({
                    'losses/total_loss_mean': float(np.mean(recent_losses)),
                    'losses/total_loss_current': float(recent_losses[-1]),
                    'losses/total_loss_std': float(np.std(recent_losses)),
                })
            
            if self.policy_losses:
                recent_policy = self.policy_losses[-10:]
                metrics.update({
                    'losses/policy_loss_mean': float(np.mean(recent_policy)),
                    'losses/policy_loss_current': float(recent_policy[-1]),
                })
            
            if self.value_losses:
                recent_value = self.value_losses[-10:]
                metrics.update({
                    'losses/value_loss_mean': float(np.mean(recent_value)),
                    'losses/value_loss_current': float(recent_value[-1]),
                })
            
            if self.entropy_losses:
                recent_entropy = self.entropy_losses[-10:]
                metrics.update({
                    'losses/entropy_loss_mean': float(np.mean(recent_entropy)),
                    'losses/entropy_loss_current': float(recent_entropy[-1]),
                })
            
            # === LEARNING METRICS ===
            if self.learning_rates:
                metrics['training/learning_rate'] = float(self.learning_rates[-1])
            
            if self.explained_variances:
                metrics['training/explained_variance'] = float(self.explained_variances[-1])
            
            if self.exploration_rates:
                metrics['training/exploration_rate'] = float(self.exploration_rates[-1])
            
            if self.clipfracs:
                metrics['training/clip_fraction'] = float(self.clipfracs[-1])
            
            if self.kl_divergences:
                metrics['training/kl_divergence'] = float(self.kl_divergences[-1])
            
            # === PERFORMANCE METRICS ===
            if self.fps_values:
                metrics['performance/fps'] = float(self.fps_values[-1])
                metrics['performance/fps_mean'] = float(np.mean(self.fps_values[-10:]))
            
            # Time per step
            if self.step_times:
                recent_step_times = self.step_times[-100:]
                metrics.update({
                    'performance/step_time_mean': float(np.mean(recent_step_times)),
                    'performance/step_time_std': float(np.std(recent_step_times)),
                })
            
            # === ACTION DISTRIBUTION METRICS ===
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                action_names = ['monitor', 'rate_limit', 'block_ips', 'shutdown_services']
                for i, action_name in enumerate(action_names):
                    action_prob = self.action_counts[i] / total_actions
                    metrics[f'actions/{action_name}_probability'] = float(action_prob)
                
                # Action entropy (policy diversity)
                probs = [self.action_counts[i] / total_actions for i in range(4)]
                action_entropy = -sum(p * np.log(p + 1e-8) for p in probs if p > 0)
                metrics['actions/entropy'] = float(action_entropy)
            
            # === STABILITY METRICS ===
            if len(self.episode_rewards) >= 50:
                # Coefficient of variation for reward stability
                recent_50 = self.episode_rewards[-50:]
                cv = np.std(recent_50) / (abs(np.mean(recent_50)) + 1e-8)
                metrics['stability/reward_coefficient_variation'] = float(cv)
                
                # Moving average convergence
                if len(self.episode_rewards) >= 100:
                    ma_20 = np.mean(self.episode_rewards[-20:])
                    ma_50 = np.mean(self.episode_rewards[-50:])
                    convergence = abs(ma_20 - ma_50) / (abs(ma_50) + 1e-8)
                    metrics['stability/convergence_metric'] = float(convergence)
            
            # Log all metrics to MLflow
            mlflow.log_metrics(metrics, step=self.num_timesteps)
            
            if self.verbose > 1:
                print(f"📊 Logged {len(metrics)} comprehensive metrics at timestep {self.num_timesteps}")
                print(f"   • Episode reward (recent): {metrics.get('rewards/episode_mean_recent', 0):.3f}")
                print(f"   • Episodes completed: {self.episode_count}")
                print(f"   • FPS: {metrics.get('performance/fps', 0):.1f}")
                
        except Exception as e:
            logger.warning(f"Failed to log comprehensive metrics: {e}")
    
    def _on_training_end(self) -> None:
        """Called when training ends with final summary"""
        try:
            final_metrics = {
                'training/completed': 1,
                'training/total_timesteps': self.num_timesteps,
                'training/total_episodes': self.episode_count,
            }
            
            if self.episode_rewards:
                final_metrics.update({
                    'final/mean_reward': float(np.mean(self.episode_rewards[-50:])),
                    'final/best_reward': float(np.max(self.episode_rewards)),
                    'final/total_episodes_logged': len(self.episode_rewards),
                    'final/final_10_episodes_mean': float(np.mean(self.episode_rewards[-10:])),
                })
                
                # Training progression
                if len(self.episode_rewards) >= 100:
                    first_100 = np.mean(self.episode_rewards[:100])
                    last_100 = np.mean(self.episode_rewards[-100:])
                    final_metrics['final/total_improvement'] = float(last_100 - first_100)
            
            # Final performance metrics
            if self.fps_values:
                final_metrics['final/average_fps'] = float(np.mean(self.fps_values))
            
            # Action distribution summary
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                action_names = ['monitor', 'rate_limit', 'block_ips', 'shutdown_services']
                for i, action_name in enumerate(action_names):
                    final_metrics[f'final/{action_name}_usage'] = float(self.action_counts[i] / total_actions)
            
            mlflow.log_metrics(final_metrics, step=self.num_timesteps)
            
            if self.verbose > 0:
                print(f"✅ Training completed - logged {len(self.episode_rewards)} episodes")
                print(f"   • Final mean reward: {final_metrics.get('final/mean_reward', 0):.3f}")
                print(f"   • Best episode reward: {final_metrics.get('final/best_reward', 0):.3f}")
                print(f"   • Total improvement: {final_metrics.get('final/total_improvement', 0):.3f}")
                
        except Exception as e:
            logger.warning(f"Failed to log final metrics: {e}")


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
        self.configured_save_path = Path(save_path)
        
        # Create a unique run ID
        self.run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Setup artifact directories using config path
        self.run_artifact_path = self.configured_save_path / self.run_id
        self.models_path = self.run_artifact_path / "models"
        self.logs_path = self.run_artifact_path / "logs"
        self.plots_path = self.run_artifact_path / "plots"
        self.save_path = self.run_artifact_path
        
        # Create directories
        self._create_directories()
        
        # MLflow tracking
        self.current_run = None
        self.best_model_metric = -np.inf
        self.best_model_path: Optional[Path] = None
        self.config: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized TrainingManager for {experiment_name}")
        logger.info(f"Models will be saved to: {self.models_path}")
    
    def _create_directories(self) -> None:
        """Create all necessary directories"""
        for path in [self.run_artifact_path, self.models_path, self.logs_path, self.plots_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> None:
        """
        Start MLflow run with comprehensive experiment setup.
        
        Args:
            run_name: Optional custom run name
            nested: Whether this is a nested run
        """
        try:
            # End any existing run first
            if mlflow.active_run() is not None:
                mlflow.end_run()
            
            mlflow.set_experiment(self.experiment_name)
            self.current_run = mlflow.start_run(
                run_name=run_name or self.run_id,
                nested=nested
            )
            
            # Log basic configuration
            if self.config:
                if hasattr(self.config, '__dict__'):
                    mlflow.log_params(self.config.__dict__)
                elif isinstance(self.config, dict):
                    mlflow.log_params(self.config)
            
            logger.info(f"Started MLflow run: {self.current_run.info.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run"""
        try:
            if self.current_run:
                mlflow.end_run()
                self.current_run = None
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
        if not self.current_run:
            logger.warning("No active MLflow run for metric logging")
            return
            
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
                logger.debug(f"Logged {len(numeric_metrics)} metrics to MLflow")
                
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow with conflict prevention.
        
        Args:
            params: Parameters to log
        """
        if not self.current_run:
            logger.warning("No active MLflow run for parameter logging")
            return
        
        try:
            # Get already logged parameters to avoid conflicts
            run_data = mlflow.get_run(self.current_run.info.run_id)
            existing_params = run_data.data.params
            
            # Filter out parameters that are already logged with same values
            params_to_log = {}
            for key, value in params.items():
                str_value = str(value)
                if key not in existing_params:
                    params_to_log[key] = str_value
                elif existing_params[key] != str_value:
                    # Log with suffix if value differs
                    new_key = f"{key}_updated"
                    params_to_log[new_key] = str_value
                    logger.warning(f"Parameter conflict: {key}={existing_params[key]} vs {str_value}, logging as {new_key}")
            
            if params_to_log:
                mlflow.log_params(params_to_log)
                logger.info(f"Logged {len(params_to_log)} parameters to MLflow")
            else:
                logger.info("All parameters already logged with same values")
                
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def train_algorithm(self, 
                       algorithm: BaseAlgorithm,
                       total_timesteps: int,
                       eval_freq: int = 10000,
                       n_eval_episodes: int = 10,
                       save_freq: int = 50000) -> Dict[str, Any]:
        """
        Train RL algorithm with comprehensive MLflow metrics logging
        
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
        
        # Log training parameters only if we have an active run
        if self.current_run:
            self.log_params({
                'algorithm_class': algorithm.__class__.__name__,
                'total_timesteps': total_timesteps,
                'eval_freq': eval_freq,
                'n_eval_episodes': n_eval_episodes,
                'save_freq': save_freq
            })
        
        # Setup callbacks
        callbacks = []
        
        # MLflow callback for comprehensive metrics
        mlflow_callback = MLflowCallback(
            log_freq=max(1000, total_timesteps // 100),  # Log ~100 times during training
            verbose=1
        )
        callbacks.append(mlflow_callback)
        
        # Evaluation callback
        if eval_freq > 0:
            eval_callback = EvalCallback(
                eval_env=algorithm.get_env(),
                best_model_save_path=str(self.models_path / "best_model"),
                log_path=str(self.logs_path / "evaluations"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        try:
            # Train the algorithm
            start_time = datetime.now()
            
            print(f"🚀 Training {algorithm.__class__.__name__} for {total_timesteps:,} timesteps...")
            print(f"📊 Comprehensive metrics will be logged every {mlflow_callback.log_freq:,} timesteps")
            print(f"📈 Tracking: rewards, losses, episode lengths, action distribution, performance")
            
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
                'success': True,
                'total_episodes': mlflow_callback.episode_count,
                'metrics_logged': len(mlflow_callback.episode_rewards)
            }
            
            # Add best model info if available
            if hasattr(eval_callback, 'best_mean_reward'):
                results['best_mean_reward'] = eval_callback.best_mean_reward
                results['best_model_path'] = str(self.models_path / "best_model")
            else:
                results['best_mean_reward'] = final_eval.get('mean_reward', 0.0)
            
            # Log final comprehensive summary
            if self.current_run and mlflow_callback.episode_rewards:
                final_metrics = {
                    'summary/training_time_seconds': training_time,
                    'summary/final_mean_reward': final_eval.get('mean_reward', 0.0),
                    'summary/final_std_reward': final_eval.get('std_reward', 0.0),
                    'summary/total_episodes_trained': mlflow_callback.episode_count,
                    'summary/episodes_logged': len(mlflow_callback.episode_rewards),
                    'summary/training_success': 1,
                    'summary/best_episode_reward': float(np.max(mlflow_callback.episode_rewards)),
                    'summary/reward_improvement': float(np.mean(mlflow_callback.episode_rewards[-10:]) - np.mean(mlflow_callback.episode_rewards[:10])) if len(mlflow_callback.episode_rewards) >= 20 else 0.0,
                    'summary/average_fps': float(np.mean(mlflow_callback.fps_values)) if mlflow_callback.fps_values else 0.0
                }
                
                self.log_metrics(final_metrics)
            
            print(f"✅ Training completed successfully!")
            print(f"   • Episodes trained: {mlflow_callback.episode_count}")
            print(f"   • Episodes with rewards logged: {len(mlflow_callback.episode_rewards)}")
            print(f"   • Training time: {training_time:.1f}s")
            if mlflow_callback.episode_rewards:
                print(f"   • Final reward: {np.mean(mlflow_callback.episode_rewards[-10:]):.3f}")
                print(f"   • Best episode: {np.max(mlflow_callback.episode_rewards):.3f}")
            
            logger.info(f"Training completed successfully in {training_time:.2f}s")
            return results
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
            logger.error(f"Training failed: {e}")
            
            # Log failure metrics
            if self.current_run:
                self.log_metrics({
                    'summary/training_time_seconds': training_time,
                    'summary/training_success': 0,
                    'summary/training_failed': 1
                })
            
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
        if not self.current_run:
            logger.warning("No active MLflow run for model logging")
            return
            
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
        if not self.current_run:
            logger.warning("No active MLflow run for figure logging")
            return
            
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
            if self.current_run:
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