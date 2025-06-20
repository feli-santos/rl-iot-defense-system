"""
DQN Algorithm Implementation using Stable Baselines3
"""

from typing import Dict, Any, Optional
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm
import os

from algorithms.base_algorithm import BaseAlgorithm
from environment.environment import IoTEnv


class DQNAlgorithm(BaseAlgorithm):
    """DQN Algorithm using Stable Baselines3"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
    def create_model(self, env, training_manager):
        """Create DQN model"""
        
        # Create DQN model with MultiInputPolicy for Dict observation space
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.DQN_LEARNING_RATE,
            buffer_size=self.config.DQN_BUFFER_SIZE,
            learning_starts=1000,  # Start learning after 1000 steps
            batch_size=self.config.DQN_BATCH_SIZE,
            tau=self.config.DQN_TAU,
            gamma=self.config.DQN_GAMMA,
            train_freq=4,  # Train every 4 steps
            gradient_steps=1,
            target_update_interval=self.config.DQN_TARGET_UPDATE_FREQ,
            exploration_fraction=0.1,  # Fraction of training to anneal epsilon
            exploration_initial_eps=self.config.EXPLORATION_EPS_START,
            exploration_final_eps=self.config.EXPLORATION_EPS_END,
            verbose=1,
            tensorboard_log=str(training_manager.logs_path),
            # Add policy_kwargs to configure the network architecture for Dict observations
            policy_kwargs={
                "net_arch": self.config.NETWORK_HIDDEN_LAYERS
            }
        )
        
        return model
        
    def train(self, model, training_manager):
        """Train DQN model"""
        
        # Calculate total timesteps from episodes and epochs
        total_timesteps = self.config.DQN_TOTAL_EPISODES * self.config.DQN_EPOCHS_PER_EPISODE
        
        # Train the model with progress tracking
        print(f"Training DQN for {total_timesteps} timesteps...")
        
        # Custom callback for logging
        class ProgressCallback:
            def __init__(self, training_manager):
                self.training_manager = training_manager
                self.last_log = 0
                
            def __call__(self, locals, globals):
                # Log every 1000 timesteps
                if locals['self'].num_timesteps - self.last_log >= 1000:
                    self.training_manager.log_metrics({
                        'timesteps': locals['self'].num_timesteps,
                        'fps': locals.get('fps', 0),
                    }, step=locals['self'].num_timesteps)
                    self.last_log = locals['self'].num_timesteps
                return True
        
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=ProgressCallback(training_manager)
        )
        
        return model
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get DQN hyperparameters"""
        return {
            'learning_rate': self.config.DQN_LEARNING_RATE,
            'buffer_size': self.config.DQN_BUFFER_SIZE,
            'batch_size': self.config.DQN_BATCH_SIZE,
            'tau': self.config.DQN_TAU,
            'gamma': self.config.DQN_GAMMA,
            'target_update_freq': self.config.DQN_TARGET_UPDATE_FREQ,
            'total_episodes': self.config.DQN_TOTAL_EPISODES,
            'epochs_per_episode': self.config.DQN_EPOCHS_PER_EPISODE,
            'total_timesteps': self.config.DQN_TOTAL_EPISODES * self.config.DQN_EPOCHS_PER_EPISODE,
            'exploration_initial_eps': self.config.EXPLORATION_EPS_START,
            'exploration_final_eps': self.config.EXPLORATION_EPS_END,
            'policy': 'MultiInputPolicy',
            'net_arch': self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def save_model(self, model, path: str):
        """Save DQN model"""
        model.save(path.replace('.zip', ''))  # SB3 adds .zip automatically
        
    def load_model(self, path: str):
        """Load DQN model"""
        return DQN.load(path.replace('.zip', ''))