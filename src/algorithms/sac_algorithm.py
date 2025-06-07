"""
SAC Algorithm Implementation using Stable Baselines3
"""

from typing import Dict, Any, Optional
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm
import os

from algorithms.base_algorithm import BaseAlgorithm
from environment import IoTEnv


class SACAlgorithm(BaseAlgorithm):
    """SAC Algorithm using Stable Baselines3"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
    def create_model(self, env, training_manager):
        """Create SAC model"""
        
        # Create SAC model
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self.config.SAC_LEARNING_RATE,
            buffer_size=self.config.SAC_BUFFER_SIZE,
            learning_starts=self.config.SAC_LEARNING_STARTS,
            batch_size=self.config.SAC_BATCH_SIZE,
            tau=self.config.SAC_TAU,
            gamma=self.config.SAC_GAMMA,
            train_freq=self.config.SAC_TRAIN_FREQ,
            gradient_steps=self.config.SAC_GRADIENT_STEPS,
            ent_coef=self.config.SAC_ENT_COEF,
            target_update_interval=self.config.SAC_TARGET_UPDATE_INTERVAL,
            target_entropy=self.config.SAC_TARGET_ENTROPY,
            verbose=1,
            tensorboard_log=str(training_manager.logs_path)
        )
        
        return model
        
    def train(self, model, training_manager):
        """Train SAC model"""
        
        total_timesteps = self.config.SAC_TOTAL_TIMESTEPS
        
        # Train the model with progress tracking
        print(f"Training SAC for {total_timesteps} timesteps...")
        
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
        """Get SAC hyperparameters"""
        return {
            'learning_rate': self.config.SAC_LEARNING_RATE,
            'buffer_size': self.config.SAC_BUFFER_SIZE,
            'learning_starts': self.config.SAC_LEARNING_STARTS,
            'batch_size': self.config.SAC_BATCH_SIZE,
            'tau': self.config.SAC_TAU,
            'gamma': self.config.SAC_GAMMA,
            'train_freq': self.config.SAC_TRAIN_FREQ,
            'gradient_steps': self.config.SAC_GRADIENT_STEPS,
            'ent_coef': self.config.SAC_ENT_COEF,
            'target_update_interval': self.config.SAC_TARGET_UPDATE_INTERVAL,
            'target_entropy': self.config.SAC_TARGET_ENTROPY,
            'total_timesteps': self.config.SAC_TOTAL_TIMESTEPS
        }
        
    def save_model(self, model, path: str):
        """Save SAC model"""
        model.save(path.replace('.zip', ''))  # SB3 adds .zip automatically
        
    def load_model(self, path: str):
        """Load SAC model"""
        return SAC.load(path.replace('.zip', ''))