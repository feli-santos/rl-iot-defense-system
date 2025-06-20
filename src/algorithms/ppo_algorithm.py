"""
PPO Algorithm Implementation using Stable Baselines3
"""

from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm
import os

from algorithms.base_algorithm import BaseAlgorithm
from environment.environment import IoTEnv


class PPOAlgorithm(BaseAlgorithm):
    """PPO Algorithm using Stable Baselines3"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
    def create_model(self, env, training_manager):
        """Create PPO model"""
        
        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.PPO_LEARNING_RATE,
            n_steps=self.config.PPO_N_STEPS,
            batch_size=self.config.PPO_BATCH_SIZE,
            n_epochs=self.config.PPO_N_EPOCHS,
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA,
            clip_range=self.config.PPO_CLIP_RANGE,
            ent_coef=self.config.PPO_ENT_COEF,
            vf_coef=self.config.PPO_VF_COEF,
            max_grad_norm=self.config.PPO_MAX_GRAD_NORM,
            verbose=1,
            tensorboard_log=str(training_manager.logs_path),
            # Add policy_kwargs to configure the network architecture for Dict observations
            policy_kwargs={
                "net_arch": dict(
                    pi=self.config.NETWORK_HIDDEN_LAYERS,  # Policy network architecture
                    vf=self.config.NETWORK_HIDDEN_LAYERS   # Value function network architecture
                )
            }
        )
        
        return model
        
    def train(self, model, training_manager):
        """Train PPO model"""
        
        total_timesteps = self.config.PPO_TOTAL_TIMESTEPS
        
        # Train the model with progress tracking
        print(f"Training PPO for {total_timesteps} timesteps...")
        
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
        """Get PPO hyperparameters"""
        return {
            'learning_rate': self.config.PPO_LEARNING_RATE,
            'n_steps': self.config.PPO_N_STEPS,
            'batch_size': self.config.PPO_BATCH_SIZE,
            'n_epochs': self.config.PPO_N_EPOCHS,
            'gamma': self.config.PPO_GAMMA,
            'gae_lambda': self.config.PPO_GAE_LAMBDA,
            'clip_range': self.config.PPO_CLIP_RANGE,
            'ent_coef': self.config.PPO_ENT_COEF,
            'vf_coef': self.config.PPO_VF_COEF,
            'max_grad_norm': self.config.PPO_MAX_GRAD_NORM,
            'total_timesteps': self.config.PPO_TOTAL_TIMESTEPS,
            'policy': 'MultiInputPolicy',
            'net_arch': self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def save_model(self, model, path: str):
        """Save PPO model"""
        model.save(path.replace('.zip', ''))  # SB3 adds .zip automatically
        
    def load_model(self, path: str):
        """Load PPO model"""
        return PPO.load(path.replace('.zip', ''))