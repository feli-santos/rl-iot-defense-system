"""
A2C Algorithm Implementation using Stable Baselines3
"""

from typing import Dict, Any, Optional
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm
import os

from algorithms.base_algorithm import BaseAlgorithm
from environment.environment import IoTEnv


class A2CAlgorithm(BaseAlgorithm):
    """A2C Algorithm using Stable Baselines3"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        
    def create_model(self, env, training_manager):
        """Create A2C model"""
        
        # Create A2C model with MultiInputPolicy for Dict observation space
        model = A2C(
            "MultiInputPolicy",
            env,
            learning_rate=0.0007,  # A2C typically uses higher learning rates
            n_steps=5,            # Number of steps to run for each environment per update
            gamma=self.config.PPO_GAMMA,  # Reuse gamma from PPO config
            gae_lambda=0.95,
            ent_coef=0.01,        # Entropy coefficient for exploration
            vf_coef=0.25,         # Value function coefficient
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(training_manager.logs_path),
            policy_kwargs={
                "net_arch": dict(
                    pi=self.config.NETWORK_HIDDEN_LAYERS,
                    vf=self.config.NETWORK_HIDDEN_LAYERS
                )
            }
        )
        
        return model
        
    def train(self, model, training_manager):
        """Train A2C model"""
        
        total_timesteps = self.config.PPO_TOTAL_TIMESTEPS  # Reuse PPO timesteps
        
        print(f"Training A2C for {total_timesteps} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )
        
        return model
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get A2C hyperparameters"""
        return {
            'learning_rate': 0.0007,
            'n_steps': 5,
            'gamma': self.config.PPO_GAMMA,
            'gae_lambda': 0.95,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'total_timesteps': self.config.PPO_TOTAL_TIMESTEPS,
            'policy': 'MultiInputPolicy',
            'net_arch': self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def save_model(self, model, path: str):
        """Save A2C model"""
        model.save(path.replace('.zip', ''))
        
    def load_model(self, path: str):
        """Load A2C model"""
        return A2C.load(path.replace('.zip', ''))