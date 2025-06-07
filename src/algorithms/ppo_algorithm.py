"""
PPO Algorithm Implementation

Proximal Policy Optimization implementation using Stable Baselines3.
"""

import torch
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm

from .base_algorithm import BaseAlgorithm


class PPOAlgorithm(BaseAlgorithm):
    """PPO algorithm implementation"""
    
    def __init__(self, config: Any):
        super().__init__(config, "PPO")
        
    def create_model(self, env: VecEnv, training_manager: Any) -> PPO:
        """Create and configure the PPO model"""
        
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
            policy_kwargs={
                "net_arch": self.config.NETWORK_HIDDEN_LAYERS,
                "activation_fn": torch.nn.ReLU
            },
            verbose=self.config.TRAINING_VERBOSE,
            seed=self.config.TRAINING_SEED,
            device=self.config.TRAINING_DEVICE,
            tensorboard_log=training_manager.logs_path
        )
        
        self.model = model
        return model
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get PPO-specific hyperparameters"""
        return {
            "learning_rate": self.config.PPO_LEARNING_RATE,
            "n_steps": self.config.PPO_N_STEPS,
            "batch_size": self.config.PPO_BATCH_SIZE,
            "n_epochs": self.config.PPO_N_EPOCHS,
            "gamma": self.config.PPO_GAMMA,
            "gae_lambda": self.config.PPO_GAE_LAMBDA,
            "clip_range": self.config.PPO_CLIP_RANGE,
            "ent_coef": self.config.PPO_ENT_COEF,
            "vf_coef": self.config.PPO_VF_COEF,
            "max_grad_norm": self.config.PPO_MAX_GRAD_NORM,
            "network_arch": self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def train(self, model: PPO, training_manager: Any) -> PPO:
        """Train the PPO model"""
        total_timesteps = self.get_total_timesteps()
        
        # Create custom callback for MLflow logging
        from ..training import MLflowCallback
        mlflow_callback = MLflowCallback(training_manager)
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=mlflow_callback,
            log_interval=10
        )
        
        return model
        
    def get_total_timesteps(self) -> int:
        """Get total training timesteps for PPO"""
        return self.config.PPO_TOTAL_TIMESTEPS
        
    def load_model(self, path: str, env: VecEnv) -> PPO:
        """Load a trained PPO model"""
        return PPO.load(path, env=env)