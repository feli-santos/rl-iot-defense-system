"""
SAC Algorithm Implementation

Soft Actor-Critic implementation using Stable Baselines3.
"""

import torch
from typing import Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm

from .base_algorithm import BaseAlgorithm


class SACAlgorithm(BaseAlgorithm):
    """SAC algorithm implementation"""
    
    def __init__(self, config: Any):
        super().__init__(config, "SAC")
        
    def create_model(self, env: VecEnv, training_manager: Any) -> SAC:
        """Create and configure the SAC model"""
        
        model = SAC(
            "MultiInputPolicy",
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
        """Get SAC-specific hyperparameters"""
        return {
            "learning_rate": self.config.SAC_LEARNING_RATE,
            "buffer_size": self.config.SAC_BUFFER_SIZE,
            "batch_size": self.config.SAC_BATCH_SIZE,
            "tau": self.config.SAC_TAU,
            "gamma": self.config.SAC_GAMMA,
            "train_freq": self.config.SAC_TRAIN_FREQ,
            "gradient_steps": self.config.SAC_GRADIENT_STEPS,
            "ent_coef": self.config.SAC_ENT_COEF,
            "target_update_interval": self.config.SAC_TARGET_UPDATE_INTERVAL,
            "target_entropy": self.config.SAC_TARGET_ENTROPY,
            "network_arch": self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def train(self, model: SAC, training_manager: Any) -> SAC:
        """Train the SAC model"""
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
        """Get total training timesteps for SAC"""
        return self.config.SAC_TOTAL_TIMESTEPS
        
    def load_model(self, path: str, env: VecEnv) -> SAC:
        """Load a trained SAC model"""
        return SAC.load(path, env=env)