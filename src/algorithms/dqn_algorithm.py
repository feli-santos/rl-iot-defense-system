"""
DQN Algorithm Implementation

Deep Q-Network implementation using Stable Baselines3.
"""

import torch
from typing import Dict, Any
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm as SB3BaseAlgorithm

from .base_algorithm import BaseAlgorithm


class DQNAlgorithm(BaseAlgorithm):
    """DQN algorithm implementation"""
    
    def __init__(self, config: Any):
        super().__init__(config, "DQN")
        
    def create_model(self, env: VecEnv, training_manager: Any) -> DQN:
        """Create and configure the DQN model"""
        
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=self.config.DQN_LEARNING_RATE,
            buffer_size=self.config.DQN_BUFFER_SIZE,
            learning_starts=1000,
            batch_size=self.config.DQN_BATCH_SIZE,
            tau=self.config.DQN_TAU,
            gamma=self.config.DQN_GAMMA,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=self.config.DQN_TARGET_UPDATE_FREQ,
            exploration_fraction=0.1,
            exploration_initial_eps=self.config.EXPLORATION_EPS_START,
            exploration_final_eps=self.config.EXPLORATION_EPS_END,
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
        """Get DQN-specific hyperparameters"""
        return {
            "learning_rate": self.config.DQN_LEARNING_RATE,
            "buffer_size": self.config.DQN_BUFFER_SIZE,
            "batch_size": self.config.DQN_BATCH_SIZE,
            "tau": self.config.DQN_TAU,
            "gamma": self.config.DQN_GAMMA,
            "exploration_eps_start": self.config.EXPLORATION_EPS_START,
            "exploration_eps_end": self.config.EXPLORATION_EPS_END,
            "target_update_freq": self.config.DQN_TARGET_UPDATE_FREQ,
            "network_arch": self.config.NETWORK_HIDDEN_LAYERS
        }
        
    def train(self, model: DQN, training_manager: Any) -> DQN:
        """Train the DQN model"""
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
        """Get total training timesteps for DQN"""
        return self.config.DQN_TOTAL_EPISODES * self.config.DQN_EPOCHS_PER_EPISODE
        
    def load_model(self, path: str, env: VecEnv) -> DQN:
        """Load a trained DQN model"""
        return DQN.load(path, env=env)