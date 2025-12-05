"""
Adversarial Algorithm for Blue Team Training.

This module provides a unified interface for training RL agents
(DQN, PPO, A2C) on the AdversarialIoTEnv. Unlike the original
implementation that used MultiInputPolicy for Dict observations,
this uses MlpPolicy for the Box observation space.

Key differences from original algorithms:
- Uses MlpPolicy (Box observations) instead of MultiInputPolicy (Dict)
- Simplified hyperparameter configuration
- Unified interface for all three algorithms
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)

# Type alias for SB3 algorithms
SB3Algorithm = Union[DQN, PPO, A2C]


@dataclass
class AdversarialAlgorithmConfig:
    """Configuration for adversarial RL training.
    
    Attributes:
        algorithm_type: Type of algorithm ('dqn', 'ppo', 'a2c').
        policy: Policy type (always 'MlpPolicy' for Box obs).
        total_timesteps: Total training timesteps.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor.
        verbose: Verbosity level.
        tensorboard_log: Path for TensorBoard logs.
        
        DQN-specific:
            buffer_size: Replay buffer size.
            learning_starts: Steps before training starts.
            batch_size: Training batch size.
            tau: Soft update coefficient.
            target_update_interval: Steps between target updates.
            exploration_fraction: Fraction of training for epsilon decay.
            exploration_initial_eps: Starting epsilon.
            exploration_final_eps: Final epsilon.
        
        PPO/A2C-specific:
            n_steps: Steps per rollout.
            n_epochs: Epochs per update (PPO only).
            gae_lambda: GAE lambda.
            ent_coef: Entropy coefficient.
            vf_coef: Value function coefficient.
            max_grad_norm: Gradient clipping.
    """
    
    algorithm_type: str = "ppo"
    policy: str = "MlpPolicy"
    total_timesteps: int = 50000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    verbose: int = 1
    tensorboard_log: Optional[str] = None
    
    # DQN-specific
    buffer_size: int = 50000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    
    # PPO/A2C-specific
    n_steps: int = 2048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class AdversarialAlgorithm:
    """Unified RL algorithm interface for adversarial training.
    
    This class provides a simple interface for creating, training,
    saving, and loading RL models for the AdversarialIoTEnv.
    
    Supported algorithms:
    - DQN (Deep Q-Network)
    - PPO (Proximal Policy Optimization)
    - A2C (Advantage Actor-Critic)
    
    Example:
        >>> config = AdversarialAlgorithmConfig(algorithm_type="ppo")
        >>> alg = AdversarialAlgorithm(config)
        >>> model = alg.create_model(env)
        >>> model = alg.train(model, total_timesteps=10000)
        >>> alg.save_model(model, "models/ppo_agent")
    """
    
    ALGORITHMS = {
        "dqn": DQN,
        "ppo": PPO,
        "a2c": A2C,
    }
    
    def __init__(
        self,
        config: Optional[AdversarialAlgorithmConfig] = None,
    ) -> None:
        """Initialize the algorithm.
        
        Args:
            config: Algorithm configuration.
        """
        self._config = config or AdversarialAlgorithmConfig()
        
        if self._config.algorithm_type not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self._config.algorithm_type}. "
                f"Supported: {list(self.ALGORITHMS.keys())}"
            )
        
        logger.info(
            f"AdversarialAlgorithm initialized: {self._config.algorithm_type}"
        )
    
    @property
    def config(self) -> AdversarialAlgorithmConfig:
        """Get algorithm configuration."""
        return self._config
    
    @property
    def algorithm_name(self) -> str:
        """Get algorithm name in uppercase."""
        return self._config.algorithm_type.upper()
    
    def create_model(
        self,
        env: gym.Env,
    ) -> SB3Algorithm:
        """Create an RL model for the given environment.
        
        Args:
            env: Gymnasium environment.
        
        Returns:
            Stable Baselines3 model.
        """
        alg_class = self.ALGORITHMS[self._config.algorithm_type]
        
        if self._config.algorithm_type == "dqn":
            model = alg_class(
                self._config.policy,
                env,
                learning_rate=self._config.learning_rate,
                buffer_size=self._config.buffer_size,
                learning_starts=self._config.learning_starts,
                batch_size=self._config.batch_size,
                tau=self._config.tau,
                gamma=self._config.gamma,
                target_update_interval=self._config.target_update_interval,
                exploration_fraction=self._config.exploration_fraction,
                exploration_initial_eps=self._config.exploration_initial_eps,
                exploration_final_eps=self._config.exploration_final_eps,
                verbose=self._config.verbose,
                tensorboard_log=self._config.tensorboard_log,
            )
        
        elif self._config.algorithm_type == "ppo":
            model = alg_class(
                self._config.policy,
                env,
                learning_rate=self._config.learning_rate,
                n_steps=self._config.n_steps,
                batch_size=self._config.batch_size,
                n_epochs=self._config.n_epochs,
                gamma=self._config.gamma,
                gae_lambda=self._config.gae_lambda,
                ent_coef=self._config.ent_coef,
                vf_coef=self._config.vf_coef,
                max_grad_norm=self._config.max_grad_norm,
                verbose=self._config.verbose,
                tensorboard_log=self._config.tensorboard_log,
            )
        
        elif self._config.algorithm_type == "a2c":
            model = alg_class(
                self._config.policy,
                env,
                learning_rate=self._config.learning_rate,
                n_steps=min(self._config.n_steps, 5),  # A2C uses smaller n_steps
                gamma=self._config.gamma,
                gae_lambda=self._config.gae_lambda,
                ent_coef=self._config.ent_coef,
                vf_coef=self._config.vf_coef,
                max_grad_norm=self._config.max_grad_norm,
                verbose=self._config.verbose,
                tensorboard_log=self._config.tensorboard_log,
            )
        
        logger.info(f"Created {self.algorithm_name} model with MlpPolicy")
        return model
    
    def train(
        self,
        model: SB3Algorithm,
        total_timesteps: Optional[int] = None,
        progress_bar: bool = True,
    ) -> SB3Algorithm:
        """Train the model.
        
        Args:
            model: Model to train.
            total_timesteps: Override for training timesteps.
            progress_bar: Whether to show progress bar.
        
        Returns:
            Trained model.
        """
        timesteps = total_timesteps or self._config.total_timesteps
        
        logger.info(f"Training {self.algorithm_name} for {timesteps} timesteps")
        
        model.learn(
            total_timesteps=timesteps,
            progress_bar=progress_bar,
        )
        
        logger.info("Training complete")
        return model
    
    def save_model(
        self,
        model: SB3Algorithm,
        path: Union[str, Path],
    ) -> None:
        """Save model to disk.
        
        Args:
            model: Model to save.
            path: Path to save model (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load_model(
        self,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
    ) -> SB3Algorithm:
        """Load model from disk.
        
        Args:
            path: Path to saved model.
            env: Environment for loaded model.
        
        Returns:
            Loaded model.
        """
        path = Path(path)
        alg_class = self.ALGORITHMS[self._config.algorithm_type]
        
        model = alg_class.load(str(path), env=env)
        logger.info(f"Model loaded from {path}")
        
        return model
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters as dictionary.
        
        Returns:
            Dictionary of hyperparameters.
        """
        base_params = {
            "algorithm": self._config.algorithm_type,
            "policy": self._config.policy,
            "learning_rate": self._config.learning_rate,
            "gamma": self._config.gamma,
            "total_timesteps": self._config.total_timesteps,
        }
        
        if self._config.algorithm_type == "dqn":
            base_params.update({
                "buffer_size": self._config.buffer_size,
                "batch_size": self._config.batch_size,
                "tau": self._config.tau,
                "target_update_interval": self._config.target_update_interval,
                "exploration_fraction": self._config.exploration_fraction,
                "exploration_initial_eps": self._config.exploration_initial_eps,
                "exploration_final_eps": self._config.exploration_final_eps,
            })
        else:
            base_params.update({
                "n_steps": self._config.n_steps,
                "gae_lambda": self._config.gae_lambda,
                "ent_coef": self._config.ent_coef,
                "vf_coef": self._config.vf_coef,
                "max_grad_norm": self._config.max_grad_norm,
            })
            if self._config.algorithm_type == "ppo":
                base_params["n_epochs"] = self._config.n_epochs
                base_params["batch_size"] = self._config.batch_size
        
        return base_params


def create_algorithm(
    algorithm_type: str,
    **kwargs,
) -> AdversarialAlgorithm:
    """Factory function to create an algorithm.
    
    Args:
        algorithm_type: Type of algorithm ('dqn', 'ppo', 'a2c').
        **kwargs: Additional configuration parameters.
    
    Returns:
        Configured AdversarialAlgorithm instance.
    """
    config = AdversarialAlgorithmConfig(
        algorithm_type=algorithm_type,
        **kwargs,
    )
    return AdversarialAlgorithm(config)
