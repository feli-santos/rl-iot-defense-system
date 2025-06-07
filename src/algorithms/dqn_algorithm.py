"""
DQN Algorithm Implementation
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from algorithms.base_algorithm import BaseAlgorithm
from environment import IoTEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class DQNNetwork(nn.Module):
    """DQN Neural Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DQNAlgorithm(BaseAlgorithm):
    """DQN Algorithm Implementation"""
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _get_observation_dim(self, observation_space):
        """Calculate total observation dimension from different space types"""
        if isinstance(observation_space, spaces.Box):
            # Simple box space
            if len(observation_space.shape) == 1:
                return observation_space.shape[0]
            else:
                return np.prod(observation_space.shape)
        elif isinstance(observation_space, spaces.Dict):
            # Dictionary space - sum all dimensions
            total_dim = 0
            for key, space in observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    total_dim += np.prod(space.shape)
                elif isinstance(space, spaces.Discrete):
                    total_dim += 1
            return total_dim
        elif isinstance(observation_space, spaces.Discrete):
            return observation_space.n
        else:
            # Fallback
            return self.config.ENVIRONMENT_NUM_STATES
    
    def _flatten_observation(self, obs):
        """Flatten observation from dict or other format to vector"""
        if isinstance(obs, dict):
            # Handle dictionary observations
            flattened = []
            for key in sorted(obs.keys()):  # Sort keys for consistency
                if isinstance(obs[key], np.ndarray):
                    flattened.append(obs[key].flatten())
                else:
                    flattened.append(np.array([obs[key]]))
            return np.concatenate(flattened)
        elif isinstance(obs, np.ndarray):
            return obs.flatten()
        else:
            return np.array(obs).flatten()
        
    def create_model(self, env, training_manager):
        """Create DQN model"""
        # Handle different environment types
        if hasattr(env, 'envs') and len(env.envs) > 0:
            # VecEnv case - get the base environment
            base_env = env.envs[0]
            if hasattr(base_env, 'env'):
                # Monitor wrapper case
                actual_env = base_env.env
            else:
                actual_env = base_env
        else:
            # Single environment case
            actual_env = env
        
        # Get environment dimensions
        if hasattr(actual_env, 'observation_space') and actual_env.observation_space is not None:
            state_dim = self._get_observation_dim(actual_env.observation_space)
        else:
            # Fallback to config values
            print("Warning: Could not get observation_space from env, using config values")
            state_dim = self.config.ENVIRONMENT_NUM_STATES
        
        if hasattr(actual_env, 'action_space') and actual_env.action_space is not None:
            if hasattr(actual_env.action_space, 'n'):
                action_dim = actual_env.action_space.n
            else:
                action_dim = actual_env.action_space.shape[0]
        else:
            # Fallback to config values
            print("Warning: Could not get action_space from env, using config values")
            action_dim = self.config.ENVIRONMENT_NUM_ACTIONS
        
        print(f"DQN Model - State dim: {state_dim}, Action dim: {action_dim}")
        print(f"Observation space type: {type(actual_env.observation_space)}")
        
        # Create Q-network
        q_network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=self.config.NETWORK_HIDDEN_LAYERS
        ).to(self.device)
        
        # Create target network
        target_network = DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_layers=self.config.NETWORK_HIDDEN_LAYERS
        ).to(self.device)
        
        # Copy weights
        target_network.load_state_dict(q_network.state_dict())
        
        return {
            'q_network': q_network,
            'target_network': target_network,
            'optimizer': torch.optim.Adam(q_network.parameters(), lr=self.config.DQN_LEARNING_RATE),
            'state_dim': state_dim,
            'action_dim': action_dim
        }
        
    def train(self, model, training_manager):
        """Train DQN model"""
        # Simple DQN training implementation
        q_network = model['q_network']
        target_network = model['target_network']
        optimizer = model['optimizer']
        
        # Create fresh environment for training (not wrapped)
        train_env = IoTEnv(self.config)
        
        episodes = self.config.DQN_TOTAL_EPISODES
        epsilon = self.config.EXPLORATION_EPS_START
        epsilon_decay = self.config.EXPLORATION_EPS_DECAY
        epsilon_min = self.config.EXPLORATION_EPS_END
        
        print(f"Training DQN for {episodes} episodes...")
        
        for episode in tqdm(range(episodes), desc="Training DQN"):
            # Handle new Gymnasium reset format
            reset_result = train_env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result  # Fallback for old format
                
            state = self._flatten_observation(obs)
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 500:  # Max steps per episode
                # Epsilon-greedy action selection
                if torch.rand(1).item() < epsilon:
                    action = train_env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = q_network(state_tensor)
                        action = q_values.argmax().item()
                
                # Handle new Gymnasium step format
                step_result = train_env.step(action)
                if len(step_result) == 5:
                    # New Gymnasium format: (obs, reward, terminated, truncated, info)
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # Old gym format: (obs, reward, done, info)
                    next_obs, reward, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step result format: {len(step_result)} values")
                
                next_state = self._flatten_observation(next_obs)
                episode_reward += reward
                state = next_state
                step_count += 1
            
            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Log metrics
            if episode % 10 == 0:
                training_manager.log_metrics({
                    'dqn_episode_reward': episode_reward,
                    'dqn_epsilon': epsilon,
                    'dqn_episode_length': step_count
                }, step=episode)
                
            # Update target network
            if episode % self.config.DQN_TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(q_network.state_dict())
                
            # Print progress
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.4f}")
        
        print("DQN Training completed!")
        return model
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get DQN hyperparameters"""
        return {
            'learning_rate': self.config.DQN_LEARNING_RATE,
            'gamma': self.config.DQN_GAMMA,
            'batch_size': self.config.DQN_BATCH_SIZE,
            'total_episodes': self.config.DQN_TOTAL_EPISODES,
            'target_update_freq': self.config.DQN_TARGET_UPDATE_FREQ,
            'hidden_layers': self.config.NETWORK_HIDDEN_LAYERS,
            'eps_start': self.config.EXPLORATION_EPS_START,
            'eps_end': self.config.EXPLORATION_EPS_END,
            'eps_decay': self.config.EXPLORATION_EPS_DECAY
        }
        
    def save_model(self, model, path: str):
        """Save DQN model"""
        torch.save({
            'q_network_state_dict': model['q_network'].state_dict(),
            'target_network_state_dict': model['target_network'].state_dict(),
            'optimizer_state_dict': model['optimizer'].state_dict(),
            'state_dim': model['state_dim'],
            'action_dim': model['action_dim']
        }, path)
        
    def load_model(self, path: str):
        """Load DQN model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate the networks
        q_network = DQNNetwork(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_layers=self.config.NETWORK_HIDDEN_LAYERS
        ).to(self.device)
        
        target_network = DQNNetwork(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_layers=self.config.NETWORK_HIDDEN_LAYERS
        ).to(self.device)
        
        # Load the state dictionaries
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        optimizer = torch.optim.Adam(q_network.parameters(), lr=self.config.DQN_LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'q_network': q_network,
            'target_network': target_network,
            'optimizer': optimizer,
            'state_dim': checkpoint['state_dim'],
            'action_dim': checkpoint['action_dim']
        }