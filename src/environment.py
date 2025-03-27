import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from attack import generate_attack_graph, get_critical_paths
import random

class IoTEnv(gym.Env):
    """Custom IoT environment for trigger-action attack simulation"""
    
    def __init__(self, config):
        super(IoTEnv, self).__init__()
        self.config = config
        
        # Define action and observation space
        self.action_space = spaces.Discrete(config.ENVIRONMENT_NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(config.ENVIRONMENT_NUM_STATES,), 
            dtype=np.float32
        )
        
        # Initialize attack graph and critical paths
        self.attack_graph = generate_attack_graph(config.ENVIRONMENT_NUM_DEVICES)
        self.critical_paths = get_critical_paths(self.attack_graph)
        self.current_attack_path = self._select_attack_path()
        
        # Initialize environment state
        self.current_state = np.zeros(config.ENVIRONMENT_NUM_STATES, dtype=np.float32)
        self.attack_proximity = 0.0
        self.goal_node = config.ENVIRONMENT_NUM_DEVICES - 1  # Last device is goal
        self.attack_step = 0  # Track progress along attack path
        
        # Track action counts for reward calculation
        self.action_counts = {
            0: 0,  # a1: event injection
            1: 0,  # a2: checking device accessibility
            2: 0,  # a3: monitoring security status
            3: 0   # a4: blocking triggers
        }
        
        # Store episode history
        self.episode_history = []
        
    def _select_attack_path(self) -> List[int]:
        """Select an attack path from critical paths"""
        return random.choice(self.critical_paths)
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state
        
        Args:
            seed: Optional seed for random number generator
            options: Optional dictionary of additional options
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Handle seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset environment state
        self.current_state = np.zeros(self.config.ENVIRONMENT_NUM_STATES, dtype=np.float32)
        self.attack_proximity = 0.0
        self.attack_step = 0
        self.current_attack_path = self._select_attack_path()
        self.action_counts = {i: 0 for i in range(self.config.ENVIRONMENT_NUM_ACTIONS)}
        self.episode_history = []
        
        # Return observation and empty info dict
        return self.current_state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment
        
        Returns:
            observation: The new observation
            reward: The reward for this step
            terminated: Whether the episode has ended (goal achieved/failed)
            truncated: Whether the episode was truncated (max steps reached)
            info: Additional information
        """
        self.action_counts[action] += 1
        self.episode_history.append(action)
        
        # Update state based on action
        new_state = self._update_state(action)
        reward = self._calculate_reward(action)
        terminated = self._check_termination()
        truncated = len(self.episode_history) >= self.config.DQN_EPOCHS_PER_EPISODE
        
        info = {
            'action_counts': self.action_counts,
            'attack_proximity': self.attack_proximity,
            'attack_path': self.current_attack_path,
            'attack_step': self.attack_step
        }
        
        self.current_state = new_state
        return new_state, reward, terminated, truncated, info
    
    def _update_state(self, action: int) -> np.ndarray:
        """Update the environment state based on the action taken"""
        new_state = self.current_state.copy()
        
        if action == 0:  # Event injection (a1)
            # Follow the attack path if available
            if self.attack_step < len(self.current_attack_path):
                target = self.current_attack_path[self.attack_step]
                if new_state[target] == 0:  # Only inject if not already compromised
                    new_state[target] = 1.0
                    self.attack_step += 1
            else:
                # Random injection if attack path completed
                available_targets = np.where(new_state == 0)[0]
                if len(available_targets) > 0:
                    target = np.random.choice(available_targets)
                    new_state[target] = 1.0
            
            self.attack_proximity = np.sum(new_state) / len(new_state)
            
        elif action == 3:  # Blocking triggers (a4)
            # Block the most advanced compromised node in attack path
            compromised_in_path = [n for n in self.current_attack_path if new_state[n] == 1]
            if compromised_in_path:
                target = compromised_in_path[-1]  # Block the most recent compromise
                new_state[target] = 0.0
                self.attack_proximity = np.sum(new_state) / len(new_state)
            
        # Actions a2 and a3 don't directly change the state
        return new_state
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on the paper's equation (1)"""
        n_a1, n_a2, n_a3, n_a4 = (
            self.action_counts[0], self.action_counts[1],
            self.action_counts[2], self.action_counts[3]
        )
        
        # Immediate rewards for each action
        r_a1, r_a2, r_a3, r_a4 = -1, 0.1, 0.5, 1.0
        p = self.attack_proximity
        k = self.config.REWARD_INJECTION_THRESHOLD
        G_r = self.config.REWARD_GOAL_REWARD
        
        # Check if goal node is compromised
        if self.current_state[self.goal_node] == 1.0:
            return G_r  # Large penalty if goal is compromised
        
        # Handle division by zero cases
        denominator = (n_a1 + n_a2)
        if denominator == 0:
            # If no actions have been taken yet, return neutral reward
            return 0.0
        
        # Calculate reward based on paper's equation (1)
        if (n_a1 * p) / denominator < k:
            reward = n_a3 * r_a3 - (p * n_a1 * r_a1) / denominator - G_r
        else:
            denominator2 = (n_a4 + n_a3)
            term1 = (n_a4 * r_a4) * (n_a3 * r_a3) / denominator2 if denominator2 != 0 else 0
            term2 = max(n_a2 * r_a2, (p * n_a1 * r_a1) / denominator + G_r)
            reward = term1 - term2
        
        # Add scaling factor to reward
        scaled_reward = reward / 100
        return float(np.clip(scaled_reward, -10, 10))  # Prevent extreme values
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to environment conditions"""
        # Episode ends if goal node is compromised
        return self.current_state[self.goal_node] == 1.0
    
    def render(self, mode='human'):
        """Render the current state of the environment"""
        print(f"Current state: {self.current_state}")
        print(f"Compromised nodes: {np.where(self.current_state == 1)[0].tolist()}")
        print(f"Current attack path: {self.current_attack_path}")
        print(f"Attack step: {self.attack_step}/{len(self.current_attack_path)}")
        print(f"Attack proximity: {self.attack_proximity:.2f}")
        print(f"Action counts: {self.action_counts}")