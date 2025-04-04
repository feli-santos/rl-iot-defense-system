import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from collections import deque
from attack_simulation import generate_attack_graph, get_critical_paths, AdaptiveAttacker
import random

class IoTEnv(gym.Env):
    """Custom IoT environment for trigger-action attack simulation"""
    
    def __init__(self, config, history_length: int = 5):
        super(IoTEnv, self).__init__()
        self.config = config
        self.history_length = history_length
        
        # Define action and observation space with temporal features
        self.action_space = spaces.Discrete(config.ENVIRONMENT_NUM_ACTIONS)
        
        # Enhanced observation space with temporal features
        self.observation_space = spaces.Dict({
            'current_state': spaces.Box(
                low=0, high=1, 
                shape=(config.ENVIRONMENT_NUM_STATES,), 
                dtype=np.float32
            ),
            'state_history': spaces.Box(
                low=0, high=1,
                shape=(history_length, config.ENVIRONMENT_NUM_STATES),
                dtype=np.float32
            ),
            'action_history': spaces.Box(
                low=0, high=config.ENVIRONMENT_NUM_ACTIONS-1,
                shape=(history_length,),
                dtype=np.float32
            )
        })
        
        # Initialize attack graph and critical paths
        self.attack_graph = generate_attack_graph(config.ENVIRONMENT_NUM_DEVICES)
        self.critical_paths = get_critical_paths(self.attack_graph)
        self.current_attack_path = self._select_attack_path()
        
        # Initialize environment state
        self.current_state = np.zeros(config.ENVIRONMENT_NUM_STATES, dtype=np.float32)
        self.state_history = np.zeros((history_length, config.ENVIRONMENT_NUM_STATES), dtype=np.float32)
        self.action_history = np.zeros(history_length, dtype=np.float32)
        self.attack_proximity = 0.0
        self.goal_node = config.ENVIRONMENT_NUM_DEVICES - 1  # Last device is goal
        self.attack_step = 0  # Track progress along attack path
        
        # Use adaptive attacker
        self.attacker = AdaptiveAttacker(config.ENVIRONMENT_NUM_DEVICES, self.goal_node)
        
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
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state"""
        # Handle seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset environment state
        self.current_state = np.zeros(self.config.ENVIRONMENT_NUM_STATES, dtype=np.float32)
        self.state_history = np.zeros((self.history_length, self.config.ENVIRONMENT_NUM_STATES), dtype=np.float32)
        self.action_history = np.zeros(self.history_length, dtype=np.float32)
        self.attack_proximity = 0.0
        self.attack_step = 0
        self.current_attack_path = self._select_attack_path()
        self.action_counts = {i: 0 for i in range(self.config.ENVIRONMENT_NUM_ACTIONS)}
        self.episode_history = []
        
        # Initial attack on a random node
        initial_node = np.random.randint(0, self.config.ENVIRONMENT_NUM_STATES - 1)
        self.current_state[initial_node] = 1.0
        
        # Return observation and empty info dict
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        self.action_counts[action] += 1
        self.episode_history.append(action)
        
        # Update action history
        self.action_history = np.roll(self.action_history, -1)
        self.action_history[-1] = action
        
        # Execute defensive action
        self._execute_defense(action)
        
        # Get attacker's next move based on current state and defense action
        attack_node = self.attacker.get_next_action(self.current_state, action)
        if attack_node >= 0:
            self.current_state[attack_node] = 1.0
        
        # Update state history
        self.state_history = np.roll(self.state_history, -1, axis=0)
        self.state_history[-1] = self.current_state.copy()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = len(self.episode_history) >= self.config.DQN_EPOCHS_PER_EPISODE
        
        info = {
            'action_counts': self.action_counts,
            'attack_proximity': self.attack_proximity,
            'attack_path': self.current_attack_path,
            'attack_step': self.attack_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Return the current observation including temporal features"""
        return {
            'current_state': self.current_state.copy(),
            'state_history': self.state_history.copy(),
            'action_history': self.action_history.copy()
        }
    
    def _execute_defense(self, action: int) -> None:
        """Execute defensive action on the network"""
        if action == 3:  # Blocking action
            # Block the most advanced compromised node in attack path
            compromised_in_path = [n for n in self.current_attack_path if self.current_state[n] == 1]
            if compromised_in_path:
                target = compromised_in_path[-1]  # Block the most recent compromise
                self.current_state[target] = 0.0
                self.attack_proximity = np.sum(self.current_state) / len(self.current_state)
        
        # Other actions may not directly modify the state
        # but could update internal information tracking
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on the paper's equation (1)"""
        
        n_a1, n_a2, n_a3, n_a4 = (
            max(1, self.action_counts[0]),  # Avoid division by zero
            max(1, self.action_counts[1]),
            max(1, self.action_counts[2]),
            max(1, self.action_counts[3])
        )
        
        # Immediate rewards for each action
        r_a1, r_a2, r_a3, r_a4 = -0.1, 0.01, 0.05, 0.1
        p = self.attack_proximity
        k = self.config.REWARD_INJECTION_THRESHOLD
        G_r = self.config.REWARD_GOAL_REWARD
        
        # Calculate reward based on paper's equation (1)
        if (n_a1 * p) / (n_a1 + n_a2) < k:
            reward = n_a3 * r_a3 - (p * n_a1 * r_a1) / (n_a1 + n_a2) - G_r
        else:
            term1 = (n_a4 * r_a4) * (n_a3 * r_a3) / (n_a4 + n_a3) if (n_a4 + n_a3) != 0 else 0
            term2 = max(n_a2 * r_a2, (p * n_a1 * r_a1) / (n_a1 + n_a2) + G_r)
            reward = term1 - term2

        return float(np.clip(reward / 50, -2, 2))  # Scale down but preserve ratios
    
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