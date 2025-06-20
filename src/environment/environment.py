"""
Enhanced IoT Defense Environment with Real Attack Prediction

Integrates trained LSTM attack predictor with CICIoT2023 data for realistic
IoT network simulation and defense training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import random
from dataclasses import dataclass

# Import will be handled with try/except to avoid dependency issues
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentConfig:
    """Configuration for IoT Defense Environment"""
    max_steps: int = 1000
    attack_probability: float = 0.3
    sequence_length: int = 10
    reward_scale: float = 1.0
    model_path: str = "models/saved/lstm_real_data.pth"
    data_path: str = "data/processed/ciciot2023"
    state_history_length: int = 10
    action_history_length: int = 5


class IoTEnv(gym.Env):
    """
    Enhanced IoT Defense Environment using real attack prediction.
    
    Simulates an IoT network where agents must defend against various attack types
    using real network flow patterns from CICIoT2023 dataset.
    
    Action Space:
        0: No action (monitor only)
        1: Rate limiting
        2: Block suspicious IPs
        3: Shutdown affected services
    
    Observation Space:
        Dict with keys:
        - 'current_state': Current network state (22 features)
        - 'state_history': History of network states
        - 'action_history': History of defense actions
        - 'attack_prediction': LSTM attack risk assessment
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None) -> None:
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        
        # Initialize attack predictor
        self._initialize_attack_predictor()
        
        # Define action space (4 discrete defense actions: 0, 1, 2, 3)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'current_state': spaces.Box(
                low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
            ),
            'state_history': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.config.state_history_length, 22), 
                dtype=np.float32
            ),
            'action_history': spaces.Box(
                low=0, high=3, 
                shape=(self.config.action_history_length,), 
                dtype=np.int32
            ),
            'attack_prediction': spaces.Box(
                low=0.0, high=1.0, shape=(6,), dtype=np.float32
            )
        })
        
        # Initialize environment state
        self.reset()
        
        logger.info("Enhanced IoT Defense Environment initialized")
    
    def _initialize_attack_predictor(self) -> None:
        """Initialize the enhanced attack predictor"""
        try:
            model_path = Path(self.config.model_path)
            data_path = Path(self.config.data_path)
            
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}. Using mock predictor.")
                self.attack_predictor = None
                return
            
            # Import here to avoid circular imports
            from predictor.interface import AttackPredictorInterface
            
            self.attack_predictor = AttackPredictorInterface(model_path, data_path)
            logger.info("Attack predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize attack predictor: {e}")
            self.attack_predictor = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize environment state
        self.current_step = 0
        self.total_reward = 0.0
        
        # Initialize network state (22 features from RL state representation)
        self.current_network_state = self._generate_initial_network_state()
        
        # Initialize history buffers
        self.state_history = [self.current_network_state.copy() 
                             for _ in range(self.config.state_history_length)]
        self.action_history = [0] * self.config.action_history_length
        
        # Initialize attack state
        self.current_attack_active = False
        self.attack_type = None
        self.attack_severity = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Validate action (0, 1, 2, 3 are valid)
        if not isinstance(action, (int, np.integer)) or action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, 2, or 3.")
        
        # Convert to regular int if numpy integer
        action = int(action)
        
        # Simulate network state evolution
        self._simulate_network_evolution()
        
        # Determine if attack occurs
        attack_occurred = self._determine_attack_occurrence()
        
        # Get attack prediction from LSTM
        attack_prediction = self._get_attack_prediction()
        
        # Calculate reward
        reward = self._calculate_reward(action, attack_occurred, attack_prediction)
        
        # Update history
        self._update_history(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.config.max_steps
        
        # Update step counter
        self.current_step += 1
        self.total_reward += reward
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'attack_occurred': attack_occurred,
            'attack_prediction': attack_prediction,
            'defense_action': action,
            'step_reward': reward
        })
        
        return observation, reward, terminated, truncated, info
    
    def _generate_initial_network_state(self) -> np.ndarray:
        """Generate initial network state (22 features)"""
        # Simulate realistic initial network state
        state = np.random.normal(0, 0.1, 22)  # Start with small values
        state = np.clip(state, -2, 2)  # Reasonable bounds
        return state.astype(np.float32)
    
    def _simulate_network_evolution(self) -> None:
        """Simulate network state evolution over time"""
        # Add some temporal dynamics
        noise = np.random.normal(0, 0.05, 22)
        
        # Simulate network traffic patterns
        if self.current_step % 100 < 20:  # Peak hours
            traffic_multiplier = 1.5
        else:
            traffic_multiplier = 1.0
        
        # Update network state
        self.current_network_state += noise * traffic_multiplier
        self.current_network_state = np.clip(self.current_network_state, -5, 5)
    
    def _determine_attack_occurrence(self) -> bool:
        """Determine if an attack occurs in current step"""
        # Use configurable attack probability
        base_probability = self.config.attack_probability
        
        # Increase probability during peak hours
        if self.current_step % 100 < 20:
            attack_probability = base_probability * 1.5
        else:
            attack_probability = base_probability
        
        attack_occurred = random.random() < attack_probability
        
        if attack_occurred:
            self._simulate_attack_effects()
        
        return attack_occurred
    
    def _simulate_attack_effects(self) -> None:
        """Simulate effects of attack on network state"""
        self.current_attack_active = True
        
        # Simulate different attack types affecting different features
        attack_types = ['ddos', 'dos', 'botnet', 'mitm', 'reconnaissance']
        self.attack_type = random.choice(attack_types)
        
        # Modify network state based on attack type
        if self.attack_type in ['ddos', 'dos']:
            # Increase traffic-related features
            self.current_network_state[:5] += np.random.uniform(0.5, 2.0, 5)
            self.attack_severity = 0.8
        elif self.attack_type == 'botnet':
            # Increase connection-related features
            self.current_network_state[5:10] += np.random.uniform(0.3, 1.5, 5)
            self.attack_severity = 0.7
        elif self.attack_type == 'mitm':
            # Modify protocol-related features
            self.current_network_state[10:15] += np.random.uniform(0.2, 1.0, 5)
            self.attack_severity = 0.6
        else:  # reconnaissance
            # Subtle changes in various features
            self.current_network_state += np.random.uniform(0.1, 0.3, 22)
            self.attack_severity = 0.3
    
    def _get_attack_prediction(self) -> Dict[str, Any]:
        """Get attack prediction from LSTM"""
        if self.attack_predictor is None:
            # Mock prediction when no real predictor available
            return {
                'risk_score': random.uniform(0.1, 0.9),
                'confidence': random.uniform(0.5, 0.95),
                'is_attack': random.choice([True, False]),
                'severity_level': random.choice(['low', 'medium', 'high', 'critical']),
                'predicted_attack': 'mock_prediction',
                'attack_category': 'unknown'
            }
        
        try:
            # Convert network states to format expected by predictor
            network_sequence = []
            for state in self.state_history[-self.config.sequence_length:]:
                # Convert state array to feature dictionary
                feature_dict = {f'feature_{i}': float(state[i]) for i in range(len(state))}
                network_sequence.append(feature_dict)
            
            # Get prediction
            prediction = self.attack_predictor.predict_attack_risk(network_sequence)
            return prediction
            
        except Exception as e:
            logger.error(f"Attack prediction failed: {e}")
            # Return safe default
            return {
                'risk_score': 0.5,
                'confidence': 0.5,
                'is_attack': False,
                'severity_level': 'medium',
                'predicted_attack': 'prediction_error',
                'attack_category': 'unknown'
            }
    
    def _calculate_reward(self, action: int, attack_occurred: bool, 
                         attack_prediction: Dict[str, Any]) -> float:
        """Calculate reward based on defense effectiveness"""
        if self.attack_predictor is not None:
            try:
                # Use enhanced predictor's reward calculation
                network_sequence = []
                for state in self.state_history[-self.config.sequence_length:]:
                    feature_dict = {f'feature_{i}': float(state[i]) for i in range(len(state))}
                    network_sequence.append(feature_dict)
                
                reward = self.attack_predictor.calculate_rl_reward(
                    network_sequence, action, attack_occurred
                )
                return reward * self.config.reward_scale
                
            except Exception as e:
                logger.error(f"Reward calculation failed: {e}")
        
        # Fallback reward calculation
        return self._calculate_fallback_reward(action, attack_occurred, attack_prediction)
    
    def _calculate_fallback_reward(self, action: int, attack_occurred: bool,
                                  attack_prediction: Dict[str, Any]) -> float:
        """Fallback reward calculation when predictor unavailable"""
        reward = 0.0
        
        # Prediction accuracy reward
        predicted_attack = attack_prediction.get('is_attack', False)
        confidence = attack_prediction.get('confidence', 0.5)
        
        if attack_occurred and predicted_attack:
            reward += 10.0 * confidence  # Correct attack prediction
        elif not attack_occurred and not predicted_attack:
            reward += 5.0  # Correct benign prediction
        elif attack_occurred and not predicted_attack:
            reward -= 15.0  # Missed attack
        else:
            reward -= 5.0  # False alarm
        
        # Defense action effectiveness
        if attack_occurred:
            action_effectiveness = [0.0, 0.7, 0.8, 0.9]  # Effectiveness per action
            reward += 5.0 * action_effectiveness[action]
        else:
            action_penalty = [0.0, -1.0, -2.0, -3.0]  # Penalty for unnecessary actions
            reward += action_penalty[action]
        
        return reward * self.config.reward_scale
    
    def _update_history(self, action: int) -> None:
        """Update state and action history"""
        # Update state history
        self.state_history.append(self.current_network_state.copy())
        if len(self.state_history) > self.config.state_history_length:
            self.state_history.pop(0)
        
        # Update action history
        self.action_history.append(action)
        if len(self.action_history) > self.config.action_history_length:
            self.action_history.pop(0)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if critical attack succeeded without proper defense
        if (self.current_attack_active and 
            self.attack_severity > 0.8 and 
            self.action_history[-1] == 0):  # No defense action
            return True
        
        return False
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # Get attack prediction
        attack_prediction = self._get_attack_prediction()
        
        # Convert attack prediction to numerical format
        pred_array = np.array([
            attack_prediction.get('risk_score', 0.0),
            attack_prediction.get('confidence', 0.0),
            float(attack_prediction.get('is_attack', False)),
            {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}.get(
                attack_prediction.get('severity_level', 'medium'), 0.5
            ),
            hash(attack_prediction.get('attack_category', 'unknown')) % 10 / 10.0,  # Encode category
            max(attack_prediction.get('all_probabilities', {}).values()) if 
            attack_prediction.get('all_probabilities') else 0.0
        ], dtype=np.float32)
        
        return {
            'current_state': self.current_network_state.copy(),
            'state_history': np.array(self.state_history, dtype=np.float32),
            'action_history': np.array(self.action_history, dtype=np.int32),
            'attack_prediction': pred_array
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'attack_active': self.current_attack_active,
            'attack_type': self.attack_type,
            'attack_severity': self.attack_severity,
            'has_real_predictor': self.attack_predictor is not None
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Attack Active: {self.current_attack_active}")
            print(f"Attack Type: {self.attack_type}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"Current State: {self.current_network_state[:5]}...")
            print("-" * 40)
        
        return None
    
    def close(self) -> None:
        """Close the environment"""
        pass