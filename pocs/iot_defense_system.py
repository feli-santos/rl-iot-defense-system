import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class IoTDefenseEnv(gym.Env):
    """
    A custom Gym environment simulating an IoT network defense scenario.
    
    State: A 10-dimensional vector.
      - state[0] represents the "attack level" (ranging from 0 to 1)
      - state[1:] represent additional network features.
      
    Action space (Discrete 4):
      0: Monitor
      1: Block
      2: Allow
      3: Escalate

    Reward:
      +1 if the chosen action matches the optimal action;
      -1 if it does not.
      
    State Transition:
      - Correct action reduces the attack level.
      - Incorrect action increases the attack level.
    """
    
    def __init__(self):
        super(IoTDefenseEnv, self).__init__()
        # Define observation space: 10-dimensional, values between 0 and 1.
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        # Define action space: 4 discrete actions.
        self.action_space = spaces.Discrete(4)
        self.max_steps = 100
        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Initialize attack level randomly between 0 and 1.
        attack_level = np.random.uniform(0, 1)
        # Other features: initialize with random values.
        other_features = np.random.rand(9)
        self.state = np.concatenate(([attack_level], other_features))
        # Return observation and an empty info dict.
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        attack_level = self.state[0]
        
        # Determine optimal action based on attack level:
        # - Low (<0.3): optimal action is 2 (Allow)
        # - Moderate (0.3 to <0.7): optimal action is 0 (Monitor)
        # - High (>=0.7): optimal action is 1 (Block)
        if attack_level < 0.3:
            correct_action = 2  # Allow
        elif attack_level < 0.7:
            correct_action = 0  # Monitor
        else:
            correct_action = 1  # Block

        # Reward logic: +1 for correct action, -1 for incorrect.
        if action == correct_action:
            reward = 1.0
            # Correct action reduces the attack level.
            delta = np.random.uniform(0.1, 0.3)
            new_attack_level = max(0, attack_level - delta)
        else:
            reward = -1.0
            # Incorrect action increases the attack level.
            delta = np.random.uniform(0.05, 0.2)
            new_attack_level = min(1, attack_level + delta)
        
        # Update the rest of the state with some noise.
        noise = np.random.normal(0, 0.05, 9)
        new_other_features = np.clip(self.state[1:] + noise, 0, 1)
        next_state = np.concatenate(([new_attack_level], new_other_features))
        self.state = next_state
        
        # Episode termination condition.
        done = self.current_step >= self.max_steps
        terminated = done
        truncated = False
        
        # Additional info for debugging.
        info = {"correct_action": correct_action, "attack_level": new_attack_level}
        # Gymnasium step returns (observation, reward, terminated, truncated, info)
        return next_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        # Simple rendering: print current step and attack level.
        print(f"Step: {self.current_step}, Attack Level: {self.state[0]:.2f}, Full State: {self.state}")

if __name__ == "__main__":
    # Create the custom environment.
    env = IoTDefenseEnv()
    
    # Initialize a PPO agent with an MLP policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent for 10,000 timesteps.
    model.learn(total_timesteps=100_000)
    
    # Evaluate the trained agent.
    obs, info = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break
    print("Total reward over the evaluation episode:", total_reward)
