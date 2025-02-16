import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class IoTDefenseEnv(gym.Env):
    """
    A custom Gym environment simulating an IoT network defense scenario.
    
    State: An 11-dimensional vector.
      - state[0]: "attack level" (a float between 0 and 1).
      - state[1]: "attack progress" (a cumulative measure of attack advancement, 0 to 1).
      - state[2:]: 9 additional network features.
      
    Action space (Discrete 4):
      0: Monitor
      1: Block
      2: Allow
      3: Escalate

    Reward:
      - Base reward: +1 for a correct (optimal) defensive action, -1 for an incorrect action.
      - Additional reward is given based on how much the "attack progress" is reduced (or increased).
      - In "defense" mode, only actions 0 (Monitor) and 1 (Block) are allowed. Other actions incur extra penalty.

    State Transition:
      - Correct action reduces both the attack level and the attack progress.
      - Incorrect action increases both.
    """
    
    def __init__(self, mode="defense", strict_threshold=0.8):
        super(IoTDefenseEnv, self).__init__()
        # The state is 11-dimensional: [attack_level, attack_progress, 9 other features]
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.max_steps = 100
        self.current_step = 0
        self.mode = mode  # "defense" or "strict"
        self.strict_threshold = strict_threshold  # Threshold to switch to strict mode.
        self.state = None
        self.attack_progress = 0.0  # Initialize attack progress

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Initialize attack level randomly between 0 and 1.
        attack_level = np.random.uniform(0, 1)
        # Reset attack progress.
        self.attack_progress = 0.0
        # Other features: initialize with random values.
        other_features = np.random.rand(9)
        # State: [attack_level, attack_progress, other_features]
        self.state = np.concatenate(([attack_level, self.attack_progress], other_features))
        self.mode = "defense"  # Start in defense mode.
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        attack_level = self.state[0]
        old_progress = self.attack_progress
        
        # Determine the optimal action based on attack level:
        # - Low (<0.3): optimal action is 2 (Allow)
        # - Moderate (0.3 to <0.7): optimal action is 0 (Monitor)
        # - High (>=0.7): optimal action is 1 (Block)
        if attack_level < 0.3:
            correct_action = 2  # Allow
        elif attack_level < 0.7:
            correct_action = 0  # Monitor
        else:
            correct_action = 1  # Block
        
        # Dynamic mode switching: if attack progress exceeds threshold, switch to "strict" mode.
        if self.attack_progress > self.strict_threshold:
            self.mode = "strict"
        else:
            self.mode = "defense"
            
        extra_penalty = 0.0
        # In defense or strict mode, only actions 0 (Monitor) and 1 (Block) are allowed.
        if self.mode in ["defense", "strict"]:
            allowed_actions = [0, 1]
            if action not in allowed_actions:
                extra_penalty = 2.0  # Penalty for disallowed actions.
        
        # Base reward logic.
        if action == correct_action:
            base_reward = 1.0
            # Correct action reduces the attack level.
            delta_level = np.random.uniform(0.1, 0.3)
            new_attack_level = max(0, attack_level - delta_level)
            # Correct action reduces attack progress.
            delta_progress = np.random.uniform(0.05, 0.2)
            new_progress = max(0, self.attack_progress - delta_progress)
        else:
            base_reward = -1.0
            # Incorrect action increases the attack level.
            delta_level = np.random.uniform(0.05, 0.2)
            new_attack_level = min(1, attack_level + delta_level)
            # Incorrect action increases attack progress.
            delta_progress = np.random.uniform(0.05, 0.2)
            new_progress = min(1, self.attack_progress + delta_progress)
        
        # Update attack progress.
        self.attack_progress = new_progress
        
        # Adjust reward based on the change in attack progress.
        progress_change = old_progress - new_progress
        reward = base_reward + progress_change - extra_penalty

        # Update other features with noise.
        noise = np.random.normal(0, 0.05, 9)
        new_other_features = np.clip(self.state[2:] + noise, 0, 1)
        # Form new state.
        next_state = np.concatenate(([new_attack_level, self.attack_progress], new_other_features))
        self.state = next_state
        
        done = self.current_step >= self.max_steps
        terminated = done
        truncated = False
        
        info = {
            "correct_action": correct_action,
            "attack_level": new_attack_level,
            "attack_progress": self.attack_progress,
            "mode": self.mode
        }
        return next_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Attack Level: {self.state[0]:.2f}, Attack Progress: {self.state[1]:.2f}, Mode: {self.mode}")
        print(f"Full State: {self.state}")

# Custom callback for logging additional metrics.
class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.attack_levels = []  # Record attack levels.
        self.rewards = []        # Record rewards.

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None and len(infos) > 0:
            info = infos[0]
            attack_level = info.get("attack_level")
            if attack_level is not None:
                self.logger.record("custom/attack_level", attack_level)
                self.attack_levels.append(attack_level)
        reward = self.locals.get("reward")
        if reward is not None:
            self.rewards.append(reward)
        return True

if __name__ == "__main__":
    # Create the environment.
    env = IoTDefenseEnv(mode="defense", strict_threshold=0.8)
    
    # Specify TensorBoard log directory.
    tensorboard_log = "./tensorboard/"
    
    # Create the logging callback.
    callback = LoggingCallback(verbose=1)
    
    # Initialize PPO agent using an LSTM-based policy for temporal modeling.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
    
    # Train the agent.
    model.learn(total_timesteps=100_000, callback=callback)
    
    # Evaluation: run an evaluation episode.
    obs, info = env.reset()
    total_reward = 0
    eval_attack_levels = []
    eval_rewards = []
    steps = []
    
    for step in range(100):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        eval_attack_levels.append(info["attack_level"])
        eval_rewards.append(reward)
        steps.append(step)
        env.render()
        if terminated or truncated:
            break

    print("Total reward over the evaluation episode:", total_reward)
    
    # Visualization of attack levels and rewards.
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, eval_attack_levels, label="Attack Level", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Attack Level")
    plt.title("Attack Level over Steps")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(steps, eval_rewards, label="Reward per Step", color="orange", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward over Steps")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
