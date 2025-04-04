import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment import IoTEnv
from config_loader import config
import os

def plot_training_results(log_dir):
    """Plot training rewards and episode lengths"""
    results = load_results(log_dir)
    
    # Plot rewards
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['r'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    
    # Plot episode lengths
    plt.subplot(1, 2, 2)
    plt.plot(results['l'])
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_metrics.png'))
    plt.close()

def evaluate_model(model_path, log_dir, num_episodes=10):
    """Evaluate the trained DQN model with VecEnv"""
    # Create vectorized environment
    env = DummyVecEnv([lambda: IoTEnv(config)])
    model = DQN.load(model_path, env=env)

    episode_rewards = []
    action_counts = []
    attack_proximities = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0.0
        ep_action_counts = np.zeros(config.ENVIRONMENT_NUM_ACTIONS)
        ep_attack_proximities = []
        
        while not done[0]:
            # VecEnv returns actions as array of shape (n_envs, action)
            action, _ = model.predict(obs, deterministic=True) # type: ignore
            obs, reward, done, infos = env.step(action)
            
            # For single environment VecEnv, we take first element
            total_reward += reward[0]
            ep_action_counts[action[0]] += 1
            ep_attack_proximities.append(infos[0]['attack_proximity'])
        
        episode_rewards.append(total_reward)
        action_counts.append(ep_action_counts)
        attack_proximities.append(np.mean(ep_attack_proximities))
        
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Rewards
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Rewards')
    
    # Action counts
    plt.subplot(1, 3, 2)
    avg_action_counts = np.mean(action_counts, axis=0)
    plt.bar(['a1', 'a2', 'a3', 'a4'], avg_action_counts)
    plt.xlabel('Action')
    plt.ylabel('Average Count')
    plt.title('Action Distribution')
    
    # Attack proximity
    plt.subplot(1, 3, 3)
    plt.plot(attack_proximities)
    plt.xlabel('Episode')
    plt.ylabel('Attack Proximity')
    plt.title('Attack Proximity Over Episodes')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'evaluation_metrics.png'))
    plt.close()
    
    # Also calculate mean reward using evaluate_policy for consistency
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"\nMean reward from evaluate_policy: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'action_distribution': avg_action_counts,
        'avg_attack_proximity': np.mean(attack_proximities),
        'mean_reward': mean_reward,
        'std_reward': std_reward
    }

# Modifying evaluation.py to work with MLflow
def evaluate_model_with_mlflow(model_path, log_dir, num_episodes=10, experiment_name="model_evaluation"):
    """Evaluate the trained DQN model with VecEnv and log to MLflow"""
    import mlflow
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"evaluation_{os.path.basename(model_path)}"):
        # Log the model being evaluated
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("num_episodes", num_episodes)
        
        # Run the standard evaluation
        results = evaluate_model(model_path, log_dir, num_episodes)
        
        # Log metrics to MLflow
        mlflow.log_metric("avg_reward", float(results['avg_reward']))
        mlflow.log_metric("avg_attack_proximity", float(results['avg_attack_proximity']))
        mlflow.log_metric("mean_reward", float(results['mean_reward']))
        mlflow.log_metric("std_reward", float(results['std_reward']))
        
        # Log action distribution as a parameter (or as separate metrics)
        for i, count in enumerate(results['action_distribution']):
            mlflow.log_metric(f"action_{i+1}_frequency", float(count))
        
        # Log generated plots as artifacts
        mlflow.log_artifact(os.path.join(log_dir, 'evaluation_metrics.png'))
        
        return results

def main():
    log_dir = os.path.join(config.TRAINING_LOGS_DIR, "monitor")
    model_path = os.path.join(config.TRAINING_MODEL_DIR, "final_model.zip")
    
    # Plot training results
    plot_training_results(log_dir)
    
    # Evaluate the trained model with MLflow integration
    results = evaluate_model_with_mlflow(
        model_path=model_path, 
        log_dir=log_dir,
        experiment_name="iot_defense_evaluation"
    )
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Action Distribution: {results['action_distribution']}")
    print(f"Average Attack Proximity: {results['avg_attack_proximity']:.2f}")

if __name__ == "__main__":
    os.makedirs(config.TRAINING_LOGS_DIR, exist_ok=True)
    main()