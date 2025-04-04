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
    # Create vectorized environment with same normalization as training
    env = DummyVecEnv([lambda: IoTEnv(config)])
    
    # Try to load model with more robust error handling
    try:
        print(f"Loading model from: {model_path}")
        model = DQN.load(model_path, env=env)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    episode_rewards = []
    action_counts = []
    attack_proximities = []
    
    for ep in range(num_episodes):
        # Reset with better error handling for different gym/SB3 versions
        try:
            obs = env.reset()
            # Handle different reset() return types
            if isinstance(obs, tuple):
                obs = obs[0]  # Extract observation if reset returns (obs, info)
        except Exception as e:
            print(f"Error during reset: {e}")
            raise
            
        done = False
        total_reward = 0.0
        ep_action_counts = np.zeros(config.ENVIRONMENT_NUM_ACTIONS)
        ep_attack_proximities = []
        
        step_count = 0
        
        while not done:
            # Predict action with error handling
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception as e:
                print(f"Error during prediction (step {step_count}): {e}")
                print(f"Observation shape: {np.array(obs).shape}")
                raise
                
            # Step environment with error handling
            try:
                obs, reward, terminated, info = env.step(action)
                
                # Handle different step() return types (SB3 compatibility)
                if isinstance(terminated, tuple):
                    terminated, truncated = terminated
                    done = terminated or truncated
                else:
                    done = terminated[0]  # Extract boolean from array
                    
                # Extract reward safely
                reward_value = reward[0] if hasattr(reward, "__getitem__") else reward
                total_reward += reward_value
                
                # Extract and record action
                action_idx = action[0] if hasattr(action, "__getitem__") else action
                ep_action_counts[action_idx] += 1
                
                # Extract proximity data if available
                if isinstance(info, list) and len(info) > 0:
                    if 'attack_proximity' in info[0]:
                        ep_attack_proximities.append(info[0]['attack_proximity'])
            except Exception as e:
                print(f"Error during step {step_count}: {e}")
                raise
                
            step_count += 1
            
            # Safety exit for very long episodes
            if step_count > 1000:
                print(f"Warning: Episode {ep+1} exceeded 1000 steps, terminating")
                break
        
        episode_rewards.append(total_reward)
        action_counts.append(ep_action_counts)
        
        if ep_attack_proximities:
            attack_proximities.append(np.mean(ep_attack_proximities))
        
        print(f"Episode {ep+1}: Steps={step_count}, Reward={total_reward:.2f}")
    
    # Generate plot with better error handling
    try:
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
        action_labels = [f'a{i+1}' for i in range(len(avg_action_counts))]
        plt.bar(action_labels, avg_action_counts)
        plt.xlabel('Action')
        plt.ylabel('Average Count')
        plt.title('Action Distribution')
        
        # Attack proximity
        plt.subplot(1, 3, 3)
        if attack_proximities:
            plt.plot(attack_proximities)
            plt.xlabel('Episode')
            plt.ylabel('Attack Proximity')
            plt.title('Attack Proximity Over Episodes')
        else:
            plt.text(0.5, 0.5, "No proximity data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Attack Proximity (No Data)')
        
        plt.tight_layout()
        plot_path = os.path.join(log_dir, 'evaluation_metrics.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved evaluation plot to: {plot_path}")
    except Exception as e:
        print(f"Error generating plots: {e}")
        
    # Also calculate mean reward using evaluate_policy for consistency
    try:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
        print(f"\nMean reward from evaluate_policy: {mean_reward:.2f} +/- {std_reward:.2f}")
    except Exception as e:
        print(f"Error during evaluate_policy: {e}")
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        print(f"Falling back to manual calculation: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return {
        'avg_reward': float(np.mean(episode_rewards)),
        'action_distribution': avg_action_counts.tolist(),
        'avg_attack_proximity': float(np.mean(attack_proximities)) if attack_proximities else 0.0,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward)
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
    """Main function to run evaluation"""
    # Update paths to match the current folder structure from TrainingManager
    
    # Find the latest run directory in artifacts
    artifacts_dir = "./artifacts"
    if not os.path.exists(artifacts_dir):
        print(f"Error: Artifacts directory '{artifacts_dir}' not found!")
        return
        
    # Get the most recent run directory
    run_dirs = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
    if not run_dirs:
        print(f"Error: No run directories found in '{artifacts_dir}'!")
        return
        
    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(artifacts_dir, x)), reverse=True)
    latest_run = os.path.join(artifacts_dir, run_dirs[0])
    
    print(f"Using latest run: {latest_run}")
    
    # Update paths for the latest run
    monitor_dir = os.path.join(latest_run, "logs", "monitor")
    models_dir = os.path.join(latest_run, "models")
    
    # Check if model exists
    model_path = os.path.join(models_dir, "dqn_final.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Create output directory
    evaluation_dir = os.path.join(latest_run, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Check if monitor files exist
    if not os.path.exists(monitor_dir) or not any(f.endswith('.csv') for f in os.listdir(monitor_dir)):
        print(f"Warning: No monitor files found in '{monitor_dir}'")
        print("Skipping training plot generation")
    else:
        # Plot training results
        plot_training_results(monitor_dir)
        
    # Evaluate the trained model with MLflow integration
    results = evaluate_model_with_mlflow(
        model_path=model_path, 
        log_dir=evaluation_dir,
        experiment_name="iot_defense_evaluation"
    )
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Action Distribution: {results['action_distribution']}")
    print(f"Average Attack Proximity: {results['avg_attack_proximity']:.2f}")

if __name__ == "__main__":
    os.makedirs(config.TRAINING_LOGS_DIR, exist_ok=True)
    main()