import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import IoTEnv
from config_loader import config
from models.attack_predictor import LSTMAttackPredictor
from utils.data_generator import RealisticAttackDataGenerator
from utils.training_manager import TrainingManager
import mlflow
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import warnings
import logging
from tqdm import tqdm

# Suppress MLflow warnings
warnings.filterwarnings("ignore", message=".*Model logged without a signature and input example.*")
warnings.filterwarnings("ignore", message=".*Encountered an unexpected error while inferring pip requirements.*")
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

class MLflowCallback(BaseCallback):
    """Custom callback for logging to MLflow during RL training"""
    
    def __init__(self, training_manager: TrainingManager, verbose: int = 0):
        super(MLflowCallback, self).__init__(verbose)
        self.training_manager = training_manager
        self.step_count = 0
        
    def _on_step(self) -> bool:
        # Log every 10 steps
        if self.step_count % 10 == 0:
            # Log episode reward and length
            if len(self.model.ep_info_buffer) > 0:
                ep_infos = self.model.ep_info_buffer[-1]
                metrics = {
                    "episode_reward": ep_infos.get('r', 0),
                    "episode_length": ep_infos.get('l', 0)
                }
                self.training_manager.log_metrics(metrics, self.step_count)
            
        self.step_count += 1
        return True
    
    def _on_training_end(self) -> None:
        # Save the final model
        model_path = os.path.join(self.training_manager.models_path, "final_model.zip")
        self.model.save(model_path)
        mlflow.log_artifact(model_path, "models")

def train_lstm_attack_predictor(training_manager: TrainingManager):
    """Train the LSTM model to predict attack sequences with better tracking"""
    print("Training LSTM attack predictor...")
    
    with training_manager.start_run(run_name="lstm_training"):
        # Generate data
        data_gen = RealisticAttackDataGenerator(config.ENVIRONMENT_NUM_STATES)
        num_samples = 5000
        seq_length = 10
        
        # Log data generation parameters
        training_manager.log_metrics({
            "num_samples": num_samples,
            "seq_length": seq_length
        })
        
        # Create training data
        X_batch, y_batch = data_gen.generate_batch(num_samples, seq_length)
        
        # Split into train/val
        split = int(0.8 * num_samples)
        X_train = X_batch[:split]
        y_train = y_batch[:split]
        X_val = X_batch[split:]
        y_val = y_batch[split:]
        
        # Convert 3D tensors to 2D for the LSTM input
        train_data = np.reshape(X_train, (X_train.shape[0], seq_length * X_train.shape[2]))
        val_data = np.reshape(X_val, (X_val.shape[0], seq_length * X_val.shape[2]))
        
        # Use the last timestep for prediction targets
        train_labels = np.argmax(y_train[:, -1], axis=1)
        val_labels = np.argmax(y_val[:, -1], axis=1)
        
        # Print shapes to validate
        print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
        print(f"Val data shape: {val_data.shape}, Val labels shape: {val_labels.shape}")
        
        # Initialize and train LSTM model
        lstm_model = LSTMAttackPredictor(config)
        
        # For tracking metrics during training
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        # Train for specified epochs
        epochs = 100
        batch_size = 32
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.LongTensor(train_data),
                torch.LongTensor(train_labels)
            ),
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.LongTensor(val_data),
                torch.LongTensor(val_labels)
            ),
            batch_size=batch_size, 
            shuffle=False
        )
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        
        # Add progress bar to your training loops
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training phase
            lstm_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                lstm_model.optimizer.zero_grad()
                outputs = lstm_model(inputs)
                loss = lstm_model.criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), config.LSTM_GRADIENT_CLIP_NORM)
                
                lstm_model.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            lstm_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = lstm_model(inputs)
                    loss = lstm_model.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss_avg = val_loss / len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc)
            
            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss_avg,
                "val_acc": val_acc,
                "epoch": epoch + 1
            }
            training_manager.log_metrics(metrics, epoch + 1)
            
            # Check for best model
            is_best_loss = training_manager.save_best_model(lstm_model, val_loss_avg, "val_loss", "min")
            is_best_acc = training_manager.save_best_model(lstm_model, val_acc, "val_acc", "max")
            
            # Output progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.4f}")
                print(f"Best Val Loss: {min(val_losses):.4f}, Best Val Acc: {max(val_accs):.4f}")
                print("------------------------")
        
        # Plot training curves
        training_manager.plot_training_curves(
            {"loss": train_losses, "accuracy": train_accs},
            {"loss": val_losses, "accuracy": val_accs},
            "LSTM Training Metrics"
        )
        
        # Save final model
        final_model_path = training_manager.log_model(lstm_model, "lstm_final")
        print(f"Final model saved to: {final_model_path}")
        
        return lstm_model

def train_dqn_policy(lstm_model, training_manager: TrainingManager):
    """Train the DQN defense policy with enhanced monitoring"""
    print("Training DQN defense policy...")
    
    with training_manager.start_run(run_name="dqn_training", nested=True):
        # Create log directory for monitor files
        monitor_dir = os.path.join(training_manager.logs_path, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        
        # Create environment
        env = IoTEnv(config)
        env = Monitor(env, monitor_dir)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Log environment parameters
        training_manager.log_metrics({
            "num_devices": config.ENVIRONMENT_NUM_DEVICES,
            "num_actions": config.ENVIRONMENT_NUM_ACTIONS,
            "num_states": config.ENVIRONMENT_NUM_STATES
        })
        
        # Custom callback for MLflow logging
        mlflow_callback = MLflowCallback(training_manager)
        
        # Define DQN model
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate=config.DQN_LEARNING_RATE,
            buffer_size=config.DQN_BUFFER_SIZE,
            learning_starts=1000,
            batch_size=config.DQN_BATCH_SIZE,
            tau=config.DQN_TAU,
            gamma=config.DQN_GAMMA,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=config.DQN_TARGET_UPDATE_FREQ,
            exploration_fraction=0.1,
            exploration_initial_eps=config.EXPLORATION_EPS_START,
            exploration_final_eps=config.EXPLORATION_EPS_END,
            policy_kwargs={
                "net_arch": config.NETWORK_HIDDEN_LAYERS,
                "activation_fn": torch.nn.ReLU
            },
            verbose=config.TRAINING_VERBOSE,
            seed=config.TRAINING_SEED,
            device=config.TRAINING_DEVICE,
            tensorboard_log=training_manager.logs_path
        )
        
        # Train the model
        model.learn(
            total_timesteps=config.DQN_TOTAL_EPISODES * config.DQN_EPOCHS_PER_EPISODE,
            callback=mlflow_callback,
            log_interval=10
        )
        
        # Save the trained model
        model_path = os.path.join(training_manager.models_path, "dqn_final.zip")
        model.save(model_path)
        env_path = os.path.join(training_manager.models_path, "environment.pkl")
        env.save(env_path)
        
        # Log artifacts
        mlflow.log_artifact(model_path, "models")
        mlflow.log_artifact(env_path, "models")
        
        return model, env

def main():
    # Create the training manager
    training_manager = TrainingManager(
        experiment_name="iot_defense_system",
        base_artifact_path="./artifacts",
        config=config
    )
    
    # Step 1: Train LSTM attack predictor with enhanced monitoring
    lstm_model = train_lstm_attack_predictor(training_manager)
    
    # Step 2: Train DQN defense policy with enhanced monitoring
    dqn_model, env = train_dqn_policy(lstm_model, training_manager)
    
    # Close the training manager
    training_manager.end_run()
    
    print("Training completed successfully!")
    print(f"All artifacts saved to: {training_manager.run_artifact_path}")

if __name__ == "__main__":
    # Install MLflow if not already installed
    try:
        import mlflow
    except ImportError:
        print("Installing MLflow...")
        import subprocess
        subprocess.check_call(["pip", "install", "mlflow"])
        import mlflow
    
    main()