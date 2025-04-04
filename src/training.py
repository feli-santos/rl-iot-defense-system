import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import IoTEnv
from config_loader import config  # Changed from config
from models.attack_predictor import LSTMAttackPredictor  # Changed from policy
from utils.training_data_generator import RealisticAttackDataGenerator  # Changed from data_generator

def train_lstm_attack_predictor():
    """Train the LSTM model to predict attack sequences"""
    print("Training LSTM attack predictor...")
    
    # Instead of random data, use your realistic data generator
    data_gen = RealisticAttackDataGenerator(config.ENVIRONMENT_NUM_STATES)
    
    # Generate more data samples
    num_samples = 5000  
    seq_length = 10
    
    # Create training data
    X_batch, y_batch = data_gen.generate_batch(num_samples, seq_length)
    
    # Split into train/val
    split = int(0.8 * num_samples)
    X_train = X_batch[:split]
    y_train = y_batch[:split]  # Don't select only the last timestep yet
    X_val = X_batch[split:]
    y_val = y_batch[split:]
    
    # Convert 3D tensors to 2D for the LSTM input
    train_data = np.reshape(X_train, (X_train.shape[0], seq_length * X_train.shape[2]))
    val_data = np.reshape(X_val, (X_val.shape[0], seq_length * X_val.shape[2]))
    
    # Use the last timestep for prediction targets
    train_labels = np.argmax(y_train[:, -1], axis=1)  # Get final timestep labels
    val_labels = np.argmax(y_val[:, -1], axis=1)
    
    # Print shapes to validate
    print(f"Train data shape: {train_data.shape}, Train labels shape: {train_labels.shape}")
    print(f"Val data shape: {val_data.shape}, Val labels shape: {val_labels.shape}")
    
    # Initialize and train LSTM model
    lstm_model = LSTMAttackPredictor(config)
    (train_losses, train_accs), (val_losses, val_accs) = lstm_model.train_model(
        train_data, train_labels, val_data, val_labels, epochs=100)
    
    return lstm_model

def train_dqn_policy(lstm_model):
    """Train the DQN defense policy"""
    print("Training DQN defense policy...")
    
    # Create environment
    env = IoTEnv(config)

    # Create log directory for monitor files
    monitor_dir = os.path.join(config.TRAINING_LOGS_DIR, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    # Wrap environment for monitoring and vecNormalize
    env = Monitor(env, monitor_dir)
    env = DummyVecEnv([lambda: env])  # Vectorized environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Define callbacks
    eval_callback = EvalCallback(
        env,
        eval_freq=1000,
        best_model_save_path=os.path.join(config.TRAINING_MODEL_DIR, "best_model"),
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Define DQN model with paper's hyperparameters
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
        tensorboard_log=config.TRAINING_TENSORBOARD_DIR
    )

    # Train the model
    model.learn(
        total_timesteps=config.DQN_TOTAL_EPISODES * config.DQN_EPOCHS_PER_EPISODE,
        callback=[eval_callback, ProgressBarCallback()],
        log_interval=10
    )
    
    # Save the trained model
    model.save(os.path.join(config.TRAINING_MODEL_DIR, "final_model"))
    env.save(os.path.join(config.TRAINING_MODEL_DIR, "environment"))
    
    return model, env

def main():
    # Create directories
    os.makedirs(config.TRAINING_MODEL_DIR, exist_ok=True)
    
    # Step 1: Train LSTM attack predictor
    lstm_model = train_lstm_attack_predictor()
    
    # Step 2: Train DQN defense policy
    #dqn_model = train_dqn_policy(lstm_model)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()