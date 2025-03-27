import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import IoTEnv
from config import config
from policy import LSTMAttackPredictor

def train_lstm_attack_predictor():
    """Train the LSTM model to predict attack sequences"""
    print("Training LSTM attack predictor...")
    
    # Simulate loading PEEVES dataset (in practice, you would load real data)
    num_samples = 1000
    seq_length = 10
    num_events = 227  # From paper
    
    # Generate synthetic training data (replace with real data)
    train_data = np.random.randint(0, num_events, size=(int(num_samples*0.8), seq_length))
    train_labels = np.random.randint(0, config.LSTM_OUTPUT_CLASSES, size=int(num_samples*0.8))
    
    val_data = np.random.randint(0, num_events, size=(int(num_samples*0.2), seq_length))
    val_labels = np.random.randint(0, config.LSTM_OUTPUT_CLASSES, size=int(num_samples*0.2))
    
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
    env = Monitor(env)  # Wrap for logging
    env = DummyVecEnv([lambda: env])  # Vectorized environment
    
    # Define callbacks
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10)
    eval_callback = EvalCallback(
        env,
        callback_after_eval=stop_callback,
        eval_freq=1000,
        best_model_save_path=os.path.join(config.TRAINING_LOG_DIR, "best_model"),
        deterministic=True,
        render=False
    )
    
    # Define DQN model with paper's hyperparameters
    model = DQN(
        "MlpPolicy",
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
        device=config.TRAINING_DEVICE
    )
    
    # Train the model
    model.learn(
        total_timesteps=config.DQN_TOTAL_EPISODES * config.DQN_EPOCHS_PER_EPISODE,
        callback=eval_callback,
        log_interval=10
    )
    
    # Save the trained model
    model.save(os.path.join(config.TRAINING_LOG_DIR, "dqn_iot_warden"))
    
    return model

def main():
    # Create directories
    os.makedirs(config.TRAINING_LOG_DIR, exist_ok=True)
    
    # Step 1: Train LSTM attack predictor
    lstm_model = train_lstm_attack_predictor()
    
    # Step 2: Train DQN defense policy
    dqn_model = train_dqn_policy(lstm_model)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()