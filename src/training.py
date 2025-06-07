import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import IoTEnv
from config_loader import config
from models.attack_predictor import LSTMAttackPredictor
from utils.data_generator import RealisticAttackDataGenerator
from utils.training_manager import TrainingManager
from algorithms.algorithm_factory import AlgorithmFactory
from benchmarking.benchmark_runner import BenchmarkRunner
import mlflow
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import warnings
import logging
from tqdm import tqdm
import argparse
from typing import Optional, List

# Suppress MLflow warnings
warnings.filterwarnings("ignore", message=".*Model logged without a signature and input example.*")
warnings.filterwarnings("ignore", message=".*Encountered an unexpected error while inferring pip requirements.*")
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

class MLflowCallback(BaseCallback):
    """Custom callback for logging to MLflow during RL training"""
    
    def __init__(self, training_manager: TrainingManager, verbose=0):
        super(MLflowCallback, self).__init__(verbose)
        self.training_manager = training_manager
        
    def _on_step(self) -> bool:
        # Log metrics every 100 steps
        if self.n_calls % 100 == 0:
            # Get info from the logger if available
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                self.training_manager.log_metrics({
                    "episode_reward": ep_info.get("r", 0),
                    "episode_length": ep_info.get("l", 0),
                    "step": self.n_calls
                }, step=self.n_calls)
        return True


def train_lstm_attack_predictor(training_manager: TrainingManager):
    """Train the LSTM model to predict attack sequences with better tracking"""
    print("Training LSTM attack predictor...")
    
    with training_manager.start_run(run_name="lstm_training", nested=True):
        # Create and train LSTM model
        lstm_model = LSTMAttackPredictor(config)
        
        # Train with progress tracking
        print("Training LSTM model...")
        train_metrics, val_metrics = lstm_model.train_model(epochs=100, batch_size=32)
        
        # Log training metrics - FIX: Use string keys instead of integer indices
        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
            train_metrics['loss'], train_metrics['accuracy'], 
            val_metrics['loss'], val_metrics['accuracy']
        )):
            training_manager.log_metrics({
                "lstm_train_loss": train_loss,
                "lstm_train_acc": train_acc,
                "lstm_val_loss": val_loss,
                "lstm_val_acc": val_acc
            }, step=epoch)
            
        # Save model with proper tracking
        model_path = training_manager.log_model(lstm_model, "lstm_attack_predictor")
        
        # Create and log training curves - FIX: Use string keys
        fig = training_manager.plot_training_curves(
            train_metrics={"loss": train_metrics['loss'], "accuracy": train_metrics['accuracy']},
            val_metrics={"loss": val_metrics['loss'], "accuracy": val_metrics['accuracy']},
            title="LSTM Training Curves"
        )
        
        # Track best models - FIX: Use string keys
        best_val_loss_idx = np.argmin(val_metrics['loss'])
        best_val_acc_idx = np.argmax(val_metrics['accuracy'])
        
        training_manager.save_best_model(
            lstm_model, 
            val_metrics['loss'][best_val_loss_idx], 
            "val_loss", 
            mode="min"
        )
        
        training_manager.save_best_model(
            lstm_model, 
            val_metrics['accuracy'][best_val_acc_idx], 
            "val_acc", 
            mode="max"
        )
        
        print(f"LSTM training completed. Model saved to: {model_path}")
        print(f"Best validation loss: {min(val_metrics['loss']):.4f}")
        print(f"Best validation accuracy: {max(val_metrics['accuracy']):.4f}")
        
        return lstm_model


def train_single_algorithm(algorithm_name: str, lstm_model: LSTMAttackPredictor, 
                         training_manager: TrainingManager):
    """Train a single RL algorithm"""
    print(f"Training {algorithm_name} defense policy...")
    
    with training_manager.start_run(run_name=f"{algorithm_name.lower()}_training", nested=True):
        # Create algorithm instance
        algorithm = AlgorithmFactory.create_algorithm(algorithm_name, config)
        
        # Log algorithm hyperparameters as parameters (not metrics)
        hyperparams = algorithm.get_hyperparameters()
        
        # Log hyperparameters properly
        for key, value in hyperparams.items():
            if isinstance(value, (list, tuple)):
                # Log lists as parameters with string representation
                mlflow.log_param(key, str(value))
                # Also log useful derived metrics
                if 'layers' in key.lower() or 'units' in key.lower():
                    mlflow.log_param(f"{key}_count", len(value))
            elif isinstance(value, (dict,)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)
        
        # Create environment
        env = create_training_environment(training_manager)
        
        # Log environment parameters as metrics (these are scalar)
        env_metrics = {
            "num_devices": config.ENVIRONMENT_NUM_DEVICES,
            "num_actions": config.ENVIRONMENT_NUM_ACTIONS,
            "num_states": config.ENVIRONMENT_NUM_STATES
        }
        training_manager.log_metrics(env_metrics)
        
        # Create and train model
        model = algorithm.create_model(env, training_manager)
        trained_model = algorithm.train(model, training_manager)
        
        # Save the trained model
        model_path = os.path.join(training_manager.models_path, f"{algorithm_name.lower()}_final.zip")
        algorithm.save_model(trained_model, model_path)
        
        # REMOVED: Save environment - VecEnv doesn't support save()
        # Instead, save environment configuration
        env_config_path = os.path.join(training_manager.models_path, "environment_config.json")
        env_config = {
            "environment_type": "IoTEnv",
            "num_devices": config.ENVIRONMENT_NUM_DEVICES,
            "num_actions": config.ENVIRONMENT_NUM_ACTIONS,
            "num_states": config.ENVIRONMENT_NUM_STATES,
            "history_length": config.ENVIRONMENT_HISTORY_LENGTH,
            "observation_space_type": str(type(env.envs[0].env.observation_space)),
            "action_space_type": str(type(env.envs[0].env.action_space))
        }
        
        import json
        with open(env_config_path, 'w') as f:
            json.dump(env_config, f, indent=2)
        
        # Log artifacts
        mlflow.log_artifact(model_path, "models")
        mlflow.log_artifact(env_config_path, "models")
        
        print(f"{algorithm_name} training completed!")
        return trained_model, env


def create_training_environment(training_manager: TrainingManager):
    """Create and configure the training environment"""
    # Create log directory for monitor files
    monitor_dir = os.path.join(training_manager.logs_path, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Create base environment
    base_env = IoTEnv(config)
    
    # Verify environment is created properly
    print(f"Environment created: {type(base_env)}")
    print(f"Observation space: {base_env.observation_space}")
    print(f"Action space: {base_env.action_space}")
    
    # Wrap with Monitor for logging
    env = Monitor(base_env, monitor_dir)
    
    # For Stable Baselines3 algorithms (PPO, SAC), wrap in VecEnv
    env = DummyVecEnv([lambda: env])
    
    # Optional: normalize observations and rewards
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    return env


def run_benchmark_comparison(lstm_model: LSTMAttackPredictor):
    """Run comprehensive benchmark comparison"""
    print("\n" + "="*60)
    print("STARTING ALGORITHM BENCHMARK COMPARISON")
    print("="*60)
    
    # Create benchmark runner
    benchmark_runner = BenchmarkRunner(config, lstm_model)
    
    # Run benchmark
    algorithms_to_compare = config.ALGORITHM_ALGORITHMS_TO_COMPARE
    num_runs = config.BENCHMARKING_NUM_RUNS
    
    metrics_collector = benchmark_runner.run_benchmark(
        algorithms=algorithms_to_compare,
        num_runs=num_runs
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    comparison_data = metrics_collector.get_comparison_data()
    for algorithm_name, summary in comparison_data.items():
        print(f"\n{algorithm_name}:")
        print(f"  Average Reward: {summary.get('avg_reward_mean', 0):.3f} ± {summary.get('avg_reward_std', 0):.3f}")
        print(f"  Training Time: {summary.get('training_time_mean', 0):.2f}s ± {summary.get('training_time_std', 0):.2f}s")
        print(f"  Number of Runs: {summary.get('num_runs', 0)}")
    
    return metrics_collector


def main():
    """Main training function with command-line interface"""
    parser = argparse.ArgumentParser(description='IoT Defense System Training')
    
    # Algorithm selection
    parser.add_argument('--algorithm', type=str, 
                       choices=['DQN', 'PPO', 'A2C', 'ALL'],
                       help='Algorithm to train (overrides config)')
    
    # Benchmark options
    parser.add_argument('--algorithms', nargs='+',
                       choices=['DQN', 'PPO', 'A2C'],
                       help='Algorithms for benchmark mode')
    parser.add_argument('--runs', type=int,
                       help='Number of runs per algorithm (overrides config)')
    
    # Training options
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM training (use existing model)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing benchmark results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle analyze-only mode
    if args.analyze_only:
        from benchmarking.metrics_collector import MetricsCollector
        from benchmarking.benchmark_analyzer import BenchmarkAnalyzer
        
        print("Analyzing existing benchmark results...")
        metrics_collector = MetricsCollector()
        metrics_collector.load_results()
        
        analyzer = BenchmarkAnalyzer(metrics_collector)
        analyzer.generate_comparison_report()
        return
    
    # Create training manager
    training_manager = TrainingManager(
        experiment_name="iot_defense_system",
        base_artifact_path="./artifacts",
        config=config
    )
    
    try:
        # Handle LSTM training
        if not args.skip_lstm:
            lstm_model = train_lstm_attack_predictor(training_manager)
        else:
            print("Skipping LSTM training - using existing model")
            lstm_model = LSTMAttackPredictor(config)
            # In practice, load pre-trained weights here
        
        # Determine algorithm mode
        if args.algorithm:
            algorithm_type = args.algorithm.upper()
        else:
            algorithm_type = config.ALGORITHM_TYPE.upper()
        
        # Override config with command-line args
        if args.algorithms:
            algorithms_to_compare = args.algorithms
        else:
            algorithms_to_compare = config.ALGORITHM_ALGORITHMS_TO_COMPARE
            
        if args.runs:
            num_runs = args.runs
        else:
            num_runs = config.BENCHMARKING_NUM_RUNS
        
        # Execute based on algorithm type
        if algorithm_type == "ALL" or args.algorithms:
            print(f"Running benchmark with algorithms: {algorithms_to_compare}")
            print(f"Number of runs per algorithm: {num_runs}")
            
            # Run benchmark with overridden parameters
            benchmark_runner = BenchmarkRunner(config, lstm_model)
            metrics_collector = benchmark_runner.run_benchmark(
                algorithms=algorithms_to_compare,
                num_runs=num_runs
            )
            
            # Analyze results
            from benchmarking.benchmark_analyzer import BenchmarkAnalyzer
            analyzer = BenchmarkAnalyzer(metrics_collector)
            analyzer.generate_comparison_report()
            
        elif algorithm_type in AlgorithmFactory.get_available_algorithms():
            # Train single algorithm
            trained_model, env = train_single_algorithm(algorithm_type, lstm_model, training_manager)
            print(f"Single algorithm training completed: {algorithm_type}")
            
        else:
            available_algorithms = AlgorithmFactory.get_available_algorithms()
            raise ValueError(f"Invalid algorithm type: {algorithm_type}. Available: {available_algorithms + ['ALL']}")
            
    finally:
        training_manager.end_run()
    
    print("Training completed successfully!")
    print(f"All artifacts saved to: {training_manager.run_artifact_path}")


if __name__ == "__main__":    
    main()