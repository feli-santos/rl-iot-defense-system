"""
Benchmark Runner

Runs benchmarking experiments across multiple RL algorithms.
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Use absolute imports
from algorithms.algorithm_factory import AlgorithmFactory
from utils.training_manager import TrainingManager
from environment import IoTEnv
from models.attack_predictor import LSTMAttackPredictor
from benchmarking.metrics_collector import MetricsCollector
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy


class BenchmarkRunner:
    """Runs benchmarking experiments for multiple RL algorithms"""
    
    def __init__(self, config: Any, lstm_model: LSTMAttackPredictor):
        self.config = config
        self.lstm_model = lstm_model
        self.metrics_collector = MetricsCollector()
        
    def run_benchmark(self, algorithms: Optional[List[str]] = None, 
                     num_runs: int = 3) -> MetricsCollector:
        """
        Run benchmark comparison across multiple algorithms
        
        Args:
            algorithms: List of algorithm names to benchmark. If None, uses config.
            num_runs: Number of independent runs per algorithm
            
        Returns:
            MetricsCollector: Collected metrics from all runs
        """
        if algorithms is None:
            algorithms = self.config.ALGORITHM_ALGORITHMS_TO_COMPARE
            
        print(f"Starting benchmark with algorithms: {algorithms}")
        print(f"Number of runs per algorithm: {num_runs}")
        
        # Create base training manager for experiment tracking
        base_training_manager = TrainingManager(
            experiment_name="algorithm_benchmark",
            base_artifact_path="./artifacts/benchmark",
            config=self.config
        )
        
        try:
            base_training_manager.start_run(run_name="benchmark_comparison")
            
            # Log benchmark configuration  
            base_training_manager.log_metrics({
                "num_algorithms": len(algorithms),
                "num_runs_per_algorithm": num_runs,
            })
            
            # Run experiments for each algorithm
            for algorithm_name in algorithms:
                print(f"\n{'='*50}")
                print(f"Benchmarking {algorithm_name}")
                print(f"{'='*50}")
                
                self._run_algorithm_benchmark(
                    algorithm_name, 
                    num_runs, 
                    base_training_manager
                )
                
            # Save final results
            self.metrics_collector.save_results()
            
            # Log comparison summary
            comparison_data = self.metrics_collector.get_comparison_data()
            for alg_name, summary in comparison_data.items():
                base_training_manager.log_metrics({
                    f"{alg_name}_avg_reward_mean": summary.get('avg_reward_mean', 0),
                    f"{alg_name}_avg_reward_std": summary.get('avg_reward_std', 0),
                    f"{alg_name}_training_time_mean": summary.get('training_time_mean', 0)
                })
                
        finally:
            base_training_manager.end_run()
            
        return self.metrics_collector
        
    def _run_algorithm_benchmark(self, algorithm_name: str, num_runs: int, 
                               parent_training_manager: TrainingManager) -> None:
        """Run benchmark for a specific algorithm"""
        
        for run_id in range(num_runs):
            print(f"\nRun {run_id + 1}/{num_runs} for {algorithm_name}")
            
            # Create algorithm instance
            try:
                algorithm = AlgorithmFactory.create_algorithm(algorithm_name, self.config)
            except ValueError as e:
                print(f"Error creating algorithm {algorithm_name}: {e}")
                continue
                
            # Create training manager for this specific run
            run_training_manager = TrainingManager(
                experiment_name=f"benchmark_{algorithm_name.lower()}",
                base_artifact_path=f"./artifacts/benchmark/{algorithm_name.lower()}",
                config=self.config
            )
            
            try:
                run_training_manager.start_run(run_name=f"{algorithm_name.lower()}_run_{run_id}")
                
                # Start metrics collection
                self.metrics_collector.start_run(
                    algorithm_name, 
                    run_id, 
                    algorithm.get_hyperparameters()
                )
                
                # Train the algorithm
                start_time = time.time()
                trained_model, env = self._train_algorithm(algorithm, run_training_manager)
                training_time = time.time() - start_time
                
                # Evaluate the algorithm
                evaluation_results = self._evaluate_algorithm(
                    algorithm_name, 
                    trained_model, 
                    env, 
                    run_training_manager
                )
                
                # Update metrics
                self.metrics_collector.update_training_metrics(
                    algorithm_name, 
                    run_id, 
                    training_time
                )
                
                self.metrics_collector.update_evaluation_metrics(
                    algorithm_name, 
                    run_id, 
                    evaluation_results
                )
                
                # Save model
                model_path = run_training_manager.models_path / f"{algorithm_name.lower()}_final.zip"
                algorithm.save_model(trained_model, str(model_path))
                
                print(f"Completed {algorithm_name} run {run_id + 1}")
                print(f"Training time: {training_time:.2f}s")
                print(f"Average reward: {evaluation_results.get('avg_reward', 0):.2f}")
                
            except Exception as e:
                print(f"Error in {algorithm_name} run {run_id}: {e}")
                continue
            finally:
                run_training_manager.end_run()
                
    def _train_algorithm(self, algorithm, training_manager: TrainingManager):
        """Train a specific algorithm"""
        print(f"Training {algorithm.algorithm_name}...")
        
        # Create environment
        env = self._create_environment(training_manager)
        
        # Create and train model
        model = algorithm.create_model(env, training_manager)
        trained_model = algorithm.train(model, training_manager)
        
        return trained_model, env
        
    def _create_environment(self, training_manager: TrainingManager):
        """Create and configure the training environment"""
        # Create log directory for monitor files
        monitor_dir = os.path.join(training_manager.logs_path, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        
        # Create environment
        env = IoTEnv(self.config)
        env = Monitor(env, monitor_dir)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        return env
        
    def _evaluate_algorithm(self, algorithm_name: str, model, env, 
                          training_manager: TrainingManager) -> Dict[str, Any]:
        """Evaluate a trained algorithm"""
        print(f"Evaluating {algorithm_name}...")
        
        num_episodes = self.config.BENCHMARKING_EVALUATION_EPISODES
        
        try:
            # Use stable-baselines3 evaluate_policy for consistent evaluation
            episode_rewards, episode_lengths = evaluate_policy(
                model, 
                env, 
                n_eval_episodes=num_episodes,
                return_episode_rewards=True
            )
            
            # Calculate metrics
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            final_reward = episode_rewards[-1] if episode_rewards else 0.0
            
            results = {
                'avg_reward': float(avg_reward),
                'std_reward': float(std_reward),
                'final_reward': float(final_reward),
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': [int(l) for l in episode_lengths],
                'evaluation_metrics': {
                    'success_rate': float(np.mean([r > 0 for r in episode_rewards])),
                    'avg_episode_length': float(np.mean(episode_lengths))
                }
            }
            
            # Log to MLflow
            training_manager.log_metrics({
                'eval_avg_reward': avg_reward,
                'eval_std_reward': std_reward,
                'eval_success_rate': results['evaluation_metrics']['success_rate'],
                'eval_avg_episode_length': results['evaluation_metrics']['avg_episode_length']
            })
            
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'final_reward': 0.0,
                'episode_rewards': [],
                'episode_lengths': [],
                'evaluation_metrics': {}
            }