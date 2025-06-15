"""
Fixed Benchmark Runner with Proper Integration

Addresses the import and integration issues in your current benchmark system.
"""

import os
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

# Fixed imports to match new structure
from algorithms.algorithm_factory import AlgorithmFactory
from training.training_manager import TrainingManager
from environment import IoTEnv, EnvironmentConfig
from predictors.lstm_predictor import LSTMAttackPredictor
from benchmarking.metrics_collector import MetricsCollector
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Fixed benchmark runner with proper integration."""
    
    def __init__(self, config: Dict[str, Any], 
                 lstm_model_path: Optional[Path] = None):
        """
        Initialize benchmark runner.
        
        Args:
            config: Configuration dictionary
            lstm_model_path: Path to trained LSTM model
        """
        self.config = config
        self.lstm_model_path = lstm_model_path
        self.metrics_collector = MetricsCollector()
        
        # Create results directory
        self.results_path = Path("results/benchmarks")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Benchmark runner initialized")
    
    def run_benchmark(self, algorithms: Optional[List[str]] = None, 
                     num_runs: int = 3) -> MetricsCollector:
        """
        Run benchmark comparison across multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to benchmark
            num_runs: Number of independent runs per algorithm
            
        Returns:
            MetricsCollector with results
        """
        if algorithms is None:
            algorithms = ['dqn', 'ppo', 'a2c']  # Default algorithms
        
        print(f"🚀 Starting Benchmark with {len(algorithms)} algorithms")
        print(f"📊 Algorithms: {algorithms}")
        print(f"🔄 Runs per algorithm: {num_runs}")
        print("=" * 60)
        
        try:
            # Run experiments for each algorithm
            for algorithm_name in algorithms:
                print(f"\n{'='*50}")
                print(f"🧪 Benchmarking {algorithm_name.upper()}")
                print(f"{'='*50}")
                
                self._run_algorithm_benchmark(algorithm_name, num_runs)
            
            # Save results
            self.metrics_collector.save_results(
                str(self.results_path / "benchmark_results.json")
            )
            
            print(f"\n🎉 Benchmark completed successfully!")
            print(f"📁 Results saved to: {self.results_path}")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        
        return self.metrics_collector
    
    def _run_algorithm_benchmark(self, algorithm_name: str, num_runs: int) -> None:
        """Run benchmark for specific algorithm."""
        
        for run_id in range(num_runs):
            print(f"\n🏃 Run {run_id + 1}/{num_runs} for {algorithm_name}")
            
            try:
                # Create environment configuration  
                env_config = EnvironmentConfig(
                    max_steps=self.config.get('max_steps', 1000),
                    attack_probability=self.config.get('attack_probability', 0.3),
                    model_path=str(self.lstm_model_path) if self.lstm_model_path else None,
                    data_path=self.config.get('data_path', 'data/processed/ciciot2023')
                )
                
                # Get algorithm hyperparameters from config
                algo_config = self.config.get('rl', {}).get('algorithms', {}).get(algorithm_name, {})
                
                # Create algorithm and environment
                algorithm, env = AlgorithmFactory.create_algorithm_with_env(
                    algorithm_name=algorithm_name,
                    env_config=env_config,
                    hyperparams=algo_config,
                    verbose=0  # Reduce verbosity for benchmarking
                )
                
                # Start metrics collection
                self.metrics_collector.start_run(
                    algorithm_name=algorithm_name,
                    run_id=run_id,
                    hyperparameters=algo_config
                )
                
                # Train algorithm
                start_time = time.time()
                training_timesteps = self.config.get('rl', {}).get('training', {}).get('total_timesteps', 50000)
                
                print(f"  🏋️ Training for {training_timesteps:,} timesteps...")
                algorithm.learn(total_timesteps=training_timesteps)
                
                training_time = time.time() - start_time
                
                # Evaluate algorithm
                print(f"  📊 Evaluating...")
                evaluation_results = self._evaluate_algorithm(algorithm, env)
                
                # Update metrics
                self.metrics_collector.update_training_metrics(
                    algorithm_name=algorithm_name,
                    run_id=run_id,
                    training_time=training_time
                )
                
                self.metrics_collector.update_evaluation_metrics(
                    algorithm_name=algorithm_name,
                    run_id=run_id,
                    evaluation_results=evaluation_results
                )
                
                # Save model
                model_path = self.results_path / f"{algorithm_name}_run_{run_id}.zip"
                algorithm.save(str(model_path))
                
                print(f"  ✅ Completed: Reward={evaluation_results['avg_reward']:.2f}, Time={training_time:.1f}s")
                
                # Clean up
                env.close()
                
            except Exception as e:
                logger.error(f"Error in {algorithm_name} run {run_id}: {e}")
                print(f"  ❌ Run failed: {e}")
                continue
    
    def _evaluate_algorithm(self, algorithm, env) -> Dict[str, Any]:
        """Evaluate trained algorithm."""
        num_episodes = self.config.get('evaluation_episodes', 20)
        
        try:
            episode_rewards, episode_lengths = evaluate_policy(
                algorithm, 
                env, 
                n_eval_episodes=num_episodes,
                return_episode_rewards=True,
                deterministic=True
            )
            
            return {
                'avg_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'final_reward': float(episode_rewards[-1]) if episode_rewards else 0.0,
                'episode_rewards': [float(r) for r in episode_rewards],
                'episode_lengths': [int(l) for l in episode_lengths],
                'evaluation_metrics': {
                    'success_rate': float(np.mean([r > 0 for r in episode_rewards])),
                    'avg_episode_length': float(np.mean(episode_lengths))
                }
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'avg_reward': 0.0,
                'std_reward': 0.0,
                'final_reward': 0.0,
                'episode_rewards': [],
                'episode_lengths': [],
                'evaluation_metrics': {}
            }