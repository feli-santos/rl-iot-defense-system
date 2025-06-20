"""
Benchmark Runner

This module provides a flexible benchmark runner for comparing different RL algorithms.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import json

from algorithms.algorithm_factory import AlgorithmFactory
from training.training_manager import TrainingManager
from environment.environment import IoTEnv, EnvironmentConfig
from predictor.attack import LSTMAttackPredictor
from benchmarking.metrics_collector import MetricsCollector
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Benchmark Runner for comparing RL algorithms on IoT environments.
    
    Modes:
    - 'train': Train algorithms from scratch then evaluate
    - 'evaluate': Load pre-trained models and evaluate only
    - 'mixed': Mix of training and loading based on availability
    """
    
    def __init__(self, config: Dict[str, Any], 
                 lstm_model_path: Optional[Path] = None,
                 mode: str = 'evaluate'):
        """
        Initialize benchmark runner.
        
        Args:
            config: Configuration dictionary
            lstm_model_path: Path to trained LSTM model
            mode: Benchmark mode ('train', 'evaluate', 'mixed')
        """
        self.config = config
        self.lstm_model_path = lstm_model_path
        self.mode = mode
        
        # Create results directory
        self.results_path = Path("results/benchmark")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics collector with correct path - FIXED
        self.metrics_collector = MetricsCollector(save_path=self.results_path)
        
        # Model storage paths
        self.models_path = Path(config['models']['rl']['save_dir'])
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Benchmark runner initialized in {mode} mode")
        logger.info(f"Results will be saved to: {self.results_path}")
    
    def run_benchmark(self, algorithms: Optional[List[str]] = None, 
                     num_runs: int = 3,
                     model_paths: Optional[Dict[str, List[str]]] = None) -> MetricsCollector:
        """
        Run benchmark comparison across multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to benchmark
            num_runs: Number of independent runs per algorithm (for training mode)
            model_paths: Dict mapping algorithm names to lists of model paths (for evaluate mode)
            
        Returns:
            MetricsCollector with results
        """
        if algorithms is None:
            algorithms = ['dqn', 'ppo', 'a2c']
        
        print(f"🚀 Starting Benchmark in {self.mode.upper()} mode")
        print(f"📊 Algorithms: {algorithms}")
        print(f"🔄 Runs per algorithm: {num_runs}")
        print("=" * 60)
        
        try:
            if self.mode == 'train':
                self._run_training_benchmark(algorithms, num_runs)
            elif self.mode == 'evaluate':
                if not model_paths:
                    model_paths = self._discover_trained_models(algorithms)
                self._run_evaluation_benchmark(model_paths)
            elif self.mode == 'mixed':
                self._run_mixed_benchmark(algorithms, num_runs, model_paths)
            else:
                raise ValueError(f"Unknown benchmark mode: {self.mode}")
            
            # Save results
            self.metrics_collector.save_results(
                f"{self.mode}.json"
            )
            
            print(f"\n🎉 Benchmark completed successfully!")
            print(f"📁 Results saved to: {self.results_path}")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise
        
        return self.metrics_collector
    
    def _discover_trained_models(self, algorithms: List[str]) -> Dict[str, List[str]]:
        """
        Discover pre-trained models in the models directory.
        
        Args:
            algorithms: List of algorithm names to look for
            
        Returns:
            Dict mapping algorithm names to lists of model paths
        """
        model_paths = {}
        
        for algorithm in algorithms:
            algorithm_models = []
            
            # Look for models in different locations
            search_paths = [
                self.models_path / algorithm,
                self.models_path,
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    # Look for .zip files (SB3 format) with algorithm name
                    model_files = list(search_path.glob(f"*{algorithm}*.zip"))
                    model_files.extend(search_path.glob(f"{algorithm}_*.zip"))
                    
                    # Look in subdirectories but filter by algorithm name
                    for subdir in search_path.iterdir():
                        if subdir.is_dir():
                            # Only include subdirectories that contain the algorithm name
                            if algorithm.lower() in subdir.name.lower():
                                model_files.extend(subdir.glob("*.zip"))
                                model_files.extend(subdir.glob("**/final_model.zip"))
                                model_files.extend(subdir.glob("**/best_model.zip"))
                
                    algorithm_models.extend([str(p) for p in model_files])
        
        # Remove duplicates and sort
        algorithm_models = sorted(list(set(algorithm_models)))
        
        if algorithm_models:
            model_paths[algorithm] = algorithm_models
            print(f"🔍 Found {len(algorithm_models)} {algorithm.upper()} models")
        else:
            print(f"⚠️  No pre-trained {algorithm.upper()} models found")
    
        return model_paths
    
    def _run_training_benchmark(self, algorithms: List[str], num_runs: int) -> None:
        """Run benchmark by training algorithms from scratch."""
        for algorithm_name in algorithms:
            print(f"\n{'='*50}")
            print(f"🧪 Training & Benchmarking {algorithm_name.upper()}")
            print(f"{'='*50}")
            
            self._run_algorithm_benchmark(algorithm_name, num_runs)
    
    def _run_evaluation_benchmark(self, model_paths: Dict[str, List[str]]) -> None:
        """Run benchmark by evaluating pre-trained models."""
        for algorithm_name, paths in model_paths.items():
            print(f"\n{'='*50}")
            print(f"🧪 Evaluating {algorithm_name.upper()} Models")
            print(f"📁 Found {len(paths)} models")
            print(f"{'='*50}")
            
            for i, model_path in enumerate(paths):
                model_full_path = Path(model_path).resolve()
                model_name = Path(model_path).name
                print(f"\n📊 Evaluating model {i+1}/{len(paths)}")
                print(f"📁 File: {model_name}")
                print(f"🗂️  Path: {model_full_path}")
                
                try:
                    evaluation_results = self._evaluate_pretrained_model(algorithm_name, model_path)
                    avg_reward = evaluation_results.get('avg_reward', 0.0)
                    
                    # Collect metrics
                    self.metrics_collector.start_run(
                        algorithm_name=algorithm_name,
                        run_id=i,
                        hyperparameters={"model_path": str(model_full_path)}
                    )
                    
                    self.metrics_collector.update_evaluation_metrics(
                        algorithm_name=algorithm_name,
                        run_id=i,
                        evaluation_results=evaluation_results
                    )
                    
                    print(f"  ✅ Evaluation completed: Reward={avg_reward:.3f}")
                    
                except Exception as e:
                    print(f"  ❌ Evaluation failed: {e}")
                    logger.error(f"Failed to evaluate {model_full_path}: {e}")
                    continue
    
    def _run_mixed_benchmark(self, algorithms: List[str], num_runs: int,
                           model_paths: Optional[Dict[str, List[str]]]) -> None:
        """Run mixed benchmark (evaluate existing, train missing)."""
        discovered_models = self._discover_trained_models(algorithms)
        
        for algorithm_name in algorithms:
            print(f"\n{'='*50}")
            print(f"🔍 Processing {algorithm_name.upper()}")
            print(f"{'='*50}")
            
            if algorithm_name in discovered_models and discovered_models[algorithm_name]:
                # Evaluate existing models
                print(f"📁 Found {len(discovered_models[algorithm_name])} existing models - evaluating")
                paths = discovered_models[algorithm_name][:num_runs]  # Limit to num_runs
                
                for i, model_path in enumerate(paths):
                    model_full_path = Path(model_path).resolve()
                    model_name = Path(model_path).name
                    print(f"\n📊 Evaluating existing model {i+1}/{len(paths)}")
                    print(f"📁 File: {model_name}")
                    print(f"🗂️  Path: {model_full_path}")
                    
                    try:
                        evaluation_results = self._evaluate_pretrained_model(algorithm_name, model_path)
                        avg_reward = evaluation_results.get('avg_reward', 0.0)
                        
                        # Collect metrics
                        self.metrics_collector.start_run(
                            algorithm_name=algorithm_name,
                            run_id=i,
                            hyperparameters={"model_path": str(model_full_path)}
                        )
                        
                        self.metrics_collector.update_evaluation_metrics(
                            algorithm_name=algorithm_name,
                            run_id=i,
                            evaluation_results=evaluation_results
                        )
                        
                        print(f"  ✅ Evaluation completed: Reward={avg_reward:.3f}")
                        
                    except Exception as e:
                        print(f"  ❌ Evaluation failed: {e}")
                        logger.error(f"Failed to evaluate {model_full_path}: {e}")

            else:
                # Train new models
                print(f"🔄 No existing models found - training {num_runs} new models")
                self._run_algorithm_benchmark(algorithm_name, num_runs)
    
    def _evaluate_pretrained_model(self, algorithm_name: str, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a pre-trained model.
        
        Args:
            algorithm_name: Name of the algorithm
            model_path: Path to the trained model
            
        Returns:
            Evaluation results dictionary
        """
        # Create environment
        env_config = EnvironmentConfig(
            max_steps=self.config.get('environment', {}).get('max_steps', 1000),
            attack_probability=self.config.get('environment', {}).get('attack_probability', 0.3),
            state_history_length=self.config.get('environment', {}).get('state_history_length', 10),
            action_history_length=self.config.get('environment', {}).get('action_history_length', 5),
            reward_scale=self.config.get('environment', {}).get('reward_scale', 1.0),
            model_path=str(self.lstm_model_path) if self.lstm_model_path else None,
            data_path=self.config.get('data_path', 'data/processed/ciciot2023')
        )
        
        # Create environment
        env = IoTEnv(env_config)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        try:
            # Load algorithm based on type
            if algorithm_name.lower() == 'dqn':
                from stable_baselines3 import DQN
                algorithm = DQN.load(model_path, env=env)
            elif algorithm_name.lower() == 'ppo':
                from stable_baselines3 import PPO
                algorithm = PPO.load(model_path, env=env)
            elif algorithm_name.lower() == 'a2c':
                from stable_baselines3 import A2C
                algorithm = A2C.load(model_path, env=env)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Evaluate
            evaluation_results = self._evaluate_algorithm(algorithm, env)
            
            # Clean up
            env.close()
            
            return evaluation_results
            
        except Exception as e:
            env.close()
            raise e
    
    def _run_algorithm_benchmark(self, algorithm_name: str, num_runs: int) -> None:
        """Run benchmark for specific algorithm (training mode)."""
        
        for run_id in range(num_runs):
            print(f"\n🏃 Run {run_id + 1}/{num_runs} for {algorithm_name}")
            
            try:
                # Create environment configuration  
                env_config = EnvironmentConfig(
                    max_steps=self.config.get('environment', {}).get('max_steps', 1000),
                    attack_probability=self.config.get('environment', {}).get('attack_probability', 0.3),
                    state_history_length=self.config.get('environment', {}).get('state_history_length', 10),
                    action_history_length=self.config.get('environment', {}).get('action_history_length', 5),
                    reward_scale=self.config.get('environment', {}).get('reward_scale', 1.0),
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
                
                # Save trained model
                model_save_path = self.models_path / f"{algorithm_name}_benchmark_run_{run_id}.zip"
                algorithm.save(str(model_save_path))
                print(f"  💾 Model saved: {model_save_path}")
                
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