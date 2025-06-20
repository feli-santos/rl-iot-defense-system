"""
Metrics Collector

Collects and organizes metrics from algorithm training and evaluation.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class RunMetrics:
    """Metrics for a single algorithm run"""
    algorithm_name: str
    run_id: int
    hyperparameters: Dict[str, Any]
    training_time: float
    convergence_step: Optional[int] = None
    final_reward: float = 0.0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "algorithm_name": self.algorithm_name,
            "run_id": self.run_id,
            "hyperparameters": self.hyperparameters,
            "training_time": self.training_time,
            "convergence_step": self.convergence_step,
            "final_reward": self.final_reward,
            "avg_reward": self.avg_reward,
            "std_reward": self.std_reward,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "evaluation_metrics": self.evaluation_metrics
        }


class MetricsCollector:
    """Collects and manages metrics across multiple algorithm runs"""
    
    def __init__(self, save_path: Optional[Path] = None):
        self.metrics: Dict[str, List[RunMetrics]] = {}
        self.save_path = save_path or Path("results/benchmark")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def start_run(self, algorithm_name: str, run_id: int, hyperparameters: Dict[str, Any]) -> None:
        """Start tracking a new run"""
        if algorithm_name not in self.metrics:
            self.metrics[algorithm_name] = []
            
        # Initialize run metrics
        run_metrics = RunMetrics(
            algorithm_name=algorithm_name,
            run_id=run_id,
            hyperparameters=hyperparameters,
            training_time=0.0
        )
        
        self.metrics[algorithm_name].append(run_metrics)
        
    def update_training_metrics(self, algorithm_name: str, run_id: int, 
                              training_time: float, convergence_step: Optional[int] = None) -> None:
        """Update training-related metrics"""
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics:
            run_metrics.training_time = training_time
            run_metrics.convergence_step = convergence_step
            
    def update_evaluation_metrics(self, algorithm_name: str, run_id: int, 
                                evaluation_results: Dict[str, Any]) -> None:
        """Update evaluation metrics"""
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics:
            run_metrics.avg_reward = evaluation_results.get('avg_reward', 0.0)
            run_metrics.std_reward = evaluation_results.get('std_reward', 0.0)
            run_metrics.final_reward = evaluation_results.get('final_reward', 0.0)
            run_metrics.episode_rewards = evaluation_results.get('episode_rewards', [])
            run_metrics.episode_lengths = evaluation_results.get('episode_lengths', [])
            run_metrics.evaluation_metrics = evaluation_results.get('evaluation_metrics', {})
            
    def _get_run_metrics(self, algorithm_name: str, run_id: int) -> Optional[RunMetrics]:
        """Get run metrics for specific algorithm and run"""
        if algorithm_name in self.metrics:
            for run_metrics in self.metrics[algorithm_name]:
                if run_metrics.run_id == run_id:
                    return run_metrics
        return None
        
    def get_algorithm_summary(self, algorithm_name: str) -> Dict[str, Any]:
        """Get summary statistics for an algorithm across all runs"""
        if algorithm_name not in self.metrics:
            return {}
            
        runs = self.metrics[algorithm_name]
        if not runs:
            return {}
            
        avg_rewards = [run.avg_reward for run in runs]
        training_times = [run.training_time for run in runs]
        
        return {
            "algorithm_name": algorithm_name,
            "num_runs": len(runs),
            "avg_reward_mean": np.mean(avg_rewards),
            "avg_reward_std": np.std(avg_rewards),
            "avg_reward_min": np.min(avg_rewards),
            "avg_reward_max": np.max(avg_rewards),
            "training_time_mean": np.mean(training_times),
            "training_time_std": np.std(training_times),
            "convergence_steps": [run.convergence_step for run in runs if run.convergence_step]
        }
        
    def get_comparison_data(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison data for all algorithms"""
        comparison = {}
        for algorithm_name in self.metrics.keys():
            comparison[algorithm_name] = self.get_algorithm_summary(algorithm_name)
        return comparison
        
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save all metrics to file"""
        results = {}
        for algorithm_name, runs in self.metrics.items():
            results[algorithm_name] = [run.to_dict() for run in runs]
            
        save_file = self.save_path / filename
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Benchmarking results saved to: {save_file}")
        
    def load_results(self, filename: str = "benchmark_results.json") -> None:
        """Load metrics from file"""
        load_file = self.save_path / filename
        if not load_file.exists():
            print(f"Results file not found: {load_file}")
            return
            
        with open(load_file, 'r') as f:
            results = json.load(f)
            
        self.metrics = {}
        for algorithm_name, runs_data in results.items():
            self.metrics[algorithm_name] = []
            for run_data in runs_data:
                run_metrics = RunMetrics(**run_data)
                self.metrics[algorithm_name].append(run_metrics)
                
        print(f"Benchmarking results loaded from: {load_file}")