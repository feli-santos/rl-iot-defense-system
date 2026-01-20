"""
Benchmark Runner for Adversarial RL Evaluation.

This module provides evaluation capabilities for RL models trained on
the AdversarialIoTEnv. It supports both single-model detailed evaluation
and multi-model comparison benchmarking.

Usage:
    - Single model: Detailed metrics with episode-by-episode analysis
    - Multiple models: Comparison across algorithms with summary statistics
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.adversarial_env import (
    AdversarialEnvConfig,
    AdversarialIoTEnv,
)
from src.benchmarking.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.
    
    Attributes:
        num_episodes: Number of evaluation episodes per model.
        generator_path: Path to Attack Sequence Generator.
        dataset_path: Path to processed dataset.
        results_path: Path for saving benchmark results.
        env_config: Environment configuration overrides.
        deterministic: Whether to use deterministic actions.
    """
    num_episodes: int = 20
    generator_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    results_path: Path = Path("results/benchmark")
    env_config: Optional[AdversarialEnvConfig] = None
    deterministic: bool = True


# =============================================================================
# ALGORITHM DETECTION
# =============================================================================

ALGORITHM_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


def detect_algorithm_type(model_path: Path) -> Optional[str]:
    """Detect algorithm type from model path or metadata.
    
    Args:
        model_path: Path to the model file.
    
    Returns:
        Algorithm name ('dqn', 'ppo', 'a2c') or None if unknown.
    """
    # Check if algorithm name is in the path
    path_str = str(model_path).lower()
    
    for alg_name in ALGORITHM_CLASSES:
        if alg_name in path_str:
            return alg_name
    
    # Try to infer from parent directory names
    for parent in model_path.parents:
        parent_name = parent.name.lower()
        for alg_name in ALGORITHM_CLASSES:
            if parent_name.startswith(alg_name):
                return alg_name
    
    return None


def load_model(
    model_path: Path,
    algorithm_type: Optional[str] = None,
    env: Optional[DummyVecEnv] = None,
) -> BaseAlgorithm:
    """Load a trained model from disk.
    
    Args:
        model_path: Path to the model file.
        algorithm_type: Algorithm type (auto-detected if None).
        env: Optional environment to attach.
    
    Returns:
        Loaded SB3 model.
    
    Raises:
        ValueError: If algorithm type cannot be determined.
    """
    if algorithm_type is None:
        algorithm_type = detect_algorithm_type(model_path)
    
    if algorithm_type is None:
        raise ValueError(
            f"Cannot determine algorithm type for {model_path}. "
            f"Please specify algorithm_type explicitly."
        )
    
    if algorithm_type not in ALGORITHM_CLASSES:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}. "
            f"Supported: {list(ALGORITHM_CLASSES.keys())}"
        )
    
    alg_class = ALGORITHM_CLASSES[algorithm_type]
    model = alg_class.load(str(model_path), env=env)
    
    logger.info(f"Loaded {algorithm_type.upper()} model from {model_path}")
    return model


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class BenchmarkRunner:
    """Runner for evaluating and comparing RL models.
    
    This class provides a unified interface for:
    - Evaluating single models with detailed metrics
    - Comparing multiple models with summary statistics
    - Discovering and auto-loading trained models
    
    The runner uses the new metrics collector with PRD 7.2 metrics:
    - Attack Mitigation Rate
    - False Positive Rate
    - Mean Time to Contain
    - Availability Score
    
    Example:
        >>> runner = BenchmarkRunner(config, benchmark_config)
        >>> results = runner.evaluate_model("path/to/model.zip")
        >>> comparison = runner.run_comparison(["dqn", "ppo", "a2c"])
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        benchmark_config: Optional[BenchmarkConfig] = None,
    ) -> None:
        """Initialize the benchmark runner.
        
        Args:
            config: System configuration dictionary.
            benchmark_config: Benchmark-specific configuration.
        """
        self.config = config
        self.benchmark_config = benchmark_config or BenchmarkConfig()
        
        # Setup paths from config or benchmark_config
        self._setup_paths()
        
        # Initialize metrics collector
        self.results_path = self.benchmark_config.results_path
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.metrics_collector = MetricsCollector(save_path=self.results_path)
        
        logger.info(f"BenchmarkRunner initialized")
        logger.info(f"  Generator: {self.generator_path}")
        logger.info(f"  Dataset: {self.dataset_path}")
        logger.info(f"  Results: {self.results_path}")
    
    def _setup_paths(self) -> None:
        """Setup generator and dataset paths from config."""
        # Generator path
        if self.benchmark_config.generator_path:
            self.generator_path = self.benchmark_config.generator_path
        else:
            gen_config = self.config.get("attack_generator", {})
            self.generator_path = Path(gen_config.get("output_dir", "artifacts/generator"))
        
        # Dataset path
        if self.benchmark_config.dataset_path:
            self.dataset_path = self.benchmark_config.dataset_path
        else:
            ds_config = self.config.get("dataset", {})
            self.dataset_path = Path(ds_config.get("processed_path", "data/processed/ciciot2023"))
    
    def _create_env(self) -> AdversarialIoTEnv:
        """Create an evaluation environment.
        
        Returns:
            Configured AdversarialIoTEnv instance.
        """
        env_config = self.benchmark_config.env_config
        if env_config is None:
            # Load from config
            adv_config = self.config.get("adversarial_environment", {})
            env_config = AdversarialEnvConfig(
                max_steps=adv_config.get("max_steps", 500),
                window_size=adv_config.get("observation", {}).get("window_size", 5),
                include_deltas=adv_config.get("observation", {}).get("include_deltas", True),
                num_actions=adv_config.get("actions", {}).get("num_actions", 5),
                action_cost_scale=adv_config.get("reward", {}).get("action_cost_scale", 1.0),
                impact_penalty=adv_config.get("reward", {}).get("impact_penalty", 5.0),
                defense_success_bonus=adv_config.get("reward", {}).get("defense_success_bonus", 2.0),
                false_positive_penalty=adv_config.get("reward", {}).get("false_positive_penalty", 10.0),
                penalty_block_benign=adv_config.get("reward", {}).get("penalty_block_benign", 100.0),
                penalty_block_recon=adv_config.get("reward", {}).get("penalty_block_recon", 50.0),
                patience_bonus=adv_config.get("reward", {}).get("patience_bonus", 0.5),
                correct_escalation_reward=adv_config.get("reward", {}).get("defense_reward", {}).get(
                    "correct_escalation", 1.0
                ),
                correct_de_escalation_reward=adv_config.get("reward", {}).get("defense_reward", {}).get(
                    "correct_de_escalation", 0.5
                ),
                maintained_defense_reward=adv_config.get("reward", {}).get("defense_reward", {}).get(
                    "maintained_defense", 0.2
                ),
            )
        
        env = AdversarialIoTEnv(
            generator_path=self.generator_path,
            dataset_path=self.dataset_path,
            config=env_config,
        )
        
        return env
    
    def evaluate_model(
        self,
        model_path: Union[str, Path],
        algorithm_type: Optional[str] = None,
        num_episodes: Optional[int] = None,
        run_id: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate a single model with detailed episode-by-episode metrics.
        
        Args:
            model_path: Path to the trained model.
            algorithm_type: Algorithm type (auto-detected if None).
            num_episodes: Override for number of episodes.
            run_id: Run identifier for metrics collection.
        
        Returns:
            Dictionary with evaluation results and PRD metrics.
        """
        model_path = Path(model_path)
        num_episodes = num_episodes or self.benchmark_config.num_episodes
        
        # Detect algorithm type
        if algorithm_type is None:
            algorithm_type = detect_algorithm_type(model_path)
        
        if algorithm_type is None:
            raise ValueError(f"Cannot determine algorithm for {model_path}")
        
        print(f"\n{'='*60}")
        print(f"📊 Evaluating {algorithm_type.upper()} Model")
        print(f"📁 Path: {model_path}")
        print(f"🔄 Episodes: {num_episodes}")
        print(f"{'='*60}")
        
        # Create environment
        env = self._create_env()
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        try:
            # Load model
            model = load_model(model_path, algorithm_type, env=vec_env)
            
            # Start metrics collection
            self.metrics_collector.start_run(
                algorithm_name=algorithm_type,
                run_id=run_id,
                hyperparameters={"model_path": str(model_path)},
            )
            
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            
            for ep_idx in range(num_episodes):
                episode_data = self._run_episode(model, env)
                
                episode_rewards.append(episode_data["reward"])
                episode_lengths.append(episode_data["length"])
                
                # Add episode to metrics collector
                self.metrics_collector.add_episode(
                    algorithm_name=algorithm_type,
                    run_id=run_id,
                    attack_stages=episode_data["attack_stages"],
                    actions=episode_data["actions"],
                    rewards=episode_data["step_rewards"],
                )
                
                if (ep_idx + 1) % 5 == 0:
                    print(f"  Episode {ep_idx + 1}/{num_episodes}: "
                          f"Reward={episode_data['reward']:.2f}")
            
            # Finalize run to compute PRD metrics
            self.metrics_collector.finalize_run(algorithm_type, run_id)
            
            # Get results
            run_metrics = self.metrics_collector.metrics[algorithm_type][run_id]
            
            results = {
                "algorithm": algorithm_type,
                "model_path": str(model_path),
                "num_episodes": num_episodes,
                "avg_reward": float(np.mean(episode_rewards)),
                "std_reward": float(np.std(episode_rewards)),
                "avg_length": float(np.mean(episode_lengths)),
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                # PRD 7.2 Metrics
                "attack_mitigation_rate": run_metrics.attack_mitigation_rate,
                "false_positive_rate": run_metrics.false_positive_rate,
                "mean_time_to_contain": run_metrics.mean_time_to_contain,
                "availability_score": run_metrics.availability_score,
            }
            
            self._print_results(results)
            
            return results
            
        finally:
            env.close()
            vec_env.close()
    
    def _run_episode(
        self,
        model: BaseAlgorithm,
        env: AdversarialIoTEnv,
    ) -> Dict[str, Any]:
        """Run a single evaluation episode.
        
        Args:
            model: Trained model to evaluate.
            env: Environment to run in.
        
        Returns:
            Episode data dictionary.
        """
        obs, info = env.reset()
        
        attack_stages = [info["attack_stage"]]
        actions = []
        step_rewards = []
        
        total_reward = 0.0
        done = False
        step_count = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=self.benchmark_config.deterministic)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record data
            attack_stages.append(info["attack_stage"])
            actions.append(action)
            step_rewards.append(float(reward))
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        
        return {
            "reward": total_reward,
            "length": step_count,
            "attack_stages": attack_stages,
            "actions": actions,
            "step_rewards": step_rewards,
        }
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results summary."""
        print(f"\n📈 Results for {results['algorithm'].upper()}")
        print(f"   Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"   Average Length: {results['avg_length']:.1f} steps")
        print(f"\n🎯 PRD 7.2 Metrics:")
        print(f"   Attack Mitigation Rate: {results['attack_mitigation_rate']:.1%}")
        print(f"   False Positive Rate:    {results['false_positive_rate']:.1%}")
        print(f"   Mean Time to Contain:   {results['mean_time_to_contain']:.1f} steps")
        print(f"   Availability Score:     {results['availability_score']:.3f}")
    
    def run_comparison(
        self,
        algorithms: Optional[List[str]] = None,
        model_paths: Optional[Dict[str, List[Path]]] = None,
    ) -> MetricsCollector:
        """Run comparison benchmark across multiple algorithms.
        
        Args:
            algorithms: List of algorithm names to compare.
            model_paths: Dict mapping algorithm names to model paths.
        
        Returns:
            MetricsCollector with all results.
        """
        if algorithms is None:
            algorithms = ["dqn", "ppo", "a2c"]
        
        # Auto-discover models if not provided
        if model_paths is None:
            model_paths = self._discover_models(algorithms)
        
        print(f"\n🚀 Starting Algorithm Comparison")
        print(f"📊 Algorithms: {algorithms}")
        print(f"🔄 Episodes per model: {self.benchmark_config.num_episodes}")
        print("=" * 60)
        
        for algorithm in algorithms:
            if algorithm not in model_paths or not model_paths[algorithm]:
                print(f"⚠️  No models found for {algorithm.upper()}")
                continue
            
            paths = model_paths[algorithm]
            print(f"\n{'='*50}")
            print(f"🧪 Evaluating {algorithm.upper()}: {len(paths)} model(s)")
            print(f"{'='*50}")
            
            for run_id, model_path in enumerate(paths):
                try:
                    self.evaluate_model(
                        model_path=model_path,
                        algorithm_type=algorithm,
                        run_id=run_id,
                    )
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_path}: {e}")
                    print(f"❌ Failed: {e}")
        
        # Save results
        self.metrics_collector.save_results("comparison.json")
        
        print(f"\n🎉 Comparison completed!")
        print(f"📁 Results saved to: {self.results_path}")
        
        return self.metrics_collector
    
    def _discover_models(
        self,
        algorithms: List[str],
    ) -> Dict[str, List[Path]]:
        """Discover trained models in artifacts directory.
        
        Args:
            algorithms: List of algorithm names to search for.
        
        Returns:
            Dict mapping algorithm names to lists of model paths.
        """
        model_paths: Dict[str, List[Path]] = {}
        artifacts_path = Path("artifacts/rl")
        
        if not artifacts_path.exists():
            logger.warning(f"Artifacts directory not found: {artifacts_path}")
            return model_paths
        
        for algorithm in algorithms:
            algorithm_models: List[Path] = []
            
            # Search for algorithm-specific directories
            for exp_dir in artifacts_path.glob(f"{algorithm}_*"):
                if exp_dir.is_dir():
                    # Prefer explicit models/ subdir, but also scan root for backwards compatibility
                    candidate_dirs = [exp_dir / "models", exp_dir]
                    for candidate in candidate_dirs:
                        if candidate.exists():
                            for model_file in candidate.rglob("*.zip"):
                                algorithm_models.append(model_file)
            
            # Sort by modification time (newest first)
            algorithm_models = sorted(
                list(set(algorithm_models)),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            
            if algorithm_models:
                model_paths[algorithm] = algorithm_models
                print(f"🔍 Found {len(algorithm_models)} {algorithm.upper()} model(s)")
            else:
                print(f"⚠️  No {algorithm.upper()} models found")
        
        return model_paths
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all evaluated models.
        
        Returns:
            Dictionary with summary statistics per algorithm.
        """
        return self.metrics_collector.get_comparison_data()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_single_model(
    model_path: Union[str, Path],
    config: Dict[str, Any],
    generator_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    num_episodes: int = 20,
) -> Dict[str, Any]:
    """Convenience function to evaluate a single model.
    
    Args:
        model_path: Path to the model file.
        config: System configuration.
        generator_path: Path to generator (optional).
        dataset_path: Path to dataset (optional).
        num_episodes: Number of evaluation episodes.
    
    Returns:
        Evaluation results dictionary.
    """
    benchmark_config = BenchmarkConfig(
        num_episodes=num_episodes,
        generator_path=generator_path,
        dataset_path=dataset_path,
    )
    
    runner = BenchmarkRunner(config, benchmark_config)
    return runner.evaluate_model(model_path)


def run_algorithm_comparison(
    config: Dict[str, Any],
    algorithms: Optional[List[str]] = None,
    num_episodes: int = 20,
) -> MetricsCollector:
    """Convenience function to run algorithm comparison.
    
    Args:
        config: System configuration.
        algorithms: List of algorithms to compare.
        num_episodes: Episodes per model.
    
    Returns:
        MetricsCollector with results.
    """
    benchmark_config = BenchmarkConfig(num_episodes=num_episodes)
    runner = BenchmarkRunner(config, benchmark_config)
    return runner.run_comparison(algorithms)
