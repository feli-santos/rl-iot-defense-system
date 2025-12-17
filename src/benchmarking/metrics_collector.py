"""
Metrics Collector

Collects and organizes metrics from algorithm evaluation for the
adversarial IoT defense system.

PRD 7.2 Evaluation Metrics:
1. Mean Cumulative Reward: Overall efficacy
2. Attack Mitigation Rate: % attacks broken before reaching IMPACT
3. False Positive Rate: % BENIGN steps met with active defense (action > 0)
4. Mean Time to Contain: Average steps to reset attack to BENIGN
5. Availability Score: Inverse of sum of action costs
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode.
    
    Attributes:
        episode_id: Unique identifier for the episode.
        attack_stages: Sequence of attack stages during episode.
        actions: Sequence of actions taken by agent.
        rewards: Sequence of rewards received.
        total_reward: Sum of rewards for the episode.
        episode_length: Number of steps in episode.
        reached_impact: Whether attack reached IMPACT stage (4).
        containment_steps: Steps taken to reset attack to BENIGN.
        false_positive_count: Number of active actions on BENIGN state.
        total_action_cost: Sum of action costs incurred.
        benign_steps: Number of steps in BENIGN state.
        attack_steps: Number of steps in non-BENIGN state.
    """
    
    episode_id: int = 0
    attack_stages: List[int] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    episode_length: int = 0
    reached_impact: bool = False
    containment_steps: List[int] = field(default_factory=list)
    false_positive_count: int = 0
    total_action_cost: float = 0.0
    benign_steps: int = 0
    attack_steps: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "episode_id": self.episode_id,
            "attack_stages": self.attack_stages,
            "actions": self.actions,
            "rewards": self.rewards,
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "reached_impact": self.reached_impact,
            "containment_steps": self.containment_steps,
            "false_positive_count": self.false_positive_count,
            "total_action_cost": self.total_action_cost,
            "benign_steps": self.benign_steps,
            "attack_steps": self.attack_steps,
        }


@dataclass
class RunMetrics:
    """Metrics for a single algorithm evaluation run.
    
    Includes PRD 7.2 metrics:
    - Mean cumulative reward
    - Attack mitigation rate
    - False positive rate
    - Mean time to contain
    - Availability score
    """
    
    algorithm_name: str
    run_id: int
    hyperparameters: Dict[str, Any]
    training_time: float = 0.0
    convergence_step: Optional[int] = None
    
    # Basic reward metrics
    final_reward: float = 0.0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    
    # PRD 7.2 Metrics
    attack_mitigation_rate: float = 0.0  # % attacks stopped before IMPACT
    false_positive_rate: float = 0.0  # % BENIGN met with active defense
    mean_time_to_contain: float = 0.0  # Steps to reset to BENIGN
    availability_score: float = 0.0  # 1 / (1 + total_action_cost)
    
    # Detailed episode data
    episode_metrics: List[EpisodeMetrics] = field(default_factory=list)
    
    # Legacy evaluation metrics dict
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            "attack_mitigation_rate": self.attack_mitigation_rate,
            "false_positive_rate": self.false_positive_rate,
            "mean_time_to_contain": self.mean_time_to_contain,
            "availability_score": self.availability_score,
            "episode_metrics": [ep.to_dict() for ep in self.episode_metrics],
            "evaluation_metrics": self.evaluation_metrics,
        }


# Action costs for availability score calculation (force continuum)
ACTION_COSTS = [0.0, 0.1, 0.3, 0.5, 0.8]


class MetricsCollector:
    """Collects and manages metrics across multiple algorithm runs.
    
    This class tracks both run-level and episode-level metrics for
    evaluating RL agents on the adversarial IoT defense environment.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.start_run("ppo", 0, {"learning_rate": 3e-4})
        >>> collector.add_episode(
        ...     "ppo", 0, 
        ...     attack_stages=[0, 1, 2, 1, 0],
        ...     actions=[0, 1, 2, 2, 0],
        ...     rewards=[0.0, -0.1, 0.5, 0.3, 0.0]
        ... )
        >>> collector.finalize_run("ppo", 0)
        >>> collector.save_results("evaluation_results.json")
    """
    
    def __init__(self, save_path: Optional[Path] = None) -> None:
        """Initialize metrics collector.
        
        Args:
            save_path: Directory to save results.
        """
        self.metrics: Dict[str, List[RunMetrics]] = {}
        self.save_path = Path(save_path) if save_path else Path("results/benchmark")
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MetricsCollector initialized, saving to: {self.save_path}")
    
    def start_run(
        self,
        algorithm_name: str,
        run_id: int,
        hyperparameters: Dict[str, Any],
    ) -> None:
        """Start tracking a new evaluation run.
        
        Args:
            algorithm_name: Name of the algorithm (dqn, ppo, a2c).
            run_id: Unique run identifier.
            hyperparameters: Algorithm hyperparameters.
        """
        if algorithm_name not in self.metrics:
            self.metrics[algorithm_name] = []
        
        run_metrics = RunMetrics(
            algorithm_name=algorithm_name,
            run_id=run_id,
            hyperparameters=hyperparameters,
        )
        
        self.metrics[algorithm_name].append(run_metrics)
        logger.debug(f"Started run {run_id} for {algorithm_name}")
    
    def add_episode(
        self,
        algorithm_name: str,
        run_id: int,
        attack_stages: List[int],
        actions: List[int],
        rewards: List[float],
        episode_id: Optional[int] = None,
    ) -> None:
        """Add episode data to a run.
        
        Args:
            algorithm_name: Algorithm name.
            run_id: Run identifier.
            attack_stages: Sequence of attack stages (0-4).
            actions: Sequence of actions taken (0-4).
            rewards: Sequence of rewards received.
            episode_id: Optional episode identifier.
        """
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics is None:
            logger.warning(f"Run {run_id} for {algorithm_name} not found")
            return
        
        # Auto-assign episode ID
        if episode_id is None:
            episode_id = len(run_metrics.episode_metrics)
        
        # Calculate episode metrics
        episode = self._calculate_episode_metrics(
            episode_id, attack_stages, actions, rewards
        )
        
        run_metrics.episode_metrics.append(episode)
        run_metrics.episode_rewards.append(episode.total_reward)
        run_metrics.episode_lengths.append(episode.episode_length)
    
    def _calculate_episode_metrics(
        self,
        episode_id: int,
        attack_stages: List[int],
        actions: List[int],
        rewards: List[float],
    ) -> EpisodeMetrics:
        """Calculate metrics for a single episode.
        
        Args:
            episode_id: Episode identifier.
            attack_stages: Attack stage sequence.
            actions: Action sequence.
            rewards: Reward sequence.
        
        Returns:
            Calculated episode metrics.
        """
        episode = EpisodeMetrics(
            episode_id=episode_id,
            attack_stages=attack_stages.copy(),
            actions=actions.copy(),
            rewards=rewards.copy(),
            total_reward=sum(rewards),
            episode_length=len(attack_stages),
        )
        
        # Track containment and false positives
        in_attack = False
        attack_start_step = 0
        
        for step, (stage, action) in enumerate(zip(attack_stages, actions)):
            # Count BENIGN vs attack steps
            if stage == 0:
                episode.benign_steps += 1
                
                # Check for false positive (active action on BENIGN)
                if action > 0:
                    episode.false_positive_count += 1
                
                # Check if we just contained an attack
                if in_attack:
                    containment_time = step - attack_start_step
                    episode.containment_steps.append(containment_time)
                    in_attack = False
            else:
                episode.attack_steps += 1
                
                # Start tracking attack if not already
                if not in_attack:
                    in_attack = True
                    attack_start_step = step
                
                # Check if attack reached IMPACT
                if stage == 4:
                    episode.reached_impact = True
            
            # Accumulate action cost
            if action < len(ACTION_COSTS):
                episode.total_action_cost += ACTION_COSTS[action]
        
        return episode
    
    def finalize_run(
        self,
        algorithm_name: str,
        run_id: int,
    ) -> None:
        """Finalize run and compute PRD 7.2 aggregate metrics.
        
        Args:
            algorithm_name: Algorithm name.
            run_id: Run identifier.
        """
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics is None or not run_metrics.episode_metrics:
            logger.warning(f"No episodes found for {algorithm_name} run {run_id}")
            return
        
        episodes = run_metrics.episode_metrics
        
        # Basic reward metrics
        run_metrics.avg_reward = float(np.mean(run_metrics.episode_rewards))
        run_metrics.std_reward = float(np.std(run_metrics.episode_rewards))
        if run_metrics.episode_rewards:
            run_metrics.final_reward = run_metrics.episode_rewards[-1]
        
        # PRD 7.2 Metric 1: Mean Cumulative Reward (already computed above)
        
        # PRD 7.2 Metric 2: Attack Mitigation Rate
        # % of episodes where attack didn't reach IMPACT
        impact_count = sum(1 for ep in episodes if ep.reached_impact)
        run_metrics.attack_mitigation_rate = 1.0 - (impact_count / len(episodes))
        
        # PRD 7.2 Metric 3: False Positive Rate
        # % of BENIGN steps met with active defense (action > 0)
        total_benign_steps = sum(ep.benign_steps for ep in episodes)
        total_false_positives = sum(ep.false_positive_count for ep in episodes)
        if total_benign_steps > 0:
            run_metrics.false_positive_rate = total_false_positives / total_benign_steps
        
        # PRD 7.2 Metric 4: Mean Time to Contain
        # Average steps to reset attack to BENIGN
        all_containment_steps = []
        for ep in episodes:
            all_containment_steps.extend(ep.containment_steps)
        if all_containment_steps:
            run_metrics.mean_time_to_contain = float(np.mean(all_containment_steps))
        
        # PRD 7.2 Metric 5: Availability Score
        # Inverse of sum of action costs: 1 / (1 + total_cost)
        total_action_cost = sum(ep.total_action_cost for ep in episodes)
        run_metrics.availability_score = 1.0 / (1.0 + total_action_cost)
        
        # Store in evaluation_metrics for compatibility
        run_metrics.evaluation_metrics = {
            "avg_reward": run_metrics.avg_reward,
            "attack_mitigation_rate": run_metrics.attack_mitigation_rate,
            "false_positive_rate": run_metrics.false_positive_rate,
            "mean_time_to_contain": run_metrics.mean_time_to_contain,
            "availability_score": run_metrics.availability_score,
            "total_episodes": len(episodes),
        }
        
        logger.info(
            f"Finalized {algorithm_name} run {run_id}: "
            f"reward={run_metrics.avg_reward:.3f}, "
            f"mitigation={run_metrics.attack_mitigation_rate:.2%}, "
            f"fpr={run_metrics.false_positive_rate:.2%}"
        )
    
    def update_training_metrics(
        self,
        algorithm_name: str,
        run_id: int,
        training_time: float,
        convergence_step: Optional[int] = None,
    ) -> None:
        """Update training-related metrics.
        
        Args:
            algorithm_name: Algorithm name.
            run_id: Run identifier.
            training_time: Time spent training.
            convergence_step: Step when convergence was reached.
        """
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics:
            run_metrics.training_time = training_time
            run_metrics.convergence_step = convergence_step
    
    def update_evaluation_metrics(
        self,
        algorithm_name: str,
        run_id: int,
        evaluation_results: Dict[str, Any],
    ) -> None:
        """Update evaluation metrics from legacy format.
        
        Args:
            algorithm_name: Algorithm name.
            run_id: Run identifier.
            evaluation_results: Dictionary with evaluation results.
        """
        run_metrics = self._get_run_metrics(algorithm_name, run_id)
        if run_metrics:
            run_metrics.avg_reward = evaluation_results.get('avg_reward', 0.0)
            run_metrics.std_reward = evaluation_results.get('std_reward', 0.0)
            run_metrics.final_reward = evaluation_results.get('final_reward', 0.0)
            run_metrics.episode_rewards = evaluation_results.get('episode_rewards', [])
            run_metrics.episode_lengths = evaluation_results.get('episode_lengths', [])
            run_metrics.evaluation_metrics = evaluation_results.get('evaluation_metrics', {})
            
            # Update PRD metrics if provided
            run_metrics.attack_mitigation_rate = evaluation_results.get(
                'attack_mitigation_rate', 0.0
            )
            run_metrics.false_positive_rate = evaluation_results.get(
                'false_positive_rate', 0.0
            )
            run_metrics.mean_time_to_contain = evaluation_results.get(
                'mean_time_to_contain', 0.0
            )
            run_metrics.availability_score = evaluation_results.get(
                'availability_score', 0.0
            )
    
    def _get_run_metrics(
        self,
        algorithm_name: str,
        run_id: int,
    ) -> Optional[RunMetrics]:
        """Get run metrics for specific algorithm and run.
        
        Args:
            algorithm_name: Algorithm name.
            run_id: Run identifier.
        
        Returns:
            RunMetrics if found, None otherwise.
        """
        if algorithm_name in self.metrics:
            for run_metrics in self.metrics[algorithm_name]:
                if run_metrics.run_id == run_id:
                    return run_metrics
        return None
    
    def get_algorithm_summary(self, algorithm_name: str) -> Dict[str, Any]:
        """Get summary statistics for an algorithm across all runs.
        
        Args:
            algorithm_name: Algorithm name.
        
        Returns:
            Summary statistics dictionary.
        """
        if algorithm_name not in self.metrics:
            return {}
        
        runs = self.metrics[algorithm_name]
        if not runs:
            return {}
        
        avg_rewards = [run.avg_reward for run in runs]
        training_times = [run.training_time for run in runs]
        mitigation_rates = [run.attack_mitigation_rate for run in runs]
        fpr_rates = [run.false_positive_rate for run in runs]
        ttc_times = [run.mean_time_to_contain for run in runs]
        availability_scores = [run.availability_score for run in runs]
        
        return {
            "algorithm_name": algorithm_name,
            "num_runs": len(runs),
            # Basic metrics
            "avg_reward_mean": float(np.mean(avg_rewards)),
            "avg_reward_std": float(np.std(avg_rewards)),
            "avg_reward_min": float(np.min(avg_rewards)),
            "avg_reward_max": float(np.max(avg_rewards)),
            "training_time_mean": float(np.mean(training_times)),
            "training_time_std": float(np.std(training_times)),
            # PRD 7.2 metrics
            "attack_mitigation_rate_mean": float(np.mean(mitigation_rates)),
            "attack_mitigation_rate_std": float(np.std(mitigation_rates)),
            "false_positive_rate_mean": float(np.mean(fpr_rates)),
            "false_positive_rate_std": float(np.std(fpr_rates)),
            "mean_time_to_contain_mean": float(np.mean(ttc_times)),
            "mean_time_to_contain_std": float(np.std(ttc_times)),
            "availability_score_mean": float(np.mean(availability_scores)),
            "availability_score_std": float(np.std(availability_scores)),
            "convergence_steps": [
                run.convergence_step for run in runs if run.convergence_step
            ],
        }
    
    def get_comparison_data(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison data for all algorithms.
        
        Returns:
            Dictionary mapping algorithm names to summary statistics.
        """
        comparison = {}
        for algorithm_name in self.metrics.keys():
            comparison[algorithm_name] = self.get_algorithm_summary(algorithm_name)
        return comparison
    
    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save all metrics to file.
        
        Args:
            filename: Output filename.
        """
        results = {}
        for algorithm_name, runs in self.metrics.items():
            results[algorithm_name] = [run.to_dict() for run in runs]
        
        if Path(filename).is_absolute():
            save_file = Path(filename)
        else:
            save_file = self.save_path / filename
        
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {save_file}")
        print(f"Benchmarking results saved to: {save_file}")
    
    def load_results(self, filename: str = "benchmark_results.json") -> None:
        """Load metrics from file.
        
        Args:
            filename: Input filename.
        """
        if Path(filename).is_absolute():
            load_file = Path(filename)
        else:
            load_file = self.save_path / filename
        
        if not load_file.exists():
            logger.warning(f"Results file not found: {load_file}")
            print(f"Results file not found: {load_file}")
            return
        
        with open(load_file, 'r') as f:
            results = json.load(f)
        
        self.metrics = {}
        for algorithm_name, runs_data in results.items():
            self.metrics[algorithm_name] = []
            for run_data in runs_data:
                # Handle episode_metrics conversion
                episode_metrics_data = run_data.pop('episode_metrics', [])
                episode_metrics = [
                    EpisodeMetrics(**ep) for ep in episode_metrics_data
                ]
                
                run_metrics = RunMetrics(**run_data)
                run_metrics.episode_metrics = episode_metrics
                self.metrics[algorithm_name].append(run_metrics)
        
        logger.info(f"Results loaded from: {load_file}")
        print(f"Benchmarking results loaded from: {load_file}")
