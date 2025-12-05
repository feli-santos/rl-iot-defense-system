"""Tests for the PRD 7.2 metrics collector."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.benchmarking.metrics_collector import (
    ACTION_COSTS,
    EpisodeMetrics,
    MetricsCollector,
    RunMetrics,
)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of EpisodeMetrics."""
        episode = EpisodeMetrics()
        
        assert episode.episode_id == 0
        assert episode.attack_stages == []
        assert episode.actions == []
        assert episode.rewards == []
        assert episode.total_reward == 0.0
        assert episode.episode_length == 0
        assert episode.reached_impact is False
        assert episode.containment_steps == []
        assert episode.false_positive_count == 0
        assert episode.total_action_cost == 0.0
        assert episode.benign_steps == 0
        assert episode.attack_steps == 0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        episode = EpisodeMetrics(
            episode_id=1,
            attack_stages=[0, 1, 2],
            actions=[0, 1, 2],
            rewards=[0.0, -0.1, 0.5],
            total_reward=0.4,
            episode_length=3,
        )
        
        d = episode.to_dict()
        
        assert d["episode_id"] == 1
        assert d["attack_stages"] == [0, 1, 2]
        assert d["actions"] == [0, 1, 2]
        assert d["total_reward"] == 0.4
        assert d["episode_length"] == 3


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of RunMetrics."""
        run = RunMetrics(
            algorithm_name="ppo",
            run_id=0,
            hyperparameters={"learning_rate": 3e-4},
        )
        
        assert run.algorithm_name == "ppo"
        assert run.run_id == 0
        assert run.attack_mitigation_rate == 0.0
        assert run.false_positive_rate == 0.0
        assert run.mean_time_to_contain == 0.0
        assert run.availability_score == 0.0
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        run = RunMetrics(
            algorithm_name="dqn",
            run_id=1,
            hyperparameters={"buffer_size": 10000},
            attack_mitigation_rate=0.85,
            false_positive_rate=0.10,
        )
        
        d = run.to_dict()
        
        assert d["algorithm_name"] == "dqn"
        assert d["run_id"] == 1
        assert d["attack_mitigation_rate"] == 0.85
        assert d["false_positive_rate"] == 0.10


class TestMetricsCollector:
    """Tests for MetricsCollector class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """Create a MetricsCollector instance."""
        return MetricsCollector(save_path=temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test collector initialization."""
        collector = MetricsCollector(save_path=temp_dir)
        
        assert collector.metrics == {}
        assert collector.save_path == temp_dir
    
    def test_start_run(self, collector):
        """Test starting a new run."""
        collector.start_run("ppo", 0, {"learning_rate": 3e-4})
        
        assert "ppo" in collector.metrics
        assert len(collector.metrics["ppo"]) == 1
        assert collector.metrics["ppo"][0].algorithm_name == "ppo"
        assert collector.metrics["ppo"][0].run_id == 0
    
    def test_start_multiple_runs(self, collector):
        """Test starting multiple runs for same algorithm."""
        collector.start_run("dqn", 0, {})
        collector.start_run("dqn", 1, {})
        collector.start_run("ppo", 0, {})
        
        assert len(collector.metrics["dqn"]) == 2
        assert len(collector.metrics["ppo"]) == 1
    
    def test_add_episode(self, collector):
        """Test adding episode data."""
        collector.start_run("ppo", 0, {})
        
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 1, 0],
            actions=[0, 1, 2, 2, 0],
            rewards=[0.0, -0.1, 0.5, 0.3, 0.0],
        )
        
        run = collector.metrics["ppo"][0]
        assert len(run.episode_metrics) == 1
        assert len(run.episode_rewards) == 1
        assert run.episode_rewards[0] == 0.7  # sum of rewards
    
    def test_episode_metrics_calculation(self, collector):
        """Test episode metrics calculation."""
        collector.start_run("ppo", 0, {})
        
        # Episode with attack stages: 0 -> 1 -> 2 -> 1 -> 0
        # Actions: observe on benign, log on recon, throttle on access, throttle on recon, observe on benign
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 1, 0],
            actions=[0, 1, 2, 2, 0],
            rewards=[0.0, -0.1, 0.5, 0.3, 0.0],
        )
        
        episode = collector.metrics["ppo"][0].episode_metrics[0]
        
        assert episode.episode_length == 5
        assert episode.benign_steps == 2
        assert episode.attack_steps == 3
        assert episode.reached_impact is False
        assert episode.false_positive_count == 0  # No active action on BENIGN
        
        # Action costs: 0.0 + 0.1 + 0.3 + 0.3 + 0.0 = 0.7
        expected_cost = ACTION_COSTS[0] + ACTION_COSTS[1] + ACTION_COSTS[2] + ACTION_COSTS[2] + ACTION_COSTS[0]
        assert episode.total_action_cost == expected_cost
    
    def test_episode_reaches_impact(self, collector):
        """Test detection of attack reaching IMPACT."""
        collector.start_run("dqn", 0, {})
        
        # Episode reaches IMPACT (stage 4)
        collector.add_episode(
            "dqn", 0,
            attack_stages=[0, 1, 2, 3, 4],
            actions=[0, 0, 0, 0, 4],
            rewards=[0.0, -0.1, -0.3, -0.5, -5.0],
        )
        
        episode = collector.metrics["dqn"][0].episode_metrics[0]
        
        assert episode.reached_impact is True
    
    def test_false_positive_detection(self, collector):
        """Test detection of false positives."""
        collector.start_run("a2c", 0, {})
        
        # Active actions on BENIGN states (false positives)
        collector.add_episode(
            "a2c", 0,
            attack_stages=[0, 0, 0, 0, 0],  # All BENIGN
            actions=[1, 2, 0, 3, 0],  # 3 active actions
            rewards=[-0.1, -0.3, 0.0, -0.5, 0.0],
        )
        
        episode = collector.metrics["a2c"][0].episode_metrics[0]
        
        assert episode.benign_steps == 5
        assert episode.false_positive_count == 3
    
    def test_containment_tracking(self, collector):
        """Test tracking of attack containment."""
        collector.start_run("ppo", 0, {})
        
        # Attack from step 1-3, contained at step 4
        # Then another attack at step 5, never contained
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 1, 0, 2, 3],
            actions=[0, 1, 2, 2, 0, 2, 3],
            rewards=[0.0, -0.1, 0.5, 0.3, 0.0, 0.5, 0.3],
        )
        
        episode = collector.metrics["ppo"][0].episode_metrics[0]
        
        # Attack started at step 1, contained at step 4 -> 3 steps
        assert len(episode.containment_steps) == 1
        assert episode.containment_steps[0] == 3
    
    def test_finalize_run(self, collector):
        """Test run finalization with PRD 7.2 metrics."""
        collector.start_run("ppo", 0, {})
        
        # Add multiple episodes
        # Episode 1: Attack mitigated (no IMPACT)
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 1, 0],
            actions=[0, 1, 2, 2, 0],
            rewards=[0.0, -0.1, 0.5, 0.3, 0.2],
        )
        
        # Episode 2: Attack reaches IMPACT
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 3, 4],
            actions=[0, 0, 0, 0, 4],
            rewards=[0.0, -0.1, -0.3, -0.5, -5.0],
        )
        
        # Episode 3: All BENIGN with false positives
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 0, 0, 0, 0],
            actions=[1, 0, 2, 0, 0],
            rewards=[-0.1, 0.0, -0.3, 0.0, 0.0],
        )
        
        collector.finalize_run("ppo", 0)
        
        run = collector.metrics["ppo"][0]
        
        # Attack Mitigation Rate: 2/3 episodes didn't reach IMPACT
        assert run.attack_mitigation_rate == pytest.approx(2/3, rel=1e-5)
        
        # False Positive Rate: 2 FPs out of (2+1+5) = 8 BENIGN steps
        # Episode 1: 2 BENIGN (stages 0,0,2,3 has 0 at positions 0,3), 0 FPs
        # Episode 2: 1 BENIGN (stage 0 at position 0), 0 FPs  
        # Episode 3: 5 BENIGN (all 0s), 2 FPs (actions 1 and 2)
        assert run.false_positive_rate == pytest.approx(2/8, rel=1e-5)
        
        # Availability Score: 1 / (1 + total_cost)
        assert run.availability_score > 0
        assert run.availability_score < 1
        
        # Evaluation metrics dict populated
        assert "attack_mitigation_rate" in run.evaluation_metrics
        assert "false_positive_rate" in run.evaluation_metrics
        assert "availability_score" in run.evaluation_metrics
    
    def test_get_algorithm_summary(self, collector):
        """Test algorithm summary statistics."""
        collector.start_run("ppo", 0, {})
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 0],
            actions=[0, 2, 0],
            rewards=[0.0, 0.5, 0.0],
        )
        collector.finalize_run("ppo", 0)
        
        collector.start_run("ppo", 1, {})
        collector.add_episode(
            "ppo", 1,
            attack_stages=[0, 1, 2, 0],
            actions=[0, 1, 2, 0],
            rewards=[0.0, 0.3, 0.5, 0.0],
        )
        collector.finalize_run("ppo", 1)
        
        summary = collector.get_algorithm_summary("ppo")
        
        assert summary["algorithm_name"] == "ppo"
        assert summary["num_runs"] == 2
        assert "avg_reward_mean" in summary
        assert "attack_mitigation_rate_mean" in summary
        assert "false_positive_rate_mean" in summary
        assert "availability_score_mean" in summary
    
    def test_get_comparison_data(self, collector):
        """Test comparison data for multiple algorithms."""
        for algo in ["dqn", "ppo", "a2c"]:
            collector.start_run(algo, 0, {})
            collector.add_episode(
                algo, 0,
                attack_stages=[0, 1, 0],
                actions=[0, 2, 0],
                rewards=[0.0, 0.5, 0.0],
            )
            collector.finalize_run(algo, 0)
        
        comparison = collector.get_comparison_data()
        
        assert len(comparison) == 3
        assert "dqn" in comparison
        assert "ppo" in comparison
        assert "a2c" in comparison
    
    def test_save_and_load_results(self, collector, temp_dir):
        """Test saving and loading results."""
        # Add some data
        collector.start_run("ppo", 0, {"lr": 3e-4})
        collector.add_episode(
            "ppo", 0,
            attack_stages=[0, 1, 2, 0],
            actions=[0, 1, 2, 0],
            rewards=[0.0, 0.3, 0.5, 0.0],
        )
        collector.finalize_run("ppo", 0)
        
        # Save
        collector.save_results("test_results.json")
        
        # Load into new collector
        new_collector = MetricsCollector(save_path=temp_dir)
        new_collector.load_results("test_results.json")
        
        # Verify data
        assert "ppo" in new_collector.metrics
        assert len(new_collector.metrics["ppo"]) == 1
        assert new_collector.metrics["ppo"][0].algorithm_name == "ppo"
        assert new_collector.metrics["ppo"][0].attack_mitigation_rate > 0
    
    def test_update_training_metrics(self, collector):
        """Test updating training metrics."""
        collector.start_run("dqn", 0, {})
        collector.update_training_metrics("dqn", 0, training_time=100.5, convergence_step=5000)
        
        run = collector.metrics["dqn"][0]
        assert run.training_time == 100.5
        assert run.convergence_step == 5000
    
    def test_update_evaluation_metrics_legacy(self, collector):
        """Test updating evaluation metrics from legacy format."""
        collector.start_run("a2c", 0, {})
        
        evaluation_results = {
            "avg_reward": 15.5,
            "std_reward": 2.3,
            "final_reward": 18.0,
            "episode_rewards": [10.0, 15.0, 18.0],
            "episode_lengths": [50, 60, 55],
            "attack_mitigation_rate": 0.9,
            "false_positive_rate": 0.05,
        }
        
        collector.update_evaluation_metrics("a2c", 0, evaluation_results)
        
        run = collector.metrics["a2c"][0]
        assert run.avg_reward == 15.5
        assert run.attack_mitigation_rate == 0.9
        assert run.false_positive_rate == 0.05


class TestActionCosts:
    """Tests for action costs constant."""
    
    def test_action_costs_values(self):
        """Test that action costs match PRD specification."""
        assert ACTION_COSTS[0] == 0.0   # OBSERVE
        assert ACTION_COSTS[1] == 0.1   # LOG
        assert ACTION_COSTS[2] == 0.3   # THROTTLE
        assert ACTION_COSTS[3] == 0.5   # BLOCK
        assert ACTION_COSTS[4] == 0.8   # ISOLATE
    
    def test_action_costs_length(self):
        """Test that there are 5 action costs."""
        assert len(ACTION_COSTS) == 5
    
    def test_action_costs_monotonic(self):
        """Test that action costs increase monotonically."""
        for i in range(len(ACTION_COSTS) - 1):
            assert ACTION_COSTS[i] < ACTION_COSTS[i + 1]
