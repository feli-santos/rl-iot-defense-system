"""
Tests for EpisodeGenerator.

This module tests the episode generation functionality that creates
synthetic attack sequences following Kill Chain grammar rules.
Per PRD Section 3.2, episodes must follow:
1. Progression: P(S_{t+1} > S_t) > 0 (attacks escalate)
2. Persistence: P(S_{t+1} = S_t) > 0 (attacks may sustain)
3. Reset: S_{t+1} = 0 only via external intervention
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig


class TestEpisodeGeneratorConfig:
    """Tests for EpisodeGeneratorConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = EpisodeGeneratorConfig()
        
        assert config.num_episodes > 0
        assert config.min_length >= 1
        assert config.max_length > config.min_length
        assert config.num_stages == 5

    def test_custom_config(self) -> None:
        """Should accept custom configuration."""
        config = EpisodeGeneratorConfig(
            num_episodes=5000,
            min_length=10,
            max_length=50,
        )
        
        assert config.num_episodes == 5000
        assert config.min_length == 10
        assert config.max_length == 50


class TestEpisodeGeneratorInitialization:
    """Tests for EpisodeGenerator initialization."""

    def test_init_default(self) -> None:
        """Should initialize with default config."""
        generator = EpisodeGenerator()
        assert generator is not None

    def test_init_with_config(self) -> None:
        """Should initialize with custom config."""
        config = EpisodeGeneratorConfig(num_episodes=100)
        generator = EpisodeGenerator(config=config)
        
        assert generator.config.num_episodes == 100

    def test_init_with_stage_distribution(self) -> None:
        """Should initialize with dataset stage distribution."""
        # Simulate dataset distribution (from EDA)
        stage_distribution = {
            0: 1000,   # BENIGN: many samples
            1: 500,    # RECON
            2: 200,    # ACCESS
            3: 100,    # MANEUVER
            4: 5000,   # IMPACT: most samples (attacks)
        }
        
        generator = EpisodeGenerator(stage_distribution=stage_distribution)
        
        assert generator.stage_distribution is not None

    def test_init_with_seed(self) -> None:
        """Should be reproducible with seed."""
        gen1 = EpisodeGenerator(seed=42)
        gen2 = EpisodeGenerator(seed=42)
        
        ep1 = gen1.generate_episode()
        ep2 = gen2.generate_episode()
        
        assert ep1 == ep2


class TestEpisodeGeneratorSingleEpisode:
    """Tests for single episode generation."""

    @pytest.fixture
    def generator(self) -> EpisodeGenerator:
        """Create generator with fixed seed."""
        return EpisodeGenerator(seed=42)

    def test_generate_episode_returns_list(
        self, generator: EpisodeGenerator
    ) -> None:
        """Should return a list of integers."""
        episode = generator.generate_episode()
        
        assert isinstance(episode, list)
        assert all(isinstance(s, int) for s in episode)

    def test_episode_length_within_bounds(
        self, generator: EpisodeGenerator
    ) -> None:
        """Episode length should be within configured bounds."""
        config = generator.config
        
        for _ in range(20):
            episode = generator.generate_episode()
            assert config.min_length <= len(episode) <= config.max_length

    def test_episode_starts_with_benign(
        self, generator: EpisodeGenerator
    ) -> None:
        """Most episodes should start with BENIGN (0)."""
        starts = [generator.generate_episode()[0] for _ in range(50)]
        
        # At least 80% should start with BENIGN
        benign_starts = sum(1 for s in starts if s == 0)
        assert benign_starts >= 40

    def test_episode_values_in_valid_range(
        self, generator: EpisodeGenerator
    ) -> None:
        """All stages should be 0-4."""
        episode = generator.generate_episode()
        
        assert all(0 <= s <= 4 for s in episode)

    def test_no_self_reset_to_benign(
        self, generator: EpisodeGenerator
    ) -> None:
        """Episodes should not spontaneously reset to BENIGN.
        
        Per PRD: Reset to 0 only via external intervention.
        Once attacking, should not return to BENIGN within episode.
        """
        for _ in range(50):
            episode = generator.generate_episode()
            
            # Find first attack (non-zero)
            attack_started = False
            for stage in episode:
                if stage > 0:
                    attack_started = True
                elif attack_started and stage == 0:
                    # Found a reset after attack started - not allowed
                    pytest.fail(
                        f"Episode reset to BENIGN after attack: {episode}"
                    )


class TestEpisodeGeneratorGrammar:
    """Tests for Kill Chain grammar rules."""

    @pytest.fixture
    def generator(self) -> EpisodeGenerator:
        """Create generator with fixed seed."""
        return EpisodeGenerator(seed=42)

    def test_progression_rule(self, generator: EpisodeGenerator) -> None:
        """Attacks should generally escalate (progression rule).
        
        P(S_{t+1} > S_t) > 0
        """
        escalations = 0
        total_transitions = 0
        
        for _ in range(100):
            episode = generator.generate_episode()
            
            for i in range(len(episode) - 1):
                if episode[i] > 0:  # Only count attack transitions
                    total_transitions += 1
                    if episode[i + 1] > episode[i]:
                        escalations += 1
        
        # At least some escalations should occur
        assert escalations > 0, "No escalations found"
        escalation_rate = escalations / total_transitions
        assert escalation_rate > 0.05, f"Escalation rate too low: {escalation_rate}"

    def test_persistence_rule(self, generator: EpisodeGenerator) -> None:
        """Attacks may persist at same stage.
        
        P(S_{t+1} = S_t) > 0
        """
        persistences = 0
        total_transitions = 0
        
        for _ in range(100):
            episode = generator.generate_episode()
            
            for i in range(len(episode) - 1):
                if episode[i] > 0:  # Only count attack transitions
                    total_transitions += 1
                    if episode[i + 1] == episode[i]:
                        persistences += 1
        
        # Some persistence should occur
        assert persistences > 0, "No persistence found"

    def test_no_regression_allowed(self, generator: EpisodeGenerator) -> None:
        """Once escalated, should not regress to lower stage (except via reset).
        
        Note: Regression to lower non-zero stage is not allowed within episode.
        """
        for _ in range(50):
            episode = generator.generate_episode()
            max_stage_seen = 0
            
            for stage in episode:
                if stage > 0 and stage < max_stage_seen:
                    pytest.fail(
                        f"Regression from {max_stage_seen} to {stage} in {episode}"
                    )
                max_stage_seen = max(max_stage_seen, stage)

    def test_can_skip_stages(self, generator: EpisodeGenerator) -> None:
        """Should be possible to skip intermediate stages.
        
        e.g., RECON -> IMPACT (skipping ACCESS and MANEUVER)
        """
        skips_found = False
        
        for _ in range(200):
            episode = generator.generate_episode()
            
            for i in range(len(episode) - 1):
                if episode[i] > 0 and episode[i + 1] > episode[i] + 1:
                    skips_found = True
                    break
            
            if skips_found:
                break
        
        assert skips_found, "No stage skips found in episodes"


class TestEpisodeGeneratorBatch:
    """Tests for batch episode generation."""

    @pytest.fixture
    def generator(self) -> EpisodeGenerator:
        """Create generator with fixed seed."""
        config = EpisodeGeneratorConfig(num_episodes=100)
        return EpisodeGenerator(config=config, seed=42)

    def test_generate_batch_returns_list(
        self, generator: EpisodeGenerator
    ) -> None:
        """Should return list of episodes."""
        episodes = generator.generate_batch(n=10)
        
        assert isinstance(episodes, list)
        assert len(episodes) == 10

    def test_generate_all_episodes(
        self, generator: EpisodeGenerator
    ) -> None:
        """Should generate configured number of episodes."""
        episodes = generator.generate_all()
        
        assert len(episodes) == generator.config.num_episodes

    def test_batch_diversity(self, generator: EpisodeGenerator) -> None:
        """Batch should contain diverse episodes."""
        episodes = generator.generate_batch(n=50)
        
        # Convert to tuples for set operations
        unique_episodes = set(tuple(ep) for ep in episodes)
        
        # Should have reasonable diversity
        assert len(unique_episodes) > 30


class TestEpisodeGeneratorDistribution:
    """Tests for dataset-driven transition probabilities."""

    def test_uses_stage_distribution(self) -> None:
        """Transition probabilities should reflect dataset distribution."""
        # Dataset heavily weighted toward IMPACT
        stage_distribution = {
            0: 100,
            1: 100,
            2: 100,
            3: 100,
            4: 10000,  # Much more IMPACT
        }
        
        generator = EpisodeGenerator(
            stage_distribution=stage_distribution,
            seed=42,
        )
        
        # Generate many episodes and count final stages
        final_stages = []
        for _ in range(100):
            episode = generator.generate_episode()
            final_stages.append(episode[-1])
        
        # IMPACT should be most common final stage
        impact_count = sum(1 for s in final_stages if s == 4)
        assert impact_count > 30  # At least 30% should end at IMPACT

    def test_laplace_smoothing_for_missing_stages(self) -> None:
        """Should handle missing stages via Laplace smoothing."""
        # Distribution missing MANEUVER stage
        stage_distribution = {
            0: 100,
            1: 100,
            2: 100,
            # 3: missing!
            4: 100,
        }
        
        generator = EpisodeGenerator(
            stage_distribution=stage_distribution,
            seed=42,
        )
        
        # Should still be able to generate episodes
        episodes = generator.generate_batch(n=10)
        assert len(episodes) == 10


class TestEpisodeGeneratorForTraining:
    """Tests for generating training data for LSTM."""

    @pytest.fixture
    def generator(self) -> EpisodeGenerator:
        """Create generator for training data."""
        config = EpisodeGeneratorConfig(
            num_episodes=100,
            min_length=10,
            max_length=30,
        )
        return EpisodeGenerator(config=config, seed=42)

    def test_to_training_sequences(
        self, generator: EpisodeGenerator
    ) -> None:
        """Should convert episodes to input-target pairs for LSTM."""
        episodes = generator.generate_batch(n=10)
        
        sequences, targets = generator.to_training_sequences(
            episodes,
            sequence_length=5,
        )
        
        assert len(sequences) == len(targets)
        assert all(len(seq) == 5 for seq in sequences)

    def test_training_sequences_targets_are_next_token(
        self, generator: EpisodeGenerator
    ) -> None:
        """Targets should be the next token after each sequence."""
        episode = [0, 1, 1, 2, 3, 4]
        
        sequences, targets = generator.to_training_sequences(
            [episode],
            sequence_length=3,
        )
        
        # For sequence [0, 1, 1], target should be 2
        # For sequence [1, 1, 2], target should be 3
        # etc.
        expected_pairs = [
            ([0, 1, 1], 2),
            ([1, 1, 2], 3),
            ([1, 2, 3], 4),
        ]
        
        for i, (expected_seq, expected_target) in enumerate(expected_pairs):
            assert list(sequences[i]) == expected_seq
            assert targets[i] == expected_target

    def test_to_numpy_arrays(self, generator: EpisodeGenerator) -> None:
        """Should convert to numpy arrays for training."""
        episodes = generator.generate_batch(n=10)
        
        X, y = generator.to_numpy(episodes, sequence_length=5)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.dtype == np.int64 or X.dtype == np.int32
        assert y.dtype == np.int64 or y.dtype == np.int32

    def test_to_numpy_alignment_simple_episode(self) -> None:
        """Verify exact alignment: y is next stage after X window."""
        generator = EpisodeGenerator(seed=42)
        episode = [0, 1, 2, 3, 4]
        
        X, y = generator.to_numpy([episode], sequence_length=3)
        
        # Episode [0,1,2,3,4] with seq_len=3 should produce:
        # X[0]=[0,1,2], y[0]=3
        # X[1]=[1,2,3], y[1]=4
        assert X.shape == (2, 3)
        assert y.shape == (2,)
        
        assert list(X[0]) == [0, 1, 2]
        assert y[0] == 3
        assert list(X[1]) == [1, 2, 3]
        assert y[1] == 4
