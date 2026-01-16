"""
Tests for TransitionMask.

This module tests transition masking for Kill Chain grammar enforcement.
"""

import numpy as np
import pytest
import torch

from src.generator.transition_mask import TransitionMask


class TestTransitionMaskFromStrictGrammar:
    """Tests for strict grammar mask."""

    def test_strict_mask_creation(self) -> None:
        """Should create strict forward-only grammar."""
        mask = TransitionMask.from_strict_grammar()
        
        assert mask is not None
        assert mask.mask.shape == (5, 5)

    def test_strict_mask_allows_persistence(self) -> None:
        """Strict mask should allow staying in same stage."""
        mask = TransitionMask.from_strict_grammar()
        
        mask_matrix = mask.mask
        
        # Diagonal should all be True (can persist)
        for i in range(5):
            assert mask_matrix[i, i], f"Stage {i} should allow persistence"

    def test_strict_mask_allows_progression(self) -> None:
        """Strict mask should allow forward progression."""
        mask = TransitionMask.from_strict_grammar()
        
        mask_matrix = mask.mask
        
        # Can move from any stage to higher stage
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask_matrix[i, j], f"Should allow {i} -> {j}"

    def test_strict_mask_blocks_regression(self) -> None:
        """Strict mask should block backward transitions."""
        mask = TransitionMask.from_strict_grammar()
        
        mask_matrix = mask.mask
        
        # Cannot move backward (except IMPACT->BENIGN handled elsewhere)
        for i in range(1, 5):
            for j in range(1, i):
                assert not mask_matrix[i, j], f"Should block {i} -> {j}"


class TestTransitionMaskFromEmpiricalData:
    """Tests for empirical data-based mask."""

    @pytest.fixture
    def sample_episodes(self) -> list:
        """Sample episodes with known transitions."""
        return [
            [0, 1, 2, 3, 4],  # Full progression
            [0, 1, 1, 2, 4],  # Skip stage 3
            [1, 2, 2, 2, 4],  # Persistence
            [0, 4, 4, 4],     # Direct to IMPACT
        ]

    def test_empirical_mask_creation(self, sample_episodes: list) -> None:
        """Should create mask from episode data."""
        mask = TransitionMask.from_empirical_data(
            sample_episodes,
            threshold=0.1,
        )
        
        assert mask is not None
        assert mask.mask.shape == (5, 5)

    def test_empirical_mask_reflects_data(self, sample_episodes: list) -> None:
        """Mask should reflect observed transitions."""
        mask = TransitionMask.from_empirical_data(
            sample_episodes,
            threshold=0.0,  # Allow all observed
        )
        
        mask_matrix = mask.mask
        
        # 0 -> 1 observed
        assert mask_matrix[0, 1]
        
        # 0 -> 4 observed
        assert mask_matrix[0, 4]
        
        # 1 -> 2 observed
        assert mask_matrix[1, 2]


class TestTransitionMaskApplication:
    """Tests for applying masks to logits."""

    @pytest.fixture
    def strict_mask(self) -> TransitionMask:
        """Get strict grammar mask."""
        return TransitionMask.from_strict_grammar()

    def test_apply_mask_to_single_logits(self, strict_mask: TransitionMask) -> None:
        """Should mask single logits vector."""
        # Current stage = 2 (ACCESS)
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        masked_logits = strict_mask.apply(logits, current_stage=2)
        
        # Should allow: 2, 3, 4 (persist, MANEUVER, IMPACT)
        # Should block: 0, 1 (BENIGN, RECON)
        assert torch.isinf(masked_logits[0]) and masked_logits[0] < 0
        assert torch.isinf(masked_logits[1]) and masked_logits[1] < 0
        assert not torch.isinf(masked_logits[2])
        assert not torch.isinf(masked_logits[3])
        assert not torch.isinf(masked_logits[4])

    def test_apply_mask_to_batch_logits(self, strict_mask: TransitionMask) -> None:
        """Should mask batch of logits."""
        # Batch of 3 samples, all at stage 1 (RECON)
        logits = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        
        masked_logits = strict_mask.apply(logits, current_stage=1)
        
        # Should block stage 0 for all samples
        assert torch.all(torch.isinf(masked_logits[:, 0]))
        
        # Should allow 1, 2, 3, 4
        assert not torch.any(torch.isinf(masked_logits[:, 1]))
        assert not torch.any(torch.isinf(masked_logits[:, 2]))

    def test_masked_softmax_is_valid_distribution(
        self, strict_mask: TransitionMask
    ) -> None:
        """Masked logits should produce valid probability distribution."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        masked_logits = strict_mask.apply(logits, current_stage=2)
        
        # Softmax should work and sum to 1
        probs = torch.softmax(masked_logits, dim=-1)
        
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
        
        # Masked positions should have zero probability
        assert probs[0] == 0.0
        assert probs[1] == 0.0
        
        # Allowed positions should have positive probability
        assert probs[2] > 0.0
        assert probs[3] > 0.0
        assert probs[4] > 0.0


class TestTransitionMaskWithGenerator:
    """Integration tests with AttackSequenceGenerator."""

    def test_generator_with_mask_respects_constraints(self) -> None:
        """Generator with mask should only produce valid transitions."""
        from src.generator.attack_sequence_generator import AttackSequenceGenerator
        
        model = AttackSequenceGenerator()
        mask = TransitionMask.from_strict_grammar()
        model.set_transition_mask(mask)
        
        # Generate from stage 2 (ACCESS) many times
        history = [0, 1, 2]
        
        sampled_stages = []
        for _ in range(100):
            next_stage = model.sample_next(history)
            sampled_stages.append(next_stage)
        
        # Should never sample 0 or 1 (regression blocked)
        assert 0 not in sampled_stages
        assert 1 not in sampled_stages
        
        # Should only sample 2, 3, or 4
        assert all(s in [2, 3, 4] for s in sampled_stages)

    def test_generator_without_mask_can_regress(self) -> None:
        """Generator without mask can produce any transition (including invalid)."""
        from src.generator.attack_sequence_generator import AttackSequenceGenerator
        
        model = AttackSequenceGenerator()
        # No mask set
        
        # With enough samples, untrained model could sample any stage
        # (This test is probabilistic but with 1000 samples should be reliable)
        history = [0, 1, 2, 3, 4]
        
        sampled_stages = set()
        for _ in range(1000):
            next_stage = model.sample_next(history)
            sampled_stages.add(next_stage)
        
        # Should be able to sample most stages (untrained model is random)
        assert len(sampled_stages) >= 3, "Untrained model should sample multiple stages"
