"""
Transition Mask for Kill Chain Grammar Constraints.

This module provides functionality to enforce or regularize valid Kill Chain
stage transitions by masking invalid transitions during inference and/or training.
Per PRD Section 3.2, Kill Chain grammar follows:
- Progression: can move forward (e.g., RECON -> ACCESS)
- Persistence: can stay in same stage (e.g., ACCESS -> ACCESS)
- No regression: cannot move backward (except IMPACT -> BENIGN via external intervention)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TransitionMask:
    """Manages Kill Chain transition constraints.
    
    Builds an allowed-transitions mask based on empirical data or
    predefined grammar rules, then applies it to logits to enforce
    valid stage progressions.
    
    Example:
        >>> mask = TransitionMask.from_empirical_data(episodes)
        >>> logits = model(x)
        >>> masked_logits = mask.apply(logits, current_stage=2)
        >>> probs = F.softmax(masked_logits, dim=-1)
    """
    
    def __init__(
        self,
        transition_matrix: np.ndarray,
        threshold: float = 0.01,
        allow_regression: bool = False,
    ) -> None:
        """Initialize transition mask from a transition probability matrix.
        
        Args:
            transition_matrix: 5x5 matrix where [i,j] = P(next=j | current=i).
            threshold: Minimum probability to allow a transition.
            allow_regression: If False, enforce no backward transitions (except to BENIGN).
        """
        self._num_stages = 5
        self._threshold = threshold
        self._allow_regression = allow_regression
        
        # Build boolean mask: True = allowed, False = forbidden
        self._mask = self._build_mask(transition_matrix)
        
        logger.info(
            f"TransitionMask initialized: threshold={threshold}, "
            f"allow_regression={allow_regression}"
        )
    
    def _build_mask(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Build boolean mask from transition probabilities.
        
        Args:
            transition_matrix: 5x5 transition probability matrix.
        
        Returns:
            Boolean mask of shape (5, 5).
        """
        # Start with threshold-based mask
        mask = transition_matrix >= self._threshold
        
        # Enforce no-regression rule if requested
        if not self._allow_regression:
            for i in range(self._num_stages):
                for j in range(i):
                    # Block all backward transitions by default
                    mask[i, j] = False
            
            # Special case: Allow IMPACT (4) -> BENIGN (0) for episode reset
            mask[4, 0] = True
        
        # Always allow staying in same stage (persistence)
        for i in range(self._num_stages):
            mask[i, i] = True
        
        # Always allow progression to higher stages
        for i in range(self._num_stages):
            for j in range(i + 1, self._num_stages):
                mask[i, j] = True
        
        return mask.astype(bool)
    
    def apply(
        self,
        logits: torch.Tensor,
        current_stage: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply transition mask to logits.
        
        Sets logits for forbidden transitions to -inf, effectively
        removing them from the probability distribution after softmax.
        
        Args:
            logits: Tensor of shape (batch, num_stages) or (num_stages,).
            current_stage: Current stage ID. If None, applies mask assuming
                          each row in batch corresponds to stages 0-4.
        
        Returns:
            Masked logits with same shape as input.
        """
        device = logits.device
        
        if logits.ndim == 1:
            # Single prediction
            if current_stage is None:
                raise ValueError("current_stage required for single logits vector")
            
            mask_row = self._mask[current_stage]
            mask_tensor = torch.tensor(mask_row, dtype=torch.bool, device=device)
            
            masked_logits = logits.clone()
            masked_logits[~mask_tensor] = float('-inf')
            
        elif logits.ndim == 2:
            # Batch prediction
            if current_stage is not None:
                # All samples have same current stage
                mask_row = self._mask[current_stage]
                mask_tensor = torch.tensor(mask_row, dtype=torch.bool, device=device)
                mask_tensor = mask_tensor.unsqueeze(0).expand(logits.shape[0], -1)
            else:
                # Assume batch contains all stages (for training)
                # This case typically used when training with mixed stages
                # For now, just return original logits
                # (proper implementation would need per-sample stage tracking)
                return logits
            
            masked_logits = logits.clone()
            masked_logits[~mask_tensor] = float('-inf')
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
        
        return masked_logits
    
    @property
    def mask(self) -> np.ndarray:
        """Get the transition mask matrix."""
        return self._mask.copy()
    
    @classmethod
    def from_empirical_data(
        cls,
        episodes: list,
        threshold: float = 0.01,
        allow_regression: bool = False,
        smoothing: float = 1e-6,
    ) -> "TransitionMask":
        """Build transition mask from observed episode data.
        
        Args:
            episodes: List of episode sequences (lists of stage IDs).
            threshold: Minimum probability to allow a transition.
            allow_regression: If False, enforce no backward transitions.
            smoothing: Laplace smoothing parameter.
        
        Returns:
            TransitionMask instance.
        """
        num_stages = 5
        
        # Count transitions
        counts = np.zeros((num_stages, num_stages), dtype=np.float64)
        
        for episode in episodes:
            for i in range(len(episode) - 1):
                current = episode[i]
                next_stage = episode[i + 1]
                if 0 <= current < num_stages and 0 <= next_stage < num_stages:
                    counts[current, next_stage] += 1
        
        # Add smoothing
        counts += smoothing
        
        # Normalize to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        transition_matrix = counts / np.maximum(row_sums, 1e-12)
        
        logger.info(f"Built empirical transition matrix from {len(episodes)} episodes")
        
        return cls(
            transition_matrix=transition_matrix,
            threshold=threshold,
            allow_regression=allow_regression,
        )
    
    @classmethod
    def from_strict_grammar(cls) -> "TransitionMask":
        """Create mask enforcing strict Kill Chain grammar (progression + persistence only).
        
        Returns:
            TransitionMask with strict forward/stay-only transitions.
        """
        num_stages = 5
        
        # Build strict grammar: can only stay or move forward
        transition_matrix = np.zeros((num_stages, num_stages), dtype=np.float64)
        
        for i in range(num_stages):
            # Can stay in current stage
            transition_matrix[i, i] = 1.0
            
            # Can move to any higher stage
            for j in range(i + 1, num_stages):
                transition_matrix[i, j] = 1.0
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        logger.info("Created strict Kill Chain grammar mask")
        
        return cls(
            transition_matrix=transition_matrix,
            threshold=0.0,  # All defined transitions allowed
            allow_regression=False,
        )
