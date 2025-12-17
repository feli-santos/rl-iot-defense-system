"""
Episode Generator for Attack Sequence Training.

This module generates synthetic attack episodes following Kill Chain
grammar rules (PRD Section 3.2). Episodes are integer sequences
representing attack progression through Kill Chain stages.

Grammar Rules:
1. Progression: P(S_{t+1} > S_t) > 0 (attacks escalate)
2. Persistence: P(S_{t+1} = S_t) > 0 (attacks may sustain)
3. Reset: S_{t+1} = 0 only via external intervention (not within episode)

Episodes are used to train the Attack Sequence Generator (LSTM)
as a next-token predictor.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeGeneratorConfig:
    """Configuration for Episode Generator.
    
    Attributes:
        num_episodes: Total number of episodes to generate.
        min_length: Minimum episode length.
        max_length: Maximum episode length.
        num_stages: Number of Kill Chain stages (always 5).
        benign_start_prob: Probability of starting with BENIGN.
        progression_weight: Weight for escalation transitions.
        persistence_weight: Weight for staying at same stage.
        skip_weight: Weight for skipping intermediate stages.
    """
    
    num_episodes: int = 10000
    min_length: int = 5
    max_length: int = 30
    num_stages: int = 5
    benign_start_prob: float = 0.8
    progression_weight: float = 0.5
    persistence_weight: float = 0.3
    skip_weight: float = 0.2


class EpisodeGenerator:
    """Generates synthetic attack episodes following Kill Chain grammar.
    
    This class creates training data for the Attack Sequence Generator
    by producing integer sequences representing attack progressions.
    Transition probabilities can be influenced by dataset class distribution
    (Laplace-smoothed) to ensure generated episodes reflect real attack patterns.
    
    Attributes:
        config: Episode generation configuration.
        stage_distribution: Optional dataset stage distribution for weighting.
    
    Example:
        >>> generator = EpisodeGenerator(seed=42)
        >>> episode = generator.generate_episode()
        >>> episode
        [0, 0, 1, 1, 2, 4, 4]
    """
    
    def __init__(
        self,
        config: Optional[EpisodeGeneratorConfig] = None,
        stage_distribution: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the Episode Generator.
        
        Args:
            config: Generator configuration. Uses defaults if not provided.
            stage_distribution: Optional mapping of stage ID to sample count
                from dataset. Used to weight transition probabilities.
            seed: Random seed for reproducibility.
        """
        self._config = config or EpisodeGeneratorConfig()
        self._rng = np.random.default_rng(seed)
        
        # Store stage distribution (with Laplace smoothing)
        self._stage_distribution = self._apply_laplace_smoothing(
            stage_distribution
        )
        
        # Build transition probability matrix
        self._transition_probs = self._build_transition_matrix()
        
        logger.info(
            f"EpisodeGenerator initialized: "
            f"{self._config.num_episodes} episodes, "
            f"length [{self._config.min_length}, {self._config.max_length}]"
        )
    
    def _apply_laplace_smoothing(
        self,
        distribution: Optional[Dict[int, int]],
    ) -> Dict[int, float]:
        """Apply Laplace smoothing to stage distribution.
        
        Ensures all stages have non-zero probability, even if missing
        from the dataset.
        
        Args:
            distribution: Raw stage counts from dataset.
        
        Returns:
            Smoothed probability distribution over stages.
        """
        smoothing_alpha = 1.0  # Laplace smoothing parameter
        
        if distribution is None:
            # Uniform distribution if no dataset info
            return {i: 1.0 / self._config.num_stages for i in range(self._config.num_stages)}
        
        # Add smoothing and normalize
        total = sum(distribution.values()) + smoothing_alpha * self._config.num_stages
        
        smoothed = {}
        for stage_id in range(self._config.num_stages):
            count = distribution.get(stage_id, 0) + smoothing_alpha
            smoothed[stage_id] = count / total
            
            # Log warning for underrepresented stages
            if distribution.get(stage_id, 0) == 0:
                logger.warning(
                    f"Stage {stage_id} has no samples in dataset, "
                    f"using Laplace smoothing"
                )
        
        return smoothed
    
    def _build_transition_matrix(self) -> np.ndarray:
        """Build state transition probability matrix.
        
        Creates a 5x5 matrix where entry [i][j] is P(next=j | current=i).
        Respects Kill Chain grammar:
        - From BENIGN (0): can stay or transition to any attack stage
        - From attack stage: can persist or escalate (no regression)
        
        Returns:
            Transition probability matrix of shape (5, 5).
        """
        num_stages = self._config.num_stages
        trans = np.zeros((num_stages, num_stages))
        
        # Transition weights
        persist_w = self._config.persistence_weight
        progress_w = self._config.progression_weight
        skip_w = self._config.skip_weight
        
        # From BENIGN (0): can go to any stage
        # Weight attack stages by dataset distribution
        trans[0, 0] = 0.4  # Stay benign
        for j in range(1, num_stages):
            trans[0, j] = 0.6 * self._stage_distribution.get(j, 0.25)
        trans[0] /= trans[0].sum()  # Normalize
        
        # From attack stages: persist, progress, or skip
        for i in range(1, num_stages):
            # Persist at current stage
            trans[i, i] = persist_w
            
            # Progress/skip to higher stages
            higher_stages = list(range(i + 1, num_stages))
            if higher_stages:
                # Weight by distance (closer = more likely)
                for j in higher_stages:
                    distance = j - i
                    if distance == 1:
                        trans[i, j] = progress_w
                    else:
                        # Skip weight decreases with distance
                        trans[i, j] = skip_w / distance
            
            # If at IMPACT (4), must persist
            if i == num_stages - 1:
                trans[i, i] = 1.0
            else:
                # Normalize
                trans[i] /= trans[i].sum() if trans[i].sum() > 0 else 1.0
        
        return trans
    
    @property
    def config(self) -> EpisodeGeneratorConfig:
        """Get generator configuration."""
        return self._config
    
    @property
    def stage_distribution(self) -> Optional[Dict[int, float]]:
        """Get smoothed stage distribution."""
        return self._stage_distribution
    
    # =========================================================================
    # Episode Generation
    # =========================================================================
    
    def generate_episode(self) -> List[int]:
        """Generate a single attack episode.
        
        Returns:
            List of integers representing Kill Chain stages.
        """
        # Random episode length
        length = self._rng.integers(
            self._config.min_length,
            self._config.max_length + 1,
        )
        
        episode = []
        
        # Starting state
        if self._rng.random() < self._config.benign_start_prob:
            current_state = 0  # Start BENIGN
        else:
            # Start with an attack stage (weighted by distribution)
            attack_probs = np.array([
                self._stage_distribution.get(i, 0.25)
                for i in range(1, self._config.num_stages)
            ])
            attack_probs /= attack_probs.sum()
            current_state = int(self._rng.choice(range(1, self._config.num_stages), p=attack_probs))
        
        episode.append(current_state)
        
        # Generate remaining steps
        for _ in range(length - 1):
            # Sample next state from transition probabilities
            next_state = int(self._rng.choice(
                self._config.num_stages,
                p=self._transition_probs[current_state],
            ))
            
            episode.append(next_state)
            current_state = next_state
        
        return episode
    
    def generate_batch(self, n: int) -> List[List[int]]:
        """Generate a batch of episodes.
        
        Args:
            n: Number of episodes to generate.
        
        Returns:
            List of episode lists.
        """
        return [self.generate_episode() for _ in range(n)]
    
    def generate_all(self) -> List[List[int]]:
        """Generate all configured episodes.
        
        Returns:
            List of all episodes (config.num_episodes).
        """
        return self.generate_batch(self._config.num_episodes)
    
    # =========================================================================
    # Training Data Conversion
    # =========================================================================
    
    def to_training_sequences(
        self,
        episodes: List[List[int]],
        sequence_length: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """Convert episodes to input-target pairs for LSTM training.
        
        Creates sliding window sequences where the target is the
        next token after each sequence (next-token prediction).
        
        Args:
            episodes: List of episode lists.
            sequence_length: Length of input sequences.
        
        Returns:
            Tuple of (sequences, targets) where each sequence is
            `sequence_length` tokens and target is the next token.
        """
        sequences = []
        targets = []
        
        for episode in episodes:
            if len(episode) <= sequence_length:
                continue
            
            for i in range(len(episode) - sequence_length):
                seq = episode[i:i + sequence_length]
                target = episode[i + sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        return sequences, targets
    
    def to_numpy(
        self,
        episodes: List[List[int]],
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert episodes to numpy arrays for training.
        
        Args:
            episodes: List of episode lists.
            sequence_length: Length of input sequences.
        
        Returns:
            Tuple of (X, y) numpy arrays.
        """
        sequences, targets = self.to_training_sequences(
            episodes, sequence_length
        )
        
        X = np.array(sequences, dtype=np.int64)
        y = np.array(targets, dtype=np.int64)
        
        return X, y
