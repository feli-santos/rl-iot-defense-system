"""First-order Markov attacker over kill-chain stages.

This module provides :class:`MarkovAttacker`, a small, seedable attacker that
samples the next kill-chain stage from a fixed first-order transition matrix.

The transition matrix is built to mirror
:meth:`src.generator.episode_generator.EpisodeGenerator._build_transition_matrix`
*exactly*, so the adversarial dynamics seen by the defender are unchanged from
the earlier LSTM-based generator (which was itself a high-fidelity imitator of
this same hand-built kill-chain grammar). Using the matrix directly removes the
need to train, version, and hash a neural checkpoint while preserving behaviour.

Kill-chain grammar (stages ``0 BENIGN, 1 RECON, 2 ACCESS, 3 MANEUVER, 4 IMPACT``):

- From ``BENIGN`` the attacker mostly stays benign and otherwise begins an
  attack, weighted by the dataset stage distribution.
- From an attack stage the attacker may persist or escalate to a higher stage
  (no regression); ``IMPACT`` is absorbing.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

# Defaults mirror EpisodeGeneratorConfig so dynamics match the legacy generator.
DEFAULT_NUM_STAGES = 5
DEFAULT_PERSISTENCE_WEIGHT = 0.3
DEFAULT_PROGRESSION_WEIGHT = 0.5
DEFAULT_SKIP_WEIGHT = 0.2


class MarkovAttacker:
    """Seedable first-order Markov attacker over kill-chain stages.

    Args:
        num_stages: Number of kill-chain stages (default 5).
        stage_distribution: Optional mapping ``stage_id -> prior weight`` used to
            weight the BENIGN row's attack-onset probabilities. Missing attack
            stages default to ``0.25`` (matching the legacy episode generator).
        persistence_weight: Probability mass for staying at the current attack
            stage.
        progression_weight: Weight for advancing exactly one stage.
        skip_weight: Weight for skipping ahead more than one stage (decreased by
            distance).
    """

    def __init__(
        self,
        num_stages: int = DEFAULT_NUM_STAGES,
        stage_distribution: Optional[Dict[int, float]] = None,
        persistence_weight: float = DEFAULT_PERSISTENCE_WEIGHT,
        progression_weight: float = DEFAULT_PROGRESSION_WEIGHT,
        skip_weight: float = DEFAULT_SKIP_WEIGHT,
    ) -> None:
        if num_stages < 2:
            raise ValueError(f"num_stages must be >= 2, got {num_stages}")
        self.num_stages = int(num_stages)
        self._stage_distribution: Dict[int, float] = dict(stage_distribution or {})
        self.persistence_weight = float(persistence_weight)
        self.progression_weight = float(progression_weight)
        self.skip_weight = float(skip_weight)
        self._transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self) -> np.ndarray:
        """Build the 5x5 transition matrix ``trans[i][j] = P(next=j | current=i)``.

        Mirrors ``EpisodeGenerator._build_transition_matrix`` exactly.
        """
        num_stages = self.num_stages
        trans = np.zeros((num_stages, num_stages))

        persist_w = self.persistence_weight
        progress_w = self.progression_weight
        skip_w = self.skip_weight

        # From BENIGN (0): stay benign or begin an attack, weighted by prior.
        trans[0, 0] = 0.4
        for j in range(1, num_stages):
            trans[0, j] = 0.6 * self._stage_distribution.get(j, 0.25)
        trans[0] /= trans[0].sum()

        # From attack stages: persist, progress, or skip (no regression).
        for i in range(1, num_stages):
            trans[i, i] = persist_w
            for j in range(i + 1, num_stages):
                distance = j - i
                if distance == 1:
                    trans[i, j] = progress_w
                else:
                    trans[i, j] = skip_w / distance

            if i == num_stages - 1:
                # IMPACT is absorbing.
                trans[i, i] = 1.0
            else:
                trans[i] /= trans[i].sum() if trans[i].sum() > 0 else 1.0

        return trans

    @property
    def transition_matrix(self) -> np.ndarray:
        """Read-only copy of the transition matrix."""
        return self._transition_matrix.copy()

    def sample_next(self, current_stage: int, rng: np.random.Generator) -> int:
        """Sample the next stage given the current stage.

        Args:
            current_stage: Current kill-chain stage id in ``[0, num_stages)``.
            rng: Seeded NumPy ``Generator`` for deterministic sampling.

        Returns:
            The next stage id.
        """
        stage = int(current_stage)
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"current_stage must be in [0, {self.num_stages}), got {stage}")
        return int(rng.choice(self.num_stages, p=self._transition_matrix[stage]))
