"""
Adversarial RL Algorithms.

This package provides RL algorithms for training Blue Team defense agents.
"""

from src.algorithms.adversarial_algorithm import (
    AdversarialAlgorithm,
    AdversarialAlgorithmConfig,
    create_algorithm,
)

__all__ = [
    "AdversarialAlgorithm",
    "AdversarialAlgorithmConfig",
    "create_algorithm",
]
