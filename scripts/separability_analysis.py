"""
Separability analysis for BENIGN vs RECON feature distributions.

This script compares class-conditional feature distributions sampled from
RealizationEngine and reports distance metrics that approximate class
separability.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.label_mapper import KillChainStage
from src.utils.realization_engine import RealizationEngine


@dataclass(frozen=True)
class SeparabilityConfig:
    """Configuration for separability analysis.

    Attributes:
        data_path: Path to processed dataset directory.
        sample_size: Number of samples per class.
        seed: Random seed for reproducibility.
        top_k_features: Number of top mean-diff features to display.
    """

    data_path: Path
    sample_size: int = 5000
    seed: int = 42
    top_k_features: int = 15


def _softmax(vector: np.ndarray) -> np.ndarray:
    """Compute softmax for a vector.

    Args:
        vector: Input vector.

    Returns:
        Softmax-normalized vector.
    """
    shifted = vector - np.max(vector)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL divergence with numerical stability.

    Args:
        p: First probability distribution.
        q: Second probability distribution.
        eps: Small constant to avoid log(0).

    Returns:
        KL divergence $D_{KL}(p || q)$.
    """
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_feature_columns(data_path: Path) -> list[str]:
    """Load feature column names from metadata.

    Args:
        data_path: Path to processed dataset directory.

    Returns:
        List of feature column names if available.
    """
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        return []
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return list(metadata.get("feature_columns", []))


def _find_keyword_features(
    feature_columns: Iterable[str],
    keywords: Iterable[str],
) -> list[str]:
    """Find feature columns containing any keyword.

    Args:
        feature_columns: Available feature column names.
        keywords: Keywords to search for.

    Returns:
        List of matching feature names.
    """
    lowered = [col.lower() for col in feature_columns]
    matches = []
    for original, lowered_col in zip(feature_columns, lowered):
        if any(keyword in lowered_col for keyword in keywords):
            matches.append(original)
    return matches


def run_analysis(config: SeparabilityConfig) -> None:
    """Run separability analysis and print results.

    Args:
        config: Separability configuration.
    """
    rng = np.random.default_rng(config.seed)
    engine = RealizationEngine(config.data_path, seed=config.seed)

    benign_samples = engine.sample_batch(
        KillChainStage.BENIGN,
        batch_size=config.sample_size,
        normalize=True,
    )
    recon_samples = engine.sample_batch(
        KillChainStage.RECON,
        batch_size=config.sample_size,
        normalize=True,
    )

    benign_mean = benign_samples.mean(axis=0)
    recon_mean = recon_samples.mean(axis=0)

    cosine_sim = _cosine_similarity(benign_mean, recon_mean)
    cosine_dist = 1.0 - cosine_sim

    benign_prob = _softmax(benign_mean)
    recon_prob = _softmax(recon_mean)
    kl_benign_recon = _safe_kl(benign_prob, recon_prob)
    kl_recon_benign = _safe_kl(recon_prob, benign_prob)
    sym_kl = 0.5 * (kl_benign_recon + kl_recon_benign)

    mean_diff = np.abs(benign_mean - recon_mean)
    top_indices = np.argsort(mean_diff)[::-1][: config.top_k_features]

    feature_columns = _load_feature_columns(config.data_path)
    if feature_columns:
        top_features = [feature_columns[idx] for idx in top_indices]
    else:
        top_features = [str(idx) for idx in top_indices]

    print("=" * 72)
    print("Separability Analysis: BENIGN vs RECON")
    print("=" * 72)
    print(f"Samples per class: {config.sample_size}")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    print(f"Cosine distance:   {cosine_dist:.6f}")
    print(f"KL(BENIGN||RECON): {kl_benign_recon:.6f}")
    print(f"KL(RECON||BENIGN): {kl_recon_benign:.6f}")
    print(f"Symmetric KL:      {sym_kl:.6f}")
    print("\nTop mean-difference features:")
    for rank, (idx, name) in enumerate(zip(top_indices, top_features), start=1):
        print(f"  {rank:02d}. {name} (|Δμ|={mean_diff[idx]:.6f})")

    keywords = ["rate", "per_second", "pps", "packets", "frequency"]
    if feature_columns:
        keyword_matches = _find_keyword_features(feature_columns, keywords)
        print("\nFeature keyword scan (rate/frequency):")
        if keyword_matches:
            for feature in keyword_matches:
                print(f"  - {feature}")
        else:
            print("  (no keyword matches found)")

    print("=" * 72)


def parse_args() -> SeparabilityConfig:
    """Parse CLI arguments.

    Returns:
        SeparabilityConfig instance.
    """
    parser = argparse.ArgumentParser(description="Separability analysis for BENIGN vs RECON")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/ciciot2023"),
        help="Path to processed dataset directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Samples per class",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=15,
        help="Number of top mean-diff features to display",
    )

    args = parser.parse_args()
    return SeparabilityConfig(
        data_path=args.data_path,
        sample_size=args.sample_size,
        seed=args.seed,
        top_k_features=args.top_k_features,
    )


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    run_analysis(config)


if __name__ == "__main__":
    main()
