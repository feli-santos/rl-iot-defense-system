"""
Measure improved target distribution with temperature + coverage.

Diagnostic script to verify that tempered distribution and minimum stage
coverage produce training targets where all stages are well-represented.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig


def measure_target_distribution(
    stage_distribution: Dict[int, int],
    temperature: float,
    min_stage_coverage: Dict[int, float],
    num_episodes: int = 10000,
    sequence_length: int = 5,
    seed: int = 42,
) -> Dict[int, int]:
    """Measure training target distribution."""
    config = EpisodeGeneratorConfig(
        num_episodes=num_episodes,
        distribution_temperature=temperature,
        min_stage_coverage=min_stage_coverage,
    )
    
    generator = EpisodeGenerator(
        config=config,
        stage_distribution=stage_distribution,
        seed=seed,
    )
    
    episodes = generator.generate_all()
    X, y = generator.to_numpy(episodes, sequence_length=sequence_length)
    
    # Count y targets
    unique, counts = np.unique(y, return_counts=True)
    y_counts = dict(zip(unique.tolist(), counts.tolist()))
    
    # Ensure all stages present
    for stage in range(5):
        if stage not in y_counts:
            y_counts[stage] = 0
    
    return y_counts


def main():
    # Load actual CICIoT2023 stage distribution
    metadata_path = Path("data/processed/ciciot2023/metadata.json")
    if not metadata_path.exists():
        print("Metadata not found. Using synthetic distribution.")
        stage_distribution = {0: 21268, 1: 3885, 2: 425, 3: 24042, 4: 450380}
    else:
        with open(metadata_path) as f:
            metadata = json.load(f)
        stage_distribution = metadata.get("stage_counts", metadata.get("stage_distribution", {}))
        # Convert string keys to int
        stage_distribution = {int(k): v for k, v in stage_distribution.items()}
    
    print("=" * 70)
    print("Target Distribution Measurement")
    print("=" * 70)
    print(f"\nDataset stage distribution: {stage_distribution}")
    
    # Scenario 1: Baseline (no mitigation)
    print("\n[Scenario 1] Baseline: temperature=1.0, no coverage enforcement")
    y_baseline = measure_target_distribution(
        stage_distribution=stage_distribution,
        temperature=1.0,
        min_stage_coverage=None,
    )
    print(f"y target counts: {dict(sorted(y_baseline.items()))}")
    total = sum(y_baseline.values())
    print(f"y target fractions: {[f'{y_baseline.get(i, 0)/total:.4f}' for i in range(5)]}")
    
    # Scenario 2: Temperature only
    print("\n[Scenario 2] Temperature=0.4 only (no coverage)")
    y_temp = measure_target_distribution(
        stage_distribution=stage_distribution,
        temperature=0.4,
        min_stage_coverage=None,
    )
    print(f"y target counts: {dict(sorted(y_temp.items()))}")
    total = sum(y_temp.values())
    print(f"y target fractions: {[f'{y_temp.get(i, 0)/total:.4f}' for i in range(5)]}")
    
    # Scenario 3: Temperature + minimum coverage
    print("\n[Scenario 3] Temperature=0.4 + min_stage_coverage={1:0.3, 2:0.3, 3:0.3}")
    y_combined = measure_target_distribution(
        stage_distribution=stage_distribution,
        temperature=0.4,
        min_stage_coverage={1: 0.3, 2: 0.3, 3: 0.3},
    )
    print(f"y target counts: {dict(sorted(y_combined.items()))}")
    total = sum(y_combined.values())
    print(f"y target fractions: {[f'{y_combined.get(i, 0)/total:.4f}' for i in range(5)]}")
    
    # Scenario 4: More aggressive settings
    print("\n[Scenario 4] Temperature=0.3 + min_stage_coverage={0:0.4, 1:0.4, 2:0.4, 3:0.4}")
    y_aggressive = measure_target_distribution(
        stage_distribution=stage_distribution,
        temperature=0.3,
        min_stage_coverage={0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4},
    )
    print(f"y target counts: {dict(sorted(y_aggressive.items()))}")
    total = sum(y_aggressive.values())
    print(f"y target fractions: {[f'{y_aggressive.get(i, 0)/total:.4f}' for i in range(5)]}")
    
    print("\n" + "=" * 70)
    print("Improvement Summary")
    print("=" * 70)
    
    # Check if minority stages have sufficient representation
    for stage in [0, 1, 2, 3]:
        baseline_count = y_baseline.get(stage, 0)
        aggressive_count = y_aggressive.get(stage, 0)
        improvement = aggressive_count - baseline_count
        print(f"Stage {stage}: {baseline_count} → {aggressive_count} (+{improvement} targets)")
    
    # Check if all stages have at least 500 targets
    all_sufficient = all(y_aggressive.get(i, 0) >= 500 for i in range(5))
    if all_sufficient:
        print("\n✓ All stages have ≥500 training targets!")
    else:
        print("\n✗ Some stages still have <500 training targets")
        for i in range(5):
            if y_aggressive.get(i, 0) < 500:
                print(f"  Stage {i}: {y_aggressive.get(i, 0)} < 500")


if __name__ == "__main__":
    main()
