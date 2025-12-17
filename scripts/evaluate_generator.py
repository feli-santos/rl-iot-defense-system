"""
Evaluate trained generator model and print per-stage metrics.

Loads the trained LSTM and evaluates it on episodes generated with
the same temperature/coverage settings to verify per-stage F1 improvements.
"""

import json
from pathlib import Path
import torch

from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from src.generator.attack_sequence_generator import AttackSequenceGenerator
from src.training.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig


def main():
    # Load metadata
    metadata_path = Path("data/processed/ciciot2023/metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    stage_distribution = metadata.get("stage_counts", {})
    stage_distribution = {int(k): v for k, v in stage_distribution.items()}
    
    # Create episode generator with same settings as training
    episode_config = EpisodeGeneratorConfig(
        num_episodes=5000,  # Smaller for evaluation
        distribution_temperature=0.3,
        min_stage_coverage={0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4},
    )
    
    print("=" * 70)
    print("Generator Evaluation - Per-Stage Metrics")
    print("=" * 70)
    print(f"\nGenerating {episode_config.num_episodes} evaluation episodes...")
    print(f"Temperature={episode_config.distribution_temperature}, Coverage={episode_config.min_stage_coverage}")
    
    generator = EpisodeGenerator(
        config=episode_config,
        stage_distribution=stage_distribution,
        seed=123,  # Different seed from training
    )
    episodes = generator.generate_all()
    
    # Load trained model
    model_path = "artifacts/generator/attack_sequence_generator.pth"
    
    print(f"\nLoading model from {model_path}")
    model = AttackSequenceGenerator.load(model_path, device="cpu")
    model.eval()
    
    # Create trainer just for evaluate() function
    training_config = GeneratorTrainingConfig(
        epochs=1,  # Dummy
        output_dir=Path("artifacts/generator"),
        device="cpu",
        sequence_length=5,
    )
    trainer = GeneratorTrainer(config=training_config, model_config=None)
    trainer._model = model
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(episodes)
    
    # Print results
    print("\n" + "=" * 70)
    print("Overall Metrics")
    print("=" * 70)
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Perplexity:    {metrics['perplexity']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Transition Acc:{metrics['transition_accuracy']:.4f}")
    
    print("\n" + "=" * 70)
    print("Per-Stage F1 Scores")
    print("=" * 70)
    f1_scores = []
    for stage_id in range(5):
        f1 = metrics[f'f1_stage_{stage_id}']
        f1_scores.append(f1)
        print(f"Stage {stage_id}: {f1:.4f}")
    
    # Check if all stages have F1 > 0.1
    min_f1 = min(f1_scores)
    if min_f1 > 0.1:
        print(f"\n✓ All stages have F1 > 0.1 (min={min_f1:.4f})")
    else:
        print(f"\n✗ Some stages have F1 ≤ 0.1 (min={min_f1:.4f})")
    
    # Print target distribution
    import numpy as np
    X, y = generator.to_numpy(episodes, sequence_length=5)
    unique, counts = np.unique(y, return_counts=True)
    y_counts = dict(zip(unique.tolist(), counts.tolist()))
    print("\n" + "=" * 70)
    print("Target Distribution")
    print("=" * 70)
    for stage in range(5):
        count = y_counts.get(stage, 0)
        frac = count / len(y)
        print(f"Stage {stage}: {count:6d} ({frac:.4f})")


if __name__ == "__main__":
    main()
