#!/usr/bin/env python3
"""
IoT Defense System - Adversarial Training Pipeline

Main entry point for the adversarial RL defense system. This system uses:
- Attacker: first-order Markov chain over kill-chain stages (see MarkovAttacker)
- Blue Team: RL agents (DQN/PPO/A2C) to learn defense policies

Modes:
- process-data: Prepare CICIoT2023 dataset for adversarial environment

For Blue-Team training, benchmark evaluation and ablation sweeps, use the
canonical Makefile targets: ``make blue-team``, ``make benchmark-eval``,
``make ablation-ood-eval``, or the scripts directly.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging(log_level: str = "INFO", log_dir: Path = Path("results/logs")) -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "adversarial_training.log"),
            logging.StreamHandler(),
        ],
    )


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IoT Defense System - Adversarial Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw CICIoT2023 dataset
  python main.py --mode process-data

  # For Blue-Team training, benchmark/ablation evaluation, use the Makefile:
  #   make blue-team
  #   make benchmark-eval
  #   make ablation-ood-eval
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["process-data"],
        default="process-data",
        help="Training mode",
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, default="config.yml", help="Path to configuration file"
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/ciciot2023",
        help="Path to processed dataset",
    )

    parser.add_argument(
        "--rl-path", type=str, default="artifacts/rl", help="Path to RL model directory"
    )

    # RL training options
    parser.add_argument(
        "--algorithm",
        choices=["dqn", "ppo", "a2c"],
        default="ppo",
        help="RL algorithm (dqn, ppo, or a2c)",
    )

    parser.add_argument(
        "--timesteps", type=int, default=None, help="RL training timesteps (overrides config)"
    )

    # General options
    parser.add_argument("--force", action="store_true", help="Force retrain even if models exist")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device for training"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {config_path}")
    return config


def process_data(config: dict, args: argparse.Namespace) -> bool:
    """Process CICIoT2023 dataset for adversarial environment."""
    print("\n📊 Processing Dataset for Adversarial Environment")
    print("=" * 60)

    try:
        from src.utils.dataset_processor import CICIoTProcessor, DataProcessingConfig

        # Create processing config
        processing_config = DataProcessingConfig(
            dataset_path=Path(config["dataset"]["raw_path"]),
            output_path=Path(args.data_path),
            sample_size=config["dataset"]["sample_size"],
            train_split=config["dataset"]["train_split"],
            val_split=config["dataset"]["val_split"],
            test_split=config["dataset"]["test_split"],
            feature_selection=config["dataset"].get("feature_selection", False),
            variance_threshold=config["dataset"].get("variance_threshold", 0.01),
            correlation_threshold=config["dataset"].get("correlation_threshold", 0.95),
            feature_keep_keywords=config["dataset"].get("feature_keep_keywords", None),
            sampling_strategy=config["dataset"].get("sampling_strategy", None),
            sampling_mode=config["dataset"].get("sampling_mode", "default"),
            benign_target_count=config["dataset"].get("benign_target_count", None),
            max_samples_per_attack_class=config["dataset"].get(
                "max_samples_per_attack_class", None
            ),
        )

        # Process dataset
        processor = CICIoTProcessor(processing_config)

        # Check if already processed
        output_path = Path(args.data_path)
        if (output_path / "features.npy").exists() and not args.force:
            print("✅ Dataset already processed. Use --force to reprocess.")
            return True

        # Process for adversarial environment
        results = processor.process_for_adversarial_env()

        print("✅ Dataset processing completed!")
        print(f"   - Total samples: {results['total_samples']:,}")
        print(f"   - Features: {results['num_features']}")
        stage_distribution = results.get("stage_distribution", results.get("stage_counts", {}))
        print(f"   - Stage distribution: {stage_distribution}")
        print(f"   - Output path: {args.data_path}")

        return True

    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        print(f"❌ Dataset processing failed: {e}")
        return False


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    print("\n" + "=" * 60)
    print("🚀 IoT Defense System - Adversarial Training Pipeline")
    print("=" * 60)
    print(f"   Mode: {args.mode}")
    print(f"   Device: {args.device}")
    print("=" * 60)

    try:
        # Load configuration
        config = load_config(args.config)

        if args.mode == "process-data":
            process_data(config, args)

        print("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
