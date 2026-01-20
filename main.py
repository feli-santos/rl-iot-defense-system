#!/usr/bin/env python3
"""
IoT Defense System - Adversarial Training Pipeline

Main entry point for the adversarial RL defense system. This system uses:
- Red Team: Attack Sequence Generator (LSTM) to produce attack sequences
- Blue Team: RL agents (DQN/PPO/A2C) to learn defense policies

Modes:
- process-data: Prepare CICIoT2023 dataset for adversarial environment
- train-generator: Train the Attack Sequence Generator (Red Team)
- train-rl: Train single RL defense agent (Blue Team)
- train-all-rl: Train all RL algorithms (DQN, PPO, A2C)
- train-all: Run complete training pipeline
- evaluate: Evaluate trained models (single or comparison)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging(log_level: str = "INFO", log_dir: Path = Path("results/logs")) -> None:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'adversarial_training.log'),
            logging.StreamHandler()
        ]
    )


logger = logging.getLogger(__name__)


def get_generator_path(args: argparse.Namespace) -> Path:
    """Resolve generator path with auto-detection of latest.
    
    Args:
        args: Parsed command line arguments.
    
    Returns:
        Path to generator directory containing attack_sequence_generator.pth
    """
    if args.generator_path:
        path = Path(args.generator_path)
        
        # If it's a .pth file, return parent directory
        if path.suffix == '.pth':
            return path.parent
        
        # If it's a directory with the model, use it directly
        if path.is_dir() and (path / "attack_sequence_generator.pth").exists():
            return path
        
        # If it's a directory with timestamped subdirs, find latest
        if path.is_dir():
            timestamped_dirs = sorted(
                [d for d in path.iterdir() if d.is_dir() and (d / "attack_sequence_generator.pth").exists()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if timestamped_dirs:
                latest = timestamped_dirs[0]
                logger.info(f"Auto-detected latest generator: {latest}")
                return latest
    
    # Default path
    default_path = Path("artifacts/generator")
    
    # Check if default has the model
    if (default_path / "attack_sequence_generator.pth").exists():
        return default_path
    
    # Check for timestamped subdirs in default
    if default_path.exists():
        timestamped_dirs = sorted(
            [d for d in default_path.iterdir() if d.is_dir() and (d / "attack_sequence_generator.pth").exists()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if timestamped_dirs:
            latest = timestamped_dirs[0]
            logger.info(f"Auto-detected latest generator: {latest}")
            return latest
    
    return default_path


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='IoT Defense System - Adversarial Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw CICIoT2023 dataset
  python main.py --mode process-data
  
  # Train Attack Sequence Generator (Red Team)
  python main.py --mode train-generator
  
  # Train single RL agent (Blue Team)
  python main.py --mode train-rl --algorithm ppo
  
  # Train all RL algorithms
  python main.py --mode train-all-rl
  
  # Full pipeline
  python main.py --mode train-all
  
  # Evaluate single model (detailed metrics)
  python main.py --mode evaluate --model-path artifacts/rl/ppo_*.zip
  
  # Compare algorithms (auto-discover models)
  python main.py --mode evaluate --algorithms dqn ppo a2c
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['process-data', 'train-generator', 'train-rl', 'train-all-rl', 'train-all', 'evaluate'],
        default='train-all',
        help='Training mode'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration file'
    )
    
    # Paths
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/ciciot2023',
        help='Path to processed dataset'
    )
    
    parser.add_argument(
        '--generator-path',
        type=str,
        default=None,
        help='Path to generator model directory or .pth file. Auto-detects latest if directory.'
    )
    
    parser.add_argument(
        '--rl-path',
        type=str,
        default='artifacts/rl',
        help='Path to RL model directory'
    )
    
    # Evaluation options
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to specific model file for single-model evaluation'
    )
    
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['dqn', 'ppo', 'a2c'],
        default=None,
        help='Algorithms to evaluate/compare (auto-discovers models)'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes per model'
    )
    
    # Generator training options
    parser.add_argument(
        '--generator-epochs',
        type=int,
        default=None,
        help='Generator training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Number of episodes to generate for training'
    )
    
    # RL training options
    parser.add_argument(
        '--algorithm',
        choices=['dqn', 'ppo', 'a2c'],
        default='ppo',
        help='RL algorithm to train (for train-rl mode)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='RL training timesteps (overrides config)'
    )
    
    # General options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retrain even if models exist'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        default='cpu',
        help='Device for training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
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
            dataset_path=Path(config['dataset']['raw_path']),
            output_path=Path(args.data_path),
            sample_size=config['dataset']['sample_size'],
            sequence_length=config['dataset']['sequence_length'],
            train_split=config['dataset']['train_split'],
            val_split=config['dataset']['val_split'],
            test_split=config['dataset']['test_split'],
            feature_selection=config['dataset'].get('feature_selection', False),
            variance_threshold=config['dataset'].get('variance_threshold', 0.01),
            correlation_threshold=config['dataset'].get('correlation_threshold', 0.95),
            feature_keep_keywords=config['dataset'].get('feature_keep_keywords', None),
            sampling_strategy=config['dataset'].get('sampling_strategy', None)
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
        print(f"   - Stage distribution: {results['stage_distribution']}")
        print(f"   - Output path: {args.data_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        print(f"❌ Dataset processing failed: {e}")
        return False


def train_generator(config: dict, args: argparse.Namespace) -> bool:
    """Train the Attack Sequence Generator (Red Team)."""
    print("\n🔴 Training Attack Sequence Generator (Red Team)")
    print("=" * 60)
    
    try:
        from src.generator.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
        from src.generator.attack_sequence_generator import AttackSequenceGeneratorConfig
        from src.training.generator_trainer import GeneratorTrainer, GeneratorTrainingConfig
        
        # Check if already trained
        generator_path = Path(args.generator_path)
        model_file = generator_path / "attack_sequence_generator.pth"
        
        if model_file.exists() and not args.force:
            print(f"✅ Generator already trained at {model_file}. Use --force to retrain.")
            return True
        
        # Get config values
        ep_config = config.get('episode_generation', {})
        gen_config = config.get('attack_generator', {})
        
        # Load stage distribution from metadata
        data_path = Path(args.data_path)
        metadata_path = data_path / "metadata.json"
        stage_distribution = None
        
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            stage_distribution = metadata.get("stage_counts", metadata.get("stage_distribution", {}))
            # Convert string keys to int
            if stage_distribution:
                stage_distribution = {int(k): v for k, v in stage_distribution.items()}
                print(f"   Loaded stage distribution: {stage_distribution}")
        
        # Create episode generator
        episode_config = EpisodeGeneratorConfig(
            num_episodes=args.num_episodes or ep_config.get('num_episodes', 10000),
            min_length=ep_config.get('min_length', 5),
            max_length=ep_config.get('max_length', 30),
            benign_start_prob=ep_config.get('benign_start_prob', 0.8),
            distribution_temperature=ep_config.get('distribution_temperature', 1.0),
            min_stage_coverage=ep_config.get('min_stage_coverage', None),
        )
        
        print(f"   Generating {episode_config.num_episodes:,} training episodes...")
        print(f"   Temperature={episode_config.distribution_temperature}, Coverage={episode_config.min_stage_coverage}")
        episode_generator = EpisodeGenerator(
            config=episode_config,
            stage_distribution=stage_distribution,
            seed=42,
        )
        episodes = episode_generator.generate_all()
        
        # Create model config
        model_config_dict = gen_config.get('model', {})
        model_config = AttackSequenceGeneratorConfig(
            num_stages=model_config_dict.get('num_stages', 5),
            embedding_dim=model_config_dict.get('embedding_dim', 32),
            hidden_size=model_config_dict.get('hidden_size', 64),
            num_layers=model_config_dict.get('num_layers', 2),
            dropout=model_config_dict.get('dropout', 0.1),
            temperature=model_config_dict.get('temperature', 1.0),
        )
        
        # Create training config
        training_config_dict = gen_config.get('training', {})
        training_config = GeneratorTrainingConfig(
            epochs=args.generator_epochs or training_config_dict.get('epochs', 50),
            batch_size=training_config_dict.get('batch_size', 32),
            learning_rate=training_config_dict.get('learning_rate', 0.001),
            sequence_length=training_config_dict.get('sequence_length', 5),
            val_split=training_config_dict.get('val_split', 0.2),
            early_stopping_patience=training_config_dict.get('early_stopping_patience', 10),
            output_dir=generator_path,
            device=args.device,
            # Imbalance mitigation
            use_class_weights=training_config_dict.get('use_class_weights', True),
            use_weighted_sampler=training_config_dict.get('use_weighted_sampler', True),
            class_weight_smoothing=training_config_dict.get('class_weight_smoothing', 0.5),
            grad_clip_norm=training_config_dict.get('grad_clip_norm', 1.0),
            use_lr_scheduler=training_config_dict.get('use_lr_scheduler', True),
            scheduler_patience=training_config_dict.get('scheduler_patience', 5),
            seed=training_config_dict.get('seed', 42),
            # Balanced validation
            balanced_validation=training_config_dict.get('balanced_validation', False),
            val_samples_per_class=training_config_dict.get('val_samples_per_class', 80),
            # Macro-F1 early stopping with recall gates
            use_macro_f1_stopping=training_config_dict.get('use_macro_f1_stopping', False),
            min_recall_stage_1=training_config_dict.get('min_recall_stage_1', 0.5),
            min_recall_stage_2=training_config_dict.get('min_recall_stage_2', 0.5),
        )
        
        # Train generator
        trainer = GeneratorTrainer(config=training_config, model_config=model_config)
        
        print(f"   Training for {training_config.epochs} epochs...")
        results = trainer.train(episodes)
        
        print("✅ Generator training completed!")
        if training_config.use_macro_f1_stopping and results.get('best_macro_f1') is not None:
            print(f"   - Best macro F1: {results['best_macro_f1']:.4f}")
        print(f"   - Best validation loss: {results['best_val_loss']:.4f}")
        print(f"   - Epochs trained: {results['epochs_trained']}")
        print(f"   - Model saved to: {generator_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Generator training failed: {e}")
        print(f"❌ Generator training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_rl(config: dict, args: argparse.Namespace) -> bool:
    """Train RL defense agent (Blue Team)."""
    print("\n🔵 Training RL Defense Agent (Blue Team)")
    print("=" * 60)
    
    try:
        from src.environment.adversarial_env import AdversarialIoTEnv, AdversarialEnvConfig
        from src.algorithms.adversarial_algorithm import AdversarialAlgorithm, AdversarialAlgorithmConfig
        from datetime import datetime
        import uuid
        
        # Check dependencies
        generator_path = Path(args.generator_path)
        if not (generator_path / "attack_sequence_generator.pth").exists():
            print("❌ Generator not found. Train generator first.")
            return False
        
        data_path = Path(args.data_path)
        if not (data_path / "features.npy").exists():
            print("❌ Processed data not found. Process data first.")
            return False
        
        # Get config values
        rl_config = config.get('rl', {})
        env_config = config.get('adversarial_environment', {})
        algo_config = rl_config.get('algorithms', {}).get(args.algorithm, {})
        
        # Create environment config
        adversarial_env_config = AdversarialEnvConfig(
            max_steps=env_config.get('max_steps', 500),
            window_size=env_config.get('observation', {}).get('window_size', 5),
            include_deltas=env_config.get('observation', {}).get('include_deltas', True),
            num_actions=env_config.get('actions', {}).get('num_actions', 5),
            patience_bonus=env_config.get('reward', {}).get('patience_bonus', 0.5),
        )
        
        # Create environment
        print(f"   Creating environment...")
        env = AdversarialIoTEnv(
            generator_path=generator_path,
            dataset_path=data_path,
            config=adversarial_env_config,
            device=args.device,
        )
        
        # Get training timesteps
        timesteps = args.timesteps or rl_config.get('training', {}).get('total_timesteps', 50000)
        
        # Create algorithm config
        algorithm_config = AdversarialAlgorithmConfig(
            algorithm_type=args.algorithm,
            total_timesteps=timesteps,
            learning_rate=algo_config.get('learning_rate', 3e-4),
            gamma=algo_config.get('gamma', 0.99),
            verbose=1,
        )
        
        # Merge algorithm-specific params
        if args.algorithm == 'dqn':
            algorithm_config.buffer_size = algo_config.get('buffer_size', 50000)
            algorithm_config.batch_size = algo_config.get('batch_size', 32)
            algorithm_config.target_update_interval = algo_config.get('target_update_interval', 1000)
        elif args.algorithm == 'ppo':
            algorithm_config.n_steps = algo_config.get('n_steps', 2048)
            algorithm_config.n_epochs = algo_config.get('n_epochs', 10)
            algorithm_config.batch_size = algo_config.get('batch_size', 64)
        elif args.algorithm == 'a2c':
            algorithm_config.n_steps = algo_config.get('n_steps', 5)
        
        # Create algorithm
        algorithm = AdversarialAlgorithm(config=algorithm_config)
        
        print(f"   Algorithm: {args.algorithm.upper()}")
        print(f"   Training for {timesteps:,} timesteps...")
        
        # Create model and train
        model = algorithm.create_model(env)
        trained_model = algorithm.train(model, total_timesteps=timesteps)
        
        # Save model with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:6]
        model_dir = Path(args.rl_path) / f"{args.algorithm}_{timestamp}_{run_id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        algorithm.save_model(trained_model, model_dir / f"{args.algorithm}_agent")
        
        # Quick evaluation
        print("   Running quick evaluation...")
        total_reward = 0.0
        n_episodes = 10
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = trained_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
        
        avg_reward = total_reward / n_episodes
        
        print("✅ RL training completed!")
        print(f"   - Algorithm: {args.algorithm.upper()}")
        print(f"   - Average reward (10 episodes): {avg_reward:.2f}")
        print(f"   - Model saved to: {model_dir}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        print(f"❌ RL training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_all_rl(config: dict, args: argparse.Namespace) -> bool:
    """Train all RL algorithms (DQN, PPO, A2C)."""
    print("\n🔵 Training All RL Algorithms (Blue Team)")
    print("=" * 60)
    
    algorithms = ['dqn', 'ppo', 'a2c']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*50}")
        print(f"Training {algorithm.upper()}...")
        print(f"{'='*50}")
        
        # Create a modified args for this algorithm
        algo_args = argparse.Namespace(**vars(args))
        algo_args.algorithm = algorithm
        
        success = train_rl(config, algo_args)
        results[algorithm] = success
        
        if not success:
            print(f"⚠️  {algorithm.upper()} training failed, continuing with next...")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for algo, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {algo.upper()}")
    
    all_success = all(results.values())
    return all_success


def run_evaluate(config: dict, args: argparse.Namespace) -> bool:
    """Evaluate trained models (single or comparison mode).
    
    - Single model: Detailed PRD 7.2 metrics for one model
    - Comparison: Summary metrics across multiple algorithms
    """
    print("\n📊 Evaluating Trained Models")
    print("=" * 60)
    
    try:
        from src.benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkConfig
        from src.benchmarking.benchmark_analyzer import BenchmarkAnalyzer
        
        # Resolve generator path
        generator_path = get_generator_path(args)
        
        if not (generator_path / "attack_sequence_generator.pth").exists():
            print(f"❌ Generator not found at {generator_path}. Train generator first.")
            return False
        
        data_path = Path(args.data_path)
        if not (data_path / "features.npy").exists():
            print("❌ Processed data not found. Process data first.")
            return False
        
        # Create benchmark config
        benchmark_config = BenchmarkConfig(
            num_episodes=args.eval_episodes,
            generator_path=generator_path,
            dataset_path=data_path,
        )
        
        runner = BenchmarkRunner(config, benchmark_config)
        
        # Determine mode: single model or comparison
        if args.model_path:
            # Single model evaluation
            print(f"   Mode: Single Model Evaluation")
            print(f"   Model: {args.model_path}")
            
            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                return False
            
            results = runner.evaluate_model(model_path)
            
            # Use analyzer for detailed report
            analyzer = BenchmarkAnalyzer(runner.metrics_collector)
            algorithm = results.get('algorithm', 'unknown')
            analyzer.generate_single_model_report(algorithm, run_id=0)
            
        else:
            # Comparison mode
            algorithms = args.algorithms or ['dqn', 'ppo', 'a2c']
            print(f"   Mode: Algorithm Comparison")
            print(f"   Algorithms: {', '.join(algorithms)}")
            
            runner.run_comparison(algorithms)
            
            # Generate comparison report with visualizations
            analyzer = BenchmarkAnalyzer(runner.metrics_collector)
            analyzer.generate_comparison_report()
        
        print("\n✅ Evaluation completed!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Resolve generator path early
    resolved_generator_path = get_generator_path(args)
    args.generator_path = resolved_generator_path
    
    print("\n" + "=" * 60)
    print("🚀 IoT Defense System - Adversarial Training Pipeline")
    print("=" * 60)
    print(f"   Mode: {args.mode}")
    if args.mode == 'train-rl':
        print(f"   Algorithm: {args.algorithm}")
    print(f"   Device: {args.device}")
    print(f"   Generator: {args.generator_path}")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        if args.mode == 'process-data':
            process_data(config, args)
            
        elif args.mode == 'train-generator':
            train_generator(config, args)
            
        elif args.mode == 'train-rl':
            train_rl(config, args)
            
        elif args.mode == 'train-all-rl':
            train_all_rl(config, args)
            
        elif args.mode == 'train-all':
            # Full pipeline
            if not process_data(config, args):
                return
            if not train_generator(config, args):
                return
            if not train_rl(config, args):
                return
            
        elif args.mode == 'evaluate':
            run_evaluate(config, args)
        
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
