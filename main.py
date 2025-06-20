"""
IoT Defense System - Main Entry Point

Unified training pipeline for LSTM attack prediction and RL defense agents
using real CICIoT2023 dataset.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.config_loader import ConfigLoader
from training.lstm_trainer import LSTMTrainer
from training.rl_trainer import RLTrainer
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.benchmark_analyzer import BenchmarkAnalyzer
from utils.dataset_processor import CICIoTProcessor, DataProcessingConfig

# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'main.log'),
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='IoT Defense System - Train LSTM and/or RL agents'
    )
    
    # Training mode selection
    parser.add_argument(
        '--mode',
        choices=['lstm', 'rl', 'both', 'benchmark', 'process-data'],
        default='both',
        help='Training mode: lstm, rl, both, benchmark, or process-data'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration file'
    )
    
    # LSTM specific options
    parser.add_argument(
        '--lstm-epochs',
        type=int,
        default=None,
        help='Number of LSTM training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--lstm-batch-size',
        type=int,
        default=None,
        help='LSTM batch size (overrides config)'
    )
    
    # RL specific options
    parser.add_argument(
        '--rl-algorithm',
        choices=['dqn', 'ppo', 'a2c'],
        default=None,
        help='RL algorithm to use (overrides config)'
    )
    
    parser.add_argument(
        '--rl-timesteps',
        type=int,
        default=None,
        help='RL training timesteps (overrides config)'
    )
    
    # Benchmark options
    parser.add_argument(
        '--benchmark-mode',
        choices=['train', 'evaluate', 'mixed'],
        default='mixed',
        help='Benchmark mode: train (from scratch), evaluate (pre-trained), mixed (auto)'
    )
    
    parser.add_argument(
        '--benchmark-algorithms',
        nargs='+',
        choices=['dqn', 'ppo', 'a2c'],
        default=['dqn', 'ppo', 'a2c'],
        help='Algorithms to benchmark'
    )
    
    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=3,
        help='Number of runs per algorithm'
    )
    
    # Data options
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/ciciot2023',
        help='Path to processed dataset'
    )
    
    parser.add_argument(
        '--force-retrain-lstm',
        action='store_true',
        help='Force retrain LSTM even if model exists'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def process_dataset(config: dict, args: argparse.Namespace) -> None:
    """Process raw CICIoT2023 dataset with configurable splits and EDA recommendations."""
    print("📊 Starting Dataset Processing")
    print("=" * 60)
    
    try:
        # Create processing config with EDA recommendations from config file
        processing_config = DataProcessingConfig(
            dataset_path=Path(config['dataset']['raw_path']),
            output_path=Path(args.data_path),
            sample_size=config['dataset']['sample_size'],
            sequence_length=config['dataset']['sequence_length'],
            train_split=config['dataset']['train_split'],
            val_split=config['dataset']['val_split'],
            test_split=config['dataset']['test_split'],
            # Add EDA recommendations
            feature_selection=config['dataset'].get('feature_selection', False),
            sampling_strategy=config['dataset'].get('sampling_strategy', None)
        )
        
        # Process dataset
        processor = CICIoTProcessor(processing_config)
        results = processor.process_dataset()
        
        print("✅ Dataset processing completed!")
        print(f"   - Total samples: {results['total_samples']:,}")
        print(f"   - Train samples: {results['train_samples']:,}")
        print(f"   - Val samples: {results['val_samples']:,}")
        print(f"   - Test samples: {results['test_samples']:,}")
        print(f"   - Features: {results['feature_count']}")
        print(f"   - Classes: {results['class_count']}")
        print(f"   - Splits: {results['splits']}")
        
        if config['dataset'].get('feature_selection', False):
            print(f"   - ✅ Feature selection applied")
        if config['dataset'].get('sampling_strategy') == 'balanced':
            print(f"   - ✅ Class balancing applied")
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        raise


def train_lstm(config: dict, args: argparse.Namespace) -> Optional[Path]:
    """Train LSTM attack predictor."""
    print("🧠 Starting LSTM Attack Predictor Training")
    print("=" * 60)
    
    try:
        # Check if model already exists
        model_path = Path(config['models']['lstm']['save_path'])
        
        if model_path.exists() and not args.force_retrain_lstm:
            logger.info(f"LSTM model already exists at {model_path}")
            print(f"✅ Using existing LSTM model: {model_path}")
            return model_path
        
        # Create LSTM trainer
        trainer = LSTMTrainer(
            config=config,
            data_path=Path(args.data_path)
        )
        
        # Override config with command line arguments
        if args.lstm_epochs:
            trainer.config['lstm']['training']['epochs'] = args.lstm_epochs
        if args.lstm_batch_size:
            trainer.config['lstm']['training']['batch_size'] = args.lstm_batch_size
        
        # Train LSTM
        training_results = trainer.train()
        
        print(f"✅ LSTM training completed!")
        print(f"   - Final accuracy: {training_results['test_accuracy']:.4f}")
        print(f"   - Model saved to: {training_results['model_path']}")
        
        return Path(training_results['model_path'])
        
    except Exception as e:
        logger.error(f"LSTM training failed: {e}")
        print(f"❌ LSTM training failed: {e}")
        return None


def train_rl_agents(config: dict, args: argparse.Namespace, 
                   lstm_model_path: Optional[Path]) -> bool:
    """Train RL defense agents."""
    print("\n🤖 Starting RL Defense Agent Training")
    print("=" * 60)
    
    try:
        # Create RL trainer
        trainer = RLTrainer(
            config=config,
            data_path=Path(args.data_path),
            lstm_model_path=lstm_model_path
        )
        
        # Override config with command line arguments
        if args.rl_algorithm:
            trainer.config['rl']['algorithm'] = args.rl_algorithm
        if args.rl_timesteps:
            trainer.config['rl']['training']['total_timesteps'] = args.rl_timesteps
        
        # Train RL agents
        training_results = trainer.train()
        
        print(f"✅ RL training completed!")
        print(f"   - Algorithm: {training_results['algorithm']}")
        print(f"   - Final reward: {training_results['final_reward']:.2f}")
        print(f"   - Model saved to: {training_results['model_path']}")
        
        return True
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        print(f"❌ RL training failed: {e}")
        return False


def run_benchmark(config: dict, args: argparse.Namespace) -> None:
    """Run algorithm benchmark comparison with flexible modes."""
    print("🏆 Starting Algorithm Benchmark")
    print("=" * 60)
    
    try:
        # Check for LSTM model
        lstm_model_path = Path(config['models']['lstm']['save_path'])
        if not lstm_model_path.exists():
            print("❌ LSTM model not found. Train LSTM first or use --mode both")
            return
        
        # Determine benchmark mode
        benchmark_mode = getattr(args, 'benchmark_mode', 'mixed')
        
        # Create benchmark runner
        runner = BenchmarkRunner(
            config=config,
            lstm_model_path=lstm_model_path,
            mode=benchmark_mode
        )
        
        # Run benchmark
        metrics_collector = runner.run_benchmark(
            algorithms=args.benchmark_algorithms,
            num_runs=args.benchmark_runs
        )
        
        # Analyze results
        analyzer = BenchmarkAnalyzer(metrics_collector)
        analyzer.generate_comparison_report()
        
        print("🎉 Benchmark analysis completed!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"❌ Benchmark failed: {e}")
        raise


def main() -> None:
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🚀 IoT Defense System Training Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Config: {args.config}")
    print(f"Data path: {args.data_path}")
    print("=" * 60)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(Path(args.config))
        
        logger.info("Configuration loaded successfully")
        
        if args.mode == 'process-data':
            process_dataset(config, args)
            
        elif args.mode == 'lstm':
            train_lstm(config, args)
            
        elif args.mode == 'rl':
            # Check for existing LSTM model
            lstm_model_path = Path(config['models']['lstm']['save_path'])
            if not lstm_model_path.exists():
                print("❌ No LSTM model found. Train LSTM first or use --mode both")
                return
            
            train_rl_agents(config, args, lstm_model_path)
            
        elif args.mode == 'both':
            # Train LSTM first
            lstm_model_path = train_lstm(config, args)
            
            if lstm_model_path is None:
                print("❌ Cannot proceed with RL training without LSTM model")
                return
            
            # Then train RL agents
            success = train_rl_agents(config, args, lstm_model_path)
            
            if not success:
                print("❌ RL training failed")
                return
                
        elif args.mode == 'benchmark':
            run_benchmark(config, args)
        
        print("\n🎉 Training pipeline completed successfully!")
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()