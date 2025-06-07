"""
Benchmark Runner Script

Standalone script to run algorithm benchmarks.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config_loader import config
from models.attack_predictor import LSTMAttackPredictor
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.benchmark_analyzer import BenchmarkAnalyzer
from utils.training_manager import TrainingManager
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run RL Algorithm Benchmark')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['DQN', 'PPO', 'A2C'],  # Changed SAC to A2C
                       default=['DQN', 'PPO', 'A2C'],  # Changed SAC to A2C
                       help='Algorithms to benchmark')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per algorithm')
    parser.add_argument('--skip-lstm', action='store_true',
                       help='Skip LSTM training (use pre-trained)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run analysis on existing results')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Load existing results and analyze
        from benchmarking.metrics_collector import MetricsCollector
        metrics_collector = MetricsCollector()
        metrics_collector.load_results()
        
        analyzer = BenchmarkAnalyzer(metrics_collector)
        analyzer.generate_comparison_report()
        return
    
    print("Starting Algorithm Benchmark")
    print(f"Algorithms: {args.algorithms}")
    print(f"Runs per algorithm: {args.runs}")
    
    # Train or load LSTM model
    if not args.skip_lstm:
        print("Training LSTM attack predictor...")
        training_manager = TrainingManager(
            experiment_name="lstm_for_benchmark",
            base_artifact_path="./artifacts/lstm_benchmark",
            config=config
        )
        
        with training_manager.start_run(run_name="lstm_training"):
            lstm_model = LSTMAttackPredictor(config)
            lstm_model.train_model(epochs=100, batch_size=32)
            training_manager.log_model(lstm_model, "lstm_attack_predictor")
        
        training_manager.end_run()
    else:
        # Load pre-trained LSTM
        print("Loading pre-trained LSTM model...")
        lstm_model = LSTMAttackPredictor(config)
        # Note: In practice, you'd load weights here
    
    # Run benchmark
    benchmark_runner = BenchmarkRunner(config, lstm_model)
    metrics_collector = benchmark_runner.run_benchmark(
        algorithms=args.algorithms,
        num_runs=args.runs
    )
    
    # Analyze results
    print("Analyzing benchmark results...")
    analyzer = BenchmarkAnalyzer(metrics_collector)
    analyzer.generate_comparison_report()
    
    print("Benchmark completed successfully!")
    print("Check ./benchmark_analysis/ for detailed results and plots.")


if __name__ == "__main__":
    main()