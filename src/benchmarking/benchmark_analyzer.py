"""
Benchmark Analyzer

Analyzes and visualizes benchmark results across algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from .metrics_collector import MetricsCollector


class BenchmarkAnalyzer:
    """Analyzes and visualizes benchmark results"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.results_path = Path("results/benchmark/analysis")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def generate_comparison_report(self) -> None:
        """Generate comprehensive comparison report"""
        print("Generating benchmark comparison report...")
        
        # Check if we have any data first
        if not self.metrics_collector.metrics:
            print("Warning: No benchmark data available. Cannot generate comparison report.")
            return
        
        # Check if any algorithms have valid data
        valid_algorithms = {
            name: runs for name, runs in self.metrics_collector.metrics.items() 
            if runs  # At least some runs exist
        }
        
        if not valid_algorithms:
            print("Warning: No valid algorithm data found. Cannot generate comparison report.")
            return
        
        # Create comparison plots
        self._plot_performance_comparison()
        self._plot_training_time_comparison()
        self._plot_convergence_analysis()
        self._plot_reward_distributions()
        
        # Generate summary statistics
        self._generate_summary_table()
        
        print(f"Benchmark analysis completed. Results saved to: {self.results_path}")
        
    def _plot_performance_comparison(self) -> None:
        """Plot performance comparison across algorithms"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = list(self.metrics_collector.metrics.keys())
        avg_rewards = []
        std_rewards = []
        
        for algorithm in algorithms:
            summary = self.metrics_collector.get_algorithm_summary(algorithm)
            avg_rewards.append(summary.get('avg_reward_mean', 0))
            std_rewards.append(summary.get('avg_reward_std', 0))
        
        # Bar plot with error bars
        ax1.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Box plot for detailed distribution
        all_rewards = []
        labels = []
        for algorithm in algorithms:
            runs = self.metrics_collector.metrics[algorithm]
            rewards = [run.avg_reward for run in runs]
            all_rewards.extend(rewards)
            labels.extend([algorithm] * len(rewards))
            
        df = pd.DataFrame({'Algorithm': labels, 'Reward': all_rewards})
        sns.boxplot(data=df, x='Algorithm', y='Reward', ax=ax2)
        ax2.set_title('Reward Distribution by Algorithm')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_training_time_comparison(self) -> None:
        """Plot training time comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        algorithms = list(self.metrics_collector.metrics.keys())
        training_times = []
        std_times = []
        
        for algorithm in algorithms:
            summary = self.metrics_collector.get_algorithm_summary(algorithm)
            training_times.append(summary.get('training_time_mean', 0))
            std_times.append(summary.get('training_time_std', 0))
        
        bars = ax.bar(algorithms, training_times, yerr=std_times, capsize=5, alpha=0.7)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(std_times) * 0.1,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_convergence_analysis(self) -> None:
        """Plot convergence analysis if convergence data is available"""
        # Check if we have any data
        if not self.metrics_collector.metrics:
            print("Warning: No metrics data available for convergence analysis")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        has_data = False
        for algorithm_name, runs in self.metrics_collector.metrics.items():
            for i, run in enumerate(runs):
                if run.episode_rewards:
                    has_data = True
                    # Calculate moving average for smoother curves
                    window_size = min(50, len(run.episode_rewards) // 10)
                    if window_size > 1:
                        smoothed_rewards = pd.Series(run.episode_rewards).rolling(window=window_size).mean()
                    else:
                        smoothed_rewards = run.episode_rewards
                        
                    alpha = 0.3 if i > 0 else 1.0  # Make first run more visible
                    label = f'{algorithm_name} Run {i+1}' if i == 0 else ''
                    ax.plot(smoothed_rewards, label=label, 
                           alpha=alpha, linewidth=1.5 if i == 0 else 1.0)
        
        if has_data:
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward (Moving Average)')
            ax.set_title('Learning Curves Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No convergence data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves (No Data)')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_reward_distributions(self) -> None:
        """Plot reward distributions for each algorithm"""
        # Check if we have any data
        if not self.metrics_collector.metrics:
            print("Warning: No metrics data available for reward distribution plots")
            return
            
        # Filter out algorithms with no valid runs
        valid_algorithms = {
            name: runs for name, runs in self.metrics_collector.metrics.items() 
            if runs and any(run.episode_rewards for run in runs)
        }
        
        if not valid_algorithms:
            print("Warning: No algorithms with episode reward data available")
            return
        
        fig, axes = plt.subplots(1, len(valid_algorithms), 
                                figsize=(5 * len(valid_algorithms), 6))
        
        if len(valid_algorithms) == 1:
            axes = [axes]
            
        for ax, (algorithm_name, runs) in zip(axes, valid_algorithms.items()):
            all_episode_rewards = []
            for run in runs:
                if run.episode_rewards:  # Check if episode_rewards is not empty
                    all_episode_rewards.extend(run.episode_rewards)
                
            if all_episode_rewards:
                ax.hist(all_episode_rewards, bins=30, alpha=0.7, density=True)
                ax.axvline(np.mean(all_episode_rewards), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_episode_rewards):.2f}')
                ax.set_xlabel('Episode Reward')
                ax.set_ylabel('Density')
                ax.set_title(f'{algorithm_name} Reward Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No reward data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{algorithm_name} Reward Distribution (No Data)')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'reward_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_summary_table(self) -> None:
        """Generate summary statistics table"""
        summary_data = []
        
        for algorithm_name in self.metrics_collector.metrics.keys():
            summary = self.metrics_collector.get_algorithm_summary(algorithm_name)
            summary_data.append({
                'Algorithm': algorithm_name,
                'Avg Reward': f"{summary.get('avg_reward_mean', 0):.3f} ± {summary.get('avg_reward_std', 0):.3f}",
                'Training Time (s)': f"{summary.get('training_time_mean', 0):.2f} ± {summary.get('training_time_std', 0):.2f}",
                'Num Runs': summary.get('num_runs', 0),
                'Best Reward': f"{summary.get('avg_reward_max', 0):.3f}",
                'Worst Reward': f"{summary.get('avg_reward_min', 0):.3f}"
            })
            
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv(self.results_path / 'summary_table.csv', index=False)
        
        # Save as formatted text
        with open(self.results_path / 'summary_report.txt', 'w') as f:
            f.write("ALGORITHM BENCHMARK SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Add additional analysis
            f.write("DETAILED ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            # Find best performing algorithm
            best_alg = max(self.metrics_collector.metrics.keys(), 
                          key=lambda x: self.metrics_collector.get_algorithm_summary(x).get('avg_reward_mean', 0))
            f.write(f"Best performing algorithm: {best_alg}\n")
            
            # Find fastest training algorithm
            fastest_alg = min(self.metrics_collector.metrics.keys(),
                             key=lambda x: self.metrics_collector.get_algorithm_summary(x).get('training_time_mean', float('inf')))
            f.write(f"Fastest training algorithm: {fastest_alg}\n")
            
        print(f"Summary table saved to: {self.results_path / 'summary_table.csv'}")
        print(f"Detailed report saved to: {self.results_path / 'summary_report.txt'}")