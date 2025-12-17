"""
Benchmark Analyzer

Analyzes and visualizes benchmark results across algorithms.
Includes PRD 7.2 metrics visualization and attack progression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from .metrics_collector import MetricsCollector, EpisodeMetrics, ACTION_COSTS


# Kill Chain stage names for visualization
KILL_CHAIN_STAGES = ["BENIGN", "RECON", "ACCESS", "MANEUVER", "IMPACT"]

# Force Continuum action names
FORCE_CONTINUUM_ACTIONS = ["OBSERVE", "LOG", "THROTTLE", "BLOCK", "ISOLATE"]


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
        
        # PRD 7.2 visualizations
        self._plot_prd_metrics_comparison()
        self._plot_attack_progression()
        self._plot_defense_heatmap()
        
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
    
    # =========================================================================
    # PRD 7.2 Metrics Visualization
    # =========================================================================
    
    def _plot_prd_metrics_comparison(self) -> None:
        """Plot PRD 7.2 metrics comparison across algorithms.
        
        Visualizes:
        - Attack Mitigation Rate
        - False Positive Rate
        - Mean Time to Contain
        - Availability Score
        """
        if not self.metrics_collector.metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PRD 7.2 Security Metrics Comparison", fontsize=14, fontweight='bold')
        
        algorithms = list(self.metrics_collector.metrics.keys())
        
        # Collect PRD metrics for each algorithm
        metrics_data = {
            'attack_mitigation_rate': [],
            'false_positive_rate': [],
            'mean_time_to_contain': [],
            'availability_score': [],
        }
        
        for alg in algorithms:
            # Average across all runs for this algorithm
            runs = self.metrics_collector.metrics[alg]
            if runs:
                metrics_data['attack_mitigation_rate'].append(
                    np.mean([r.attack_mitigation_rate for r in runs])
                )
                metrics_data['false_positive_rate'].append(
                    np.mean([r.false_positive_rate for r in runs])
                )
                metrics_data['mean_time_to_contain'].append(
                    np.mean([r.mean_time_to_contain for r in runs])
                )
                metrics_data['availability_score'].append(
                    np.mean([r.availability_score for r in runs])
                )
            else:
                for key in metrics_data:
                    metrics_data[key].append(0.0)
        
        # Attack Mitigation Rate (higher is better)
        ax1 = axes[0, 0]
        bars1 = ax1.bar(algorithms, metrics_data['attack_mitigation_rate'], 
                       color='green', alpha=0.7)
        ax1.set_ylabel('Rate')
        ax1.set_title('Attack Mitigation Rate ↑')
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Target (80%)')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        for bar, val in zip(bars1, metrics_data['attack_mitigation_rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)
        
        # False Positive Rate (lower is better)
        ax2 = axes[0, 1]
        bars2 = ax2.bar(algorithms, metrics_data['false_positive_rate'],
                       color='red', alpha=0.7)
        ax2.set_ylabel('Rate')
        ax2.set_title('False Positive Rate ↓')
        ax2.set_ylim(0, max(0.3, max(metrics_data['false_positive_rate']) * 1.2))
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Target (<10%)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        for bar, val in zip(bars2, metrics_data['false_positive_rate']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Mean Time to Contain (lower is better)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(algorithms, metrics_data['mean_time_to_contain'],
                       color='blue', alpha=0.7)
        ax3.set_ylabel('Steps')
        ax3.set_title('Mean Time to Contain ↓')
        ax3.grid(True, alpha=0.3)
        for bar, val in zip(bars3, metrics_data['mean_time_to_contain']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Availability Score (higher is better)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(algorithms, metrics_data['availability_score'],
                       color='purple', alpha=0.7)
        ax4.set_ylabel('Score')
        ax4.set_title('Availability Score ↑')
        ax4.set_ylim(0, 1.05)
        ax4.grid(True, alpha=0.3)
        for bar, val in zip(bars4, metrics_data['availability_score']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.results_path / 'prd_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PRD metrics comparison saved to: {self.results_path / 'prd_metrics_comparison.png'}")
    
    def _plot_attack_progression(self) -> None:
        """Plot attack progression analysis across episodes.
        
        Shows how attacks progress through Kill Chain stages and
        how effectively the defense agent prevents escalation.
        """
        if not self.metrics_collector.metrics:
            return
        
        # Check if we have episode data with attack stages
        has_episode_data = False
        for alg, runs in self.metrics_collector.metrics.items():
            for run in runs:
                episodes = getattr(run, "episodes", []) or []
                if episodes and episodes[0].attack_stages:
                    has_episode_data = True
                    break
            if has_episode_data:
                break
        
        if not has_episode_data:
            print("Warning: No episode-level attack stage data available for attack progression plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Attack Kill Chain Analysis", fontsize=14, fontweight='bold')
        
        # Left: Stage distribution heatmap per algorithm
        ax1 = axes[0]
        algorithms = list(self.metrics_collector.metrics.keys())
        stage_counts = np.zeros((len(algorithms), 5))  # 5 Kill Chain stages
        
        for i, alg in enumerate(algorithms):
            for run in self.metrics_collector.metrics[alg]:
                for ep in getattr(run, "episodes", []) or []:
                    for stage in ep.attack_stages:
                        if 0 <= stage < 5:
                            stage_counts[i, stage] += 1
        
        # Normalize by total observations per algorithm
        row_sums = stage_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        stage_dist = stage_counts / row_sums
        
        sns.heatmap(stage_dist, 
                   xticklabels=KILL_CHAIN_STAGES,
                   yticklabels=algorithms,
                   annot=True, 
                   fmt='.1%',
                   cmap='RdYlGn_r',  # Red = bad (IMPACT), Green = good (BENIGN)
                   ax=ax1)
        ax1.set_title("Attack Stage Distribution by Algorithm")
        ax1.set_xlabel("Kill Chain Stage")
        ax1.set_ylabel("Algorithm")
        
        # Right: Impact reached rate (bar chart)
        ax2 = axes[1]
        impact_rates = []
        for alg in algorithms:
            total_episodes = 0
            impact_count = 0
            for run in self.metrics_collector.metrics[alg]:
                for ep in getattr(run, "episodes", []) or []:
                    total_episodes += 1
                    if ep.reached_impact:
                        impact_count += 1
            impact_rates.append(impact_count / total_episodes if total_episodes > 0 else 0)
        
        colors = ['green' if r < 0.2 else 'orange' if r < 0.4 else 'red' for r in impact_rates]
        bars = ax2.bar(algorithms, impact_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Rate')
        ax2.set_title('Impact Stage Reached Rate (Lower is Better)')
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Good (<20%)')
        ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<40%)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, impact_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.results_path / 'attack_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attack progression analysis saved to: {self.results_path / 'attack_progression.png'}")
    
    def _plot_defense_heatmap(self) -> None:
        """Plot defense action heatmap showing action selection by attack stage.
        
        Creates a heatmap showing which defensive actions are taken
        at each attack stage, revealing the learned defense policy.
        """
        if not self.metrics_collector.metrics:
            return
        
        # Check if we have episode data with actions
        has_episode_data = False
        for alg, runs in self.metrics_collector.metrics.items():
            for run in runs:
                episodes = getattr(run, "episodes", []) or []
                if episodes and episodes[0].actions:
                    has_episode_data = True
                    break
            if has_episode_data:
                break
        
        if not has_episode_data:
            print("Warning: No episode-level action data available for defense heatmap")
            return
        
        algorithms = list(self.metrics_collector.metrics.keys())
        n_algorithms = len(algorithms)
        
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 5))
        if n_algorithms == 1:
            axes = [axes]
        
        fig.suptitle("Defense Policy Heatmaps: Action by Attack Stage", fontsize=14, fontweight='bold')
        
        for ax, alg in zip(axes, algorithms):
            # Create stage-action matrix
            action_by_stage = np.zeros((5, 5))  # 5 stages x 5 actions
            
            for run in self.metrics_collector.metrics[alg]:
                for ep in getattr(run, "episodes", []) or []:
                    # Match actions with attack stages
                    # Note: attack_stages has one more element than actions (initial state)
                    for i, action in enumerate(ep.actions):
                        if i < len(ep.attack_stages):
                            stage = ep.attack_stages[i]
                            if 0 <= stage < 5 and 0 <= action < 5:
                                action_by_stage[stage, action] += 1
            
            # Normalize by row (per stage)
            row_sums = action_by_stage.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            action_dist = action_by_stage / row_sums
            
            sns.heatmap(action_dist,
                       xticklabels=FORCE_CONTINUUM_ACTIONS,
                       yticklabels=KILL_CHAIN_STAGES,
                       annot=True,
                       fmt='.0%',
                       cmap='YlOrRd',
                       ax=ax,
                       vmin=0, vmax=1)
            ax.set_title(f"{alg.upper()}")
            ax.set_xlabel("Defensive Action")
            ax.set_ylabel("Attack Stage")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.results_path / 'defense_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Defense heatmap saved to: {self.results_path / 'defense_heatmap.png'}")
    
    def generate_single_model_report(self, algorithm_name: str, run_id: int = 0) -> None:
        """Generate detailed report for a single model evaluation.
        
        Args:
            algorithm_name: Name of the algorithm.
            run_id: Run identifier.
        """
        if algorithm_name not in self.metrics_collector.metrics:
            print(f"Warning: No data for algorithm '{algorithm_name}'")
            return
        
        runs = self.metrics_collector.metrics[algorithm_name]
        if run_id >= len(runs):
            print(f"Warning: Run {run_id} not found for '{algorithm_name}'")
            return
        
        run = runs[run_id]
        
        print(f"\n{'='*60}")
        print(f"Detailed Report: {algorithm_name.upper()} (Run {run_id})")
        print(f"{'='*60}")
        
        print(f"\n📊 PRD 7.2 Security Metrics:")
        print(f"   Attack Mitigation Rate: {run.attack_mitigation_rate:.1%}")
        print(f"   False Positive Rate:    {run.false_positive_rate:.1%}")
        print(f"   Mean Time to Contain:   {run.mean_time_to_contain:.1f} steps")
        print(f"   Availability Score:     {run.availability_score:.3f}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Average Reward:         {run.avg_reward:.3f}")
        print(f"   Std Reward:             {run.std_reward:.3f}")
        print(f"   Episodes Evaluated:     {len(run.episodes)}")
        
        if run.episodes:
            total_impact = sum(1 for ep in run.episodes if ep.reached_impact)
            print(f"   Impact Reached:         {total_impact}/{len(run.episodes)} episodes")
        
        print(f"{'='*60}\n")