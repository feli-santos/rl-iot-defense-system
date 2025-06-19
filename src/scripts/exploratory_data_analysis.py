"""
CICIoT2023 Dataset EDA for Training Configuration

Focused analysis to extract optimal LSTM and RL training configurations
with 3 essential visualizations and actionable recommendations.
Supports both sample-based and full dataset analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    EDA analyzer focused on training configuration recommendations.
    
    Generates 3 key plots:
    1. Attack Distribution & Class Balance
    2. Feature Quality & Scaling Requirements  
    3. LSTM Configuration Optimization
    
    Supports both sample-based and full dataset analysis.
    """
    
    def __init__(self, data_path: Path, output_path: Path, use_full_dataset: bool = False) -> None:
        """
        Initialize analyzer.
        
        Args:
            data_path: Path to raw CICIoT2023 dataset directory
            output_path: Path to save analysis results
            use_full_dataset: Whether to analyze the complete dataset or use sampling
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.use_full_dataset = use_full_dataset
        
        # Analysis results
        self.config_recommendations: Dict[str, Any] = {}
        
        logger.info(f"EDA analyzer initialized. Output: {self.output_path}")
        logger.info(f"Full dataset analysis: {use_full_dataset}")
    
    def _process_csv_file(self, csv_file: Path, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single CSV file and return summary statistics.
        
        Args:
            csv_file: Path to CSV file
            sample_size: Optional sample size for memory efficiency
            
        Returns:
            Summary statistics for the file
        """
        try:
            # Load file
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Sample if requested and file is large
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            # Clean column names and identify label column
            label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
            label_column = None
            
            for candidate in label_candidates:
                if candidate in df.columns:
                    label_column = candidate
                    break
            
            if label_column and label_column != 'label':
                df = df.rename(columns={label_column: 'label'})
            
            # Calculate file statistics
            file_stats = {
                'file_name': csv_file.name,
                'total_samples': len(df),
                'n_features': len(df.columns) - (1 if 'label' in df.columns else 0),
                'has_labels': 'label' in df.columns,
                'missing_values': df.isnull().sum().sum(),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024**2)
            }
            
            # Label analysis
            if 'label' in df.columns:
                df['label'] = df['label'].astype(str).str.strip().str.lower()
                label_counts = df['label'].value_counts()
                file_stats.update({
                    'label_distribution': label_counts.to_dict(),
                    'n_classes': len(label_counts)
                })
            
            # Feature analysis for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_df = df[numeric_cols]
                # Handle infinite values
                numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
                
                feature_ranges = numeric_df.max() - numeric_df.min()
                feature_variance = numeric_df.var()
                
                file_stats.update({
                    'numeric_features': len(numeric_cols),
                    'zero_variance_features': (feature_variance == 0).sum(),
                    'low_variance_features': ((feature_variance > 0) & 
                                            (feature_variance < feature_variance.quantile(0.05))).sum(),
                    'high_range_features': (feature_ranges > 1000).sum(),
                    'feature_ranges_stats': feature_ranges.describe().to_dict()
                })
            
            return file_stats
            
        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")
            return {
                'file_name': csv_file.name,
                'error': str(e),
                'total_samples': 0
            }
    
    def load_dataset_parallel(self, sample_size: Optional[int] = None, 
                            max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Load dataset using parallel processing for efficiency.
        
        Args:
            sample_size: Optional sample size per file (None for full dataset)
            max_workers: Number of parallel workers (None for auto)
            
        Returns:
            Aggregated dataset statistics
        """
        logger.info(f"Loading dataset {'(full)' if not sample_size else f'(sample: {sample_size:,})'} from {self.data_path}...")
        
        # Find CSV files
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            csv_files = list(self.data_path.glob("**/*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Process files in parallel
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(csv_files))
        
        file_stats = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_csv_file, csv_file, sample_size): csv_file 
                for csv_file in csv_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                csv_file = future_to_file[future]
                try:
                    stats = future.result()
                    file_stats.append(stats)
                    if 'error' not in stats:
                        logger.info(f"Processed {csv_file.name}: {stats['total_samples']:,} samples")
                except Exception as e:
                    logger.error(f"Error processing {csv_file}: {e}")
        
        # Aggregate statistics
        total_samples = sum(stats.get('total_samples', 0) for stats in file_stats)
        total_files_processed = len([s for s in file_stats if 'error' not in s])
        
        # Aggregate label distributions
        aggregated_labels = {}
        for stats in file_stats:
            if 'label_distribution' in stats:
                for label, count in stats['label_distribution'].items():
                    aggregated_labels[label] = aggregated_labels.get(label, 0) + count
        
        # Aggregate feature statistics
        feature_stats = {
            'total_features': max((stats.get('n_features', 0) for stats in file_stats), default=0),
            'zero_variance_total': sum(stats.get('zero_variance_features', 0) for stats in file_stats),
            'low_variance_total': sum(stats.get('low_variance_features', 0) for stats in file_stats),
            'high_range_total': sum(stats.get('high_range_features', 0) for stats in file_stats),
            'missing_values_total': sum(stats.get('missing_values', 0) for stats in file_stats)
        }
        
        aggregated_stats = {
            'total_samples': total_samples,
            'total_files': len(csv_files),
            'files_processed': total_files_processed,
            'label_distribution': aggregated_labels,
            'feature_stats': feature_stats,
            'file_details': file_stats
        }
        
        logger.info(f"Dataset loaded: {total_samples:,} total samples from {total_files_processed} files")
        return aggregated_stats
    
    def load_dataset_sample(self, sample_size: int = 100000) -> pd.DataFrame:
        """
        Load representative sample from CICIoT2023 dataset (legacy method for compatibility).
        
        Args:
            sample_size: Number of samples to load
            
        Returns:
            Sample DataFrame
        """
        if self.use_full_dataset:
            # For full dataset, load a representative sample for plotting
            return self._load_representative_sample(sample_size)
        else:
            return self._load_representative_sample(sample_size)
    
    def _load_representative_sample(self, sample_size: int) -> pd.DataFrame:
        """Load a representative sample for plotting and detailed analysis."""
        logger.info(f"Loading representative sample of {sample_size:,} for detailed analysis...")
        
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            csv_files = list(self.data_path.glob("**/*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        # Load sample from multiple files for diversity
        dfs = []
        samples_per_file = max(1000, sample_size // min(len(csv_files), 10))
        
        for csv_file in csv_files[:10]:  # Use up to 10 files for diversity
            try:
                df_temp = pd.read_csv(csv_file, low_memory=False)
                if len(df_temp) > samples_per_file:
                    df_temp = df_temp.sample(n=samples_per_file, random_state=42)
                dfs.append(df_temp)
                logger.debug(f"Loaded {len(df_temp):,} samples from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
                continue
        
        # Combine and final sample
        df = pd.concat(dfs, ignore_index=True)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Clean column names and identify label column
        label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
        label_column = None
        
        for candidate in label_candidates:
            if candidate in df.columns:
                label_column = candidate
                break
        
        if label_column and label_column != 'label':
            df = df.rename(columns={label_column: 'label'})
        
        logger.info(f"Representative sample: {df.shape[0]:,} samples, {df.shape[1]} features")
        return df
    
    def analyze_attack_distribution(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze attack distribution for class balance recommendations.
        
        Args:
            df: Optional DataFrame (if None, uses full dataset statistics)
            
        Returns:
            Attack distribution analysis results
        """
        if self.use_full_dataset and df is None:
            # Use pre-computed statistics from full dataset
            dataset_stats = self.load_dataset_parallel()
            label_distribution = dataset_stats['label_distribution']
            
            if not label_distribution:
                return {'has_labels': False}
            
            # Convert to pandas Series for easier manipulation
            attack_counts = pd.Series(label_distribution)
            
        else:
            # Use provided DataFrame
            if df is None or 'label' not in df.columns:
                return {'has_labels': False}
            
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            attack_counts = df['label'].value_counts()
        
        # Categorize benign vs attacks
        benign_labels = ['benign', 'normal', 'legitimate']
        benign_count = sum(attack_counts.get(label, 0) for label in benign_labels)
        attack_count = attack_counts.sum() - benign_count
        
        # Class imbalance metrics
        imbalance_ratio = attack_counts.max() / attack_counts.min() if attack_counts.min() > 0 else float('inf')
        minority_ratio = attack_counts.min() / attack_counts.sum()
        
        return {
            'has_labels': True,
            'n_classes': len(attack_counts),
            'benign_samples': benign_count,
            'attack_samples': attack_count,
            'imbalance_ratio': imbalance_ratio,
            'minority_ratio': minority_ratio,
            'attack_distribution': attack_counts.head(10).to_dict(),
            'needs_balancing': imbalance_ratio > 10,
            'severe_imbalance': imbalance_ratio > 100
        }
    
    def analyze_feature_quality(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze feature quality for preprocessing recommendations.
        
        Args:
            df: Optional DataFrame (if None, uses full dataset statistics)
            
        Returns:
            Feature quality analysis results
        """
        if self.use_full_dataset and df is None:
            # Use pre-computed statistics from full dataset
            dataset_stats = self.load_dataset_parallel()
            feature_stats = dataset_stats['feature_stats']
            
            return {
                'has_numeric_features': True,
                'n_features': feature_stats['total_features'],
                'zero_variance_count': feature_stats['zero_variance_total'],
                'low_variance_count': feature_stats['low_variance_total'],
                'high_range_count': feature_stats['high_range_total'],
                'missing_values': feature_stats['missing_values_total'],
                'needs_scaling': feature_stats['high_range_total'] > feature_stats['total_features'] * 0.3,
                'needs_feature_selection': (feature_stats['zero_variance_total'] + 
                                          feature_stats['low_variance_total']) > feature_stats['total_features'] * 0.1,
                'feature_ranges': {}  # Would need more detailed analysis for full dataset
            }
        
        else:
            # Use provided DataFrame (original implementation)
            if df is None:
                return {'has_numeric_features': False}
            
            feature_cols = [col for col in df.columns if col != 'label']
            numeric_df = df[feature_cols].select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return {'has_numeric_features': False}
            
            # Handle infinite values
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            
            # Feature statistics
            feature_ranges = numeric_df.max() - numeric_df.min()
            feature_variance = numeric_df.var()
            
            # Quality metrics
            zero_variance_features = (feature_variance == 0).sum()
            low_variance_features = ((feature_variance > 0) & 
                                   (feature_variance < feature_variance.quantile(0.05))).sum()
            high_range_features = (feature_ranges > 1000).sum()
            missing_values = numeric_df.isnull().sum().sum()
            
            return {
                'has_numeric_features': True,
                'n_features': len(numeric_df.columns),
                'zero_variance_count': zero_variance_features,
                'low_variance_count': low_variance_features,
                'high_range_count': high_range_features,
                'missing_values': missing_values,
                'needs_scaling': high_range_features > len(numeric_df.columns) * 0.3,
                'needs_feature_selection': (zero_variance_features + low_variance_features) > len(numeric_df.columns) * 0.1,
                'feature_ranges': feature_ranges.describe().to_dict()
            }
    
    def analyze_lstm_optimization(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze optimal LSTM configuration parameters.
        
        Args:
            df: Optional DataFrame (if None, uses full dataset statistics)
            
        Returns:
            LSTM optimization recommendations
        """
        if self.use_full_dataset and df is None:
            # Use full dataset statistics
            dataset_stats = self.load_dataset_parallel()
            n_samples = dataset_stats['total_samples']
            n_features = dataset_stats['feature_stats']['total_features']
        else:
            # Use provided DataFrame
            if df is None:
                raise ValueError("DataFrame required for sample-based analysis")
            n_samples = len(df)
            n_features = len([col for col in df.columns if col != 'label'])
        
        # Memory and sequence analysis (same logic but scaled for full dataset)
        sequence_configs = []
        for seq_len in [5, 10, 15, 20]:
            for batch_size in [32, 64, 128]:
                # For full dataset, estimate based on typical sequence generation
                if self.use_full_dataset:
                    # Estimate sequences generated from full dataset
                    estimated_sequences = n_samples * 0.8  # Rough estimate
                else:
                    estimated_sequences = max(0, n_samples - seq_len + 1)
                
                memory_mb = (estimated_sequences * seq_len * n_features * 4) / (1024**2)  # float32
                training_batches = estimated_sequences // batch_size
                
                # Feasibility score (lower is better)
                memory_score = min(memory_mb / 4000, 5)  # Increased limit for full dataset
                efficiency_score = max(0, 5 - training_batches / 10000)  # Adjusted for larger dataset
                
                sequence_configs.append({
                    'seq_len': seq_len,
                    'batch_size': batch_size,
                    'n_sequences': int(estimated_sequences),
                    'memory_mb': memory_mb,
                    'training_batches': training_batches,
                    'feasible': memory_mb < 8000,  # Increased limit for full dataset
                    'score': memory_score + efficiency_score
                })
        
        # Find optimal configuration
        feasible_configs = [c for c in sequence_configs if c['feasible']]
        if feasible_configs:
            optimal_config = min(feasible_configs, key=lambda x: x['score'])
        else:
            optimal_config = min(sequence_configs, key=lambda x: x['memory_mb'])
        
        # Training recommendations based on data size (adjusted for full dataset)
        if n_samples < 100000:
            epochs_rec = 20
            patience_rec = 5
        elif n_samples < 1000000:
            epochs_rec = 30
            patience_rec = 7
        else:
            epochs_rec = 50
            patience_rec = 10
        
        return {
            'optimal_sequence_length': optimal_config['seq_len'],
            'optimal_batch_size': optimal_config['batch_size'],
            'estimated_sequences': optimal_config['n_sequences'],
            'estimated_memory_mb': optimal_config['memory_mb'],
            'recommended_epochs': epochs_rec,
            'recommended_patience': patience_rec,
            'input_size': n_features,
            'all_configs': sequence_configs
        }
    
    def generate_configuration_plots(self, df: Optional[pd.DataFrame] = None) -> None:
        """Generate 3 key plots for training configuration."""
        logger.info("Generating configuration plots...")
        
        # For plotting, always use a representative sample
        if df is None:
            df = self._load_representative_sample(50000)  # Larger sample for better plots
        
        # Run analyses (will use full dataset stats if enabled)
        attack_analysis = self.analyze_attack_distribution(df)
        feature_analysis = self.analyze_feature_quality(df)
        lstm_analysis = self.analyze_lstm_optimization(df)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Plot 1: Attack Distribution & Class Balance
        ax1 = plt.subplot(1, 3, 1)
        if attack_analysis['has_labels']:
            attack_dist = attack_analysis['attack_distribution']
            top_attacks = list(attack_dist.items())[:8]  # Top 8 for visibility
            
            labels, counts = zip(*top_attacks)
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            bars = ax1.bar(range(len(labels)), counts, color=colors)
            title_suffix = "(Full Dataset)" if self.use_full_dataset else "(Sample)"
            ax1.set_title(f'Attack Distribution {title_suffix}\n(Imbalance: {attack_analysis["imbalance_ratio"]:.1f}:1)', 
                         fontweight='bold', fontsize=12)
            ax1.set_xlabel('Attack Types')
            ax1.set_ylabel('Sample Count')
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels([l[:8] + '...' if len(l) > 8 else l for l in labels], 
                               rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=8)
            
            # Add balance indicator
            if attack_analysis['severe_imbalance']:
                balance_text = "⚠️ Severe Imbalance"
                color = 'red'
            elif attack_analysis['needs_balancing']:
                balance_text = "⚡ Moderate Imbalance"
                color = 'orange'
            else:
                balance_text = "✅ Well Balanced"
                color = 'green'
            
            ax1.text(0.02, 0.98, balance_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                    fontweight='bold', color='white')
        else:
            ax1.text(0.5, 0.5, 'No Attack Labels Found', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=14)
            ax1.set_title('Attack Distribution Analysis', fontweight='bold')
        
        # Plot 2: Feature Quality Assessment
        ax2 = plt.subplot(1, 3, 2)
        if feature_analysis['has_numeric_features']:
            quality_categories = ['Good Quality', 'Low Variance', 'Zero Variance', 'High Range']
            good_features = (feature_analysis['n_features'] - 
                           feature_analysis['zero_variance_count'] - 
                           feature_analysis['low_variance_count'])
            
            quality_counts = [
                good_features,
                feature_analysis['low_variance_count'],
                feature_analysis['zero_variance_count'],
                feature_analysis['high_range_count']
            ]
            
            colors = ['lightgreen', 'orange', 'red', 'lightblue']
            valid_data = [(cat, count, color) for cat, count, color in zip(quality_categories, quality_counts, colors) if count > 0]
            
            if valid_data:
                categories, counts, colors = zip(*valid_data)
                wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
                title_suffix = "(Full Dataset)" if self.use_full_dataset else "(Sample)"
                ax2.set_title(f'Feature Quality {title_suffix}', fontweight='bold', fontsize=12)
                
                # Add preprocessing recommendations
                preprocessing_notes = []
                if feature_analysis['needs_scaling']:
                    preprocessing_notes.append("🔧 Scaling Required")
                if feature_analysis['needs_feature_selection']:
                    preprocessing_notes.append("🎯 Feature Selection Needed")
                if feature_analysis['missing_values'] > 0:
                    preprocessing_notes.append("🔍 Handle Missing Values")
                
                if preprocessing_notes:
                    note_text = '\n'.join(preprocessing_notes)
                    ax2.text(0.02, 0.02, note_text, transform=ax2.transAxes, 
                            verticalalignment='bottom', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No Numeric Features Found', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=14)
            ax2.set_title('Feature Quality Assessment', fontweight='bold')
        
        # Plot 3: LSTM Configuration Optimization
        ax3 = plt.subplot(1, 3, 3)
        configs = lstm_analysis['all_configs']
        
        # Create configuration heatmap
        seq_lengths = sorted(list(set(c['seq_len'] for c in configs)))
        batch_sizes = sorted(list(set(c['batch_size'] for c in configs)))
        
        # Create matrix of memory usage
        memory_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
        feasible_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
        
        for config in configs:
            i = seq_lengths.index(config['seq_len'])
            j = batch_sizes.index(config['batch_size'])
            memory_matrix[i, j] = config['memory_mb']
            feasible_matrix[i, j] = 1 if config['feasible'] else 0
        
        # Plot heatmap
        im = ax3.imshow(memory_matrix, cmap='RdYlGn_r', aspect='auto')
        title_suffix = "(Full Dataset)" if self.use_full_dataset else "(Sample)"
        ax3.set_title(f'LSTM Memory Usage {title_suffix}\nby Configuration', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Sequence Length')
        
        # Set ticks
        ax3.set_xticks(range(len(batch_sizes)))
        ax3.set_xticklabels(batch_sizes)
        ax3.set_yticks(range(len(seq_lengths)))
        ax3.set_yticklabels(seq_lengths)
        
        # Add text annotations
        for i in range(len(seq_lengths)):
            for j in range(len(batch_sizes)):
                memory_val = memory_matrix[i, j]
                is_feasible = feasible_matrix[i, j]
                
                if memory_val < 1000:  # Less than 1GB
                    text_color = 'black'
                    text = f'{memory_val:.0f}'
                else:
                    text_color = 'white'
                    text = f'{memory_val/1000:.1f}G'
                
                if not is_feasible:
                    text += '\n❌'
                elif (seq_lengths[i] == lstm_analysis['optimal_sequence_length'] and 
                      batch_sizes[j] == lstm_analysis['optimal_batch_size']):
                    text += '\n⭐'
                
                ax3.text(j, i, text, ha='center', va='center', 
                        color=text_color, fontweight='bold', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Memory Usage (MB)', rotation=270, labelpad=15)
        
        # Add optimal configuration annotation
        opt_text = (f"⭐ Optimal: seq_len={lstm_analysis['optimal_sequence_length']}, "
                   f"batch_size={lstm_analysis['optimal_batch_size']}")
        ax3.text(0.02, 0.98, opt_text, transform=ax3.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_path / "training_configuration_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store analyses for config generation
        self.attack_analysis = attack_analysis
        self.feature_analysis = feature_analysis
        self.lstm_analysis = lstm_analysis
        
        logger.info("Configuration plots generated successfully")
    
    def generate_config_recommendations(self) -> Dict[str, Any]:
        """Generate training configuration recommendations."""
        logger.info("Generating configuration recommendations...")
        
        # Base configuration
        config = {
            'dataset': {
                'sequence_length': self.lstm_analysis['optimal_sequence_length'],
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15
            },
            'lstm': {
                'model': {
                    'input_size': self.lstm_analysis['input_size'],
                    'hidden_size': 128,  # Good balance
                    'num_layers': 2,
                    'dropout': 0.3 if self.feature_analysis.get('needs_feature_selection', False) else 0.2
                },
                'training': {
                    'epochs': self.lstm_analysis['recommended_epochs'],
                    'batch_size': self.lstm_analysis['optimal_batch_size'],
                    'learning_rate': 0.001,
                    'early_stopping_patience': self.lstm_analysis['recommended_patience'],
                    'weight_decay': 1e-5
                },
                'data': {
                    'sequence_length': self.lstm_analysis['optimal_sequence_length'],
                    'pin_memory': True,
                    'num_workers': 4,  
                }
            },
            'rl': {
                'training': {
                    'total_timesteps': 100000,  # Standard for IoT defense
                    'eval_freq': 10000
                }
            },
            'preprocessing': {
                'apply_scaling': self.feature_analysis.get('needs_scaling', True),
                'feature_selection': self.feature_analysis.get('needs_feature_selection', False),
                'handle_missing': self.feature_analysis.get('missing_values', 0) > 0
            }
        }
        
        # Adjust for class imbalance
        if hasattr(self, 'attack_analysis') and self.attack_analysis.get('has_labels', False):
            if self.attack_analysis['severe_imbalance']:
                config['lstm']['training']['use_class_weights'] = True
                config['lstm']['training']['focal_loss'] = True
                config['dataset']['sampling_strategy'] = 'balanced'
            elif self.attack_analysis['needs_balancing']:
                config['lstm']['training']['use_class_weights'] = True
        
        # Memory optimization (adjusted for full dataset)
        if self.lstm_analysis['estimated_memory_mb'] > 4000:
            config['lstm']['training']['gradient_accumulation_steps'] = 4
            config['lstm']['data']['num_workers'] = 2
        elif self.lstm_analysis['estimated_memory_mb'] > 2000:
            config['lstm']['training']['gradient_accumulation_steps'] = 2
            config['lstm']['data']['num_workers'] = 2
        else:
            config['lstm']['data']['num_workers'] = 4
        
        return config
    
    def generate_summary_report(self) -> None:
        """Generate concise summary report."""
        report_path = self.output_path / "eda_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("CICIoT2023 Dataset - Configuration Recommendations\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis Type: {'Full Dataset' if self.use_full_dataset else 'Sample-based'}\n\n")
            
            # Dataset Summary
            f.write("DATASET CHARACTERISTICS:\n")
            f.write("-" * 30 + "\n")
            if hasattr(self, 'attack_analysis') and self.attack_analysis.get('has_labels'):
                f.write(f"• Attack Types: {self.attack_analysis['n_classes']}\n")
                f.write(f"• Class Imbalance: {self.attack_analysis['imbalance_ratio']:.1f}:1\n")
                f.write(f"• Balance Status: {'Severe' if self.attack_analysis['severe_imbalance'] else 'Moderate' if self.attack_analysis['needs_balancing'] else 'Good'}\n")
            
            if hasattr(self, 'feature_analysis'):
                f.write(f"• Total Features: {self.feature_analysis.get('n_features', 'Unknown')}\n")
                f.write(f"• Scaling Required: {'Yes' if self.feature_analysis.get('needs_scaling') else 'No'}\n")
                f.write(f"• Feature Selection Needed: {'Yes' if self.feature_analysis.get('needs_feature_selection') else 'No'}\n")
            
            # Recommended Configuration
            f.write(f"\nRECOMMENDED LSTM CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            if hasattr(self, 'lstm_analysis'):
                f.write(f"• Sequence Length: {self.lstm_analysis['optimal_sequence_length']}\n")
                f.write(f"• Batch Size: {self.lstm_analysis['optimal_batch_size']}\n")
                f.write(f"• Training Epochs: {self.lstm_analysis['recommended_epochs']}\n")
                f.write(f"• Early Stopping Patience: {self.lstm_analysis['recommended_patience']}\n")
                f.write(f"• Estimated Memory: {self.lstm_analysis['estimated_memory_mb']:.0f} MB\n")
            
            # Key Recommendations
            f.write(f"\nKEY RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            recommendations = []
            
            if hasattr(self, 'attack_analysis') and self.attack_analysis.get('severe_imbalance'):
                recommendations.append("• Use class weights and balanced sampling for severe imbalance")
            
            if hasattr(self, 'feature_analysis'):
                if self.feature_analysis.get('needs_scaling'):
                    recommendations.append("• Apply StandardScaler for feature normalization")
                if self.feature_analysis.get('needs_feature_selection'):
                    recommendations.append("• Remove zero/low variance features")
            
            if hasattr(self, 'lstm_analysis') and self.lstm_analysis['estimated_memory_mb'] > 4000:
                recommendations.append("• Use gradient accumulation for memory optimization")
            
            if self.use_full_dataset:
                recommendations.append("• Consider distributed training for full dataset")
                recommendations.append("• Use data loading optimizations (multiple workers, prefetch)")
            
            if not recommendations:
                recommendations.append("• Configuration is well-suited for standard training")
            
            for rec in recommendations:
                f.write(f"{rec}\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete EDA analysis."""
        logger.info("Starting EDA analysis...")
        
        try:
            # Generate plots and analysis
            self.generate_configuration_plots()
            
            # Generate recommendations
            config = self.generate_config_recommendations()
            
            # Save summary report
            self.generate_summary_report()
            
            logger.info("EDA analysis completed successfully")
            
            print(f"\n🎉 EDA Analysis Complete!")
            print(f"📊 Configuration plot: {self.output_path}/training_configuration_analysis.png")
            print(f"📄 Summary report: {self.output_path}/eda_summary.txt")
            
            return config
            
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            raise


def main() -> None:
    """Main function to run EDA analysis."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CICIoT2023 EDA Analysis')
    parser.add_argument('--full-dataset', action='store_true', 
                       help='Analyze the complete dataset (default: use sampling)')
    parser.add_argument('--data-path', type=str, default='data/raw/CICIoT2023',
                       help='Path to raw dataset')
    parser.add_argument('--output-path', type=str, default='results/exploratory_data_analysis',
                       help='Output path for results')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure paths
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    analysis_type = "Full Dataset" if args.full_dataset else "Sample-based"
    print(f"🔍 Starting CICIoT2023 EDA ({analysis_type})")
    print(f"📂 Dataset: {data_path}")
    print(f"📊 Output: {output_path}")
    
    # Run analysis
    analyzer = EDAAnalyzer(data_path, output_path, use_full_dataset=args.full_dataset)
    config = analyzer.run_analysis()
    
    print(f"\n✨ Key Findings:")
    if 'lstm' in config:
        lstm_config = config['lstm']
        print(f"   • Optimal sequence length: {config['dataset']['sequence_length']}")
        print(f"   • Recommended batch size: {lstm_config['training']['batch_size']}")
        print(f"   • Training epochs: {lstm_config['training']['epochs']}")
        print(f"   • Num workers: {lstm_config['data']['num_workers']}")
    
    if 'preprocessing' in config:
        prep = config['preprocessing']
        if prep['apply_scaling']:
            print(f"   • ⚠️  Feature scaling required")
        if prep['feature_selection']:
            print(f"   • ⚠️  Feature selection recommended")


if __name__ == "__main__":
    main()