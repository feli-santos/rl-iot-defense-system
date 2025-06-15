"""
CICIoT2023 Dataset Exploratory Data Analysis

Comprehensive analysis of CICIoT2023 dataset for IoT defense system.
Generates visualizations, statistical summaries, and training recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import warnings
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class CICIoTAnalyzer:
    """
    Comprehensive analyzer for CICIoT2023 dataset.
    
    Performs statistical analysis, data quality assessment, and generates
    visualizations with training recommendations for IoT defense system.
    """
    
    def __init__(self, data_path: Path, output_path: Path) -> None:
        """
        Initialize analyzer.
        
        Args:
            data_path: Path to raw CICIoT2023 dataset directory
            output_path: Path to save analysis results
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_path / "plots"
        self.reports_dir = self.output_path / "reports"
        self.plots_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Analysis results storage
        self.analysis_results = {}
        self.recommendations = []
        
        logger.info(f"CICIoT analyzer initialized. Output: {self.output_path}")
    
    def discover_dataset_files(self) -> List[Path]:
        """
        Discover all CSV files in the dataset directory.
        
        Returns:
            List of CSV file paths
        """
        csv_files = list(self.data_path.glob("*.csv"))
        
        if not csv_files:
            # Try alternative patterns
            csv_files = list(self.data_path.glob("**/*.csv"))
        
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_path}")
        
        if csv_files:
            for file in csv_files[:5]:  # Show first 5 files
                logger.info(f"  - {file.name}")
            if len(csv_files) > 5:
                logger.info(f"  ... and {len(csv_files) - 5} more files")
        
        return csv_files
    
    def load_data_sample(self, sample_size: int = 50000) -> pd.DataFrame:
        """
        Load a sample of the CICIoT2023 dataset from multiple CSV files.
        
        Args:
            sample_size: Maximum number of samples to load for analysis
            
        Returns:
            Loaded dataset sample as DataFrame
        """
        csv_files = self.discover_dataset_files()
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        logger.info(f"Loading data sample from {len(csv_files)} files...")
        
        # Load first file to get column structure
        first_file = csv_files[0]
        logger.info(f"Reading structure from: {first_file.name}")
        
        try:
            df_first = pd.read_csv(first_file, nrows=1000)  # Small sample to check structure
        except Exception as e:
            logger.error(f"Failed to read {first_file}: {e}")
            raise
        
        logger.info(f"Dataset structure: {df_first.shape[1]} columns")
        logger.info(f"Columns: {list(df_first.columns[:10])}{'...' if len(df_first.columns) > 10 else ''}")
        
        # Determine label column (common names in CICIoT2023)
        label_candidates = ['label', 'Label', 'class', 'Class', 'attack', 'Attack']
        label_column = None
        
        for candidate in label_candidates:
            if candidate in df_first.columns:
                label_column = candidate
                break
        
        if label_column is None:
            # Try to find column with string values (likely labels)
            for col in df_first.columns:
                if df_first[col].dtype == 'object':
                    unique_vals = df_first[col].dropna().unique()
                    if len(unique_vals) > 1 and len(unique_vals) < 100:  # Reasonable number of classes
                        label_column = col
                        logger.info(f"Detected label column: {col}")
                        break
        
        # Load data from all files with sampling
        dfs = []
        total_loaded = 0
        samples_per_file = max(1, sample_size // len(csv_files))
        
        for i, file_path in enumerate(csv_files):
            if total_loaded >= sample_size:
                break
                
            try:
                logger.info(f"Loading from {file_path.name} ({i+1}/{len(csv_files)})...")
                
                # Read file with sampling
                df_temp = pd.read_csv(file_path)
                
                # Sample from file if it's large
                if len(df_temp) > samples_per_file:
                    df_temp = df_temp.sample(n=samples_per_file, random_state=42)
                
                dfs.append(df_temp)
                total_loaded += len(df_temp)
                
                logger.info(f"  Loaded {len(df_temp):,} samples (total: {total_loaded:,})")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not dfs:
            raise ValueError("Failed to load any data files")
        
        # Combine all dataframes
        logger.info("Combining dataframes...")
        df = pd.concat(dfs, ignore_index=True)
        
        # Final sampling if still too large
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Standardize label column name
        if label_column and label_column != 'label':
            df = df.rename(columns={label_column: 'label'})
            logger.info(f"Renamed '{label_column}' to 'label'")
        
        logger.info(f"Final dataset loaded: {df.shape[0]:,} samples, {df.shape[1]} features")
        
        # Store dataset info
        self.analysis_results['dataset_info'] = {
            'source_files': [f.name for f in csv_files],
            'files_loaded': len(dfs),
            'total_samples': len(df),
            'label_column': label_column,
            'has_labels': 'label' in df.columns
        }
        
        return df
    
    def analyze_dataset_overview(self, df: pd.DataFrame) -> None:
        """
        Analyze basic dataset characteristics.
        
        Args:
            df: Dataset DataFrame
        """
        logger.info("Analyzing dataset overview...")
        
        # Basic statistics
        n_samples, n_features = df.shape
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        
        # Data types analysis
        dtype_counts = df.dtypes.value_counts()
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        # Infinite values check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        # Store results
        self.analysis_results['overview'] = {
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_usage_mb': memory_usage,
            'dtype_counts': dtype_counts.to_dict(),
            'missing_values': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percent[missing_percent > 0].to_dict(),
            'infinite_values': inf_counts
        }
        
        # Create overview visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Overview Analysis', fontsize=16, fontweight='bold')
        
        # Data types distribution
        dtype_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Data Types Distribution')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage by column type
        memory_by_type = df.select_dtypes(include=['float64']).memory_usage(deep=True).sum() / 1024**2
        memory_other = memory_usage - memory_by_type
        memory_data = {'Float64': memory_by_type, 'Other': memory_other}
        
        axes[0, 1].pie([v for v in memory_data.values() if v > 0], 
                      labels=[k for k, v in memory_data.items() if v > 0], 
                      autopct='%1.1f%%')
        axes[0, 1].set_title('Memory Usage by Type (MB)')
        
        # Missing values (if any)
        if len(missing_counts[missing_counts > 0]) > 0:
            missing_top = missing_counts[missing_counts > 0].head(10)
            missing_top.plot(kind='bar', ax=axes[1, 0], color='coral')
            axes[1, 0].set_title('Features with Missing Values')
            axes[1, 0].set_ylabel('Missing Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values Found', 
                           transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Missing Values Analysis')
        
        # Dataset characteristics
        characteristics = {
            f'Samples\n({n_samples:,})': n_samples / 1000,  # Scale for visualization
            f'Features\n({n_features})': n_features,
            f'Memory\n({memory_usage:.1f} MB)': memory_usage,
            f'Files\n({self.analysis_results["dataset_info"]["files_loaded"]})': self.analysis_results["dataset_info"]["files_loaded"]
        }
        
        bars = axes[1, 1].bar(range(len(characteristics)), list(characteristics.values()), 
                             color=['lightgreen', 'lightblue', 'lightsalmon', 'lightcyan'])
        axes[1, 1].set_title('Dataset Characteristics')
        axes[1, 1].set_ylabel('Scaled Values')
        axes[1, 1].set_xticks(range(len(characteristics)))
        axes[1, 1].set_xticklabels(list(characteristics.keys()), rotation=0, ha='center')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "01_dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Data quality recommendations
        if missing_counts.sum() > 0:
            self.recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'issue': f'{missing_counts.sum():,} missing values found across {len(missing_counts[missing_counts > 0])} features',
                'recommendation': 'Handle missing values before training (imputation or removal)'
            })
        
        if inf_counts:
            self.recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'issue': f'Infinite values found in {len(inf_counts)} features',
                'recommendation': 'Replace infinite values with NaN and handle appropriately'
            })
        
        logger.info("Dataset overview analysis completed")
    
    def analyze_attack_distribution(self, df: pd.DataFrame) -> None:
        """
        Analyze attack type distribution and characteristics.
        
        Args:
            df: Dataset DataFrame
        """
        logger.info("Analyzing attack distribution...")
        
        if 'label' not in df.columns:
            logger.warning("No 'label' column found for attack analysis")
            # Try to analyze without labels
            self.analysis_results['attacks'] = {
                'has_labels': False,
                'note': 'No label column found in dataset'
            }
            return
        
        # Clean label data
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        
        # Attack distribution
        attack_counts = df['label'].value_counts()
        attack_percentages = df['label'].value_counts(normalize=True) * 100
        
        # Identify benign vs attack samples
        benign_labels = ['benign', 'normal', 'legitimate']
        benign_samples = 0
        for label in benign_labels:
            benign_samples += attack_counts.get(label, 0)
        
        attack_samples = len(df) - benign_samples
        
        # Store results
        self.analysis_results['attacks'] = {
            'has_labels': True,
            'total_attack_types': len(attack_counts),
            'benign_samples': benign_samples,
            'attack_samples': attack_samples,
            'attack_distribution': attack_counts.to_dict(),
            'attack_percentages': attack_percentages.to_dict(),
            'class_imbalance_ratio': attack_counts.max() / attack_counts.min() if attack_counts.min() > 0 else float('inf')
        }
        
        # Create attack distribution visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Attack Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Overall attack vs benign
        if benign_samples > 0:
            benign_attack_data = {'Benign': benign_samples, 'Attack': attack_samples}
            wedges, texts, autotexts = axes[0, 0].pie(
                benign_attack_data.values(), 
                labels=benign_attack_data.keys(),
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral']
            )
            axes[0, 0].set_title('Benign vs Attack Distribution')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Benign Samples Identified', 
                           transform=axes[0, 0].transAxes, ha='center', va='center')
            axes[0, 0].set_title('Benign vs Attack Distribution')
        
        # Top 15 attack types
        top_attacks = attack_counts.head(15)
        bars = axes[0, 1].bar(range(len(top_attacks)), top_attacks.values, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(top_attacks))))
        axes[0, 1].set_title(f'Top {len(top_attacks)} Attack Types Distribution')
        axes[0, 1].set_xlabel('Attack Types')
        axes[0, 1].set_ylabel('Sample Count')
        axes[0, 1].set_xticks(range(len(top_attacks)))
        axes[0, 1].set_xticklabels(top_attacks.index, rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:,}', ha='center', va='bottom', fontsize=8)
        
        # Class imbalance visualization
        attack_percentages_top15 = attack_percentages.head(15)
        y_pos = range(len(attack_percentages_top15))
        axes[1, 0].barh(y_pos, attack_percentages_top15.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(attack_percentages_top15))))
        axes[1, 0].set_title(f'Attack Types Percentage (Top {len(attack_percentages_top15)})')
        axes[1, 0].set_xlabel('Percentage (%)')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(attack_percentages_top15.index)
        
        # Cumulative distribution
        cumulative_percentages = attack_percentages.cumsum()
        axes[1, 1].plot(range(len(cumulative_percentages)), cumulative_percentages.values, 
                       marker='o', linewidth=2, markersize=4)
        axes[1, 1].set_title('Cumulative Attack Type Distribution')
        axes[1, 1].set_xlabel('Attack Type Rank')
        axes[1, 1].set_ylabel('Cumulative Percentage (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
        axes[1, 1].axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% Line')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "02_attack_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate class imbalance recommendations
        imbalance_ratio = self.analysis_results['attacks']['class_imbalance_ratio']
        if imbalance_ratio > 100:
            self.recommendations.append({
                'category': 'Data Imbalance',
                'priority': 'High',
                'issue': f'Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)',
                'recommendation': 'Use stratified sampling, class weights, or SMOTE for balancing'
            })
        elif imbalance_ratio > 10:
            self.recommendations.append({
                'category': 'Data Imbalance',
                'priority': 'Medium',
                'issue': f'Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1)',
                'recommendation': 'Consider class weights in loss function'
            })
        
        logger.info("Attack distribution analysis completed")
    
    def analyze_feature_characteristics(self, df: pd.DataFrame) -> None:
        """
        Analyze feature distributions and statistical properties.
        
        Args:
            df: Dataset DataFrame
        """
        logger.info("Analyzing feature characteristics...")
        
        # Exclude label column for feature analysis
        feature_cols = [col for col in df.columns if col not in ['label', 'Label']]
        features_df = df[feature_cols].select_dtypes(include=[np.number])  # Only numeric features
        
        if features_df.empty:
            logger.warning("No numeric features found for analysis")
            return
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Basic statistics
        basic_stats = features_df.describe()
        
        # Feature variability
        feature_variance = features_df.var().sort_values(ascending=False)
        feature_std = features_df.std().sort_values(ascending=False)
        
        # Zero/constant features
        zero_variance_features = feature_variance[feature_variance == 0].index.tolist()
        low_variance_features = feature_variance[
            (feature_variance > 0) & (feature_variance < feature_variance.quantile(0.1))
        ].index.tolist()
        
        # Feature correlations (sample for performance)
        sample_size = min(1000, len(features_df))
        correlation_sample = features_df.sample(n=sample_size, random_state=42)
        correlation_matrix = correlation_sample.corr()
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if not pd.isna(correlation_matrix.iloc[i, j]):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
        
        # Store results
        self.analysis_results['features'] = {
            'n_features': len(feature_cols),
            'n_numeric_features': len(features_df.columns),
            'zero_variance_features': zero_variance_features,
            'low_variance_features': low_variance_features,
            'high_correlation_pairs': len(high_corr_pairs),
            'mean_feature_variance': feature_variance.mean(),
            'top_variable_features': feature_variance.head(10).to_dict()
        }
        
        # Create feature analysis visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Feature Characteristics Analysis', fontsize=16, fontweight='bold')
        
        # Feature variance distribution
        valid_variances = feature_variance[feature_variance > 0]
        if len(valid_variances) > 0:
            axes[0, 0].hist(np.log10(valid_variances.values + 1e-10), bins=50, alpha=0.7, 
                           color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Feature Variance Distribution (Log Scale)')
            axes[0, 0].set_xlabel('Log10(Variance)')
            axes[0, 0].set_ylabel('Number of Features')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Top 15 most variable features
        top_var_features = feature_variance.head(15)
        if len(top_var_features) > 0:
            bars = axes[0, 1].bar(range(len(top_var_features)), np.log10(top_var_features.values + 1e-10),
                                 color='lightgreen')
            axes[0, 1].set_title('Top 15 Most Variable Features (Log Scale)')
            axes[0, 1].set_xlabel('Features')
            axes[0, 1].set_ylabel('Log10(Variance)')
            axes[0, 1].set_xticks(range(len(top_var_features)))
            axes[0, 1].set_xticklabels([f'F{i+1}' for i in range(len(top_var_features))], rotation=45)
        
        # Correlation heatmap (top 20 features by variance)
        top_features = feature_variance.head(20).index
        if len(top_features) > 1:
            corr_subset = correlation_matrix.loc[top_features, top_features]
            
            im = axes[1, 0].imshow(corr_subset.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 0].set_title('Feature Correlation Matrix (Top 20 Variable Features)')
            axes[1, 0].set_xticks(range(len(top_features)))
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_xticklabels([f'F{i+1}' for i in range(len(top_features))], rotation=45)
            axes[1, 0].set_yticklabels([f'F{i+1}' for i in range(len(top_features))])
            plt.colorbar(im, ax=axes[1, 0])
        
        # Feature quality summary
        quality_data = {
            'Good Quality': len(features_df.columns) - len(zero_variance_features) - len(low_variance_features),
            'Low Variance': len(low_variance_features),
            'Zero Variance': len(zero_variance_features)
        }
        
        colors = ['lightgreen', 'orange', 'red']
        axes[1, 1].pie([v for v in quality_data.values() if v > 0], 
                      labels=[k for k, v in quality_data.items() if v > 0],
                      autopct='%1.1f%%', colors=colors[:len([v for v in quality_data.values() if v > 0])])
        axes[1, 1].set_title('Feature Quality Distribution')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "03_feature_characteristics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature quality recommendations
        if len(zero_variance_features) > 0:
            self.recommendations.append({
                'category': 'Feature Quality',
                'priority': 'High',
                'issue': f'{len(zero_variance_features)} features have zero variance',
                'recommendation': 'Remove zero-variance features before training'
            })
        
        if len(low_variance_features) > 5:
            self.recommendations.append({
                'category': 'Feature Quality',
                'priority': 'Medium',
                'issue': f'{len(low_variance_features)} features have very low variance',
                'recommendation': 'Consider feature selection to remove low-variance features'
            })
        
        if len(high_corr_pairs) > 10:
            self.recommendations.append({
                'category': 'Feature Correlation',
                'priority': 'Medium',
                'issue': f'{len(high_corr_pairs)} highly correlated feature pairs found',
                'recommendation': 'Apply correlation-based feature selection or PCA'
            })
        
        logger.info("Feature characteristics analysis completed")
    
    def analyze_lstm_suitability(self, df: pd.DataFrame) -> None:
        """
        Analyze dataset suitability for LSTM training.
        
        Args:
            df: Dataset DataFrame
        """
        logger.info("Analyzing LSTM training suitability...")
        
        feature_cols = [col for col in df.columns if col not in ['label', 'Label']]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number])
        
        # Sequence generation simulation
        sequence_lengths = [5, 10, 15, 20]
        sequence_analysis = {}
        
        for seq_len in sequence_lengths:
            # Calculate theoretical number of sequences
            n_sequences = max(0, len(df) - seq_len + 1)
            
            # Memory estimation (rough)
            memory_per_sequence = seq_len * len(numeric_features.columns) * 4  # 4 bytes per float32
            total_memory_mb = (n_sequences * memory_per_sequence) / (1024**2)
            
            sequence_analysis[seq_len] = {
                'n_sequences': n_sequences,
                'memory_estimate_mb': total_memory_mb,
                'feasible': total_memory_mb < 4000  # 4GB threshold
            }
        
        # Class distribution for sequences
        if 'label' in df.columns:
            label_distribution = df['label'].value_counts(normalize=True)
            minority_class_ratio = label_distribution.min()
        else:
            minority_class_ratio = 0.5
        
        # Feature scaling analysis
        if not numeric_features.empty:
            sample_features = numeric_features.sample(n=min(1000, len(numeric_features)), random_state=42)
            feature_ranges = sample_features.max() - sample_features.min()
            features_need_scaling = (feature_ranges > 100).sum()
        else:
            features_need_scaling = 0
        
        # Store results
        self.analysis_results['lstm_suitability'] = {
            'sequence_analysis': sequence_analysis,
            'minority_class_ratio': minority_class_ratio,
            'features_need_scaling': features_need_scaling,
            'n_numeric_features': len(numeric_features.columns),
            'recommended_sequence_length': self._recommend_sequence_length(sequence_analysis),
            'recommended_batch_size': self._recommend_batch_size(sequence_analysis)
        }
        
        # Create LSTM suitability visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('LSTM Training Suitability Analysis', fontsize=16, fontweight='bold')
        
        # Sequence length analysis
        seq_lengths = list(sequence_analysis.keys())
        n_sequences = [sequence_analysis[sl]['n_sequences'] for sl in seq_lengths]
        memory_estimates = [sequence_analysis[sl]['memory_estimate_mb'] for sl in seq_lengths]
        
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        bars1 = ax1.bar([sl - 0.2 for sl in seq_lengths], n_sequences, 
                       width=0.4, label='# Sequences', color='lightblue', alpha=0.7)
        bars2 = ax1_twin.bar([sl + 0.2 for sl in seq_lengths], memory_estimates, 
                            width=0.4, label='Memory (MB)', color='lightcoral', alpha=0.7)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Number of Sequences', color='blue')
        ax1_twin.set_ylabel('Memory Estimate (MB)', color='red')
        ax1.set_title('Sequence Length Impact Analysis')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Memory feasibility
        feasible_seq_lengths = [sl for sl in seq_lengths if sequence_analysis[sl]['feasible']]
        infeasible_seq_lengths = [sl for sl in seq_lengths if not sequence_analysis[sl]['feasible']]
        
        feasibility_data = {
            'Feasible': len(feasible_seq_lengths),
            'Memory Intensive': len(infeasible_seq_lengths)
        }
        
        if sum(feasibility_data.values()) > 0:
            axes[0, 1].pie([v for v in feasibility_data.values() if v > 0], 
                          labels=[k for k, v in feasibility_data.items() if v > 0],
                          autopct='%1.0f', colors=['lightgreen', 'lightcoral'])
        axes[0, 1].set_title('Sequence Length Feasibility')
        
        # Feature scaling requirements
        total_features = len(numeric_features.columns)
        if total_features > 0:
            scaling_data = {
                'Need Scaling': features_need_scaling,
                'Already Scaled': total_features - features_need_scaling
            }
            
            axes[1, 0].pie([v for v in scaling_data.values() if v > 0], 
                          labels=[k for k, v in scaling_data.items() if v > 0],
                          autopct='%1.1f%%', colors=['orange', 'lightgreen'])
        axes[1, 0].set_title('Feature Scaling Requirements')
        
        # Class balance assessment
        class_balance_categories = ['Excellent (>40%)', 'Good (20-40%)', 'Poor (10-20%)', 'Critical (<10%)']
        if minority_class_ratio > 0.4:
            balance_category = 0
        elif minority_class_ratio > 0.2:
            balance_category = 1
        elif minority_class_ratio > 0.1:
            balance_category = 2
        else:
            balance_category = 3
        
        balance_scores = [0, 0, 0, 0]
        balance_scores[balance_category] = 1
        
        colors = ['green', 'yellow', 'orange', 'red']
        bars = axes[1, 1].bar(range(len(class_balance_categories)), balance_scores, 
                             color=[colors[i] if balance_scores[i] else 'lightgray' 
                                   for i in range(len(colors))])
        axes[1, 1].set_title(f'Class Balance Assessment\n(Minority Class: {minority_class_ratio:.1%})')
        axes[1, 1].set_ylabel('Current Status')
        axes[1, 1].set_xticks(range(len(class_balance_categories)))
        axes[1, 1].set_xticklabels(class_balance_categories, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "04_lstm_suitability.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # LSTM-specific recommendations
        recommended_seq_len = self.analysis_results['lstm_suitability']['recommended_sequence_length']
        recommended_batch_size = self.analysis_results['lstm_suitability']['recommended_batch_size']
        
        self.recommendations.append({
            'category': 'LSTM Configuration',
            'priority': 'High',
            'issue': 'Optimal LSTM hyperparameters based on data characteristics',
            'recommendation': f'Use sequence_length={recommended_seq_len}, batch_size={recommended_batch_size}'
        })
        
        if features_need_scaling > total_features * 0.5:
            self.recommendations.append({
                'category': 'Data Preprocessing',
                'priority': 'High',
                'issue': f'{features_need_scaling} features need scaling',
                'recommendation': 'Apply StandardScaler or MinMaxScaler before LSTM training'
            })
        
        if minority_class_ratio < 0.1:
            self.recommendations.append({
                'category': 'Class Balance',
                'priority': 'Critical',
                'issue': f'Severe class imbalance (minority: {minority_class_ratio:.1%})',
                'recommendation': 'Use weighted loss, SMOTE, or stratified sequence generation'
            })
        
        logger.info("LSTM suitability analysis completed")
    
    def _recommend_sequence_length(self, sequence_analysis: Dict) -> int:
        """Recommend optimal sequence length based on analysis."""
        feasible_lengths = [sl for sl, data in sequence_analysis.items() if data['feasible']]
        
        if not feasible_lengths:
            return 5  # Minimum safe length
        
        # Choose length that maximizes sequences while staying memory-feasible
        best_length = max(feasible_lengths)
        return min(best_length, 15)  # Cap at 15 for training efficiency
    
    def _recommend_batch_size(self, sequence_analysis: Dict) -> int:
        """Recommend optimal batch size based on memory constraints."""
        recommended_seq_len = self._recommend_sequence_length(sequence_analysis)
        estimated_memory = sequence_analysis[recommended_seq_len]['memory_estimate_mb']
        
        if estimated_memory > 2000:
            return 32  # Small batch for large memory usage
        elif estimated_memory > 1000:
            return 64  # Medium batch
        else:
            return 128  # Large batch for efficient training
    
    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.reports_dir / f"ciciot2023_analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CICIoT2023 Dataset Analysis Report\n")
            f.write("IoT Defense System - Exploratory Data Analysis\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Path: {self.data_path}\n\n")
            
            # Dataset Info
            if 'dataset_info' in self.analysis_results:
                info = self.analysis_results['dataset_info']
                f.write("DATASET SOURCE INFORMATION\n")
                f.write("-" * 50 + "\n")
                f.write(f"   • Source Files Found: {len(info['source_files'])}\n")
                f.write(f"   • Files Successfully Loaded: {info['files_loaded']}\n")
                f.write(f"   • Total Samples Analyzed: {info['total_samples']:,}\n")
                f.write(f"   • Label Column: {info['label_column'] or 'Not found'}\n")
                f.write(f"   • Has Attack Labels: {info['has_labels']}\n\n")
            
            # Dataset Overview
            if 'overview' in self.analysis_results:
                overview = self.analysis_results['overview']
                f.write("1. DATASET OVERVIEW\n")
                f.write("-" * 50 + "\n")
                f.write(f"   • Total Samples: {overview['n_samples']:,}\n")
                f.write(f"   • Total Features: {overview['n_features']}\n")
                f.write(f"   • Memory Usage: {overview['memory_usage_mb']:.1f} MB\n")
                f.write(f"   • Data Types: {overview['dtype_counts']}\n")
                
                if overview['missing_values']:
                    f.write(f"   • Missing Values: {len(overview['missing_values'])} features affected\n")
                else:
                    f.write("   • Missing Values: None detected\n")
                
                if overview['infinite_values']:
                    f.write(f"   • Infinite Values: {len(overview['infinite_values'])} features affected\n")
                f.write("\n")
            
            # Attack Distribution Analysis
            if 'attacks' in self.analysis_results:
                attacks = self.analysis_results['attacks']
                f.write("2. ATTACK DISTRIBUTION ANALYSIS\n")
                f.write("-" * 50 + "\n")
                
                if attacks['has_labels']:
                    f.write(f"   • Total Attack Types: {attacks['total_attack_types']}\n")
                    f.write(f"   • Benign Samples: {attacks['benign_samples']:,}\n")
                    f.write(f"   • Attack Samples: {attacks['attack_samples']:,}\n")
                    
                    if attacks['class_imbalance_ratio'] != float('inf'):
                        f.write(f"   • Class Imbalance Ratio: {attacks['class_imbalance_ratio']:.1f}:1\n")
                    
                    f.write("\n   Top 10 Attack Types:\n")
                    attack_dist = attacks['attack_distribution']
                    sorted_attacks = sorted(attack_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                    for i, (attack_type, count) in enumerate(sorted_attacks, 1):
                        percentage = (count / sum(attack_dist.values())) * 100
                        f.write(f"     {i:2d}. {attack_type}: {count:,} ({percentage:.1f}%)\n")
                else:
                    f.write("   • No attack labels found in dataset\n")
                f.write("\n")
            
            # Feature Characteristics
            if 'features' in self.analysis_results:
                features = self.analysis_results['features']
                f.write("3. FEATURE CHARACTERISTICS\n")
                f.write("-" * 50 + "\n")
                f.write(f"   • Total Features: {features['n_features']}\n")
                f.write(f"   • Numeric Features: {features['n_numeric_features']}\n")
                f.write(f"   • Zero Variance Features: {len(features['zero_variance_features'])}\n")
                f.write(f"   • Low Variance Features: {len(features['low_variance_features'])}\n")
                f.write(f"   • High Correlation Pairs: {features['high_correlation_pairs']}\n")
                f.write(f"   • Mean Feature Variance: {features['mean_feature_variance']:.4f}\n")
                
                if features['zero_variance_features']:
                    f.write(f"\n   Zero Variance Features: {features['zero_variance_features'][:5]}{'...' if len(features['zero_variance_features']) > 5 else ''}\n")
                f.write("\n")
            
            # LSTM Suitability
            if 'lstm_suitability' in self.analysis_results:
                lstm = self.analysis_results['lstm_suitability']
                f.write("4. LSTM TRAINING SUITABILITY\n")
                f.write("-" * 50 + "\n")
                f.write(f"   • Recommended Sequence Length: {lstm['recommended_sequence_length']}\n")
                f.write(f"   • Recommended Batch Size: {lstm['recommended_batch_size']}\n")
                f.write(f"   • Minority Class Ratio: {lstm['minority_class_ratio']:.1%}\n")
                f.write(f"   • Features Needing Scaling: {lstm['features_need_scaling']}\n")
                f.write(f"   • Numeric Features Available: {lstm['n_numeric_features']}\n")
                
                f.write("\n   Sequence Length Analysis:\n")
                for seq_len, data in lstm['sequence_analysis'].items():
                    status = "✓ Feasible" if data['feasible'] else "✗ Memory Intensive"
                    f.write(f"     Length {seq_len}: {data['n_sequences']:,} sequences, "
                           f"{data['memory_estimate_mb']:.0f} MB, {status}\n")
                f.write("\n")
            
            # Recommendations
            f.write("5. TRAINING RECOMMENDATIONS\n")
            f.write("-" * 50 + "\n")
            
            # Group recommendations by priority
            priority_groups = {'Critical': [], 'High': [], 'Medium': [], 'Low': []}
            for rec in self.recommendations:
                priority_groups[rec['priority']].append(rec)
            
            for priority in ['Critical', 'High', 'Medium', 'Low']:
                if priority_groups[priority]:
                    f.write(f"\n   {priority} Priority:\n")
                    for i, rec in enumerate(priority_groups[priority], 1):
                        f.write(f"     {i}. [{rec['category']}] {rec['issue']}\n")
                        f.write(f"        → {rec['recommendation']}\n\n")
            
            # Configuration Recommendations
            f.write("6. RECOMMENDED CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            f.write("   LSTM Configuration:\n")
            if 'lstm_suitability' in self.analysis_results:
                lstm = self.analysis_results['lstm_suitability']
                f.write(f"     sequence_length: {lstm['recommended_sequence_length']}\n")
                f.write(f"     batch_size: {lstm['recommended_batch_size']}\n")
                f.write(f"     input_size: {lstm['n_numeric_features']}\n")
            
            f.write("     hidden_size: 128  # Good balance of capacity and efficiency\n")
            f.write("     num_layers: 2     # Sufficient for IoT attack patterns\n")
            f.write("     dropout: 0.2      # Prevent overfitting\n")
            f.write("     learning_rate: 0.001  # Conservative learning rate\n")
            
            if 'attacks' in self.analysis_results and self.analysis_results['attacks']['has_labels']:
                attacks = self.analysis_results['attacks']
                if attacks.get('class_imbalance_ratio', 1) > 10:
                    f.write("     use_class_weights: true  # Address class imbalance\n")
            
            f.write("\n   RL Environment Configuration:\n")
            f.write("     max_steps: 1000\n")
            f.write("     attack_probability: 0.3  # Adjust based on observed patterns\n")
            f.write("     state_history_length: 10\n")
            f.write("     action_history_length: 5\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Analysis Report\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Comprehensive report saved to {report_path}")
    
    def run_complete_analysis(self) -> None:
        """Run complete EDA analysis pipeline."""
        logger.info("Starting complete EDA analysis...")
        
        try:
            # Load data sample
            df = self.load_data_sample(sample_size=50000)
            
            # Run all analysis components
            self.analyze_dataset_overview(df)
            self.analyze_attack_distribution(df)
            self.analyze_feature_characteristics(df)
            self.analyze_lstm_suitability(df)
            
            # Generate final report
            self.generate_comprehensive_report()
            
            logger.info("Complete EDA analysis finished successfully")
            print(f"\n🎉 Analysis Complete!")
            print(f"📊 Plots saved to: {self.plots_dir}")
            print(f"📄 Report saved to: {self.reports_dir}")
            print(f"💡 Found {len(self.recommendations)} recommendations for training optimization")
            
        except Exception as e:
            logger.error(f"EDA analysis failed: {e}")
            raise


def main() -> None:
    """Main function to run EDA analysis."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure paths - using your actual dataset structure
    data_path = Path("data/raw/CICIoT2023")
    output_path = Path("results")
    
    print(f"🔍 Starting CICIoT2023 Dataset Analysis")
    print(f"📂 Dataset path: {data_path}")
    print(f"📊 Output path: {output_path}")
    
    # Run analysis
    analyzer = CICIoTAnalyzer(data_path, output_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()