"""
CICIoT2023 Dataset Processing Utilities

Handles processing of raw CICIoT2023 dataset
into format suitable for LSTM training and RL environment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataProcessingConfig:
    """Configuration for data processing pipeline"""
    dataset_path: Path
    output_path: Path
    sample_size: int = 2_000_000  # Strategic sample size
    sequence_length: int = 10  # For LSTM sequences
    test_split: float = 0.2
    validation_split: float = 0.1
    random_state: int = 42
    max_workers: int = 4  # For parallel processing
    chunk_size: int = 50_000  # Records per chunk
    min_attack_samples: int = 100  # Minimum samples per attack class


class CICIoTProcessor:
    """
    Main processor for CICIoT2023 dataset integration with RL IoT defense system.
    
    Handles massive dataset loading, strategic sampling, feature engineering,
    and preparation for both LSTM attack prediction and RL environment integration.
    """
    
    def __init__(self, config: DataProcessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_stats: Dict[str, Any] = {}
        self.attack_categories: Dict[str, List[str]] = {}
        
        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        # IoT-relevant feature groups for state representation
        self.feature_groups = {
            'network_flow': [
                'flow_duration', 'Duration', 'Rate', 'Srate', 'Drate',
                'Header_Length', 'Tot sum', 'Tot size', 'IAT'
            ],
            'protocol_flags': [
                'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number'
            ],
            'packet_counts': [
                'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'Number'
            ],
            'protocol_types': [
                'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
                'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC'
            ],
            'statistical': [
                'Min', 'Max', 'AVG', 'Std', 'Magnitue', 'Radius',
                'Covariance', 'Variance', 'Weight'
            ],
            'protocol_indicator': ['Protocol Type']
        }
        
        # Attack taxonomy for RL reward design
        self.attack_taxonomy = {
            'ddos': ['DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 
                    'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SYN_Flood',
                    'DDoS-SynonymousIP_Flood', 'DDoS-ICMP_Fragmentation',
                    'DDoS-ACK_Fragmentation', 'DDoS-UDP_Fragmentation',
                    'DDoS-HTTP_Flood', 'DDoS-SlowLoris'],
            'dos': ['DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'DoS-HTTP_Flood'],
            'botnet': ['Mirai-greeth_flood', 'Mirai-udpplain', 'Mirai-greip_flood'],
            'reconnaissance': ['Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PortScan', 
                             'Recon-PingSweep', 'VulnerabilityScan'],
            'mitm': ['MITM-ArpSpoofing', 'DNS_Spoofing'],
            'web_attacks': ['SqlInjection', 'XSS', 'CommandInjection', 'BrowserHijacking'],
            'malware': ['Backdoor_Malware', 'Uploading_Attack'],
            'brute_force': ['DictionaryBruteForce'],
            'benign': ['BenignTraffic']
        }
    
    def discover_dataset_files(self) -> List[Path]:
        """Discover all CSV files in the dataset directory"""
        csv_files = list(self.config.dataset_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.config.dataset_path}")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.config.dataset_path}")
        
        # Sort files for consistent processing order
        csv_files.sort()
        
        # Log file sizes for processing planning
        total_size = sum(f.stat().st_size for f in csv_files)
        logger.info(f"Total dataset size: {total_size / (1024**3):.2f} GB")
        
        return csv_files
    
    def load_file_sample(self, file_path: Path, sample_fraction: float = 0.1) -> pd.DataFrame:
        """Load a sample from a single CSV file"""
        try:
            # Load with sampling for memory efficiency
            df = pd.read_csv(file_path)
            
            if sample_fraction < 1.0:
                df = df.sample(frac=sample_fraction, random_state=self.config.random_state)
            
            logger.info(f"Loaded {len(df)} records from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {str(e)}")
            return pd.DataFrame()
    
    def create_strategic_sample(self, csv_files: List[Path]) -> pd.DataFrame:
        """
        Create a strategic sample ensuring representation of all attack types
        """
        logger.info("Creating strategic sample from dataset...")
        
        # Load samples from multiple files to ensure diversity
        sample_size_per_file = max(1, self.config.sample_size // len(csv_files))
        
        all_samples = []
        total_loaded = 0
        
        for file_path in csv_files:
            if total_loaded >= self.config.sample_size:
                break
                
            # Calculate sample fraction for this file
            try:
                # Quick row count estimation
                with open(file_path, 'r') as f:
                    estimated_rows = sum(1 for _ in f) - 1  # Subtract header
                
                sample_fraction = min(1.0, sample_size_per_file / estimated_rows)
                
                df_sample = self.load_file_sample(file_path, sample_fraction)
                
                if not df_sample.empty:
                    all_samples.append(df_sample)
                    total_loaded += len(df_sample)
                    
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {str(e)}")
                continue
        
        if not all_samples:
            raise ValueError("No data could be loaded from any files")
        
        # Combine all samples
        combined_df = pd.concat(all_samples, ignore_index=True)
        logger.info(f"Combined sample size: {len(combined_df)} records")
        
        # Ensure minimum representation of each attack type
        balanced_df = self._balance_attack_classes(combined_df)
        
        # Final sample size control
        if len(balanced_df) > self.config.sample_size:
            balanced_df = balanced_df.sample(
                n=self.config.sample_size, 
                random_state=self.config.random_state
            )
        
        logger.info(f"Final strategic sample size: {len(balanced_df)} records")
        return balanced_df
    
    def _balance_attack_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure minimum representation of each attack class"""
        
        # Get class distribution
        class_counts = df['label'].value_counts()
        logger.info(f"Original class distribution: {len(class_counts)} classes")
        
        # Identify underrepresented classes
        underrepresented = class_counts[class_counts < self.config.min_attack_samples]
        
        if len(underrepresented) == 0:
            return df
        
        logger.info(f"Boosting {len(underrepresented)} underrepresented classes")
        
        # Boost underrepresented classes through upsampling
        balanced_dfs = [df]
        
        for attack_class, count in underrepresented.items():
            class_df = df[df['label'] == attack_class]
            if len(class_df) > 0:
                # Upsample to minimum threshold
                needed_samples = self.config.min_attack_samples - count
                upsampled = class_df.sample(
                    n=needed_samples, 
                    replace=True, 
                    random_state=self.config.random_state
                )
                balanced_dfs.append(upsampled)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for ML/RL integration"""
        logger.info("Preprocessing features...")
        
        # Separate features and labels
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Handle infinite values and NaNs
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Store feature statistics
        self.feature_stats = {
            'feature_names': list(X.columns),
            'feature_groups': self.feature_groups,
            'n_features': len(X.columns),
            'statistics': X.describe().to_dict()
        }
        
        # Normalize features
        X_normalized = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create attack category mapping
        self._create_attack_categories()
        
        # Combine back
        processed_df = X_normalized.copy()
        processed_df['label'] = y
        processed_df['label_encoded'] = y_encoded
        processed_df['attack_category'] = y.map(self._get_attack_category)
        
        logger.info(f"Preprocessed {len(processed_df)} records with {len(X.columns)} features")
        return processed_df
    
    def _create_attack_categories(self) -> None:
        """Create mappings between attack types and categories"""
        category_mapping = {}
        
        for category, attacks in self.attack_taxonomy.items():
            for attack in attacks:
                category_mapping[attack] = category
        
        self.attack_categories = {
            'mapping': category_mapping,
            'categories': list(self.attack_taxonomy.keys()),
            'n_categories': len(self.attack_taxonomy)
        }
    
    def _get_attack_category(self, attack_label: str) -> str:
        """Get attack category for a given attack label"""
        return self.attack_categories['mapping'].get(attack_label, 'unknown')
    
    def create_lstm_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training from network flow data
        """
        logger.info("Creating LSTM sequences...")
        
        # Sort by relevant temporal features (if available)
        # For now, we'll use the order as-is and create sliding windows
        
        feature_cols = [col for col in df.columns if col not in ['label', 'label_encoded', 'attack_category']]
        
        X = df[feature_cols].values
        y = df['label_encoded'].values
        
        # Create sequences using sliding window
        sequences = []
        labels = []
        
        for i in range(len(X) - self.config.sequence_length + 1):
            seq = X[i:i + self.config.sequence_length]
            label = y[i + self.config.sequence_length - 1]  # Predict last label in sequence
            
            sequences.append(seq)
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        logger.info(f"Created {len(sequences)} sequences of length {self.config.sequence_length}")
        return sequences, labels
    
    def create_rl_states(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create state representations for RL environment
        """
        logger.info("Creating RL state representations...")
        
        # Select key features for RL state space
        state_features = []
        for group_name, features in self.feature_groups.items():
            if group_name in ['network_flow', 'protocol_flags', 'packet_counts']:
                state_features.extend([f for f in features if f in df.columns])
        
        # Create state vectors
        states = df[state_features].values
        
        logger.info(f"Created RL states with {len(state_features)} features")
        return states
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        logger.info("Splitting data...")
        
        # Stratified split to maintain class distribution
        # First split: train + val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_split,
            stratify=df['attack_category'],
            random_state=self.config.random_state
        )
        
        # Second split: train vs val
        val_size = self.config.validation_split / (1 - self.config.test_split)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val['attack_category'],
            random_state=self.config.random_state
        )
        
        splits = {
            'train': train,
            'validation': val,
            'test': test
        }
        
        for split_name, split_df in splits.items():
            logger.info(f"{split_name.capitalize()} set: {len(split_df)} records")
        
        return splits
    
    def save_processed_data(self, data_splits: Dict[str, pd.DataFrame], 
                          lstm_data: Tuple[np.ndarray, np.ndarray],
                          rl_states: np.ndarray) -> None:
        """Save all processed data and metadata"""
        logger.info("Saving processed data...")
        
        # Save data splits
        for split_name, df in data_splits.items():
            output_file = self.config.output_path / f"{split_name}_data.parquet"
            df.to_parquet(output_file, compression='snappy')
            logger.info(f"Saved {split_name} data to {output_file}")
        
        # Save LSTM sequences
        lstm_sequences, lstm_labels = lstm_data
        np.save(self.config.output_path / "lstm_sequences.npy", lstm_sequences)
        np.save(self.config.output_path / "lstm_labels.npy", lstm_labels)
        
        # Save RL states
        np.save(self.config.output_path / "rl_states.npy", rl_states)
        
        # Save preprocessing artifacts
        artifacts = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_stats': self.feature_stats,
            'attack_categories': self.attack_categories,
            'attack_taxonomy': self.attack_taxonomy,
            'feature_groups': self.feature_groups,
            'config': self.config
        }
        
        with open(self.config.output_path / "preprocessing_artifacts.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info("All preprocessing artifacts saved")
    
    def generate_summary_report(self, data_splits: Dict[str, pd.DataFrame]) -> str:
        """Generate comprehensive processing summary"""
        
        report = []
        report.append("="*80)
        report.append("CICIoT2023 DATASET PROCESSING SUMMARY")
        report.append("="*80)
        
        # Dataset overview
        total_records = sum(len(df) for df in data_splits.values())
        report.append(f"\n📊 PROCESSED DATASET:")
        report.append(f"  • Total records: {total_records:,}")
        report.append(f"  • Features: {self.feature_stats['n_features']}")
        report.append(f"  • Attack categories: {self.attack_categories['n_categories']}")
        
        # Data splits
        report.append(f"\n📋 DATA SPLITS:")
        for split_name, df in data_splits.items():
            report.append(f"  • {split_name.capitalize()}: {len(df):,} records ({len(df)/total_records*100:.1f}%)")
        
        # Class distribution
        report.append(f"\n🎯 ATTACK DISTRIBUTION (Training Set):")
        train_df = data_splits['train']
        category_counts = train_df['attack_category'].value_counts()
        for category, count in category_counts.items():
            pct = count / len(train_df) * 100
            report.append(f"  • {category.capitalize()}: {count:,} ({pct:.1f}%)")
        
        # Feature groups
        report.append(f"\n🔧 FEATURE GROUPS:")
        for group_name, features in self.feature_groups.items():
            available_features = [f for f in features if f in self.feature_stats['feature_names']]
            report.append(f"  • {group_name.replace('_', ' ').title()}: {len(available_features)} features")
        
        # Integration readiness
        report.append(f"\n✅ INTEGRATION READY:")
        report.append(f"  • LSTM sequences: Ready for attack prediction training")
        report.append(f"  • RL states: Ready for environment integration")
        report.append(f"  • Data splits: Ready for ML/RL training pipelines")
        report.append(f"  • Preprocessing artifacts: Saved for inference")
        
        return "\n".join(report)
    
    def process_dataset(self) -> None:
        """Main processing pipeline"""
        logger.info("Starting CICIoT2023 dataset processing pipeline...")
        
        try:
            # 1. Discover dataset files
            csv_files = self.discover_dataset_files()
            
            # 2. Create strategic sample
            sample_df = self.create_strategic_sample(csv_files)
            
            # 3. Preprocess features
            processed_df = self.preprocess_features(sample_df)
            
            # 4. Create data splits
            data_splits = self.split_data(processed_df)
            
            # 5. Create LSTM sequences
            lstm_sequences, lstm_labels = self.create_lstm_sequences(data_splits['train'])
            
            # 6. Create RL states
            rl_states = self.create_rl_states(data_splits['train'])
            
            # 7. Save processed data
            self.save_processed_data(data_splits, (lstm_sequences, lstm_labels), rl_states)
            
            # 8. Generate summary report
            summary = self.generate_summary_report(data_splits)
            print(f"\n{summary}")
            
            # Save summary
            with open(self.config.output_path / "processing_summary.txt", 'w') as f:
                f.write(summary)
            
            logger.info("✅ Dataset processing completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Dataset processing failed: {str(e)}")
            raise


def main():
    """Main function to run the data processing pipeline"""
    
    # Configuration
    config = DataProcessingConfig(
        dataset_path=Path("./dataset/CICIoT2023"),
        output_path=Path("./data/processed/ciciot2023"),
        sample_size=2_000_000,  # 2M strategic sample
        sequence_length=10,
        test_split=0.2,
        validation_split=0.1,
        random_state=42
    )
    
    # Initialize processor
    processor = CICIoTProcessor(config)
    
    # Run processing pipeline
    processor.process_dataset()


if __name__ == "__main__":
    main()