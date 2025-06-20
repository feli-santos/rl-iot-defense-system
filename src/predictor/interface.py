"""
Attack Predictor Interface

Bridges between LSTM predictions and RL environment needs.
Provides real-time attack prediction and risk assessment for reward calculation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import json
import joblib
import logging
import pandas as pd

from predictor.attack import LSTMAttackPredictor, LSTMConfig
from utils.dataset_loader import CICIoTDataLoader, LoaderConfig

logger = logging.getLogger(__name__)


class AttackPredictorInterface:
    """
    Attack predictor interface that bridges between LSTM predictions 
    and RL environment needs.
    
    Provides both real-time attack prediction and risk assessment
    for RL reward function calculation.
    """
    
    def __init__(self, model_path: Path, data_path: Path) -> None:
        """
        Initialize attack predictor interface.
        
        Args:
            model_path: Path to trained LSTM model
            data_path: Path to processed dataset
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and configuration
        self._load_model()
        
        # Load preprocessing artifacts
        self._load_preprocessing_artifacts()
        
        # Attack severity mapping for RL rewards
        self.attack_severity = {
            'benign': 0.0,
            'ddos': 0.9,
            'dos': 0.8,
            'botnet': 0.85,
            'mitm': 0.75,
            'reconnaissance': 0.4,
            'web_attacks': 0.7,
            'brute_force': 0.6,
            'malware': 0.95
        }
        
        logger.info("Enhanced attack predictor initialized successfully")
    
    def _load_model(self) -> None:
        """Load the trained LSTM model with proper security settings."""
        try:
            # Add safe globals for our custom classes
            torch.serialization.add_safe_globals([LSTMConfig])
            
            # Load with weights_only=False for backward compatibility
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=False  # Allow loading custom classes
            )
            
            self.config = checkpoint['config']
            self.feature_info = checkpoint['feature_info']
            
            # Initialize model
            self.model = LSTMAttackPredictor(self.config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_preprocessing_artifacts(self) -> None:
        """Load preprocessing artifacts from new file structure."""
        try:
            # Load scaler
            scaler_path = self.data_path / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            # Load label encoder
            encoder_path = self.data_path / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("Label encoder loaded successfully")
            else:
                raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
            
            # Load metadata
            metadata_path = self.data_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.feature_columns = metadata['feature_columns']
                self.class_names = metadata['class_names']
                
                # Create attack categories mapping
                self.attack_categories = self._create_attack_categories_mapping()
                
                logger.info(f"Metadata loaded: {len(self.feature_columns)} features, {len(self.class_names)} classes")
            else:
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessing artifacts: {e}")
            raise
    
    def _create_attack_categories_mapping(self) -> Dict[str, Any]:
        """
        Create attack categories mapping from class names.
        
        Returns:
            Attack categories dictionary
        """
        # Map attack types to categories based on class names
        attack_mapping = {}
        
        for class_name in self.class_names:
            class_lower = class_name.lower()
            
            if 'benign' in class_lower or 'normal' in class_lower:
                attack_mapping[class_name] = 'benign'
            elif 'ddos' in class_lower:
                attack_mapping[class_name] = 'ddos'
            elif 'dos' in class_lower:
                attack_mapping[class_name] = 'dos'
            elif 'botnet' in class_lower or 'mirai' in class_lower:
                attack_mapping[class_name] = 'botnet'
            elif 'mitm' in class_lower or 'man_in_the_middle' in class_lower:
                attack_mapping[class_name] = 'mitm'
            elif 'recon' in class_lower or 'scan' in class_lower:
                attack_mapping[class_name] = 'reconnaissance'
            elif 'web' in class_lower or 'http' in class_lower or 'sql' in class_lower:
                attack_mapping[class_name] = 'web_attacks'
            elif 'brute' in class_lower or 'password' in class_lower:
                attack_mapping[class_name] = 'brute_force'
            elif 'malware' in class_lower or 'trojan' in class_lower:
                attack_mapping[class_name] = 'malware'
            else:
                attack_mapping[class_name] = 'unknown'
        
        return {
            'mapping': attack_mapping,
            'categories': list(set(attack_mapping.values()))
        }
    
    def preprocess_network_state(self, network_features: Dict[str, float]) -> np.ndarray:
        """
        Preprocess network state features for LSTM input.
        
        Args:
            network_features: Dictionary of network features
            
        Returns:
            Preprocessed feature vector
        """
        # Convert to feature vector matching training data format
        feature_vector = np.zeros(self.config.input_size)
        
        # Map network features to LSTM input format
        for i, feature_name in enumerate(self.feature_columns):
            if feature_name in network_features:
                feature_vector[i] = network_features[feature_name]
        
        # Create pandas DataFrame to avoid sklearn feature name warnings
        feature_df = pd.DataFrame([feature_vector], columns=self.feature_columns)
        
        # Normalize using trained scaler
        normalized_features = self.scaler.transform(feature_df)
        
        return normalized_features.flatten()
    
    def predict_attack_risk(self, network_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Predict attack risk from a sequence of network states.
        
        Args:
            network_sequence: List of network state dictionaries
            
        Returns:
            Attack risk assessment with probabilities and severity
        """
        if len(network_sequence) < self.config.sequence_length:
            # Pad sequence if too short
            while len(network_sequence) < self.config.sequence_length:
                network_sequence.insert(0, network_sequence[0] if network_sequence else {})
        
        # Take last sequence_length states
        sequence = network_sequence[-self.config.sequence_length:]
        
        # Preprocess sequence
        processed_sequence = []
        for state in sequence:
            processed_state = self.preprocess_network_state(state)
            processed_sequence.append(processed_state)
        
        processed_sequence = np.array(processed_sequence).reshape(1, self.config.sequence_length, -1)
        
        # Convert to tensor for LSTM prediction
        sequence_tensor = torch.FloatTensor(processed_sequence).to(self.device)
        
        # Predict attack probabilities using correct method name
        probabilities = self.model.predict_proba(sequence_tensor)[0]
        
        # Convert to numpy if it's a tensor
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        
        # Get predicted attack type
        predicted_class_idx = np.argmax(probabilities)
        predicted_attack = self.class_names[predicted_class_idx]
        predicted_category = self.attack_categories['mapping'].get(predicted_attack, 'unknown')
        
        # Calculate risk score
        risk_score = self.attack_severity.get(predicted_category, 0.5) * probabilities[predicted_class_idx]
        
        return {
            'predicted_attack': predicted_attack,
            'attack_category': predicted_category,
            'confidence': float(probabilities[predicted_class_idx]),
            'risk_score': float(risk_score),
            'all_probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'is_attack': predicted_category != 'benign',
            'severity_level': self._get_severity_level(risk_score)
        }
    
    def _get_severity_level(self, risk_score: float) -> str:
        """
        Convert risk score to severity level.
        
        Args:
            risk_score: Numerical risk score
            
        Returns:
            Severity level string
        """
        if risk_score < 0.2:
            return 'low'
        elif risk_score < 0.5:
            return 'medium'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def calculate_reward(self, network_sequence: List[Dict[str, float]], 
                           defense_action: int, attack_occurred: bool) -> float:
        """
        Calculate reward based on attack prediction and defense effectiveness.

        Args:
            network_sequence: Sequence of network states
            defense_action: Defense action taken (0-3)
            attack_occurred: Whether an actual attack occurred
            
        Returns:
            Reward value for training
        """
        # Get attack risk assessment
        risk_assessment = self.predict_attack_risk(network_sequence)
        
        predicted_attack = risk_assessment['is_attack']
        risk_score = risk_assessment['risk_score']
        
        # Base reward calculation
        if attack_occurred and predicted_attack:
            # Correct attack prediction
            reward = 10.0 * risk_score
        elif not attack_occurred and not predicted_attack:
            # Correct benign prediction
            reward = 5.0
        elif attack_occurred and not predicted_attack:
            # Missed attack (false negative)
            reward = -15.0 * risk_score
        else:
            # False positive
            reward = -5.0 * (1 - risk_score)
        
        # Defense action effectiveness bonus/penalty
        if attack_occurred:
            # Reward appropriate defense actions during attacks
            action_effectiveness = [0.8, 0.9, 0.7, 0.6]  # Effectiveness per action
            reward += 5.0 * action_effectiveness[defense_action]
        else:
            # Penalize unnecessary aggressive actions during benign periods
            action_penalty = [0.0, -1.0, -2.0, -3.0]  # Penalty per action
            reward += action_penalty[defense_action]
        
        return reward
    
    def get_attack_insights(self, network_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Get detailed attack insights for analysis and debugging.
        
        Args:
            network_sequence: Sequence of network states
            
        Returns:
            Detailed attack analysis
        """
        risk_assessment = self.predict_attack_risk(network_sequence)
        
        # Top attack types by probability
        all_probs = risk_assessment['all_probabilities']
        top_attacks = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'current_assessment': risk_assessment,
            'top_attack_types': top_attacks,
            'sequence_length': len(network_sequence),
            'feature_analysis': self._analyze_key_features(network_sequence[-1] if network_sequence else {}),
            'temporal_pattern': self._analyze_temporal_pattern(network_sequence)
        }
    
    def _analyze_key_features(self, current_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze key features in current network state.
        
        Args:
            current_state: Current network state dictionary
            
        Returns:
            Feature analysis results
        """
        return {
            'num_features': len(current_state),
            'non_zero_features': sum(1 for v in current_state.values() if v != 0),
            'max_feature_value': max(current_state.values()) if current_state else 0
        }
    
    def _analyze_temporal_pattern(self, network_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the network sequence.
        
        Args:
            network_sequence: Sequence of network states
            
        Returns:
            Temporal pattern analysis
        """
        if len(network_sequence) < 2:
            return {'pattern': 'insufficient_data'}
        
        return {
            'sequence_trend': 'increasing' if len(network_sequence) > 5 else 'stable',
            'variability': 'high' if len(set(str(s) for s in network_sequence)) > len(network_sequence) * 0.8 else 'low'
        }