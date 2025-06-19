"""
LSTM Attack Predictor for CICIoT2023 Dataset

Real-time attack prediction using LSTM trained on CICIoT2023 IoT network flows.
Provides sophisticated attack classification for RL environment integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataset_loader import CICIoTDataLoader, LoaderConfig

logger = logging.getLogger(__name__)

@dataclass
class LSTMConfig:
    """Configuration for LSTM attack predictor."""
    input_size: int = 46  # CICIoT2023 features
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 34  # All attack types + benign
    dropout: float = 0.2
    sequence_length: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "models/lstm/ciciot2023_attack_predictor.pth"


class RealDataLSTMPredictor(nn.Module):
    """
    LSTM-based attack predictor for real IoT network data.
    
    Predicts attack types from sequences of network flow features
    extracted from CICIoT2023 dataset.
    """
    
    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize LSTM attack predictor.
        
        Args:
            config: LSTM configuration parameters
        """
        super(RealDataLSTMPredictor, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize LSTM and linear layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Classification
        output = self.classifier(last_output)  # (batch_size, num_classes)
        
        return output
    
    def predict_attack_probability(self, sequences: np.ndarray) -> np.ndarray:
        """
        Predict attack probabilities for input sequences.
        
        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, input_size)
            
        Returns:
            Attack probabilities of shape (n_samples, num_classes)
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and move to device
            x = torch.FloatTensor(sequences).to(self.config.device)
            
            # Forward pass
            logits = self.forward(x)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            return probabilities.cpu().numpy()
    
    def predict_attack_type(self, sequences: np.ndarray, 
                           class_names: List[str]) -> List[Dict[str, Any]]:
        """
        Predict attack types with confidence scores.
        
        Args:
            sequences: Input sequences
            class_names: List of class names corresponding to indices
            
        Returns:
            List of predictions with attack type and confidence
        """
        probabilities = self.predict_attack_probability(sequences)
        
        predictions = []
        for prob in probabilities:
            predicted_class = np.argmax(prob)
            confidence = prob[predicted_class]
            
            predictions.append({
                'attack_type': class_names[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': dict(zip(class_names, prob.astype(float)))
            })
        
        return predictions


class RealDataTrainer:
    """
    Trainer for RealDataLSTMPredictor using CICIoT2023 dataset.
    Compatible with new dataset processor format.
    """
    
    def __init__(self, config: LSTMConfig, data_loader: 'CICIoTDataLoader') -> None:
        """
        Initialize trainer with data loader.
        
        Args:
            config: LSTM training configuration
            data_loader: CICIoT data loader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get feature info
        feature_info = self.data_loader.get_feature_info()
        logger.info(f"Training with {feature_info['n_features']} features, "
                   f"{feature_info['n_classes']} classes")
        
        # Store feature info for later use
        self.feature_info = feature_info
        
        # Update config with actual values
        self.config.input_size = feature_info['n_features']
        self.config.num_classes = feature_info['n_classes']
        
        # Create model
        self.model = RealDataLSTMPredictor(self.config).to(self.device)
        
        # Get data loaders
        self.dataloaders = self.data_loader.get_lstm_dataloaders()
        
        # Setup training components
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Handle class imbalance if configured
        if hasattr(self.config, 'use_class_weights') and self.config.use_class_weights:
            self.class_weights = self._calculate_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Use focal loss if configured
        if hasattr(self.config, 'focal_loss') and self.config.focal_loss:
            self.criterion = self._get_focal_loss()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Load labels to compute weights
        train_loader = self.dataloaders['train']
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        
        logger.info("Class weights calculated for imbalanced dataset")
        return torch.FloatTensor(class_weights).to(self.device)
    
    def _get_focal_loss(self):
        """Focal Loss for addressing class imbalance."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha: float = 1, gamma: float = 2):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            
            def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        logger.info("Using Focal Loss for class imbalance")
        return FocalLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            # Move data to device
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, "
                           f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the LSTM attack predictor.
        
        Returns:
            Training history with losses and accuracies
        """
        logger.info("Starting LSTM training on real CICIoT2023 data...")
        
        # Initialize MLflow tracking
        mlflow.start_run(run_name="lstm_real_data_training")
        
        try:
            # Log parameters
            mlflow.log_params({
                'model_type': 'LSTM',
                'dataset': 'CICIoT2023',
                'input_size': self.config.input_size,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'num_classes': self.config.num_classes,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs,
                'device': str(self.device)
            })
            
            # Training history
            history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            best_val_accuracy = 0.0
            
            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Training
                train_loss, train_acc = self.train_epoch(self.dataloaders['train'])
                
                # Validation
                val_loss, val_acc = self.validate_epoch(self.dataloaders['val'])
                
                # Update history
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }, step=epoch)
                
                # Save best model
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    self.save_model()
                    logger.info(f"New best validation accuracy: {val_acc:.4f}")
                
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log model
            mlflow.pytorch.log_model(self.model, "lstm_model")
            
            # Final evaluation
            test_loss, test_acc = self.validate_epoch(self.dataloaders['test'])
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_accuracy
            })
            
            logger.info(f"Training completed! Best Val Acc: {best_val_accuracy:.4f}, "
                       f"Test Acc: {test_acc:.4f}")
            
            return history
            
        finally:
            mlflow.end_run()
    
    def evaluate_detailed(self) -> Dict[str, Any]:
        """
        Perform detailed evaluation with classification report and confusion matrix.
        
        Returns:
            Detailed evaluation results
        """
        logger.info("Performing detailed evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in self.dataloaders['test']:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Get class names
        class_names = self.feature_info['class_names']
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(conf_matrix, class_names)
        
        return {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'accuracy': class_report['accuracy'],
            'macro_avg_f1': class_report['macro avg']['f1-score'],
            'weighted_avg_f1': class_report['weighted avg']['f1-score']
        }
    
    def _plot_confusion_matrix(self, conf_matrix: np.ndarray, 
                              class_names: List[str]) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix array
            class_names: List of class names
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('LSTM Attack Predictor - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("results/plots/lstm_confusion_matrix.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {plot_path}")
    
    def save_model(self) -> None:
        """Save the trained model."""
        model_path = Path(self.config.model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_info': self.feature_info
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.feature_info = checkpoint['feature_info']
        
        logger.info(f"Model loaded from {model_path}")


def create_enhanced_attack_predictor(data_path: Path, 
                                   config: Optional[LSTMConfig] = None) -> RealDataTrainer:
    """
    Factory function to create enhanced LSTM attack predictor.
    
    Args:
        data_path: Path to processed CICIoT2023 data
        config: Optional LSTM configuration
        
    Returns:
        Configured trainer ready for training
    """
    if config is None:
        config = LSTMConfig()
    
    trainer = RealDataTrainer(config, data_path)
    return trainer