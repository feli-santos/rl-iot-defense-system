"""
LSTM Attack Predictor for CICIoT2023 Dataset

Real-time attack prediction using LSTM trained on CICIoT2023 IoT network flows.
Provides sophisticated attack classification for RL environment integration.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import logging
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import sys

from utils.dataset_loader import CICIoTDataLoader, LoaderConfig

logger = logging.getLogger(__name__)

@dataclass
class LSTMConfig:
    """Configuration for LSTM attack predictor."""
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 10
    dropout: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    sequence_length: int = 10
    model_save_path: str = "models/lstm_attack_predictor.pth"
    # EDA recommendations
    use_class_weights: bool = False
    focal_loss: bool = False


class LSTMAttackPredictor(nn.Module):
    """
    LSTM-based attack predictor trained on CICIoT2023 dataset.
    
    Provides sophisticated multi-class attack classification for integration
    with RL environment as attack prediction component.
    """
    
    def __init__(self, config: LSTMConfig) -> None:
        """
        Initialize LSTM attack predictor.
        
        Args:
            config: LSTM configuration parameters
        """
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )
        
        logger.info(f"LSTM predictor initialized: {config.input_size} → "
                   f"{config.hidden_size} → {config.num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM predictor.
        
        Args:
            x: Input sequences [batch_size, seq_len, features]
            
        Returns:
            Attack classification logits [batch_size, num_classes]
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state for classification
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(last_hidden)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attack prediction probabilities.
        
        Args:
            x: Input sequences [batch_size, seq_len, features]
            
        Returns:
            Attack probabilities [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities


class DataTrainer:
    """
    Trainer for LSTM Predictor using CICIoT2023 dataset.
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
        self.model = LSTMAttackPredictor(self.config).to(self.device)

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
        
        logger.info("Calculating class weights for imbalanced dataset...")
        
        # Load labels to compute weights with simple progress
        train_loader = self.dataloaders['train']
        all_labels = []
        
        print("📊 Collecting labels for class weights...")
        for _, labels in train_loader:
            all_labels.extend(labels.numpy())
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )
        
        logger.info(f"Class weights calculated: min={class_weights.min():.3f}, "
                   f"max={class_weights.max():.3f}")
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
    
    def train_epoch(self, dataloader: DataLoader, epoch: int, 
                   total_epochs: int) -> Tuple[float, float]:
        """
        Train the model for one epoch with clean progress bar.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Create clean progress bar
        pbar = tqdm(
            dataloader, 
            desc=f"Epoch {epoch+1:2d}/{total_epochs}",
            ncols=100,  # Fixed width
            file=sys.stdout,
            leave=False,  # Don't leave progress bar after completion
            miniters=len(dataloader)//20,  # Update every 5%
            maxinterval=2.0  # Max 2 second updates
        )
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
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
            
            # Update progress bar every 5% or last batch
            if batch_idx % max(1, len(dataloader) // 20) == 0 or batch_idx == len(dataloader) - 1:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = correct_predictions / total_samples
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.3f}'
                }, refresh=True)
        
        pbar.close()  # Ensure progress bar is closed
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader: DataLoader, epoch: int, 
                      total_epochs: int, split_name: str = "Val") -> Tuple[float, float]:
        """
        Validate the model for one epoch with clean progress bar.
        
        Args:
            dataloader: Validation data loader
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            split_name: Name of the split (Val/Test)
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Create clean progress bar
        pbar = tqdm(
            dataloader, 
            desc=f"{split_name} {epoch+1:2d}/{total_epochs}",
            ncols=100,
            file=sys.stdout,
            leave=False,
            miniters=len(dataloader)//10,  # Update every 10% for validation
            maxinterval=3.0
        )
        
        with torch.no_grad():
            for batch_idx, (sequences, labels) in enumerate(pbar):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Update less frequently for validation
                if batch_idx % max(1, len(dataloader) // 10) == 0 or batch_idx == len(dataloader) - 1:
                    current_loss = total_loss / (batch_idx + 1)
                    current_acc = correct_predictions / total_samples
                    pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.3f}'
                    }, refresh=True)
        
        pbar.close()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the LSTM attack predictor with clean progress tracking.
        
        Returns:
            Training history with losses and accuracies
        """
        logger.info("🧠 Starting LSTM training on CICIoT2023 dataset...")
        print(f"\n🎯 Training Configuration:")
        print(f"   • Model: LSTM ({self.config.input_size} → {self.config.hidden_size} → {self.config.num_classes})")
        print(f"   • Dataset: {len(self.dataloaders['train'].dataset):,} train samples")
        print(f"   • Device: {self.device}")
        print(f"   • Epochs: {self.config.num_epochs}")
        print(f"   • Batch Size: {self.config.batch_size}")
        print(f"   • Learning Rate: {self.config.learning_rate}")
        if hasattr(self.config, 'use_class_weights') and self.config.use_class_weights:
            print(f"   • ⚖️  Class weights enabled (severe imbalance handling)")
        if hasattr(self.config, 'focal_loss') and self.config.focal_loss:
            print(f"   • 🎯 Focal loss enabled")
        print()
        
        # Initialize MLflow tracking
        mlflow.start_run(run_name="lstm_attack_training")

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
                'device': str(self.device),
                'use_class_weights': getattr(self.config, 'use_class_weights', False),
                'focal_loss': getattr(self.config, 'focal_loss', False)
            })
            
            # Training history
            history = {
                'train_loss': [],
                'train_accuracy': [],
                'val_loss': [],
                'val_accuracy': []
            }
            
            best_val_accuracy = 0.0
            
            # Main training loop - simple epoch counter
            print("🚀 Starting Training...")
            for epoch in range(self.config.num_epochs):
                # Training
                train_loss, train_acc = self.train_epoch(
                    self.dataloaders['train'], epoch, self.config.num_epochs
                )
                
                # Validation
                val_loss, val_acc = self.validate_epoch(
                    self.dataloaders['val'], epoch, self.config.num_epochs, "Val"
                )
                
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
                
                # Save best model and print summary
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    self.save_model()
                    print(f"✅ Epoch {epoch+1:2d}/{self.config.num_epochs} | "
                          f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | "
                          f"🎯 NEW BEST")
                else:
                    print(f"📊 Epoch {epoch+1:2d}/{self.config.num_epochs} | "
                          f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")
            
            # Final test evaluation
            print("\n🧪 Final Test Evaluation...")
            test_loss, test_acc = self.validate_epoch(
                self.dataloaders['test'], self.config.num_epochs-1, 
                self.config.num_epochs, "Test"
            )
            
            # Log model and final metrics
            mlflow.pytorch.log_model(self.model, "lstm_model")
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_accuracy
            })
            
            print(f"\n🎉 LSTM Training Complete!")
            print(f"   • Best Validation Accuracy: {best_val_accuracy:.4f}")
            print(f"   • Final Test Accuracy: {test_acc:.4f}")
            print(f"   • Model saved to: {self.config.model_save_path}")
            
            return history
            
        finally:
            mlflow.end_run()
    
    def evaluate_detailed(self) -> Dict[str, Any]:
        """
        Perform detailed evaluation with classification report and confusion matrix.
        
        Returns:
            Detailed evaluation results
        """
        logger.info("🔍 Performing detailed evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        # Evaluation with simple progress
        print("📊 Evaluating on test set...")
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
        
        # Print summary
        print(f"\n📈 Evaluation Results:")
        print(f"   • Overall Accuracy: {class_report['accuracy']:.4f}")
        print(f"   • Macro F1-Score: {class_report['macro avg']['f1-score']:.4f}")
        print(f"   • Weighted F1-Score: {class_report['weighted avg']['f1-score']:.4f}")
        
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