"""
Generator Trainer for Attack Sequence Generator.

This module provides training functionality for the Attack Sequence
Generator (LSTM next-token predictor). It handles:
- Episode-to-training-data conversion
- Model training with cross-entropy loss
- Validation and early stopping
- Model saving to artifacts/generator/

The trained model is used by the Adversarial Environment to generate
attack sequences during RL training.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.generator.attack_sequence_generator import (
    AttackSequenceGenerator,
    AttackSequenceGeneratorConfig,
)
from src.generator.episode_generator import EpisodeGenerator

logger = logging.getLogger(__name__)


@dataclass
class GeneratorTrainingConfig:
    """Configuration for Generator training.
    
    Attributes:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam optimizer learning rate.
        sequence_length: Input sequence length for LSTM.
        val_split: Fraction of data for validation.
        early_stopping_patience: Epochs without improvement before stopping.
        output_dir: Directory to save trained model.
        device: Device to train on ('cpu', 'cuda', 'mps').
    """
    
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    sequence_length: int = 5
    val_split: float = 0.2
    early_stopping_patience: int = 10
    output_dir: Optional[Path] = None
    device: str = "cpu"
    use_mlflow: bool = True
    mlflow_experiment_name: str = "lstm_attack_generator"
    
    # Imbalance mitigation
    use_class_weights: bool = False
    use_weighted_sampler: bool = False
    class_weight_smoothing: float = 1.0
    seed: Optional[int] = None
    
    # Training stability
    grad_clip_norm: Optional[float] = None
    use_lr_scheduler: bool = False
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


class GeneratorTrainer:
    """Trainer for the Attack Sequence Generator.
    
    This class handles the complete training pipeline:
    1. Convert episodes to training sequences
    2. Train LSTM with cross-entropy loss
    3. Validate and track metrics
    4. Save best model to disk
    
    Example:
        >>> trainer = GeneratorTrainer()
        >>> episodes = EpisodeGenerator().generate_all()
        >>> results = trainer.train(episodes)
        >>> trainer.save()
    """
    
    def __init__(
        self,
        config: Optional[GeneratorTrainingConfig] = None,
        model_config: Optional[AttackSequenceGeneratorConfig] = None,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            config: Training configuration.
            model_config: Model architecture configuration.
        """
        self._config = config or GeneratorTrainingConfig()
        self._model_config = model_config or AttackSequenceGeneratorConfig()
        
        # Setup output directory
        if self._config.output_dir is None:
            self._config.output_dir = Path("artifacts/generator")
        self._config.output_dir = Path(self._config.output_dir)
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self._device = torch.device(self._config.device)
        
        # Create model
        self._model = AttackSequenceGenerator(config=self._model_config)
        self._model.to(self._device)
        
        # Training state
        self._optimizer: Optional[Adam] = None
        self._criterion = nn.CrossEntropyLoss()
        self._best_val_loss = float('inf')
        self._last_confusion_matrix: Optional[np.ndarray] = None
        
        logger.info(
            f"GeneratorTrainer initialized: "
            f"epochs={self._config.epochs}, "
            f"batch_size={self._config.batch_size}, "
            f"device={self._device}"
        )
    
    @property
    def config(self) -> GeneratorTrainingConfig:
        """Get training configuration."""
        return self._config
    
    @property
    def model(self) -> AttackSequenceGenerator:
        """Get the model."""
        return self._model
    
    @property
    def last_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix from last evaluate() call."""
        return self._last_confusion_matrix
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    def prepare_data(
        self,
        episodes: List[List[int]],
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders.
        
        Args:
            episodes: List of episode sequences.
        
        Returns:
            Tuple of (train_loader, val_loader).
        """
        # Convert episodes to training sequences
        ep_gen = EpisodeGenerator()  # Use just for conversion
        X, y = ep_gen.to_numpy(episodes, self._config.sequence_length)
        
        logger.info(f"Prepared {len(X)} training sequences")
        
        # Split into train/val
        n_val = int(len(X) * self._config.val_split)
        n_train = len(X) - n_val
        
        # Shuffle with seed if provided
        if self._config.seed is not None:
            rng = np.random.RandomState(self._config.seed)
            indices = rng.permutation(len(X))
        else:
            indices = np.random.permutation(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        # Create tensors
        X_train = torch.tensor(X[train_idx], dtype=torch.long)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.long)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)
        
        # Compute class weights and/or sampler weights for imbalance mitigation
        sampler: Optional[WeightedRandomSampler] = None
        if self._config.use_class_weights or self._config.use_weighted_sampler:
            num_stages = 5
            
            # Compute inverse-frequency weights from training targets only
            class_counts = torch.bincount(y_train, minlength=num_stages).float()
            total = float(class_counts.sum().item())
            
            if total <= 0:
                raise ValueError("No training samples available to compute class weights.")
            
            eps = 1e-12
            inv_freq = total / (class_counts + eps)
            inv_freq = inv_freq.pow(float(self._config.class_weight_smoothing))
            
            # Normalize weights to mean 1.0 for stability
            inv_freq = inv_freq / inv_freq.mean()
            
            if self._config.use_class_weights:
                self._criterion = nn.CrossEntropyLoss(weight=inv_freq.to(self._device))
                logger.info(f"Using class weights: {inv_freq.tolist()}")
            
            if self._config.use_weighted_sampler:
                per_sample_weights = inv_freq[y_train].double()
                sampler = WeightedRandomSampler(
                    weights=per_sample_weights,
                    num_samples=int(len(per_sample_weights)),
                    replacement=True,
                )
                logger.info("Using WeightedRandomSampler for balanced batch sampling")
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
        )
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_loader, val_loader
    
    # =========================================================================
    # Training
    # =========================================================================
    
    def train(
        self,
        episodes: List[List[int]],
    ) -> Dict[str, Any]:
        """Train the generator model with MLflow tracking.
        
        Args:
            episodes: List of episode sequences.
        
        Returns:
            Training results dictionary with loss history.
        """
        logger.info("Starting generator training...")
        
        # Start MLflow run if enabled
        mlflow_run_active = False
        if self._config.use_mlflow:
            try:
                mlflow.set_experiment(self._config.mlflow_experiment_name)
                mlflow.start_run()
                mlflow_run_active = True
                logger.info(f"MLflow run started: {mlflow.active_run().info.run_id}")
                
                # Log training parameters
                self._log_training_params(episodes)
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}. Continuing without MLflow.")
                self._config.use_mlflow = False
        
        try:
            # Prepare data
            train_loader, val_loader = self.prepare_data(episodes)
            
            # Log dataset statistics
            if self._config.use_mlflow:
                self._log_dataset_stats(train_loader, val_loader, episodes)
            
            # Setup optimizer
            self._optimizer = Adam(
                self._model.parameters(),
                lr=self._config.learning_rate,
            )
            
            # Setup learning rate scheduler if enabled
            scheduler = None
            if self._config.use_lr_scheduler:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                scheduler = ReduceLROnPlateau(
                    self._optimizer,
                    mode='min',
                    factor=self._config.scheduler_factor,
                    patience=self._config.scheduler_patience,
                )
                logger.info("Using ReduceLROnPlateau scheduler")
            
            # Training history
            train_losses: List[float] = []
            val_losses: List[float] = []
            
            # Early stopping
            patience_counter = 0
            best_epoch = 0
            
            for epoch in range(self._config.epochs):
                # Train epoch
                train_loss = self._train_epoch(train_loader)
                train_losses.append(train_loss)
                
                # Validate
                val_loss = self._validate(val_loader)
                val_losses.append(val_loss)
                
                logger.info(
                    f"Epoch {epoch + 1}/{self._config.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )
                
                # Log metrics to MLflow
                if self._config.use_mlflow:
                    self._log_epoch_metrics(epoch, train_loss, val_loss, patience_counter)
                
                # Step scheduler if enabled
                if scheduler is not None:
                    scheduler.step(val_loss)
                
                # Check for improvement
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                    self._save_checkpoint()
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self._config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    if self._config.use_mlflow:
                        mlflow.log_metric("training/early_stopped", 1.0)
                        mlflow.log_metric("training/early_stop_epoch", epoch + 1)
                    break
            
            # Load best model
            self._load_best_checkpoint()
            
            # Save final model and config
            self._save_final()
            
            results = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": self._best_val_loss,
                "epochs_trained": len(train_losses),
                "best_epoch": best_epoch,
            }
            
            # Log final metrics and artifacts to MLflow
            if self._config.use_mlflow:
                self._log_final_results(results, train_losses, val_losses)
            
            logger.info(f"Training complete. Best val_loss: {self._best_val_loss:.4f}")
            return results
            
        finally:
            # Always end MLflow run
            if mlflow_run_active:
                try:
                    mlflow.end_run()
                    logger.info("MLflow run ended")
                except Exception as e:
                    logger.warning(f"Failed to end MLflow run: {e}")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self._model.train()
        total_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            
            # Forward pass
            self._optimizer.zero_grad()
            logits = self._model(X_batch)
            loss = self._criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if enabled
            if self._config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.grad_clip_norm,
                )
            
            self._optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                
                logits = self._model(X_batch)
                loss = self._criterion(logits, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    def evaluate(
        self,
        episodes: List[List[int]],
    ) -> Dict[str, float]:
        """Evaluate the model on episodes with comprehensive metrics.
        
        Computes:
        - Loss and accuracy
        - Per-class precision, recall, F1
        - Macro-averaged F1
        - Transition accuracy
        - Perplexity
        - Confusion matrix
        
        Args:
            episodes: List of episode sequences.
        
        Returns:
            Dictionary with evaluation metrics.
        """
        # Prepare data (no split needed)
        ep_gen = EpisodeGenerator()
        X, y = ep_gen.to_numpy(episodes, self._config.sequence_length)
        
        X_tensor = torch.tensor(X, dtype=torch.long).to(self._device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self._device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self._config.batch_size)
        
        self._model.eval()
        
        # Accumulators
        total_loss = 0.0
        total_log_probs = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self._model(X_batch)
                loss = self._criterion(logits, y_batch)
                
                total_loss += loss.item() * len(X_batch)
                
                # For perplexity
                log_probs = F.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(1, y_batch.unsqueeze(1)).squeeze(1)
                total_log_probs += target_log_probs.sum().item()
                
                # Store predictions and targets
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        total = len(all_targets)
        
        # Basic metrics
        avg_loss = total_loss / total
        accuracy = (all_preds == all_targets).sum() / total
        perplexity = np.exp(-total_log_probs / total)
        
        # Per-class metrics
        num_stages = 5
        per_class_metrics = {}
        
        for stage in range(num_stages):
            # True positives, false positives, false negatives
            tp = ((all_preds == stage) & (all_targets == stage)).sum()
            fp = ((all_preds == stage) & (all_targets != stage)).sum()
            fn = ((all_preds != stage) & (all_targets == stage)).sum()
            
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            per_class_metrics[f"precision_stage_{stage}"] = float(precision)
            per_class_metrics[f"recall_stage_{stage}"] = float(recall)
            per_class_metrics[f"f1_stage_{stage}"] = float(f1)
        
        # Macro F1
        macro_f1 = np.mean([per_class_metrics[f"f1_stage_{i}"] for i in range(num_stages)])
        
        # Transition accuracy (from episodes)
        transition_correct = 0
        transition_total = 0
        
        for episode in episodes:
            if len(episode) <= self._config.sequence_length:
                continue
            
            for i in range(len(episode) - self._config.sequence_length):
                history = episode[i:i + self._config.sequence_length]
                true_next = episode[i + self._config.sequence_length]
                
                # Predict
                x_input = torch.tensor([history], dtype=torch.long).to(self._device)
                with torch.no_grad():
                    logits = self._model(x_input)
                pred_next = logits.argmax(dim=-1).item()
                
                if pred_next == true_next:
                    transition_correct += 1
                transition_total += 1
        
        transition_accuracy = transition_correct / transition_total if transition_total > 0 else 0.0
        
        # Confusion matrix (as flat list for JSON serialization)
        confusion_matrix = np.zeros((num_stages, num_stages), dtype=np.int32)
        for pred, target in zip(all_preds, all_targets):
            confusion_matrix[target, pred] += 1
        
        # Compile results
        metrics = {
            "loss": avg_loss,
            "accuracy": float(accuracy),
            "perplexity": float(perplexity),
            "macro_f1": float(macro_f1),
            "transition_accuracy": float(transition_accuracy),
            **per_class_metrics,
        }
        
        # Store confusion matrix separately (not for primary metrics dict)
        self._last_confusion_matrix = confusion_matrix
        
        return metrics
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint during training."""
        checkpoint_path = self._config.output_dir / "checkpoint.pth"
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'best_val_loss': self._best_val_loss,
        }, checkpoint_path)
    
    def _load_best_checkpoint(self) -> None:
        """Load best checkpoint."""
        checkpoint_path = self._config.output_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self._device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_final(self) -> None:
        """Save final model and configuration."""
        # Save model
        model_path = self._config.output_dir / "attack_sequence_generator.pth"
        self._model.save(model_path, save_config=True)
        
        # Save training config
        train_config_path = self._config.output_dir / "training_config.json"
        with open(train_config_path, "w") as f:
            config_dict = asdict(self._config)
            config_dict["output_dir"] = str(self._config.output_dir)
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(
        cls,
        model_dir: Path,
        device: str = "cpu",
    ) -> "GeneratorTrainer":
        """Load a trained trainer from disk.
        
        Args:
            model_dir: Directory containing saved model.
            device: Device to load model onto.
        
        Returns:
            Loaded GeneratorTrainer instance.
        """
        model_dir = Path(model_dir)
        
        # Load training config
        train_config_path = model_dir / "training_config.json"
        if train_config_path.exists():
            with open(train_config_path, "r") as f:
                config_dict = json.load(f)
            config_dict["output_dir"] = model_dir
            config_dict["device"] = device
            training_config = GeneratorTrainingConfig(**config_dict)
        else:
            training_config = GeneratorTrainingConfig(output_dir=model_dir, device=device)
        
        # Load model config
        model_config_path = model_dir / "config.json"
        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                model_config_dict = json.load(f)
            model_config = AttackSequenceGeneratorConfig(**model_config_dict)
        else:
            model_config = None
        
        # Create trainer
        trainer = cls(config=training_config, model_config=model_config)
        
        # Load model weights
        model_path = model_dir / "attack_sequence_generator.pth"
        trainer._model = AttackSequenceGenerator.load(model_path, device=torch.device(device))
        
        logger.info(f"Trainer loaded from {model_dir}")
        return trainer
    
    # =========================================================================
    # MLflow Logging Helpers
    # =========================================================================
    
    def _log_training_params(self, episodes: List[List[int]]) -> None:
        """Log training parameters to MLflow."""
        try:
            # Training config
            mlflow.log_params({
                "epochs": self._config.epochs,
                "batch_size": self._config.batch_size,
                "learning_rate": self._config.learning_rate,
                "sequence_length": self._config.sequence_length,
                "val_split": self._config.val_split,
                "early_stopping_patience": self._config.early_stopping_patience,
                "device": self._config.device,
            })
            
            # Model config
            mlflow.log_params({
                "num_stages": self._model_config.num_stages,
                "embedding_dim": self._model_config.embedding_dim,
                "hidden_size": self._model_config.hidden_size,
                "num_layers": self._model_config.num_layers,
                "dropout": self._model_config.dropout,
                "temperature": self._model_config.temperature,
            })
            
            # Episode statistics
            episode_lengths = [len(ep) for ep in episodes]
            mlflow.log_params({
                "num_episodes": len(episodes),
                "min_episode_length": min(episode_lengths),
                "max_episode_length": max(episode_lengths),
                "mean_episode_length": np.mean(episode_lengths),
            })
            
            logger.info("Logged training parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log training params: {e}")
    
    def _log_dataset_stats(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        episodes: List[List[int]]
    ) -> None:
        """Log dataset statistics to MLflow."""
        try:
            # Dataset sizes
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            total_size = train_size + val_size
            
            mlflow.log_metrics({
                "data/train_sequences": train_size,
                "data/val_sequences": val_size,
                "data/total_sequences": total_size,
                "data/train_ratio": train_size / total_size,
            })
            
            # Stage distribution in episodes
            all_stages = [stage for ep in episodes for stage in ep]
            stage_counts = np.bincount(all_stages, minlength=5)
            stage_proportions = stage_counts / len(all_stages)
            
            for stage_id in range(5):
                mlflow.log_metrics({
                    f"data/stage_{stage_id}_count": int(stage_counts[stage_id]),
                    f"data/stage_{stage_id}_proportion": float(stage_proportions[stage_id]),
                })
            
            logger.info("Logged dataset statistics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log dataset stats: {e}")
    
    def _log_epoch_metrics(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float,
        patience_counter: int
    ) -> None:
        """Log per-epoch metrics to MLflow."""
        try:
            mlflow.log_metrics({
                "loss/train": train_loss,
                "loss/val": val_loss,
                "loss/train_val_gap": abs(train_loss - val_loss),
                "training/epoch": epoch + 1,
                "training/patience_counter": patience_counter,
            }, step=epoch)
            
            # Log improvement indicator
            if patience_counter == 0:
                mlflow.log_metric("training/improved", 1.0, step=epoch)
            else:
                mlflow.log_metric("training/improved", 0.0, step=epoch)
                
        except Exception as e:
            logger.warning(f"Failed to log epoch metrics: {e}")
    
    def _log_final_results(
        self, 
        results: Dict[str, Any],
        train_losses: List[float],
        val_losses: List[float]
    ) -> None:
        """Log final results and artifacts to MLflow."""
        try:
            # Final metrics
            mlflow.log_metrics({
                "final/best_val_loss": results["best_val_loss"],
                "final/epochs_trained": results["epochs_trained"],
                "final/best_epoch": results["best_epoch"],
                "final/final_train_loss": train_losses[-1],
                "final/final_val_loss": val_losses[-1],
                "final/min_train_loss": min(train_losses),
                "final/min_val_loss": min(val_losses),
            })
            
            # Model performance indicators
            train_val_gap = abs(train_losses[-1] - val_losses[-1])
            convergence_rate = (train_losses[0] - train_losses[-1]) / max(train_losses[0], 1e-8)
            
            mlflow.log_metrics({
                "quality/train_val_gap": train_val_gap,
                "quality/convergence_rate": convergence_rate,
                "quality/early_stop_triggered": 1.0 if results["epochs_trained"] < self._config.epochs else 0.0,
            })
            
            # Generate and log loss curves
            self._log_loss_curves(train_losses, val_losses)
            
            # Log model artifacts
            self._log_model_artifacts()
            
            # Log config files
            config_files = [
                self._config.output_dir / "config.json",
                self._config.output_dir / "training_config.json",
            ]
            for config_file in config_files:
                if config_file.exists():
                    mlflow.log_artifact(str(config_file))
            
            logger.info("Logged final results to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log final results: {e}")
    
    def _log_loss_curves(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Generate and log loss curve plots to MLflow."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss curves
            epochs = range(1, len(train_losses) + 1)
            axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
            axes[0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Train/Val gap
            gap = [abs(t - v) for t, v in zip(train_losses, val_losses)]
            axes[1].plot(epochs, gap, label='Train-Val Gap', marker='o', markersize=3, color='orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Absolute Gap')
            axes[1].set_title('Train-Val Loss Gap')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save and log
            plot_path = self._config.output_dir / "loss_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(plot_path))
            logger.info(f"Logged loss curves to MLflow: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log loss curves: {e}")
            plt.close('all')
    
    def _log_model_artifacts(self) -> None:
        """Log trained model to MLflow."""
        try:
            # Log PyTorch model
            model_path = self._config.output_dir / "attack_sequence_generator.pth"
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
                
                # Also log using MLflow's PyTorch integration
                mlflow.pytorch.log_model(
                    pytorch_model=self._model,
                    artifact_path="lstm_model",
                    registered_model_name=None,  # Optional: register model
                )
            
            logger.info("Logged model artifacts to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log model artifacts: {e}")
