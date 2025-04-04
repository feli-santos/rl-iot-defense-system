import os
import uuid
import json
import time
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

class TrainingManager:
    """Manages training experiments with versioning, logging, and artifacts storage"""
    
    def __init__(self, 
                 experiment_name: str, 
                 base_artifact_path: str = "./artifacts",
                 config: Optional[Any] = None):
        """
        Initialize the training manager
        
        Args:
            experiment_name: Name of the experiment for MLflow
            base_artifact_path: Base directory for storing artifacts
            config: Configuration object with parameters
        """
        self.experiment_name = experiment_name
        self.config = config
        
        # Create a unique run ID
        self.run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Setup artifact directories
        self.base_artifact_path = Path(base_artifact_path)
        self.run_artifact_path = self.base_artifact_path / self.run_id
        self.models_path = self.run_artifact_path / "models"
        self.logs_path = self.run_artifact_path / "logs"
        self.plots_path = self.run_artifact_path / "plots"
        
        # Create directories
        self._create_directories()
        
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        self.mlflow_run = None
    
    def _create_directories(self):
        """Create necessary directories for artifacts"""
        os.makedirs(self.run_artifact_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start an MLflow run and log config parameters"""
        run_name = run_name or self.run_id
        self.mlflow_run = mlflow.start_run(run_name=run_name, nested=nested)
        
        # Log configuration parameters if available
        if self.config:
            # Log all attributes of config object
            params = {k: v for k, v in vars(self.config).items() 
                     if not k.startswith('_') and not callable(v)}
            mlflow.log_params(params)
        
        # Log system info
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("run_id", self.run_id)
        
        return self.mlflow_run
    
    def end_run(self):
        """End the current MLflow run"""
        if self.mlflow_run:
            mlflow.end_run()
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to MLflow and save to disk"""
        # Log to MLflow
        mlflow.log_metrics(metrics, step=step)
        
        # Also save to disk
        metrics_file = self.logs_path / "metrics.jsonl"
        entry = {
            "timestamp": time.time(),
            "step": step,
            **metrics
        }
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_model(self, model: torch.nn.Module, name: str, input_example=None):
        """Save PyTorch model with proper versioning and signature"""
        # Save model architecture and weights
        model_path = self.models_path / f"{name}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Create a simple input example if none provided
        if input_example is None and hasattr(model, 'seq_length'):
            # Create a dummy input for LSTM model
            input_example = torch.zeros((1, model.seq_length), dtype=torch.long)
        
        # Log to MLflow with signature if possible
        if input_example is not None:
            from mlflow.models.signature import infer_signature
            # Get model output for this input
            model.eval()
            with torch.no_grad():
                output = model(input_example)
            # Create signature
            signature = infer_signature(input_example.numpy(), output.numpy())
            mlflow.pytorch.log_model(model, f"{name}", signature=signature, input_example=input_example)
        else:
            mlflow.pytorch.log_model(model, f"{name}")
        
        return str(model_path)
    
    def log_figure(self, figure: plt.Figure, name: str):
        """Save figure to disk and log to MLflow"""
        # Save locally
        figure_path = self.plots_path / f"{name}.png"
        figure.savefig(figure_path)
        
        # Log to MLflow
        mlflow.log_artifact(figure_path)
        
        return str(figure_path)
    
    def plot_training_curves(self, 
                             train_metrics: Dict[str, list], 
                             val_metrics: Dict[str, list], 
                             title: str = "Training Curves") -> plt.Figure:
        """
        Plot training curves with metrics
        
        Args:
            train_metrics: Dictionary mapping metric names to lists of training values
            val_metrics: Dictionary mapping metric names to lists of validation values
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(train_metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
        
        # Handle case with single metric
        if num_metrics == 1:
            axes = [axes]
        
        for ax, (metric_name, train_values) in zip(axes, train_metrics.items()):
            val_values = val_metrics.get(metric_name, [])
            
            ax.plot(train_values, label=f"Train {metric_name}")
            if val_values:
                ax.plot(val_values, label=f"Val {metric_name}")
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f"{metric_name.capitalize()} vs. Epoch")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.suptitle(title)
        fig.tight_layout()
        
        # Save and log the figure
        self.log_figure(fig, title.lower().replace(" ", "_"))
        
        return fig
    
    def save_best_model(self, 
                        model: torch.nn.Module, 
                        metric_value: float, 
                        metric_name: str = "val_loss", 
                        mode: str = "min") -> bool:
        """
        Save model if it's the best according to the metric
        
        Args:
            model: PyTorch model to save
            metric_value: Current metric value
            metric_name: Name of the metric
            mode: 'min' or 'max' depending on if lower or higher is better
        
        Returns:
            True if model was saved as best, False otherwise
        """
        # File to track best metric
        best_metric_file = self.models_path / f"best_{metric_name}.json"
        
        is_best = False
        best_so_far = float('inf') if mode == "min" else -float('inf')
        
        # Check if we have a previous best
        if os.path.exists(best_metric_file):
            with open(best_metric_file, 'r') as f:
                data = json.load(f)
                best_so_far = data["value"]
        
        # Check if current model is better
        if (mode == "min" and metric_value < best_so_far) or \
           (mode == "max" and metric_value > best_so_far):
            is_best = True
            best_so_far = metric_value
            
            # Save the best model
            model_path = self.models_path / f"best_{metric_name}_model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Update best metric file
            with open(best_metric_file, 'w') as f:
                json.dump({
                    "value": best_so_far,
                    "timestamp": time.time(),
                    "model_path": str(model_path)
                }, f)
            
            # Log to MLflow
            mlflow.log_metric(f"best_{metric_name}", best_so_far)
            mlflow.pytorch.log_model(model, f"best_{metric_name}_model")
        
        return is_best