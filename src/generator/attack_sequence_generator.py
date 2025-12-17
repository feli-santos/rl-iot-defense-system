"""
Attack Sequence Generator (Red Team Model).

This module implements the LSTM-based attack sequence generator that
predicts the next Kill Chain stage given a history of stages.
Per PRD Section 4, this is a next-token predictor:
- Architecture: Embedding -> Stacked LSTM -> Dense output head
- Training: Cross-entropy loss on next-token prediction
- Inference: Temperature-scaled softmax + categorical sampling

The generator acts as the "Red Team" in the adversarial simulation,
producing realistic multi-stage attack sequences.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttackSequenceGeneratorConfig:
    """Configuration for Attack Sequence Generator.
    
    Attributes:
        num_stages: Number of Kill Chain stages (always 5).
        embedding_dim: Dimension of stage embeddings.
        hidden_size: LSTM hidden state size.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability between LSTM layers.
        temperature: Default temperature for sampling.
    """
    
    num_stages: int = 5
    embedding_dim: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    temperature: float = 1.0


class AttackSequenceGenerator(nn.Module):
    """LSTM-based next-token predictor for Kill Chain stages.
    
    This model learns the "attack grammar" from synthetic episodes
    and generates realistic attack sequences during RL training.
    
    Architecture:
        Input (stage IDs) -> Embedding -> LSTM -> Dense -> Logits
    
    Example:
        >>> model = AttackSequenceGenerator()
        >>> history = [0, 1, 1, 2]  # BENIGN -> RECON -> RECON -> ACCESS
        >>> next_stage = model.sample_next(history)
        >>> next_stage
        3  # MANEUVER (sampled)
    """
    
    def __init__(
        self,
        config: Optional[AttackSequenceGeneratorConfig] = None,
    ) -> None:
        """Initialize the Attack Sequence Generator.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        
        self._config = config or AttackSequenceGeneratorConfig()
        
        # Embedding layer: stage ID -> embedding vector
        self.embedding = nn.Embedding(
            num_embeddings=self._config.num_stages,
            embedding_dim=self._config.embedding_dim,
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self._config.embedding_dim,
            hidden_size=self._config.hidden_size,
            num_layers=self._config.num_layers,
            dropout=self._config.dropout if self._config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output layer: hidden state -> stage logits
        self.fc = nn.Linear(
            in_features=self._config.hidden_size,
            out_features=self._config.num_stages,
        )
        
        # Store default temperature
        self._default_temperature = self._config.temperature
        
        logger.info(
            f"AttackSequenceGenerator initialized: "
            f"embed={self._config.embedding_dim}, "
            f"hidden={self._config.hidden_size}, "
            f"layers={self._config.num_layers}"
        )
    
    @property
    def config(self) -> AttackSequenceGeneratorConfig:
        """Get model configuration."""
        return self._config
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sequence of stages -> next stage logits.
        
        Args:
            x: Input tensor of shape (batch, seq_len) containing stage IDs.
        
        Returns:
            Logits tensor of shape (batch, num_stages).
        """
        # Embed stage IDs: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # LSTM forward: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use last hidden state: (batch, hidden)
        last_hidden = lstm_out[:, -1, :]
        
        # Output logits: (batch, hidden) -> (batch, num_stages)
        logits = self.fc(last_hidden)
        
        return logits
    
    # =========================================================================
    # Inference Methods
    # =========================================================================
    
    def predict_next(
        self,
        history: List[int],
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Predict probability distribution over next stage.
        
        Args:
            history: List of previous stage IDs.
            temperature: Temperature for softmax scaling.
        
        Returns:
            Probability distribution over 5 stages.
        """
        temp = temperature if temperature is not None else self._default_temperature
        
        # Prepare input
        x = torch.tensor([history], dtype=torch.long)
        
        # Forward pass
        with torch.no_grad():
            logits = self(x)
        
        # Temperature-scaled softmax
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)
        
        return probs.squeeze(0).numpy()
    
    def sample_next(
        self,
        history: List[int],
        temperature: Optional[float] = None,
    ) -> int:
        """Sample next stage from predicted distribution.
        
        Uses temperature-scaled softmax followed by categorical sampling
        to produce diverse attack patterns (PRD Section 4.2).
        
        Args:
            history: List of previous stage IDs.
            temperature: Temperature for sampling (higher = more random).
        
        Returns:
            Sampled stage ID (0-4).
        """
        probs = self.predict_next(history, temperature)
        
        # Categorical sampling
        next_stage = np.random.choice(self._config.num_stages, p=probs)
        
        return int(next_stage)
    
    def generate_sequence(
        self,
        start_history: List[int],
        length: int,
        temperature: Optional[float] = None,
    ) -> List[int]:
        """Generate an attack sequence starting from given history.
        
        Args:
            start_history: Initial sequence of stages.
            length: Total length of sequence to generate.
            temperature: Temperature for sampling.
        
        Returns:
            Complete sequence of `length` stages.
        """
        sequence = list(start_history)
        
        while len(sequence) < length:
            next_stage = self.sample_next(sequence, temperature)
            sequence.append(next_stage)
        
        return sequence[:length]
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def save(
        self,
        path: Union[str, Path],
        save_config: bool = True,
    ) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model weights.
            save_config: Whether to save config alongside model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': asdict(self._config),
        }, path)
        
        # Optionally save config as JSON
        if save_config:
            config_path = path.parent / "config.json"
            with open(config_path, "w") as f:
                json.dump(asdict(self._config), f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "AttackSequenceGenerator":
        """Load model from disk.
        
        Args:
            path: Path to saved model.
            device: Device to load model onto.
        
        Returns:
            Loaded AttackSequenceGenerator instance.
        """
        path = Path(path)
        
        if device is None:
            device = torch.device("cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # Reconstruct config
        config = AttackSequenceGeneratorConfig(**checkpoint['config'])
        
        # Create model and load weights
        model = cls(config=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model
