import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import random
from tqdm import tqdm

class LSTMAttackPredictor(nn.Module):
    """LSTM model for predicting next attack targets"""
    
    def __init__(self, config):
        super(LSTMAttackPredictor, self).__init__()
        self.config = config
        self.seq_length = config.ENVIRONMENT_HISTORY_LENGTH
        self.num_devices = config.ENVIRONMENT_NUM_DEVICES
        self.embedding_size = config.LSTM_EMBEDDING_SIZE
        self.hidden_units = config.LSTM_HIDDEN_UNITS
        self.output_classes = config.LSTM_OUTPUT_CLASSES
        
        # Device for computations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embedding layer for device IDs
        self.embedding = nn.Embedding(self.num_devices + 1, self.embedding_size)  # +1 for padding
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_units[0],
            num_layers=len(self.hidden_units),
            batch_first=True,
            dropout=0.3 if len(self.hidden_units) > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Linear(self.hidden_units[0], self.output_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_size)
        
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_size)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout and final layer
        output = self.dropout(last_output)
        output = self.fc(output)  # (batch_size, output_classes)
        
        return output
    
    def generate_attack_sequences(self, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic attack training data"""
        print("Generating realistic attack training data...")
        
        sequences = []
        labels = []
        
        # Attack patterns based on IoT network topology
        attack_patterns = [
            # Linear progression through network
            list(range(self.num_devices)),
            # Hub-focused attacks (target central devices first)
            [5, 6, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11] if self.num_devices >= 12 else list(range(self.num_devices)),
            # Random but clustered attacks
            [0, 1, 3, 2, 6, 7, 4, 5, 8, 9, 10, 11] if self.num_devices >= 12 else list(range(self.num_devices)),
        ]
        
        for _ in tqdm(range(num_samples), desc="Generating sequences"):
            # Choose random attack pattern
            pattern = random.choice(attack_patterns)
            
            # Generate sequence of length seq_length + 1 (input + target)
            start_idx = random.randint(0, max(0, len(pattern) - self.seq_length - 1))
            
            if start_idx + self.seq_length + 1 <= len(pattern):
                sequence = pattern[start_idx:start_idx + self.seq_length]
                next_target = pattern[start_idx + self.seq_length]
                
                # Ensure sequence and target are within valid range
                sequence = [min(device, self.num_devices - 1) for device in sequence]
                next_target = min(next_target, self.num_devices - 1)
                
                # Pad sequence if needed
                while len(sequence) < self.seq_length:
                    sequence.append(0)  # Pad with 0
                
                sequences.append(sequence[:self.seq_length])  # Ensure exact length
                labels.append(next_target)
            else:
                # Fallback: generate random sequence
                sequence = [random.randint(0, self.num_devices - 1) for _ in range(self.seq_length)]
                next_target = random.randint(0, self.num_devices - 1)
                
                sequences.append(sequence)
                labels.append(next_target)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.int64)
        labels = np.array(labels, dtype=np.int64)
        
        print(f"Generated data shapes:")
        print(f"Sequences: {sequences.shape}")
        print(f"Labels: {labels.shape}")
        
        # Validate shapes
        assert sequences.shape[0] == labels.shape[0], f"Sequence count mismatch: {sequences.shape[0]} vs {labels.shape[0]}"
        assert sequences.shape[1] == self.seq_length, f"Sequence length mismatch: {sequences.shape[1]} vs {self.seq_length}"
        assert len(labels.shape) == 1, f"Labels should be 1D, got shape: {labels.shape}"
        
        return sequences, labels
    
    def train_model(self, epochs: int = 100, batch_size: int = 32) -> Tuple[Dict[str, List], Dict[str, List]]:
        """Train the LSTM model"""
        print("Training LSTM model...")
        
        # Generate training data
        sequences, labels = self.generate_attack_sequences(num_samples=5000)
        
        # Split into train/validation
        split_idx = int(0.8 * len(sequences))
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"Train data shape: {train_sequences.shape}, Train labels shape: {train_labels.shape}")
        print(f"Val data shape: {val_sequences.shape}, Val labels shape: {val_labels.shape}")
        
        # Create datasets
        try:
            # Convert to tensors with proper types
            train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.long)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            val_sequences_tensor = torch.tensor(val_sequences, dtype=torch.long)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
            
            print(f"Tensor shapes before TensorDataset:")
            print(f"Train sequences tensor: {train_sequences_tensor.shape}")
            print(f"Train labels tensor: {train_labels_tensor.shape}")
            print(f"Val sequences tensor: {val_sequences_tensor.shape}")
            print(f"Val labels tensor: {val_labels_tensor.shape}")
            
            # Verify tensor compatibility
            assert train_sequences_tensor.shape[0] == train_labels_tensor.shape[0], \
                f"Train tensor size mismatch: {train_sequences_tensor.shape[0]} vs {train_labels_tensor.shape[0]}"
            assert val_sequences_tensor.shape[0] == val_labels_tensor.shape[0], \
                f"Val tensor size mismatch: {val_sequences_tensor.shape[0]} vs {val_labels_tensor.shape[0]}"
            
            train_dataset = TensorDataset(train_sequences_tensor, train_labels_tensor)
            val_dataset = TensorDataset(val_sequences_tensor, val_labels_tensor)
            
        except Exception as e:
            print(f"Error creating TensorDataset: {e}")
            print(f"Debug info:")
            print(f"train_sequences type: {type(train_sequences)}, shape: {train_sequences.shape}")
            print(f"train_labels type: {type(train_labels)}, shape: {train_labels.shape}")
            print(f"val_sequences type: {type(val_sequences)}, shape: {val_sequences.shape}")
            print(f"val_labels type: {type(val_labels)}, shape: {val_labels.shape}")
            raise
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Training metrics
        train_metrics = {'loss': [], 'accuracy': []}
        val_metrics = {'loss': [], 'accuracy': []}
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(batch_sequences)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.LSTM_GRADIENT_CLIP_NORM)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_sequences, batch_labels in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self(batch_sequences)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            
            # Store metrics
            train_metrics['loss'].append(epoch_train_loss)
            train_metrics['accuracy'].append(epoch_train_acc)
            val_metrics['loss'].append(epoch_val_loss)
            val_metrics['accuracy'].append(epoch_val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
                print(f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
                print("------------------------")
        
        return train_metrics, val_metrics
    
    def predict_next_target(self, attack_sequence: List[int]) -> int:
        """Predict the next attack target given a sequence"""
        self.eval()
        
        # Pad or truncate sequence to seq_length
        if len(attack_sequence) < self.seq_length:
            # Pad with zeros
            padded_sequence = [0] * (self.seq_length - len(attack_sequence)) + attack_sequence
        else:
            # Take last seq_length elements
            padded_sequence = attack_sequence[-self.seq_length:]
        
        # Convert to tensor
        sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            output = self(sequence_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Map class back to device ID (assuming class == device_id for simplicity)
        return min(predicted_class, self.num_devices - 1)