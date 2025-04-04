import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.training_data_generator import RealisticAttackDataGenerator  # Updated path

class LSTMAttackPredictor(nn.Module):
    """LSTM-based model to predict optimal attack sequences with gradient clipping"""
    
    def __init__(self, config, seq_length: int = 10):
        super(LSTMAttackPredictor, self).__init__()
        self.config = config
        self.seq_length = seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=227,  # From paper: 227 different event occurrences
            embedding_dim=config.LSTM_EMBEDDING_SIZE
        )
        
        # Bidirectional LSTM layers with constraints for gradient stability
        self.lstm1 = nn.LSTM(
            input_size=config.LSTM_EMBEDDING_SIZE,
            hidden_size=config.LSTM_HIDDEN_UNITS[0],
            bidirectional=True,
            batch_first=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=config.LSTM_HIDDEN_UNITS[0]*2,  # *2 for bidirectional
            hidden_size=config.LSTM_HIDDEN_UNITS[1],
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(
            in_features=config.LSTM_HIDDEN_UNITS[1]*2,  # *2 for bidirectional
            out_features=config.LSTM_OUTPUT_CLASSES
        )
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4  # Add L2 regularization
        )

        # Add dropout layers for regularization
        self.dropout = nn.Dropout(0.5)  # Change from 0.2 to 0.5
        
        # Data generator for realistic training data
        self.data_generator = RealisticAttackDataGenerator(config.ENVIRONMENT_NUM_STATES)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Take the last timestep's output
        return x
    
    def train_model(self, train_data: np.ndarray = None, train_labels: np.ndarray = None,
               val_data: np.ndarray = None, val_labels: np.ndarray = None,
               epochs: int = 100, batch_size: int = 32) -> Tuple[list, list]:
        """Train the LSTM model"""
        # If no data provided, generate realistic training data
        if train_data is None or train_labels is None:
            print("Generating realistic attack training data...")
            X_train, y_train = self.data_generator.generate_batch(
                batch_size=int(batch_size*0.8*epochs), 
                seq_length=self.seq_length
            )
            X_val, y_val = self.data_generator.generate_batch(
                batch_size=int(batch_size*0.2*epochs), 
                seq_length=self.seq_length
            )
            
            # Reshape data to match expected format
            train_data = X_train.reshape(-1, self.seq_length)
            train_labels = np.argmax(y_train[:, -1], axis=1)  # Get class labels for final timestep
            val_data = X_val.reshape(-1, self.seq_length)
            val_labels = np.argmax(y_val[:, -1], axis=1)
        
        # Convert data to PyTorch tensors
        train_dataset = TensorDataset(
            torch.LongTensor(train_data),
            torch.LongTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.LongTensor(val_data),
            torch.LongTensor(val_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print("------------------------")
            
        return (train_losses, train_accs), (val_losses, val_accs)

    def _validate(self, val_loader):
        """Helper method for validation"""
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader), correct / total
    
    def predict_attack_sequence(self, initial_event: int, max_length: int = 10) -> list:
        """Predict an attack sequence starting from an initial event"""
        self.eval()
        current_event = initial_event
        sequence = [current_event]
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                input_tensor = torch.LongTensor([sequence]).to(next(self.parameters()).device)
                output = self(input_tensor)
                next_event = torch.argmax(output, dim=1).item()
                
                if next_event == current_event:  # Stop if we're predicting the same event
                    break
                
                sequence.append(next_event)
                current_event = next_event
        
        return sequence