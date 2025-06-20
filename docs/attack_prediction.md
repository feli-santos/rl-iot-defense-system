# LSTM Attack Prediction Model

## Model Architecture

The attack prediction component uses a Long Short-Term Memory (LSTM) network, specifically the `RealDataLSTMPredictor` implemented in `src/models/lstm_attack_predictor.py`. This model is trained on the CICIoT2023 dataset to predict attack types based on sequences of network flow features. The predictions are then made available to the RL environment via the `EnhancedAttackPredictor` interface (`src/models/predictor_interface.py`).

### LSTM Network Structure (`RealDataLSTMPredictor`)

The `RealDataLSTMPredictor` architecture involves:
1.  **LSTM Layers**: One or more LSTM layers process sequences of input features. The input size corresponds to the number of features from the CICIoT2023 dataset, and hidden sizes are configurable. Bidirectionality can be used.
    -   Input: `(batch_size, sequence_length, num_features)`
2.  **Dropout Layer**: Applied for regularization.
3.  **Fully Connected (Dense) Layers**: A series of dense layers followed by activation functions (e.g., ReLU) process the LSTM output.
4.  **Output Layer**: A final dense layer with a Softmax activation function outputs a probability distribution over the possible attack classes (including benign).

The configuration for this model (input size, hidden size, number of layers, dropout, number of classes, sequence length) is managed by the `LSTMConfig` dataclass and typically sourced from `config.yml` under the `lstm.model` section.

```python
# Simplified from RealDataLSTMPredictor class in src/models/lstm_attack_predictor.py
class RealDataLSTMPredictor(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(RealDataLSTMPredictor, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.input_size,       # Number of features from CICIoT2023
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
            # Bidirectional can be added if config supports it
        )
        
        self.dropout_layer = nn.Dropout(config.dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes) # num_classes from CICIoT2023
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :] # Use output of the last time step
        last_output = self.dropout_layer(last_output)
        output = self.classifier(last_output)
        return output # Logits before Softmax for CrossEntropyLoss
```

## Mathematical Foundation
(The mathematical explanation for LSTM cells and Bidirectional Architecture from the original document remains valid and applies here.)

### LSTM Cell
Each LSTM cell processes sequence elements with the following operations:
1. **Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. **Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$; $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
3. **Cell State Update**: $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$
4. **Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$; $h_t = o_t * \tanh(C_t)$

### Softmax Output Layer
The final layer of the classifier typically uses a Softmax function (implicitly handled by `CrossEntropyLoss` during training, or applied explicitly for inference) to convert logits into probabilities:
$$P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$$
Where $z_j$ is the logit for class $j$, and $K$ is the total number of classes.

## Training Process

### Data Preparation
Training data is sourced from the preprocessed CICIoT2023 dataset. The `CICIoTDataLoader` (`src/utils/dataset_loader.py`) handles loading, batching, and splitting the data into training, validation, and test sets. It prepares sequences of network flow features and corresponding attack labels.

### Training Loop (`RealDataTrainer`)
The `RealDataTrainer` class within `src/models/lstm_attack_predictor.py` manages the training and evaluation of the `RealDataLSTMPredictor`. The overall LSTM training process is invoked by `LSTMTrainer` (`src/training/lstm_trainer.py`).
Key aspects:
1.  **Optimizer**: Typically Adam, configured via `config.yml` (`lstm.training.learning_rate`, `lstm.training.weight_decay`).
2.  **Loss Function**: `nn.CrossEntropyLoss` is used for multi-class classification.
3.  **Epochs & Batch Size**: Defined in `config.yml` (`lstm.training.epochs`, `lstm.training.batch_size`).
4.  **Train-Validation-Test Split**: Handled by `CICIoTDataLoader`.
5.  **Metrics Tracking**: Training/validation loss and accuracy are logged per epoch. MLflow integration is managed by `LSTMTrainer` or `TrainingManager` if part of a larger pipeline.
6.  **Model Saving**: The best model based on validation performance is saved.

```python
# Simplified conceptual training loop from RealDataTrainer.train_epoch
# self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
# self.criterion = nn.CrossEntropyLoss()

for epoch in range(self.config.num_epochs):
    # Training phase
    self.model.train()
    for sequences, labels in train_dataloader: # sequences: (batch, seq_len, features), labels: (batch,)
        optimizer.zero_grad()
        outputs = self.model(sequences) # outputs: (batch, num_classes)
        loss = self.criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # ... track metrics ...
    
    # Validation phase
    self.model.eval()
    with torch.no_grad():
        for sequences, labels in val_dataloader:
            outputs = self.model(sequences)
            # ... track metrics ...
    # ... log epoch metrics, save best model ...
```

## Inference Process (`AttackPredictorInterface`)

The `AttackPredictorInterface` class in `src/predictor/interface.py` serves as the bridge between the trained LSTM model and the RL environment. It handles:
1.  Receiving a sequence of network states (features).
2.  Preprocessing these states (e.g., scaling using artifacts loaded from `preprocessing_artifacts.pkl`).
3.  Feeding the sequence to the `RealDataLSTMPredictor.predict_attack_probability()` method.
4.  Obtaining a probability distribution over attack classes from the model.
5.  Interpreting these probabilities to provide a risk assessment, predicted attack type, confidence, etc., which is used by the `IoTEnv`.

```python
# Conceptual inference from EnhancedAttackPredictor using RealDataLSTMPredictor
def predict_attack_risk(self, network_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
    # ... preprocess network_sequence into a numerical numpy array `processed_sequence` ...
    # processed_sequence shape: (1, sequence_length, num_features)
    
    probabilities = self.model.predict_attack_probability(processed_sequence)[0] # self.model is RealDataLSTMPredictor
    
    predicted_class_idx = np.argmax(probabilities)
    predicted_attack = self.feature_info['class_names'][predicted_class_idx] # From loaded feature_info
    # ... further processing to calculate risk_score, severity, etc. ...
    return {
        'predicted_attack': predicted_attack,
        'confidence': float(probabilities[predicted_class_idx]),
        # ... other details ...
    }
```

## Performance Metrics

The LSTM model's performance is evaluated using:
1.  **Accuracy**: Percentage of correctly classified instances on the test set.
2.  **F1-Score (Macro and Weighted)**: To account for class imbalance.
3.  **Precision and Recall** per class.
4.  **Confusion Matrix**: To visualize classification performance across different attack types.
These metrics are typically calculated by `RealDataTrainer.evaluate_detailed()` and logged.
Example performance (from `main.py` output after LSTM training):
- Test Accuracy: (e.g., >98%)
- Macro F1-Score: (e.g., >65%, highly dependent on minority classes)
- Weighted F1-Score: (e.g., >98%)