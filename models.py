import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedNeuralNetwork(nn.Module):
    """Fixed neural network that matches the actual trained models"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 48], dropout_rate=0.3):
        super(ImprovedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Final output layer (no activation, batchnorm, or dropout)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class EnsembleModel(nn.Module):
    """Fixed ensemble model - removed duplicate forward method"""
    
    def __init__(self, input_size, config):
        super(EnsembleModel, self).__init__()
        
        # Create ensemble of neural networks
        self.models = nn.ModuleList([
            ImprovedNeuralNetwork(
                input_size,
                config['model']['hidden_sizes'],
                config['model']['dropout_rate']
            ) for _ in range(config['model']['ensemble_size'])
        ])

        # Meta-learner to combine predictions
        self.meta_learner = nn.Linear(config['model']['ensemble_size'], 1)
        
    def forward(self, x):
        # Get predictions from all models
        predictions = torch.stack([model(x) for model in self.models], dim=2)
        
        # Apply meta-learner weights
        weights = F.softmax(self.meta_learner.weight, dim=1)
        ensemble_pred = torch.sum(predictions * weights, dim=2)
        
        return ensemble_pred

    def predict(self, input_data, feature_list, norm_params):
        """
        Fixed prediction method with proper categorical encoding
        """
        # Import here to avoid circular imports
        from utils import preprocess_input_with_encoding
        
        self.eval()
        with torch.no_grad():
            # Process input with categorical encoding
            processed = preprocess_input_with_encoding(input_data, feature_list, norm_params)
            
            # Convert to tensor and predict
            tensor = torch.FloatTensor(processed).unsqueeze(0)  # Add batch dimension
            tensor = tensor.to(next(self.parameters()).device)
            
            output = self(tensor)
            return float(output.cpu().numpy().item())  # Return scalar value