import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[192, 96, 48], dropout_rate=0.3):
        super(ImprovedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Final output layer
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
    def __init__(self, input_size, config):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList([
            ImprovedNeuralNetwork(
                input_size,
                config['model']['hidden_sizes'],
                config['model']['dropout_rate']
            ) for _ in range(config['model']['ensemble_size'])
        ])

        # Meta-learner combines predictions
        self.meta_learner = nn.Linear(config['model']['ensemble_size'], 1)
        
    def forward(self, x):
        predictions = torch.stack([model(x) for model in self.models], dim=2)
        weights = F.softmax(self.meta_learner.weight, dim=1)
        ensemble_pred = torch.sum(predictions * weights, dim=2)
        return ensemble_pred
        
    def forward(self, x):
        predictions = torch.stack([model(x) for model in self.models], dim=2)
        weights = F.softmax(self.meta_learner.weight, dim=1)
        ensemble_pred = torch.sum(predictions * weights, dim=2)
        return ensemble_pred

    def predict(self, input_data, feature_list, norm_params):
        """
        Predict on single or batched input after preprocessing.
        Assumes you have a preprocess_input function.
        """
        from utils import preprocess_input

        # Ensure input_data is a dictionary
        if not isinstance(input_data, dict):
            raise ValueError(f"input_data must be a dictionary. Got {type(input_data)}")

        self.eval()
        with torch.no_grad():
            processed = preprocess_input(input_data, feature_list, norm_params)
            tensor = torch.FloatTensor(processed).to(next(self.parameters()).device)
            output = self(tensor).cpu().numpy()
            return output.tolist()