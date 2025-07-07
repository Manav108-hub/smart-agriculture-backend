import torch
import torch.nn as nn
from utils import preprocess_input
from config import (
    MODEL_PATHS, 
    CROP_FEATURES, 
    YIELD_FEATURES, 
    SUSTAINABILITY_FEATURES,
    CROP_NORM_PARAMS,
    YIELD_NORM_PARAMS,
    SUSTAINABILITY_NORM_PARAMS
)

class SingleModel(nn.Module):
    """Single model with BatchNorm layers to match the saved architecture"""
    def __init__(self, input_size):
        super().__init__()
        # Updated architecture to match your saved model
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 48),  # Changed from 64 to 48
            nn.BatchNorm1d(48),   # Changed from 64 to 48
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, 32),    # Changed from 64 to 48
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class EnsembleModel(nn.Module):
    """Ensemble model with meta-learner to match the saved architecture"""
    def __init__(self, input_size, num_models=3):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create multiple models
        self.models = nn.ModuleList([
            SingleModel(input_size) for _ in range(num_models)
        ])
        
        # Meta-learner to combine predictions
        self.meta_learner = nn.Linear(num_models, 1)
        
        self.to(self.device)
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        ensemble_input = torch.cat(predictions, dim=1)
        
        # Use meta-learner to combine
        final_output = self.meta_learner(ensemble_input)
        return final_output
    
    def predict(self, input_data, feature_list, norm_params):
        self.eval()
        with torch.no_grad():
            # Preprocess input
            processed = preprocess_input(input_data, feature_list, norm_params)
            tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)  # Add batch dimension
            output = self(tensor).cpu().numpy()[0][0]  # Remove batch and feature dimensions
            return float(output)

class ModelLoader:
    _models = {}
    
    @classmethod
    def load_model(cls, model_type):
        if model_type in cls._models:
            return cls._models[model_type]
            
        # Create model architecture
        if model_type == "crop":
            model = EnsembleModel(len(CROP_FEATURES))
        elif model_type == "yield":
            model = EnsembleModel(len(YIELD_FEATURES))
        else:  # sustainability
            model = EnsembleModel(len(SUSTAINABILITY_FEATURES))
            
        # Load weights
        try:
            checkpoint = torch.load(MODEL_PATHS[model_type], map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                else:
                    # Assume the entire dict is the state_dict
                    model.load_state_dict(checkpoint, strict=True)
            else:
                # Direct state_dict
                model.load_state_dict(checkpoint, strict=True)
                
            print(f"Successfully loaded {model_type} model")
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            
            # Try to load with strict=False and show what's missing
            try:
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        missing_keys, unexpected_keys = model.load_state_dict(
                            checkpoint['model_state_dict'], strict=False
                        )
                    elif 'state_dict' in checkpoint:
                        missing_keys, unexpected_keys = model.load_state_dict(
                            checkpoint['state_dict'], strict=False
                        )
                    else:
                        missing_keys, unexpected_keys = model.load_state_dict(
                            checkpoint, strict=False
                        )
                else:
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint, strict=False
                    )
                
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                
                if len(missing_keys) > 0:
                    print("Model architecture might not match saved model.")
                    return None
                    
            except Exception as e2:
                print(f"Failed to load model even with strict=False: {e2}")
                return None
            
        model.to(model.device)
        model.eval()
        
        cls._models[model_type] = model
        return model
    
    @classmethod
    def get_model(cls, model_type):
        """Get a loaded model"""
        if model_type not in cls._models:
            cls.load_model(model_type)
        return cls._models[model_type]
    
    @classmethod
    def get_features_and_params(cls, model_type):
        """Get feature list and normalization parameters for a model type"""
        if model_type == "crop":
            return CROP_FEATURES, CROP_NORM_PARAMS
        elif model_type == "yield":
            return YIELD_FEATURES, YIELD_NORM_PARAMS
        else:  # sustainability
            return SUSTAINABILITY_FEATURES, SUSTAINABILITY_NORM_PARAMS

# Quick fix: Update your config to match saved models
CORRECTED_CONFIG = {
    'model': {
        'ensemble_size': 3,
        'hidden_sizes': [128, 48, 32],  # Updated to match saved model
        'dropout_rate': 0.3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
    }
}

# Alternative approach: Dynamically create model architecture based on checkpoint
class DynamicModelLoader:
    """
    If you're still having issues, this class can help create the model 
    architecture based on the saved checkpoint structure
    """
    
    @staticmethod
    def inspect_checkpoint(model_path):
        """Inspect the checkpoint to understand the architecture"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        print(f"Checkpoint structure for {model_path}:")
        for key, value in state_dict.items():
            print(f"  {key}: {value.shape}")
            
        return state_dict
    
    @staticmethod
    def create_model_from_checkpoint(model_path, input_size):
        """Create model architecture based on checkpoint"""
        state_dict = DynamicModelLoader.inspect_checkpoint(model_path)
        
        # Extract layer sizes from the checkpoint
        layer_sizes = []
        
        # Look for linear layer weights to determine architecture
        for key in state_dict.keys():
            if 'models.0.network.' in key and '.weight' in key and 'Linear' in str(type(state_dict[key])):
                layer_info = key.split('.')
                layer_idx = int(layer_info[3])
                weight_shape = state_dict[key].shape
                
                if layer_idx == 0:  # First layer
                    layer_sizes.append(weight_shape[0])
                elif layer_idx % 4 == 0:  # Every 4th layer is Linear (due to BatchNorm, ReLU, Dropout)
                    layer_sizes.append(weight_shape[0])
        
        print(f"Detected layer sizes: {layer_sizes}")
        
        # Create model with detected architecture
        class DynamicSingleModel(nn.Module):
            def __init__(self, input_size, layer_sizes):
                super().__init__()
                layers = []
                prev_size = input_size
                
                for size in layer_sizes[:-1]:  # All except last
                    layers.extend([
                        nn.Linear(prev_size, size),
                        nn.BatchNorm1d(size),
                        nn.ReLU(),
                        nn.Dropout(0.3)
                    ])
                    prev_size = size
                
                # Output layer
                layers.append(nn.Linear(prev_size, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        class DynamicEnsembleModel(nn.Module):
            def __init__(self, input_size, layer_sizes, num_models=3):
                super().__init__()
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.models = nn.ModuleList([
                    DynamicSingleModel(input_size, layer_sizes) for _ in range(num_models)
                ])
                
                self.meta_learner = nn.Linear(num_models, 1)
                self.to(self.device)
            
            def forward(self, x):
                predictions = []
                for model in self.models:
                    pred = model(x)
                    predictions.append(pred)
                
                ensemble_input = torch.cat(predictions, dim=1)
                final_output = self.meta_learner(ensemble_input)
                return final_output
        
        return DynamicEnsembleModel(input_size, layer_sizes)

# Usage example for debugging:
if __name__ == "__main__":
    # Debug your models
    for model_type, path in MODEL_PATHS.items():
        print(f"\n=== Inspecting {model_type} model ===")
        try:
            DynamicModelLoader.inspect_checkpoint(path)
        except Exception as e:
            print(f"Error inspecting {model_type}: {e}")