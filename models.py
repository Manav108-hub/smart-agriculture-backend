import torch
import torch.nn as nn
from config import MODEL_PATHS, CROP_NORM_PARAMS
from utils import preprocess_input

class AgriculturalModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, input_data, feature_list):
        self.eval()
        with torch.no_grad():
            # Preprocess input
            processed = preprocess_input(
                input_data, feature_list, CROP_NORM_PARAMS
            )
            tensor = torch.FloatTensor(processed).to(self.device)
            output = self(tensor).cpu().numpy()[0]
            return float(output)

class ModelLoader:
    _models = {}
    
    @classmethod
    def load_model(cls, model_type):
        if model_type in cls._models:
            return cls._models[model_type]
            
        # Create model architecture
        if model_type == "crop":
            model = AgriculturalModel(len(config.CROP_FEATURES))
        elif model_type == "yield":
            model = AgriculturalModel(len(config.YIELD_FEATURES))
        else:
            model = AgriculturalModel(len(config.SUSTAINABILITY_FEATURES))
            
        # Load weights
        checkpoint = torch.load(config.MODEL_PATHS[model_type], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(model.device)
        model.eval()
        
        cls._models[model_type] = model
        return model