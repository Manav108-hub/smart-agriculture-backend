import numpy as np
from config import CROP_NORM_PARAMS

def preprocess_input(input_data, feature_list, norm_params):
    """Convert input data to tensor and normalize"""
    # Convert input to numpy array in correct order
    input_array = np.array([input_data[feat] for feat in feature_list])
    
    # Normalize using training parameters
    mean = np.array(norm_params["mean"])
    std = np.array(norm_params["std"])
    normalized = (input_array - mean) / (std + 1e-8)
    
    return normalized