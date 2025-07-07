import numpy as np

def preprocess_input(input_data, feature_list, norm_params):
    """Convert input data to tensor and normalize"""
    if not isinstance(input_data, dict):
        raise ValueError(f"Expected input_data to be a dictionary. Got {type(input_data)}")

    # Convert input to numpy array in correct order
    input_array = np.array([input_data[feat] for feat in feature_list])
    
    # Normalize using training parameters
    mean = np.array(norm_params["mean"])
    std = np.array(norm_params["std"])
    normalized = (input_array - mean) / (std + 1e-8)
    
    return normalized.astype(np.float32)