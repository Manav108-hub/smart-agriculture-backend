import numpy as np
from typing import Dict, List, Any
import config


def encode_categorical_features(input_data: Dict[str, Any], feature_list: List[str]) -> Dict[str, float]:
    """
    Convert categorical features to one-hot encoding
    
    Args:
        input_data: Raw input data with categorical values
        feature_list: List of all expected features (including one-hot encoded ones)
    
    Returns:
        Dictionary with one-hot encoded features
    """
    encoded_data = {}
    
    # Initialize all features to 0
    for feature in feature_list:
        encoded_data[feature] = 0.0
    
    # Copy numeric features directly
    for feature, value in input_data.items():
        if feature in feature_list:
            encoded_data[feature] = float(value)
    
    # Handle categorical encoding
    for raw_feature, value in input_data.items():
        if isinstance(value, str):  # Categorical feature
            value_lower = value.lower()
            
            if raw_feature == "Crop_Type":
                if value_lower in config.CROP_TYPE_MAPPING:
                    one_hot_feature = config.CROP_TYPE_MAPPING[value_lower]
                    if one_hot_feature in feature_list:
                        encoded_data[one_hot_feature] = 1.0
                else:
                    raise ValueError(f"Unknown crop type: {value}. Supported types: {list(config.CROP_TYPE_MAPPING.keys())}")
                    
            elif raw_feature == "Product":
                if value_lower in config.PRODUCT_MAPPING:
                    one_hot_feature = config.PRODUCT_MAPPING[value_lower]
                    if one_hot_feature in feature_list:
                        encoded_data[one_hot_feature] = 1.0
                else:
                    raise ValueError(f"Unknown product: {value}. Supported products: {list(config.PRODUCT_MAPPING.keys())}")
    
    return encoded_data


def preprocess_input_with_encoding(input_data: Dict[str, Any], feature_list: List[str], 
                                 norm_params: Dict[str, List[float]]) -> np.ndarray:
    """
    Convert input data to tensor with categorical encoding and normalization
    
    Args:
        input_data: Raw input data dictionary
        feature_list: List of expected features in correct order
        norm_params: Normalization parameters (mean, std)
    
    Returns:
        Normalized numpy array ready for model input
    """
    if not isinstance(input_data, dict):
        raise ValueError(f"Expected input_data to be a dictionary. Got {type(input_data)}")

    # Step 1: Encode categorical features
    encoded_data = encode_categorical_features(input_data, feature_list)
    
    # Step 2: Convert to array in correct feature order
    try:
        input_array = np.array([encoded_data[feat] for feat in feature_list], dtype=np.float32)
    except KeyError as e:
        missing_features = [feat for feat in feature_list if feat not in encoded_data]
        raise ValueError(f"Missing features after encoding: {missing_features}")
    
    # Step 3: Normalize using training parameters
    mean = np.array(norm_params["mean"], dtype=np.float32)
    std = np.array(norm_params["std"], dtype=np.float32)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    normalized = (input_array - mean) / std
    
    return normalized


def validate_input_data(input_data: Dict[str, Any], model_type: str) -> None:
    """
    Validate input data before processing
    
    Args:
        input_data: Input data to validate
        model_type: Type of model (crop_yield, market_price, sustainability)
    """
    required_fields = {
        "crop_yield": ["Soil_pH", "Soil_Moisture", "Temperature_C", "Rainfall_mm", 
                      "Fertilizer_Usage_kg", "Pesticide_Usage_kg", "Crop_Type"],
        "market_price": ["Market_Price_per_ton", "Demand_Index", "Supply_Index", 
                        "Competitor_Price_per_ton", "Economic_Indicator", "Weather_Impact_Score",
                        "Seasonal_Factor", "Consumer_Trend_Index", "Product"],
        "sustainability": ["Soil_pH", "Soil_Moisture", "Temperature_C", "Rainfall_mm",
                          "Fertilizer_Usage_kg", "Pesticide_Usage_kg", "Crop_Type", "Crop_Yield_ton"]
    }
    
    if model_type not in required_fields:
        raise ValueError(f"Unknown model type: {model_type}")
    
    missing_fields = []
    for field in required_fields[model_type]:
        if field not in input_data:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields for {model_type}: {missing_fields}")
    
    # Validate data types
    for field, value in input_data.items():
        if field in ["Crop_Type", "Product"]:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string, got {type(value)}")
        else:
            try:
                float(value)
            except (ValueError, TypeError):
                raise ValueError(f"{field} must be numeric, got {value}")


# Legacy function for backward compatibility
def preprocess_input(input_data, feature_list, norm_params):
    """Legacy function - redirects to new implementation"""
    return preprocess_input_with_encoding(input_data, feature_list, norm_params)