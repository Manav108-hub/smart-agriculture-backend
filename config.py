import os

# Base directory for models
MODEL_BASE_DIR = "saved_models"

# Model file paths
MODEL_PATHS = {
    "crop": os.path.join(MODEL_BASE_DIR, "Crop Yield_best.pth"),
    "yield": os.path.join(MODEL_BASE_DIR, "Market Price_best.pth"),
    "sustainability": os.path.join(MODEL_BASE_DIR, "Sustainability_best.pth")
}

# Input features from your training code
CROP_FEATURES = ["Soil_pH", "Soil_Moisture", "Temperature_C", "Rainfall_mm",
                 "Fertilizer_Usage_kg", "Pesticide_Usage_kg", "Crop_Type"] # Now 7 raw features

# YIELD_FEATURES are for the Market Price Model, not Crop Yield.
# From your model_config.yaml, the Market Price model uses:
# "Market_Price_per_ton", "Demand_Index", "Supply_Index", "Competitor_Price_per_ton",
# "Economic_Indicator", "Weather_Impact_Score", "Seasonal_Factor",
# "Consumer_Trend_Index", "Product" (encoded)
# If `Product` has 11 categories (assuming 9 numerical + 11 encoded = 20),
# then YIELD_FEATURES and YIELD_NORM_PARAMS for the Market Price model also need adjustment.
# Let's assume for now that 8 is correct based on the name "YIELD_FEATURES".
# But if "Market Price_best.pth" also has input size 20, you'll need to update this.
YIELD_FEATURES = ["soil_nitrogen", "soil_phosphorus", "soil_potassium",
                  "irrigation_hours", "fertilizer_amount", "pesticide_usage",
                  "sunlight_hours", "crop_age"] # Length 8. Verify this for Market Price model.

SUSTAINABILITY_FEATURES = ["water_usage", "carbon_footprint", "soil_health",
                             "biodiversity_score"]


# Normalization parameters from your training
CROP_NORM_PARAMS = {
    "mean": [
        # These 20 values should correspond to the means of the 20 input features
        # that your crop model was trained on, *after* one-hot encoding 'Crop_Type'
        # and any other preprocessing.
        # Example (this is a placeholder, get actual values):
        0.5, 0.6, 25.0, 150.0, 100.0, 5.0, # for the 6 numerical features
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, # assuming ~7% frequency for 14 categories
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07
    ],
    "std": [
        # These 20 values should correspond to the stds of the 20 input features
        # that your crop model was trained on, *after* one-hot encoding 'Crop_Type'
        # and any other preprocessing.
        # Example (this is a placeholder, get actual values):
        0.1, 0.2, 5.0, 50.0, 20.0, 2.0, # for the 6 numerical features
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, # for the 14 one-hot encoded features
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2
    ]
}

# TODO: Replace these with actual values from your training data
# You need to get these from your training process
YIELD_NORM_PARAMS = {
    "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Replace with actual mean values
    "std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]    # Replace with actual std values
}

SUSTAINABILITY_NORM_PARAMS = {
    "mean": [0.0, 0.0, 0.0, 0.0],  # Replace with actual mean values
    "std": [1.0, 1.0, 1.0, 1.0]    # Replace with actual std values
}

# Model configuration metadata
MODEL_CONFIG = {
    "crop": {
        "name": "Crop Yield Prediction",
        "description": "Predicts crop performance based on environmental conditions",
        # Feature count now reflects the *actual* input dimensions
        "input_features": CROP_FEATURES, # These are the raw features the API expects
        "feature_count": 20, # This must be the *effective* number of features after encoding
        "normalization": CROP_NORM_PARAMS # This now has 20 values
    },
    "yield": { # This is your Market Price Prediction model
        "name": "Market Price Prediction",
        "description": "Predicts crop yield/market price based on farming inputs",
        "input_features": YIELD_FEATURES, # These are the raw features the API expects
        "feature_count": 8, # This must be the *effective* number of features after encoding.
                             # If Product also expands to 20, this needs to be 20.
        "normalization": YIELD_NORM_PARAMS
    },
    "sustainability": {
        "name": "Sustainability Prediction",
        "description": "Predicts sustainability metrics based on environmental impact",
        "input_features": SUSTAINABILITY_FEATURES,
        "feature_count": len(SUSTAINABILITY_FEATURES), # This is 4
        "normalization": SUSTAINABILITY_NORM_PARAMS
    }
}

# Validation function to check if model files exist
def validate_model_paths():
    """Validate that all model files exist"""
    missing_files = []
    for model_name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing_files.append(f"{model_name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing model files:\n" + "\n".join(missing_files))
    
    return True

# Function to get actual normalization parameters from your training data
def get_normalization_params_from_training():
    """
    TODO: Implement this function to extract normalization parameters
    from your training process or saved training statistics

    This should return the actual mean and std values used during training
    for yield and sustainability models.
    """
    # Load the saved scaler or calculate from the training dataset
    # For example, if you saved a StandardScaler, load it and get its mean_ and scale_
    # For CROP:
    # Load your training data for the crop model.
    # Apply the same preprocessing steps (e.g., one-hot encoding for 'Crop_Type').
    # Calculate mean and std for the resulting 20 columns.

    # Placeholder for crop params (MUST BE REPLACED WITH ACTUAL 20-FEATURE VALUES)
    crop_params = {
        "mean": [0.0] * 20, # Placeholder, replace with actual 20 values
        "std": [1.0] * 20   # Placeholder, replace with actual 20 values
    }

    # For YIELD (Market Price Model):
    # Determine the correct number of input features (e.g., 9 raw, and Product encoded into 11 = 20 total)
    # Placeholder for yield params (replace with actual values for the correct feature count)
    yield_params = {
        "mean": [0.0] * 8,  # Verify this count (8) is correct for the market price model
        "std": [1.0] * 8    # Verify this count (8) is correct for the market price model
    }

    # For SUSTAINABILITY:
    # Placeholder for sustainability params (replace with actual values for 4 features)
    sustainability_params = {
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0]
    }

    return crop_params, yield_params, sustainability_params # Also return crop_params
