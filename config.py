import os
import numpy as np

# ──────────────────────────── Paths ──────────────────────────── #
MODEL_BASE_DIR = "saved_models"
MODEL_PATHS = {
    "crop_yield":     os.path.join(MODEL_BASE_DIR, "Crop_Yield_best.pth"),
    "market_price":   os.path.join(MODEL_BASE_DIR, "Market_Price_best.pth"),
    "sustainability": os.path.join(MODEL_BASE_DIR, "Sustainability_best.pth"),
}

# ──────────── CORRECTED FEATURE LISTS (based on model analysis) ──────────── #

# 1️⃣ Crop Yield Model - expects 20 inputs
CROP_FEATURES = [
    # Base environmental features (6)
    "Soil_pH",
    "Soil_Moisture", 
    "Temperature_C",
    "Rainfall_mm",
    "Fertilizer_Usage_kg",
    "Pesticide_Usage_kg",
    
    # One-hot encoded crop types (14)
    "Crop_Type_Wheat",
    "Crop_Type_Rice", 
    "Crop_Type_Corn",
    "Crop_Type_Barley",
    "Crop_Type_Soybean",
    "Crop_Type_Cotton",
    "Crop_Type_Sugarcane",
    "Crop_Type_Tomato",
    "Crop_Type_Potato",
    "Crop_Type_Onion",
    "Crop_Type_Carrot",
    "Crop_Type_Lettuce",
    "Crop_Type_Cucumber",
    "Crop_Type_Pepper"
]

# 2️⃣ Market Price Model - expects 15 inputs  
MARKET_FEATURES = [
    # Base market features (8)
    "Market_Price_per_ton",
    "Demand_Index",
    "Supply_Index", 
    "Competitor_Price_per_ton",
    "Economic_Indicator",
    "Weather_Impact_Score",
    "Seasonal_Factor",
    "Consumer_Trend_Index",
    
    # One-hot encoded products (7)
    "Product_Wheat",
    "Product_Rice",
    "Product_Corn", 
    "Product_Soybean",
    "Product_Cotton",
    "Product_Tomato",
    "Product_Potato"
]

# 3️⃣ Sustainability Model - expects 25 inputs
SUSTAINABILITY_FEATURES = [
    # Base environmental features (7)
    "Soil_pH",
    "Soil_Moisture",
    "Temperature_C", 
    "Rainfall_mm",
    "Fertilizer_Usage_kg",
    "Pesticide_Usage_kg",
    "Crop_Yield_ton",
    
    # Extended one-hot encoded crop types (18)
    "Crop_Type_Wheat",
    "Crop_Type_Rice",
    "Crop_Type_Corn", 
    "Crop_Type_Barley",
    "Crop_Type_Soybean",
    "Crop_Type_Cotton",
    "Crop_Type_Sugarcane",
    "Crop_Type_Tomato",
    "Crop_Type_Potato", 
    "Crop_Type_Onion",
    "Crop_Type_Carrot",
    "Crop_Type_Lettuce",
    "Crop_Type_Cucumber",
    "Crop_Type_Pepper",
    "Crop_Type_Cabbage",
    "Crop_Type_Spinach",
    "Crop_Type_Broccoli",
    "Crop_Type_Beans"
]

# ──────────── CATEGORICAL MAPPINGS ──────────── #
CROP_TYPE_MAPPING = {
    "wheat": "Crop_Type_Wheat",
    "rice": "Crop_Type_Rice",
    "corn": "Crop_Type_Corn",
    "barley": "Crop_Type_Barley",
    "soybean": "Crop_Type_Soybean",
    "cotton": "Crop_Type_Cotton",
    "sugarcane": "Crop_Type_Sugarcane",
    "tomato": "Crop_Type_Tomato",
    "potato": "Crop_Type_Potato",
    "onion": "Crop_Type_Onion",
    "carrot": "Crop_Type_Carrot",
    "lettuce": "Crop_Type_Lettuce",
    "cucumber": "Crop_Type_Cucumber",
    "pepper": "Crop_Type_Pepper",
    "cabbage": "Crop_Type_Cabbage",
    "spinach": "Crop_Type_Spinach",
    "broccoli": "Crop_Type_Broccoli",
    "beans": "Crop_Type_Beans"
}

PRODUCT_MAPPING = {
    "wheat": "Product_Wheat",
    "rice": "Product_Rice",
    "corn": "Product_Corn",
    "soybean": "Product_Soybean",
    "cotton": "Product_Cotton",
    "tomato": "Product_Tomato",
    "potato": "Product_Potato"
}

# ──────── PLACEHOLDER NORMALIZATION (REPLACE WITH REAL VALUES) ──────── #
# TODO: These need to be replaced with actual computed statistics from training data
CROP_NORM_PARAMS = {
    "mean": np.zeros(len(CROP_FEATURES)).tolist(),
    "std": np.ones(len(CROP_FEATURES)).tolist(),
}

MARKET_NORM_PARAMS = {
    "mean": np.zeros(len(MARKET_FEATURES)).tolist(), 
    "std": np.ones(len(MARKET_FEATURES)).tolist(),
}

SUSTAINABILITY_NORM_PARAMS = {
    "mean": np.zeros(len(SUSTAINABILITY_FEATURES)).tolist(),
    "std": np.ones(len(SUSTAINABILITY_FEATURES)).tolist(),
}

# ──────────── CORRECTED MODEL ARCHITECTURES ──────────── #
_CROP_MODEL_CFG = {
    "ensemble_size": 3,
    "hidden_sizes": [256, 128, 48],
    "dropout_rate": 0.3,
}

_MARKET_MODEL_CFG = {
    "ensemble_size": 3, 
    "hidden_sizes": [320, 192, 48],
    "dropout_rate": 0.3,
}

_SUSTAINABILITY_MODEL_CFG = {
    "ensemble_size": 3,
    "hidden_sizes": [320, 288, 176, 64],
    "dropout_rate": 0.3,
}

# ──────────── MODEL CONFIGURATION ──────────── #
MODEL_CONFIG = {
    "crop_yield": {
        "name": "Crop Yield Prediction",
        "description": "Predicts crop yield based on environmental and farming practices",
        "input_features": CROP_FEATURES,
        "feature_count": len(CROP_FEATURES),
        "normalization": CROP_NORM_PARAMS,
        "target_column": "Crop_Yield_ton", 
        "model": _CROP_MODEL_CFG,
        "categorical_features": ["Crop_Type"],
        "categorical_mappings": CROP_TYPE_MAPPING,
    },

    "market_price": {
        "name": "Market Price Prediction",
        "description": "Predicts commodity prices using market indicators",
        "input_features": MARKET_FEATURES,
        "feature_count": len(MARKET_FEATURES),
        "normalization": MARKET_NORM_PARAMS,
        "target_column": "Market_Price_per_ton",
        "model": _MARKET_MODEL_CFG,
        "categorical_features": ["Product"],
        "categorical_mappings": PRODUCT_MAPPING,
    },

    "sustainability": {
        "name": "Sustainability Assessment", 
        "description": "Evaluates environmental impact of farming practices",
        "input_features": SUSTAINABILITY_FEATURES,
        "feature_count": len(SUSTAINABILITY_FEATURES),
        "normalization": SUSTAINABILITY_NORM_PARAMS,
        "target_column": "Sustainability_Score",
        "model": _SUSTAINABILITY_MODEL_CFG,
        "categorical_features": ["Crop_Type"],
        "categorical_mappings": CROP_TYPE_MAPPING,
    },
}

# ─────────── TRAINING SETTINGS ─────────── #
GLOBAL_TRAINING_CFG = {
    "batch_size": 30,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 300,
    "patience": 30,
    "lr_patience": 15,
    "lr_factor": 0.7,
    "grad_clip": 0.5,
    "tune_hyperparams": True,
    "tuning_trials": 30,
}