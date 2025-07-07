import os
import numpy as np

# ──────────────────────────── Paths ──────────────────────────── #
MODEL_BASE_DIR = "saved_models"
MODEL_PATHS = {
    "crop_yield":     os.path.join(MODEL_BASE_DIR, "Crop_Yield_best.pth"),
    "market_price":   os.path.join(MODEL_BASE_DIR, "Market_Price_best.pth"),
    "sustainability": os.path.join(MODEL_BASE_DIR, "Sustainability_Best.pth"),
}

# ────────────────── Feature definitions (final) ───────────────── #
# 1️⃣  Farmer models (crop‑yield & sustainability) – 7 inputs
FARMER_FEATURES = [
    "Soil_pH",
    "Soil_Moisture",
    "Temperature_C",
    "Rainfall_mm",
    "Crop_Type",              # categorical, to be encoded
    "Fertilizer_Usage_kg",
    "Pesticide_Usage_kg",
]

# 2️⃣  Market model – 9 inputs
MARKET_FEATURES = [
    "Market_Price_per_ton",   # also target
    "Demand_Index",
    "Supply_Index",
    "Competitor_Price_per_ton",
    "Economic_Indicator",
    "Weather_Impact_Score",
    "Seasonal_Factor",
    "Consumer_Trend_Index",
    "Product",                # categorical, to be encoded
]

# ─────────────── Normalisation placeholders (match lengths) ───── #
CROP_NORM_PARAMS = {
    "mean": [0.0] * len(FARMER_FEATURES),
    "std":  [1.0] * len(FARMER_FEATURES),
}
MARKET_NORM_PARAMS = {
    "mean": [0.0] * len(MARKET_FEATURES),
    "std":  [1.0] * len(MARKET_FEATURES),
}
SUSTAINABILITY_NORM_PARAMS = {
    "mean": [0.0] * len(FARMER_FEATURES),
    "std":  [1.0] * len(FARMER_FEATURES),
}

# ───────────── Common model hyper‑parameters (defaults) ────────── #
_DEFAULT_MODEL_CFG = {
    "ensemble_size": 3,
    "hidden_sizes":  [256, 128, 64],
    "dropout_rate":  0.2,
}

# ──────────────── Per‑model configuration blocks ──────────────── #
MODEL_CONFIG = {
    # ––– Crop‑yield model ––––––––––––––––––––––––––––––– #
    "crop_yield": {
        "name": "Crop Yield Prediction",
        "description": "Predicts crop yield based on environmental and farming practices",
        "input_features": FARMER_FEATURES,
        "feature_count": len(FARMER_FEATURES),          # 7
        "normalization": CROP_NORM_PARAMS,
        "target_column": "Crop_Yield_ton",
        "model": {**_DEFAULT_MODEL_CFG, "hidden_sizes": [256, 128, 48]},
    },

    # ––– Market‑price model –––––––––––––––––––––––––––– #
    "market_price": {
        "name": "Market Price Prediction",
        "description": "Predicts commodity prices using market indicators",
        "input_features": MARKET_FEATURES,
        "feature_count": len(MARKET_FEATURES),          # 9
        "normalization": MARKET_NORM_PARAMS,
        "target_column": "Market_Price_per_ton",
        "model": _DEFAULT_MODEL_CFG,
    },

    # ––– Sustainability model ––––––––––––––––––––––––– #
    "sustainability": {
        "name": "Sustainability Assessment",
        "description": "Evaluates environmental impact of farming practices",
        "input_features": FARMER_FEATURES,
        "feature_count": len(FARMER_FEATURES),          # 7
        "normalization": SUSTAINABILITY_NORM_PARAMS,
        "target_column": "Sustainability_Score",
        "model": _DEFAULT_MODEL_CFG,
    },
}

# ───────────── Optional training‑time settings (unchanged) ─────── #
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
