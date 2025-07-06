MODEL_PATHS = {
    "crop": "saved_models/Crop Yield_best.pth",
    "yield": "saved_models/Market Price_best.pth",
    "sustainability": "saved_models/Sustainability_best.pth"
}

# Input features from your training code
CROP_FEATURES = ["temperature", "humidity", "soil_ph", "rainfall", 
                "altitude", "region", "season"]
YIELD_FEATURES = ["soil_nitrogen", "soil_phosphorus", "soil_potassium",
                 "irrigation_hours", "fertilizer_amount", "pesticide_usage",
                 "sunlight_hours", "crop_age"]
SUSTAINABILITY_FEATURES = ["water_usage", "carbon_footprint", "soil_health",
                          "biodiversity_score"]

# Normalization parameters from your training
CROP_NORM_PARAMS = {
    "mean": [50.55, 42.36, 48.15, 25.62, 71.48, 6.47, 103.46],
    "std": [36.92, 50.65, 49.96, 5.06, 22.26, 0.77, 54.86]
}