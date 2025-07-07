from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import models
import config

router = APIRouter()

# Request schemas
class CropRequest(BaseModel):
    temperature: float
    humidity: float
    soil_ph: float
    rainfall: float
    altitude: float
    region: float
    season: float

class YieldRequest(BaseModel):
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    irrigation_hours: float
    fertilizer_amount: float
    pesticide_usage: float
    sunlight_hours: float
    crop_age: float

class SustainabilityRequest(BaseModel):
    water_usage: float
    carbon_footprint: float
    soil_health: float
    biodiversity_score: float

# API endpoints
@router.post("/predict/crop")
async def predict_crop(request: CropRequest):
    """
    Predict crop metrics based on environmental conditions
    
    Expected features: temperature, humidity, soil_ph, rainfall, altitude, region, season
    """
    try:
        model = models.ModelLoader.get_model("crop")
        features, norm_params = models.ModelLoader.get_features_and_params("crop")
        
        # Convert request to dict and ensure correct order
        input_data = request.dict()
        result = model.predict(input_data, features, norm_params)
        
        return {
            "prediction": result,
            "model_type": "crop",
            "input_features": input_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop prediction failed: {str(e)}")

@router.post("/predict/yield")
async def predict_yield(request: YieldRequest):
    """
    Predict crop yield based on farming conditions
    
    Expected features: soil_nitrogen, soil_phosphorus, soil_potassium, irrigation_hours, 
                      fertilizer_amount, pesticide_usage, sunlight_hours, crop_age
    """
    try:
        model = models.ModelLoader.get_model("yield")
        features, norm_params = models.ModelLoader.get_features_and_params("yield")
        
        input_data = request.dict()
        result = model.predict(input_data, features, norm_params)
        
        return {
            "prediction": result,
            "model_type": "yield",
            "input_features": input_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield prediction failed: {str(e)}")

@router.post("/predict/sustainability")
async def predict_sustainability(request: SustainabilityRequest):
    """
    Predict sustainability metrics based on environmental impact factors
    
    Expected features: water_usage, carbon_footprint, soil_health, biodiversity_score
    """
    try:
        model = models.ModelLoader.get_model("sustainability")
        features, norm_params = models.ModelLoader.get_features_and_params("sustainability")
        
        input_data = request.dict()
        result = model.predict(input_data, features, norm_params)
        
        return {
            "prediction": result,
            "model_type": "sustainability",
            "input_features": input_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sustainability prediction failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    try:
        # Try to load one model to check if everything is working
        model = models.ModelLoader.get_model("crop")
        return {
            "status": "healthy",
            "message": "Agricultural ML API is running",
            "models_loaded": len(models.ModelLoader._models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Model info endpoint
@router.get("/models/info")
async def get_models_info():
    """Get information about available models and their features"""
    return {
        "available_models": {
            "crop": {
                "features": config.CROP_FEATURES,
                "description": "Predicts crop performance based on environmental conditions"
            },
            "yield": {
                "features": config.YIELD_FEATURES,
                "description": "Predicts crop yield based on farming inputs"
            },
            "sustainability": {
                "features": config.SUSTAINABILITY_FEATURES,
                "description": "Predicts sustainability metrics based on environmental impact"
            }
        }
    }