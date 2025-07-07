from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import config
import models as models  # Import updated model

router = APIRouter()

# Request schemas
class CropRequest(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Type: str  # This will be encoded in the model

class MarketPriceRequest(BaseModel):
    Market_Price_per_ton: float
    Demand_Index: float
    Supply_Index: float
    Competitor_Price_per_ton: float
    Economic_Indicator: float
    Weather_Impact_Score: float
    Seasonal_Factor: float
    Consumer_Trend_Index: float
    Product: str  # This will be encoded in the model

class SustainabilityRequest(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Type: str
    Crop_Yield_ton: float


# Helper function to load model and params
def load_model_and_params(model_type: str):
    if model_type not in ["crop_yield", "market_price", "sustainability"]:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_config = config.MODEL_CONFIG[model_type]
    model_path = config.MODEL_PATHS[model_type]
    
    input_size = model_config["feature_count"]
    model = models.EnsembleModel(input_size, model_config)
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    
    return model, model_config


@router.post("/predict/crop-yield")
async def predict_crop_yield(request: CropRequest):
    try:
        model, model_config = load_model_and_params("crop_yield")
        input_data = request.dict()
        
        # 🔍 Debugging: Log input data
        print("Input data:", input_data)
        
        result = model.predict(
            input_data=input_data,
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input": input_data,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop yield prediction failed: {str(e)}")
@router.post("/predict/market-price")
async def predict_market_price(request: MarketPriceRequest):
    try:
        model, model_config = load_model_and_params("market_price")
        result = model.predict(
            input_data=request.dict(),
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input": request.dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market price prediction failed: {str(e)}")


@router.post("/predict/sustainability")
async def predict_sustainability(request: SustainabilityRequest):
    try:
        model, model_config = load_model_and_params("sustainability")
        result = model.predict(
            input_data=request.dict(),
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input": request.dict(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sustainability prediction failed: {str(e)}")


@router.get("/health")
async def health_check():
    try:
        _, _ = load_model_and_params("crop_yield")
        return {"status": "healthy", "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/models/info")
async def get_models_info():
    return {
        "available_models": {
            "crop_yield": {
                "features": config.CROP_FEATURES,
                "description": config.MODEL_CONFIG["crop_yield"]["description"],
                "target": config.MODEL_CONFIG["crop_yield"]["target_column"]
            },
            "market_price": {
                "features": config.MARKET_FEATURES,
                "description": config.MODEL_CONFIG["market_price"]["description"],
                "target": config.MODEL_CONFIG["market_price"]["target_column"]
            },
            "sustainability": {
                "features": config.SUSTAINABILITY_FEATURES,
                "description": config.MODEL_CONFIG["sustainability"]["description"],
                "target": config.MODEL_CONFIG["sustainability"]["target_column"]
            }
        }
    }