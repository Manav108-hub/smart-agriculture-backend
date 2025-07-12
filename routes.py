from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
import torch
import config
import models as models
from utils import validate_input_data
import os

router = APIRouter()

# ──────────── REQUEST SCHEMAS ──────────── #

class CropRequest(BaseModel):
    """Crop yield prediction request"""
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Type: str
    
    @field_validator('Crop_Type')
    @classmethod
    def validate_crop_type(cls, v):
        allowed_types = list(config.CROP_TYPE_MAPPING.keys())
        if v.lower() not in allowed_types:
            raise ValueError(f"Crop_Type must be one of: {allowed_types}")
        return v.lower()
    
    @field_validator('Soil_pH')
    @classmethod
    def validate_soil_ph(cls, v):
        if not 0 <= v <= 14:
            raise ValueError("Soil_pH must be between 0 and 14")
        return v
    
    @field_validator('Soil_Moisture')
    @classmethod
    def validate_soil_moisture(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Soil_Moisture must be between 0 and 100")
        return v


class MarketPriceRequest(BaseModel):
    """Market price prediction request"""
    Market_Price_per_ton: float
    Demand_Index: float
    Supply_Index: float
    Competitor_Price_per_ton: float
    Economic_Indicator: float
    Weather_Impact_Score: float
    Seasonal_Factor: float
    Consumer_Trend_Index: float
    Product: str
    
    @field_validator('Product')
    @classmethod
    def validate_product(cls, v):
        allowed_products = list(config.PRODUCT_MAPPING.keys())
        if v.lower() not in allowed_products:
            raise ValueError(f"Product must be one of: {allowed_products}")
        return v.lower()


class SustainabilityRequest(BaseModel):
    """Sustainability assessment request"""
    Soil_pH: float
    Soil_Moisture: float
    Temperature_C: float
    Rainfall_mm: float
    Fertilizer_Usage_kg: float
    Pesticide_Usage_kg: float
    Crop_Type: str
    Crop_Yield_ton: float
    
    @field_validator('Crop_Type')
    @classmethod
    def validate_crop_type(cls, v):
        allowed_types = list(config.CROP_TYPE_MAPPING.keys())
        if v.lower() not in allowed_types:
            raise ValueError(f"Crop_Type must be one of: {allowed_types}")
        return v.lower()
    
    @field_validator('Crop_Yield_ton')
    @classmethod
    def validate_crop_yield(cls, v):
        if v < 0:
            raise ValueError("Crop_Yield_ton must be non-negative")
        return v


# ──────────── MODEL LOADING ──────────── #

def load_model_and_params(model_type: str):
    """Fixed model loading that matches actual trained architectures"""
    if model_type not in ["crop_yield", "market_price", "sustainability"]:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        model_config = config.MODEL_CONFIG[model_type]
        model_path = config.MODEL_PATHS[model_type]
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        input_size = model_config["feature_count"]
        model = models.EnsembleModel(input_size, model_config)
        
        # Load state dict with error handling
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {str(e)}")
        
        # Set device and evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return model, model_config
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed for {model_type}: {str(e)}")


# ──────────── PREDICTION ENDPOINTS ──────────── #

@router.post("/predict/crop-yield")
async def predict_crop_yield(request: CropRequest):
    """Crop yield prediction endpoint"""
    try:
        # Validate input
        input_data = request.model_dump()  # Updated for Pydantic V2
        validate_input_data(input_data, "crop_yield")
        
        # Load model
        model, model_config = load_model_and_params("crop_yield")
        
        # Make prediction
        result = model.predict(
            input_data=input_data,
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input_features_used": len(model_config["input_features"]),
            "crop_type": input_data["Crop_Type"],
            "status": "success"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crop yield prediction failed: {str(e)}")


@router.post("/predict/market-price")
async def predict_market_price(request: MarketPriceRequest):
    """Market price prediction endpoint"""
    try:
        # Validate input
        input_data = request.model_dump()  # Updated for Pydantic V2
        validate_input_data(input_data, "market_price")
        
        # Load model
        model, model_config = load_model_and_params("market_price")
        
        # Make prediction
        result = model.predict(
            input_data=input_data,
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input_features_used": len(model_config["input_features"]),
            "product": input_data["Product"],
            "status": "success"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market price prediction failed: {str(e)}")


@router.post("/predict/sustainability")
async def predict_sustainability(request: SustainabilityRequest):
    """Sustainability prediction endpoint"""
    try:
        # Validate input
        input_data = request.model_dump()  # Updated for Pydantic V2
        validate_input_data(input_data, "sustainability")
        
        # Load model
        model, model_config = load_model_and_params("sustainability")
        
        # Make prediction
        result = model.predict(
            input_data=input_data,
            feature_list=model_config["input_features"],
            norm_params=model_config["normalization"]
        )
        
        return {
            "prediction": result,
            "model": model_config["name"],
            "input_features_used": len(model_config["input_features"]),
            "crop_type": input_data["Crop_Type"],
            "status": "success"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sustainability prediction failed: {str(e)}")


# ──────────── UTILITY ENDPOINTS ──────────── #

@router.get("/health")
async def health_check():
    """Health check with model validation"""
    try:
        results = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test all models
        for model_type in ["crop_yield", "market_price", "sustainability"]:
            try:
                model, config_info = load_model_and_params(model_type)
                results[model_type] = {
                    "status": "healthy",
                    "input_size": config_info["feature_count"],
                    "model_name": config_info["name"]
                }
            except Exception as e:
                results[model_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        overall_status = "healthy" if all(r["status"] == "healthy" for r in results.values()) else "degraded"
        
        return {
            "status": overall_status,
            "device": str(device),
            "models": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/models/info")
async def get_models_info():
    """Model information endpoint"""
    return {
        "available_models": {
            "crop_yield": {
                "input_features": config.MODEL_CONFIG["crop_yield"]["input_features"],
                "feature_count": config.MODEL_CONFIG["crop_yield"]["feature_count"],
                "description": config.MODEL_CONFIG["crop_yield"]["description"],
                "target": config.MODEL_CONFIG["crop_yield"]["target_column"],
                "supported_crop_types": list(config.CROP_TYPE_MAPPING.keys())
            },
            "market_price": {
                "input_features": config.MODEL_CONFIG["market_price"]["input_features"],
                "feature_count": config.MODEL_CONFIG["market_price"]["feature_count"],
                "description": config.MODEL_CONFIG["market_price"]["description"],
                "target": config.MODEL_CONFIG["market_price"]["target_column"],
                "supported_products": list(config.PRODUCT_MAPPING.keys())
            },
            "sustainability": {
                "input_features": config.MODEL_CONFIG["sustainability"]["input_features"],
                "feature_count": config.MODEL_CONFIG["sustainability"]["feature_count"],
                "description": config.MODEL_CONFIG["sustainability"]["description"],
                "target": config.MODEL_CONFIG["sustainability"]["target_column"],
                "supported_crop_types": list(config.CROP_TYPE_MAPPING.keys())
            }
        }
    }


@router.get("/models/debug/{model_type}")
async def debug_model(model_type: str):
    """Debug endpoint to inspect model structure"""
    try:
        if model_type not in ["crop_yield", "market_price", "sustainability"]:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        model, model_config = load_model_and_params(model_type)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_type": model_type,
            "expected_input_size": model_config["feature_count"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "ensemble_size": model_config["model"]["ensemble_size"],
            "hidden_sizes": model_config["model"]["hidden_sizes"],
            "device": str(next(model.parameters()).device),
            "model_structure": str(model)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


@router.post("/test/encoding")
async def test_categorical_encoding(data: dict):
    """Test endpoint to verify categorical encoding"""
    try:
        from utils import encode_categorical_features
        
        # Test crop type encoding
        if "Crop_Type" in data:
            crop_features = config.MODEL_CONFIG["crop_yield"]["input_features"]
            encoded = encode_categorical_features(data, crop_features)
            
            return {
                "original_data": data,
                "encoded_features": {k: v for k, v in encoded.items() if v != 0 or k in data},
                "total_features": len(crop_features),
                "one_hot_features": [k for k, v in encoded.items() if k.startswith("Crop_Type_") and v == 1]
            }
        
        # Test product encoding  
        elif "Product" in data:
            market_features = config.MODEL_CONFIG["market_price"]["input_features"]
            encoded = encode_categorical_features(data, market_features)
            
            return {
                "original_data": data,
                "encoded_features": {k: v for k, v in encoded.items() if v != 0 or k in data},
                "total_features": len(market_features),
                "one_hot_features": [k for k, v in encoded.items() if k.startswith("Product_") and v == 1]
            }
        
        else:
            return {"error": "No categorical features found in data"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding test failed: {str(e)}")