from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import models

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
    try:
        model = models.ModelLoader.load_model("crop")
        result = model.predict(request.dict(), config.CROP_FEATURES)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/yield")
async def predict_yield(request: YieldRequest):
    try:
        model = models.ModelLoader.load_model("yield")
        result = model.predict(request.dict(), config.YIELD_FEATURES)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/sustainability")
async def predict_sustainability(request: SustainabilityRequest):
    try:
        model = models.ModelLoader.load_model("sustainability")
        result = model.predict(request.dict(), config.SUSTAINABILITY_FEATURES)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))