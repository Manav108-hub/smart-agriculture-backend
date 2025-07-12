from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Smart Agriculture API...")
    
    # Check if model files exist
    model_paths = [
        "saved_models/Crop_Yield_best.pth",
        "saved_models/Market_Price_best.pth", 
        "saved_models/Sustainability_best.pth"
    ]
    
    missing_models = []
    for path in model_paths:
        if not os.path.exists(path):
            missing_models.append(path)
    
    if missing_models:
        logger.warning(f"⚠️  Missing model files: {missing_models}")
        logger.warning("Some endpoints may not work until models are available")
    else:
        logger.info("✅ All model files found")
    
    logger.info("🎯 API ready to serve predictions!")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Smart Agriculture API...")


app = FastAPI(
    title="Smart Agriculture API",
    description="Fixed Machine Learning API for agricultural predictions with proper categorical encoding",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Agriculture API is running",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/api/health",
            "models_info": "/api/models/info",
            "predictions": {
                "crop_yield": "/api/predict/crop-yield",
                "market_price": "/api/predict/market-price", 
                "sustainability": "/api/predict/sustainability"
            }
        }
    }


@app.get("/status")
async def status():
    """Simple status check"""
    return {"status": "healthy", "message": "API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )