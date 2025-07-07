from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
import models
from routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up - API ready!")
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Smart Agriculture API",
    description="Machine Learning API for agricultural predictions",
    version="1.0.0",
    lifespan=lifespan
)

# Include routes
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Smart Agriculture API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)