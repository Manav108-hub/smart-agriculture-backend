from fastapi import FastAPI
import routes

app = FastAPI(title="Agricultural AI API")
app.include_router(routes.router, prefix="/api")

@app.on_event("startup")
async def load_models_on_startup():
    """Pre-load models when server starts"""
    print("Loading models...")
    models.ModelLoader.load_model("crop")
    models.ModelLoader.load_model("yield")
    models.ModelLoader.load_model("sustainability")
    print("Models loaded successfully")