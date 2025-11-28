import os
import sys
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import InferenceEngine
try:
    from inference_engine import InferenceEngine
except ImportError:
    from .inference_engine import InferenceEngine

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
DEVICE = os.getenv("DEVICE", "cpu")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("wakeword_server")

# Initialize App
app = FastAPI(
    title="Hey Katya - The Judge",
    description="False Positive Rejection Server (Stage 2)",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    
    # Suppress harmless Windows asyncio errors
    if sys.platform == "win32":
        loop = asyncio.get_running_loop()
        def handle_exception(loop, context):
            exception = context.get("exception")
            if isinstance(exception, ConnectionResetError):
                if getattr(exception, 'winerror', 0) == 10054:
                    return
            if "_ProactorBasePipeTransport" in context.get("message", ""):
                return
            loop.default_exception_handler(context)
        loop.set_exception_handler(handle_exception)

    logger.info(f"Starting server. Loading model from {MODEL_PATH} on {DEVICE}...")
    try:
        engine = InferenceEngine(model_path=MODEL_PATH, device=DEVICE)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't raise error here to allow server to start, 
        # but /verify will fail if engine is not loaded
        
@app.get("/health")
async def health_check():
    if engine is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy", "device": DEVICE}

@app.post("/verify")
async def verify_audio(file: UploadFile = File(...)):
    """
    Verify if the audio clip contains the wakeword.
    Expects raw PCM16 or WAV file (processed as bytes).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
        
    try:
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
             raise HTTPException(status_code=400, detail="Empty file received")
             
        result = engine.predict(audio_bytes)
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))