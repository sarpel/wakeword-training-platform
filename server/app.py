import os
import sys
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.logger import get_logger as setup_logger

# Import InferenceEngine
try:
    from inference_engine import InferenceEngine
except ImportError:
    from .inference_engine import InferenceEngine

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
DEVICE = os.getenv("DEVICE", "cpu")
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    ALLOWED_ORIGINS = ["http://localhost"] # Default to safe local origin

# Logging
logger = setup_logger("wakeword_server")

# Initialize App
app = FastAPI(
    title="Hey Katya - The Judge",
    description="False Positive Rejection Server (Stage 2)",
    version="1.0.0"
)

# Security Scheme
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not API_KEY:
        logger.warning("API_KEY is not set. Denying request for security.")
        raise HTTPException(status_code=503, detail="Server authentication not configured")

    # If a key is configured we require the Authorization header to be present and match
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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

@app.post("/verify", dependencies=[Depends(verify_api_key)])
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
        raise HTTPException(status_code=500, detail="Internal Server Error")