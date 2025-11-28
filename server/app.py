import os
import sys
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# Import InferenceEngine
try:
    from inference_engine import InferenceEngine
except ImportError:
    from .inference_engine import InferenceEngine

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
DEVICE = os.getenv("DEVICE", "cpu")
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 5 * 1024 * 1024)) # 5MB limit default

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

# Security Scheme
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not API_KEY:
        # Fixed: Security Vulnerability (Missing Authentication)
        # If API_KEY is not set in environment, we must strictly deny access
        # unless explicitly configured to allow anonymous (not recommended for production).
        # Assuming secure-by-default.
        logger.warning("API_KEY is not set! Rejecting all requests for security.")
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY missing")

    if credentials.credentials != API_KEY:
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
        # Fixed: Input Validation (DoS Prevention)
        # Check content length if available, though manual read check is safer for chunked uploads
        # We'll read with a limit to prevent memory exhaustion (OOM)
        audio_bytes = await file.read(MAX_FILE_SIZE + 1)
        
        if len(audio_bytes) > MAX_FILE_SIZE:
             raise HTTPException(status_code=413, detail=f"File too large. Max size is {MAX_FILE_SIZE} bytes")

        if len(audio_bytes) == 0:
             raise HTTPException(status_code=400, detail="Empty file received")
             
        # Fixed: Blocking I/O in Async Path
        # engine.predict runs CPU/GPU operations which block the event loop.
        # We must run this in a thread pool.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, engine.predict, audio_bytes)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fixed: Information Leakage
        # Do not return raw exception details to client in production
        raise HTTPException(status_code=500, detail="Internal processing error")