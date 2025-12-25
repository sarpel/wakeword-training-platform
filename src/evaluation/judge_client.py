
import requests
import io
import time
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger(__name__)

class JudgeClient:
    """
    Client for interacting with the Distributed Cascade Judge server.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def verify_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Send audio to the Judge server for secondary verification.
        """
        url = f"{self.base_url}/verify"
        
        # Convert numpy audio to WAV in-memory
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
            
        start_time = time.perf_counter()
        try:
            files = {"file": ("audio.wav", buffer, "audio/wav")}
            response = requests.post(url, files=files, headers=headers, timeout=10)
            
            end_time = time.perf_counter()
            network_latency = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                result["network_latency_ms"] = network_latency
                return result
            else:
                return {
                    "error": f"Judge server returned {response.status_code}",
                    "detail": response.text,
                    "network_latency_ms": network_latency
                }
        except Exception as e:
            logger.error(f"Failed to connect to Judge server: {e}")
            return {"error": str(e), "network_latency_ms": (time.perf_counter() - start_time) * 1000}

    def check_health(self) -> bool:
        """Check if the Judge server is alive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
