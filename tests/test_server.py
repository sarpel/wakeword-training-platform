
import pytest
import torch
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.getcwd())

# Mock imports that might be missing in some environments
sys.modules["src.models.huggingface"] = MagicMock()

from server.inference_engine import InferenceEngine
from server.app import app, verify_api_key

class TestServer:
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock(spec=torch.nn.Module)
        model.eval.return_value = None
        model.to.return_value = model
        # Mock forward pass: return logits [batch, 2]
        # Class 1 is wakeword
        model.return_value = torch.tensor([[0.1, 0.9]]) 
        return model

    @patch("server.inference_engine.Wav2VecWakeword")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_inference_engine_init(self, mock_exists, mock_load, mock_wav2vec_cls, mock_model):
        """Verify InferenceEngine initializes and loads model"""
        mock_exists.return_value = True
        mock_load.return_value = {"model_state_dict": {}}
        mock_wav2vec_cls.return_value = mock_model
        
        engine = InferenceEngine(model_path="dummy.pth", device="cpu")
        
        assert engine.model is mock_model
        mock_model.to.assert_called_with("cpu")
        mock_model.eval.assert_called()

    @patch("server.inference_engine.Wav2VecWakeword")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_inference_engine_predict(self, mock_exists, mock_load, mock_wav2vec_cls, mock_model):
        """Verify predict method returns correct structure"""
        mock_exists.return_value = True
        mock_load.return_value = {}
        mock_wav2vec_cls.return_value = mock_model
        
        engine = InferenceEngine(model_path="dummy.pth", device="cpu")
        
        # Mock preprocess to return a tensor
        with patch.object(engine, "preprocess", return_value=torch.randn(1, 16000)):
            # Mock model output for high confidence wakeword
            mock_model.return_value = torch.tensor([[ -2.0, 2.0 ]]) # Softmax -> ~0.02, ~0.98
            
            result = engine.predict(b"dummy_audio_bytes")
            
            assert result["prediction"] == 1
            assert result["label"] == "wakeword"
            assert result["confidence"] > 0.5

    def test_app_health(self):
        """Verify /health endpoint"""
        client = TestClient(app)
        
        # Mock the global engine in app
        with patch("server.app.engine", MagicMock()):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
            
        # Test unhealthy
        with patch("server.app.engine", None):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "unhealthy"

    def test_app_verify_no_auth(self):
        """Verify /verify endpoint requires auth"""
        client = TestClient(app)
        response = client.post("/verify")
        # Should be 403 because we didn't provide header, 
        # OR 503 if API_KEY not set in env (default behavior of verify_api_key mock needed)
        
        # In test env, API_KEY might be None.
        # Let's see what verify_api_key does.
        # It raises 503 if API_KEY is not set.
        
        # We expect some error code
        assert response.status_code in [403, 503]

    @patch("server.app.engine")
    def test_app_verify_success(self, mock_engine):
        """Verify /verify endpoint success flow"""
        # Mock API Key
        with patch("server.app.API_KEY", "test_secret"):
            client = TestClient(app)
            
            # Mock engine prediction
            mock_engine.predict.return_value = {"prediction": 1, "confidence": 0.99}
            
            # Send request with auth
            response = client.post(
                "/verify",
                headers={"Authorization": "Bearer test_secret"},
                files={"file": ("audio.wav", b"dummy_content", "audio/wav")}
            )
            
            assert response.status_code == 200
            assert response.json()["prediction"] == 1
