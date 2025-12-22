import pytest
import torch
import sys
import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from server.inference_engine import InferenceEngine
from server.app import app
from src.config.defaults import WakewordConfig

class TestServer:
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock(spec=torch.nn.Module)
        model.eval.return_value = None
        model.to.return_value = model
        # Mock forward pass: return logits [batch, 2]
        model.return_value = torch.tensor([[0.1, 0.9]]) 
        model.state_dict.return_value = {}
        return model

    @pytest.fixture
    def mock_checkpoint(self):
        config = WakewordConfig()
        return {
            "model_state_dict": {},
            "config": config.to_dict()
        }

    @patch("server.inference_engine.create_model")
    @patch("server.inference_engine.AudioProcessor")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_inference_engine_init(self, mock_exists, mock_load, mock_processor_cls, mock_create_model, mock_model, mock_checkpoint):
        """Verify InferenceEngine initializes and loads model using standardized factory"""
        mock_exists.return_value = True
        mock_load.return_value = mock_checkpoint
        mock_create_model.return_value = mock_model
        
        engine = InferenceEngine(model_path="dummy.pt", device="cpu")
        
        assert engine.model is mock_model
        mock_model.to.assert_called_with("cpu")
        mock_model.eval.assert_called()
        mock_create_model.assert_called()

    @patch("server.inference_engine.create_model")
    @patch("server.inference_engine.AudioProcessor")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_inference_engine_predict(self, mock_exists, mock_load, mock_processor_cls, mock_create_model, mock_model, mock_checkpoint):
        """Verify predict method returns correct structure with standardized engine"""
        mock_exists.return_value = True
        mock_load.return_value = mock_checkpoint
        mock_create_model.return_value = mock_model
        
        engine = InferenceEngine(model_path="dummy.pt", device="cpu")
        
        # Mock preprocess to return a tensor
        with patch.object(engine, "preprocess", return_value=torch.randn(1, 1, 40, 101)):
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
        """Verify /verify endpoint requires auth when API_KEY is set"""
        # In the app, verify_api_key raises 503 if API_KEY is not set, 403 if invalid
        with patch("server.app.API_KEY", "secret"):
            client = TestClient(app)
            response = client.post("/verify")
            assert response.status_code == 403

    @patch("server.app.engine")
    def test_app_verify_success(self, mock_engine):
        """Verify /verify endpoint success flow"""
        # Mock API Key
        with patch("server.app.API_KEY", "test_secret"):
            client = TestClient(app)
            
            # Mock engine prediction
            mock_engine.predict.return_value = {"prediction": 1, "confidence": 0.99, "label": "wakeword"}
            
            # Send request with auth
            headers = {"Authorization": "Bearer test_secret"}
            files = {"file": ("test.wav", b"audio_content", "audio/wav")}
            response = client.post("/verify", headers=headers, files=files)
            
            assert response.status_code == 200
            assert response.json()["prediction"] == 1