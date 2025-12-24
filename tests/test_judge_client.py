
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.evaluation.judge_client import JudgeClient

@patch("requests.post")
def test_judge_client_verify(mock_post):
    """Verify that JudgeClient sends audio correctly and handles the response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"verified": True, "confidence": 0.98}
    mock_post.return_value = mock_response
    
    client = JudgeClient("http://localhost:8000", api_key="test_key")
    audio = np.random.randn(16000) # 1s of noise
    
    result = client.verify_audio(audio)
    
    assert result["verified"] == True
    assert result["confidence"] == 0.98
    assert "network_latency_ms" in result
    
    # Check that requests.post was called with headers
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-API-Key"] == "test_key"
    assert "file" in kwargs["files"]

@patch("requests.get")
def test_judge_client_health(mock_get):
    """Verify health check logic."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    
    client = JudgeClient("http://localhost:8000")
    assert client.check_health() == True
    
    mock_response.status_code = 500
    assert client.check_health() == False
