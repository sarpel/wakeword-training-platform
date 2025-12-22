import torch
import pytest
from src.models.architectures import create_model

def test_create_conformer():
    # This should fail initially as 'conformer' is not supported
    try:
        model = create_model("conformer", num_classes=2)
        assert model is not None
    except ValueError as e:
        pytest.fail(f"Conformer model should be supported: {e}")

def test_conformer_forward():
    model = create_model("conformer", num_classes=2, input_size=40)
    # (batch, time, features)
    test_input = torch.randn(2, 50, 40)
    output = model(test_input)
    assert output.shape == (2, 2)

def test_conformer_embed():
    model = create_model("conformer", num_classes=2, input_size=40)
    test_input = torch.randn(2, 50, 40)
    embedding = model.embed(test_input)
    assert embedding.ndim == 2
    assert embedding.shape[0] == 2
