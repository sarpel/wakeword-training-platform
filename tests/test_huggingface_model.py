import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.models.huggingface import Wav2VecWakeword
    print("Successfully imported Wav2VecWakeword")
except ImportError as e:
    print(f"Failed to import Wav2VecWakeword: {e}")
    sys.exit(1)

def test_model():
    print("Initializing model (this may take a moment to download weights)...")
    # Use a smaller model or just config for testing speed if possible, 
    # but here we test the default to ensure it works.
    # To avoid downloading 300MB+ during test, we can use pretrained=False if we just want to test architecture.
    # But the user might want to see it actually work. 
    # Let's use pretrained=False for speed and just verify architecture connectivity.
    
    model = Wav2VecWakeword(num_classes=2, pretrained=False)
    model.eval()
    
    # Create dummy input: (batch_size, samples)
    # 1.5 seconds at 16kHz = 24000 samples
    batch_size = 2
    samples = 24000
    dummy_input = torch.randn(batch_size, samples)
    
    print(f"Testing forward pass with input shape {dummy_input.shape}...")
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, 2)
    if output.shape == expected_shape:
        print("Test PASSED: Output shape matches expected.")
    else:
        print(f"Test FAILED: Expected {expected_shape}, got {output.shape}")
        sys.exit(1)

if __name__ == "__main__":
    test_model()
