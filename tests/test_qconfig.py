
import torch
import torch.quantization

def test_qconfig():
    print(f"Supported engines: {torch.backends.quantized.supported_engines}")
    
    try:
        qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        print("Successfully got qnnpack qconfig")
        print(qconfig)
    except Exception as e:
        print(f"Failed to get qnnpack qconfig: {e}")

    try:
        torch.backends.quantized.engine = 'qnnpack'
        print("Successfully set engine to qnnpack")
    except RuntimeError as e:
        print(f"Failed to set engine to qnnpack: {e}")

if __name__ == "__main__":
    test_qconfig()
