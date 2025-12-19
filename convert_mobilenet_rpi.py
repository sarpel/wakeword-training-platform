import torch
import torch.onnx
import numpy as np
import os
import sys

# 1. Proje dizinini yola ekle
sys.path.append(os.getcwd())

# --- DÜZELTME BURADA ---
# mobilenetv3.py yok, sınıf architectures.py içinde tanımlı.
try:
    from src.models.architectures import MobileNetV3
    print("✅ MobileNetV3 sınıfı 'architectures.py' içinden yüklendi.")
except ImportError as e:
    print(f"❌ HATA: Model sınıfı yüklenemedi. Detay: {e}")
    print("Lütfen 'src/models/architectures.py' dosyasının içinde 'MobileNetV3' sınıfı olduğundan emin ol.")
    sys.exit(1)

# --- AYARLAR ---
CHECKPOINT_PATH = "models/checkpoints/best_model.pt"
ONNX_PATH = "hey_katya_rpi.onnx"
TFLITE_OUTPUT_FOLDER = "tflite_rpi_output"

# Model Parametreleri (Senin RPi Heavy Config'inle AYNI)
MODEL_PARAMS = {
    "num_classes": 2,
    "hidden_size": 128,       # Config: 128
    "num_layers": 3,          # Config: 3
    "bidirectional": True,    # Config: True
    "dropout": 0.2,
    "tcn_num_channels": [64, 128, 256],
    "tcn_kernel_size": 5,
    "tcn_dropout": 0.2,
    "cddnn_hidden_layers": [128, 64],
    "cddnn_context_frames": 11,
    "cddnn_dropout": 0.2
}

# Ses Ayarları (HD Model - 64 Mels)
N_MELS = 64
N_FRAMES = 101 

def run_conversion():
    print("🚀 Raspberry Pi Modeli Dönüştürülüyor...")
    
    # 1. Modeli Başlat
    try:
        model = MobileNetV3(**MODEL_PARAMS)
    except TypeError as e:
        print(f"❌ HATA: Model parametrelerinde uyuşmazlık var: {e}")
        return

    # 2. Checkpoint Yükle
    print(f"📂 Checkpoint yükleniyor: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ DOSYA BULUNAMADI: {CHECKPOINT_PATH}")
        return

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        # State dict temizliği (model. ön eklerini kaldır)
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("✅ Ağırlıklar başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ HATA: Checkpoint yüklenirken sorun çıktı: {e}")
        return

    model.eval()

    # 3. ONNX Export
    dummy_input = torch.randn(1, N_MELS, N_FRAMES)
    
    print("🔄 ONNX'e çevriliyor...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            input_names=['input'],
            output_names=['output'],
            opset_version=13,
            dynamic_axes=None 
        )
        print(f"✅ ONNX dosyası oluşturuldu: {ONNX_PATH}")
    except Exception as e:
        print(f"❌ ONNX Export hatası: {e}")
        return

    # 4. Kalibrasyon Verisi
    calib_data = np.random.randn(1, N_MELS, N_FRAMES).astype(np.float32)
    np.save("calib_data_rpi.npy", calib_data)

    # 5. TFLite Dönüşümü
    print("⏳ TFLite'a çevriliyor (onnx2tf)...")
    
    # Klasör oluştur
    if not os.path.exists(TFLITE_OUTPUT_FOLDER):
        os.makedirs(TFLITE_OUTPUT_FOLDER)
        
    # Windows'ta 'onnx2tf' komutu bazen doğrudan çalışmaz, python -m ile çağırıyoruz
    # Ayrıca Windows'ta onnxsim hatası alabilirsin, -nos (no simplification) eklenebilir ama önce normal deneyelim.
    cmd = f"onnx2tf -i {ONNX_PATH} -o {TFLITE_OUTPUT_FOLDER} -oiqt -cind input calib_data_rpi.npy 0 1"
    
    ret = os.system(cmd)
    
    if ret == 0:
        print("\n" + "="*40)
        print(f"🎉 RPi MODELİ HAZIR!")
        print(f"📂 Dosya: {TFLITE_OUTPUT_FOLDER}/hey_katya_rpi_dynamic_range_quant.tflite")
        print("="*40)
    else:
        print("\n❌ onnx2tf komutu başarısız oldu.")
        print("Eğer 'onnx2tf' bulunamadı hatası alıyorsan: 'pip install onnx2tf tensorflow' yaptığından emin ol.")
        print("Windows'ta sorun yaşarsan bu işlemi WSL (Linux) ortamında yapman gerekebilir.")

if __name__ == "__main__":
    run_conversion()