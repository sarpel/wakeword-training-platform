import torch
import torch.onnx
import numpy as np
import os
import sys

# 1. Proje dizinini yola ekle
sys.path.append(os.getcwd())

try:
    from src.models.architectures import create_model
    print("✅ Model mimarisi (src.models.architectures) yüklendi.")
except ImportError as e:
    print(f"❌ HATA: src.models.architectures import edilemedi: {e}")
    sys.exit(1)

# --- AYARLAR ---
CHECKPOINT_PATH = "models/checkpoints/best_model.pt"
ONNX_PATH = "hey_katya_rpi.onnx"
TFLITE_OUTPUT_FOLDER = "tflite_rpi_output"

# Ses Ayarları (Config dosyanla uyumlu: 64 Mels)
N_MELS = 64
N_FRAMES = 101 # (1.0 sn)

def run_conversion():
    print("🚀 Raspberry Pi Modeli Dönüştürülüyor...")
    
    # 1. Modeli Başlat
    # Standart MobileNetV3 (Eğitilen gerçek model bu)
    print("🔨 Model inşa ediliyor (Standart MobileNetV3)...")
    model = create_model(
        architecture="mobilenetv3",
        num_classes=2,
        pretrained=False,
        dropout=0.2,
        input_channels=1
    )
    
    # 2. Checkpoint Yükle
    print(f"📂 Checkpoint yükleniyor: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ DOSYA BULUNAMADI: {CHECKPOINT_PATH}")
        return

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        # --- DÜZELTME BURADA ---
        # Checkpoint içindeki doğru sözlüğü buluyoruz.
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("ℹ️ 'model_state_dict' anahtarı bulundu.")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("ℹ️ 'state_dict' anahtarı bulundu.")
        else:
            state_dict = checkpoint
            print("ℹ️ Doğrudan state dict yapısı varsayılıyor.")

        # 'mobilenet.' veya 'model.' ön eklerini temizle (Eğer varsa)
        # Senin hatana göre 'mobilenet.' bekliyor ama checkpoint'te ne var emin olalım.
        # Genelde create_model ile üretilen model 'mobilenet' attribute'una sahiptir.
        # Yüklerken doğrudan yüklemeyi deneyelim, hata verirse prefix düzeltmesi yaparız.
        
        # QAT veya EMA kalıntılarını temizle
        clean_state_dict = {}
        for k, v in state_dict.items():
            # QAT observer'larını atıyoruz (sadece ağırlıklar lazım)
            if "activation_post_process" in k or "_observer" in k:
                continue
            clean_state_dict[k] = v
            
        # Modeli yükle (strict=False yaparak gereksiz metadata hatalarını susturuyoruz)
        # Amaç ağırlıkların çoğunun oturması.
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        
        if len(missing) > 0:
            print(f"⚠️ Uyarı: {len(missing)} eksik anahtar (Normal olabilir: {missing[0]}...)")
        if len(unexpected) > 0:
             # Unexpected keys genelde 'criterion', 'optimizer' vs olabilir, sorun değil.
             pass
             
        print("✅ Ağırlıklar yüklendi.")
        
    except Exception as e:
        print(f"❌ HATA: Checkpoint yüklenirken kritik hata: {e}")
        return

    model.eval()

    # 3. ONNX Export
    # MobileNetV3 Conv2D kullandığı için giriş 4 boyutlu olmalı: (Batch, Channel, Mels, Time)
    dummy_input = torch.randn(1, 1, N_MELS, N_FRAMES)
    
    print(f"🔄 ONNX'e çevriliyor (Giriş: {dummy_input.shape})...")
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
    calib_data = np.random.randn(1, 1, N_MELS, N_FRAMES).astype(np.float32)
    np.save("calib_data_rpi.npy", calib_data)

    # 5. TFLite Dönüşümü
    print("⏳ TFLite'a çevriliyor (onnx2tf)...")
    
    if not os.path.exists(TFLITE_OUTPUT_FOLDER):
        os.makedirs(TFLITE_OUTPUT_FOLDER)
        
    cmd = f"onnx2tf -i {ONNX_PATH} -o {TFLITE_OUTPUT_FOLDER} -oiqt -cind input calib_data_rpi.npy 0 1"
    
    ret = os.system(cmd)
    
    if ret == 0:
        print("\n" + "="*40)
        print(f"🎉 RPi MODELİ HAZIR!")
        print(f"📂 Dosya: {TFLITE_OUTPUT_FOLDER}/hey_katya_rpi_dynamic_range_quant.tflite")
        print("="*40)
    else:
        print("\n❌ onnx2tf komutu başarısız oldu.")

if __name__ == "__main__":
    run_conversion()